"""
VibeMatch agent: orchestrates the full plan -> act -> check loop.

    request
      -> 1. input guardrail            (block unsafe / malformed input)
      -> 2. parse (Gemini or keyword)  (free text -> taste profile)
      -> 3. validate profile           (repair bad values, record fixes)
      -> 4. retrieve (RAG)             (knowledge notes -> related labels, hints)
      -> 5. clarify?                   (human-in-the-loop when confidence is low)
      -> 6. score + verify, retry      (up to 3 strategies; keep the best attempt)
      -> 7. explain + check            (LLM explanation, rejected if it hallucinates)
      -> 8. confidence + log           (0-1 score; JSON line in logs/vibematch.jsonl)

Every step appends to `trace`, which the CLI and UI show as "how the AI decided".
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.guardrails import check_input, hallucinated_titles, unknown_titles, validate_profile
from src.llm_client import get_llm
from src.parser import parse_request
from src.recommender import DEFAULT_WEIGHTS, load_songs, recommend_songs
from src.retriever import Retriever, build_context

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "logs" / "vibematch.jsonl"
log = logging.getLogger("vibematch.agent")

MAX_SCORE = 6.0
CLARIFY_THRESHOLD = 0.45  # parse confidence below this -> ask the user instead of guessing
ENERGY_TOLERANCE = 0.30  # a recommendation "fits" if its energy is within this of the target
MIN_TOP_SCORE = 3.0  # verification: best song should earn at least half the points
DEFAULTS = {"energy": 0.5, "likes_acoustic": False}

# Retry strategies the agent can switch to when verification fails
STRATEGIES = [
    {"name": "baseline", "weights": DEFAULT_WEIGHTS, "k": 3, "min_score": 0.15},
    {"name": "broaden_retrieval", "weights": DEFAULT_WEIGHTS, "k": 6, "min_score": 0.08},
    {
        "name": "prioritize_energy",
        "weights": {
            **DEFAULT_WEIGHTS,
            "energy": DEFAULT_WEIGHTS["energy"] * 2,
            "genre": DEFAULT_WEIGHTS["genre"] / 2,
            "mood": DEFAULT_WEIGHTS["mood"] / 2,
        },
        "k": 6,
        "min_score": 0.08,
    },
]


@dataclass
class AgentResult:
    request: str
    status: str  # "ok", "blocked", or "needs_clarification"
    message: str = ""
    profile: Dict = field(default_factory=dict)
    recommendations: List[Dict] = field(default_factory=list)
    explanation: str = ""
    explanation_source: str = ""  # "llm" or "template"
    confidence: float = 0.0
    checks: Dict[str, bool] = field(default_factory=dict)
    strategy: str = ""
    parse_method: str = ""
    trace: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class VibeMatchAgent:
    def __init__(self, songs: Optional[List[Dict]] = None, retriever: Optional[Retriever] = None, llm="auto", k: int = 5):
        self.songs = songs if songs is not None else load_songs(str(ROOT / "data" / "songs.csv"))
        self.retriever = retriever or Retriever()
        self.llm = get_llm() if llm == "auto" else llm
        self.k = k
        self.known_genres = {d.name for d in self.retriever.docs if d.type == "genre"}
        self.known_moods = {d.name for d in self.retriever.docs if d.type == "mood"}

    # ------------------------------------------------------------------ run
    def run(self, request: str, profile_overrides: Optional[Dict] = None, exclude: Iterable[str] = ()) -> AgentResult:
        result = AgentResult(request=request or "", status="ok")
        trace = result.trace

        # 1. Input guardrail
        gate = check_input(request)
        trace.append({"step": "input_guardrail", "ok": gate.ok, "reason": gate.reason})
        if not gate.ok:
            result.status, result.message = "blocked", gate.reason
            return self._finish(result)

        # 2. Parse
        parsed = parse_request(gate.cleaned, self.retriever, self.llm)
        result.parse_method = parsed.method
        trace.append({"step": "parse", "method": parsed.method, "raw_profile": parsed.profile,
                      "confidence": parsed.confidence, "notes": parsed.notes})

        # 3. Validate (and apply any human feedback overrides)
        checked = validate_profile(parsed.profile, self.known_genres, self.known_moods)
        profile = checked.profile
        if profile_overrides:
            profile.update({k: v for k, v in profile_overrides.items() if v is not None})
        trace.append({"step": "validate_profile", "profile": dict(profile), "fixes": checked.fixes})
        parse_conf = parsed.confidence
        # A label the knowledge base has never heard of is a sign the parse was a guess
        if profile["mood"] and profile["mood"] not in self.known_moods:
            parse_conf *= 0.5
            profile["mood"] = ""

        # 4. Retrieve
        ctx = build_context(self.retriever, profile, gate.cleaned)
        trace.append({"step": "retrieve", "retrieved": ctx.retrieved, "sources": ctx.sources,
                      "related_genres": sorted(ctx.related_genres), "related_moods": sorted(ctx.related_moods),
                      "energy_hint": ctx.energy_hint})
        filled = []
        energy_known = profile["energy"] is not None or ctx.energy_hint is not None
        if profile["energy"] is None:
            profile["energy"] = ctx.energy_hint if ctx.energy_hint is not None else DEFAULTS["energy"]
            filled.append(f"energy={profile['energy']} ({'retrieved' if ctx.energy_hint is not None else 'default'})")
        if profile["likes_acoustic"] is None:
            profile["likes_acoustic"] = ctx.acoustic_hint if ctx.acoustic_hint is not None else DEFAULTS["likes_acoustic"]
            filled.append(f"likes_acoustic={profile['likes_acoustic']} ({'retrieved' if ctx.acoustic_hint is not None else 'default'})")
        if filled:
            trace.append({"step": "fill_missing", "filled": filled})
        result.profile = profile

        # 5. Human-in-the-loop: ask instead of guessing when the request is too vague
        if not profile_overrides and (parse_conf < CLARIFY_THRESHOLD or not (profile["genre"] or profile["mood"])):
            result.status = "needs_clarification"
            result.message = self._clarifying_question(profile)
            result.confidence = round(parse_conf, 2)
            trace.append({"step": "clarify", "question": result.message, "parse_confidence": parse_conf})
            return self._finish(result)

        # 6. Score + verify, retrying with a different strategy on failure
        pool = [s for s in self.songs if s["title"] not in set(exclude)]
        best = None
        for strategy in STRATEGIES:
            ctx_s = build_context(self.retriever, profile, gate.cleaned, k=strategy["k"], min_score=strategy["min_score"])
            ranked = recommend_songs(profile, pool, k=self.k, weights=strategy["weights"], context=ctx_s)
            recs = [{**{key: s[key] for key in ("id", "title", "artist", "genre", "mood", "energy", "acousticness")},
                     "score": round(sc, 2), "reasons": why} for s, sc, why in ranked]
            checks = self._verify(recs, profile, energy_known)
            trace.append({"step": "score_and_verify", "strategy": strategy["name"], "checks": checks,
                          "top": [(r["title"], r["score"]) for r in recs[:3]]})
            # Rank attempts: keeping the user's requested label outranks every other
            # check, then the number of checks passed, then the top score.
            key = (checks["keeps_requested_label"], sum(checks.values()), recs[0]["score"] if recs else 0)
            if best is None or key > best[0]:
                best = (key, strategy["name"], recs, checks)
            if all(checks.values()):
                break
        _, result.strategy, result.recommendations, result.checks = best

        # 7. Explain, then check the explanation for hallucinated songs
        result.explanation, result.explanation_source = self._explain(gate.cleaned, result.recommendations, trace)

        # 8. Confidence: parse certainty + how well results scored + how many checks passed
        top = result.recommendations[0]["score"] / MAX_SCORE if result.recommendations else 0
        check_frac = sum(result.checks.values()) / max(1, len(result.checks))
        result.confidence = round(0.4 * parse_conf + 0.3 * top + 0.3 * check_frac, 2)
        if not all(result.checks.values()):
            failed = [name for name, ok in result.checks.items() if not ok]
            result.message = f"Best effort: these checks did not pass after retries: {', '.join(failed)}."
        return self._finish(result)

    # -------------------------------------------------------------- helpers
    def _verify(self, recs: List[Dict], profile: Dict, energy_known: bool = True) -> Dict[str, bool]:
        """
        Automatic checks on the agent's own output. Energy checks are skipped
        (count as passed) when the energy target is only a default the user
        never asked for.
        """
        fits = [not energy_known or abs(r["energy"] - profile["energy"]) <= ENERGY_TOLERANCE for r in recs]
        # A retry must not drop what the user asked for. The genre is the most explicit
        # request, so if the catalog has it, a song of that genre must survive; only when
        # it doesn't is a mood match enough. (Accepting either label let a Gemini-parsed
        # "folk + intense" request return zero folk songs; see README Testing summary.)
        if profile.get("genre") in {s["genre"] for s in self.songs}:
            keeps_label = any(r["genre"] == profile["genre"] for r in recs)
        elif profile.get("mood") in {s["mood"] for s in self.songs}:
            keeps_label = any(r["mood"] == profile["mood"] for r in recs)
        else:
            keeps_label = True
        return {
            "keeps_requested_label": keeps_label,
            "all_songs_in_catalog": not unknown_titles(recs, self.songs),
            "enough_results": len(recs) >= min(self.k, len(self.songs)),
            "energy_fits_majority": sum(fits) > len(fits) / 2,
            "top_pick_energy_fits": bool(fits) and fits[0],
            "top_score_strong": bool(recs) and recs[0]["score"] >= MIN_TOP_SCORE,
        }

    def _explain(self, request: str, recs: List[Dict], trace: List[Dict]):
        if self.llm is not None and recs:
            try:
                text = self.llm.explain(request, recs)
                bad = hallucinated_titles(text, recs, self.songs)
                mentions_any = any(r["title"].lower() in text.lower() for r in recs)
                ok = bool(text) and not bad and mentions_any
                trace.append({"step": "explanation_check", "source": "llm", "ok": ok, "hallucinated": bad,
                              "mentions_recommended": mentions_any})
                if ok:
                    return text, "llm"
            except Exception as exc:
                trace.append({"step": "explanation_check", "source": "llm", "ok": False, "error": type(exc).__name__})
        return self._template_explanation(recs), "template"

    @staticmethod
    def _template_explanation(recs: List[Dict]) -> str:
        if not recs:
            return "No songs matched."
        top = recs[0]
        others = ", ".join(r["title"] for r in recs[1:4])
        return (f"Top pick: {top['title']} by {top['artist']} ({top['score']:.2f}/6) because: {top['reasons']}."
                + (f" Also worth a listen: {others}." if others else ""))

    @staticmethod
    def _clarifying_question(profile: Dict) -> str:
        missing = []
        if not profile.get("genre"):
            missing.append("a genre or artist style (e.g. pop, lofi, rock)")
        if not profile.get("mood"):
            missing.append("a mood or activity (e.g. happy, chill, workout, studying)")
        ask = " and ".join(missing) or "a bit more detail"
        return f"I'm not confident I understood. Could you tell me {ask}?"

    def _finish(self, result: AgentResult) -> AgentResult:
        """Write one JSON line per run so every decision can be audited later."""
        try:
            LOG_PATH.parent.mkdir(exist_ok=True)
            entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **result.to_dict()}
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError as exc:  # logging must never crash the app
            log.warning("could not write run log: %s", exc)
        log.info("request=%r status=%s confidence=%.2f strategy=%s", result.request, result.status,
                 result.confidence, result.strategy)
        return result


def apply_feedback(profile: Dict, song: Dict, liked: bool) -> Dict:
    """
    Human-in-the-loop feedback: a thumbs-up pulls the target energy 30% of the
    way toward that song and adopts its genre/mood when the profile had none.
    A thumbs-down is handled by excluding the song on the next run.
    """
    new = dict(profile)
    if liked:
        new["energy"] = round(new["energy"] + 0.3 * (song["energy"] - new["energy"]), 2)
        if not new.get("genre"):
            new["genre"] = song["genre"]
        if not new.get("mood"):
            new["mood"] = song["mood"]
    return new
