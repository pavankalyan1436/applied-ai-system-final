"""
Turn a free-text request into a raw taste profile.

parse_request() tries the LLM first and falls back to a deterministic keyword
parser that uses the knowledge base. Either way the result still goes through
guardrails.validate_profile before the agent uses it.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.retriever import Retriever, stem, terms, tokenize

log = logging.getLogger("vibematch.parser")

HIGH_ENERGY_WORDS = {"upbeat", "energetic", "hype", "fast", "pumped", "loud", "intense", "banger", "bangers", "lively", "max", "maximum"}
LOW_ENERGY_WORDS = {"calm", "slow", "quiet", "soft", "mellow", "sleepy", "gentle", "low-key", "soothing"}
ALIASES = {"lo-fi": "lofi", "rnb": "r&b", "kpop": "k-pop", "jpop": "j-pop", "hiphop": "hip hop", "electronic": "edm"}
NEGATIONS = {"no", "not", "without", "nothing", "avoid", "less", "never"}

# How much each field contributes to keyword-parse confidence (sums to 1.0)
FIELD_WEIGHTS = {"genre": 0.3, "mood": 0.3, "energy": 0.25, "likes_acoustic": 0.15}


@dataclass
class ParseResult:
    profile: Dict
    method: str  # "llm" or "keyword"
    confidence: float
    notes: List[str] = field(default_factory=list)


def _find_label(text: str, names: List[str]) -> Optional[str]:
    """Longest label whose (stemmed) words appear in the text, e.g. 'indie pop' over 'pop'."""
    text_terms = terms(text)
    joined = " " + " ".join(text_terms) + " "
    for name in sorted(names, key=len, reverse=True):
        if " " + " ".join(terms(name)) + " " in joined:
            return name
    return None


def keyword_parse(text: str, retriever: Retriever) -> ParseResult:
    """Deterministic parser: exact label matches first, then retrieval, then word lists."""
    lowered = text.lower()
    for alias, real in ALIASES.items():
        lowered = re.sub(rf"\b{re.escape(alias)}\b", real, lowered)
    words = tokenize(lowered)
    notes: List[str] = []

    genres = [d.name for d in retriever.docs if d.type == "genre"]
    moods = [d.name for d in retriever.docs if d.type == "mood"]
    retrieved = retriever.retrieve(lowered, k=5)

    genre = _find_label(lowered, genres)
    mood = _find_label(lowered, moods)
    activity = next((d for d, _ in retrieved if d.type == "activity"), None)

    if not genre:
        hit = next(((d, s) for d, s in retrieved if d.type == "genre" and s >= 0.25), None)
        if hit:
            genre = hit[0].name
            notes.append(f"genre '{genre}' inferred from {hit[0].source}")
    if not mood:
        hit = next((d for d, s in retrieved if d.type == "mood" and s >= 0.25), None)
        if hit:
            mood = hit.name
            notes.append(f"mood '{mood}' inferred from {hit.source}")
        elif activity:
            mood = activity.related[0]
            notes.append(f"mood '{mood}' inferred from {activity.source}")

    energy = None
    stems = {stem(w) for w in words}
    if stems & {stem(w) for w in HIGH_ENERGY_WORDS} or "high energy" in lowered:
        energy = 0.85
    elif stems & {stem(w) for w in LOW_ENERGY_WORDS} or "low energy" in lowered:
        energy = 0.25
    elif activity and activity.energy is not None:
        energy = activity.energy
        notes.append(f"energy {energy} from {activity.source}")

    likes_acoustic = None
    if "acoustic" in words:
        i = words.index("acoustic")
        likes_acoustic = not any(w in NEGATIONS for w in words[max(0, i - 3): i])
    elif activity and activity.acoustic is not None:
        likes_acoustic = activity.acoustic

    profile = {"genre": genre, "mood": mood, "energy": energy, "likes_acoustic": likes_acoustic}
    confidence = sum(w for key, w in FIELD_WEIGHTS.items() if profile[key] not in (None, ""))
    return ParseResult(profile, "keyword", round(confidence, 2), notes)


def parse_request(text: str, retriever: Retriever, llm=None) -> ParseResult:
    """LLM parse when available; keyword parse if there is no LLM or it fails."""
    if llm is not None:
        genres = sorted({d.name for d in retriever.docs if d.type == "genre"})
        moods = sorted({d.name for d in retriever.docs if d.type == "mood"})
        try:
            data = llm.parse_request(text, genres, moods)
            conf = data.pop("confidence", 0.7)
            try:
                conf = min(1.0, max(0.0, float(conf)))
            except (TypeError, ValueError):
                conf = 0.5
            return ParseResult(data, "llm", round(conf, 2), ["parsed by Gemini"])
        except Exception as exc:
            log.warning("LLM parse failed (%s: %s); falling back to keyword parser",
                        type(exc).__name__, str(exc).splitlines()[0][:80] if str(exc) else "")
            result = keyword_parse(text, retriever)
            result.notes.insert(0, f"LLM parse failed ({type(exc).__name__}); used keyword fallback")
            return result
    return keyword_parse(text, retriever)
