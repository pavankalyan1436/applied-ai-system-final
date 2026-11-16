"""
Specialization experiment: does a few-shot prompt make Gemini a better parser?

Runs every held-out request in eval/parse_cases.json through
  - Gemini with the plain (zero-shot) prompt,
  - Gemini with the few-shot prompt (rules + 6 worked examples), and
  - the offline keyword parser (reference, deterministic),
and scores each output field against gold labels that list every acceptable
answer. None of the requests are few-shot examples.

    python -m src.eval_parser              # needs GEMINI_API_KEY; ~4 min (rate-limited)
    python -m src.eval_parser --repeat 1   # faster, no consistency measurement

Writes eval/report_parser.md and eval/results_parser.json.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"
FIELDS = ("genre", "mood", "energy", "likes_acoustic", "confidence")


def score_fields(gold: Dict, out: Optional[Dict], known_moods: set) -> Dict[str, Optional[bool]]:
    """
    Per-field correctness (None = field not graded for this case).
    genre/mood: value must be in the gold list (null allowed only if listed).
    energy: inside the gold [low, high] range. likes_acoustic: exact match.
    confidence: only graded when gold has max_confidence (vague requests).
    """
    if out is None:
        return {f: (False if (f in gold or (f == "confidence" and "max_confidence" in gold)) else None) for f in FIELDS}
    res: Dict[str, Optional[bool]] = {}
    for f in ("genre", "mood"):
        val = out.get(f)
        val = val.strip().lower() if isinstance(val, str) and val.strip() else None
        res[f] = (val in gold[f]) if f in gold else None
    energy = out.get("energy")
    if "energy" in gold:
        lo, hi = gold["energy"]
        res["energy"] = isinstance(energy, (int, float)) and lo <= float(energy) <= hi
    else:
        res["energy"] = None
    res["likes_acoustic"] = (out.get("likes_acoustic") is gold["likes_acoustic"]) if "likes_acoustic" in gold else None
    if "max_confidence" in gold:
        try:
            res["confidence"] = float(out.get("confidence", 1.0)) <= gold["max_confidence"]
        except (TypeError, ValueError):
            res["confidence"] = False
    else:
        res["confidence"] = None
    return res


def invented_mood(out: Optional[Dict], known_moods: set) -> bool:
    mood = (out or {}).get("mood")
    return isinstance(mood, str) and mood.strip() != "" and mood.strip().lower() not in known_moods


def normalize(out: Optional[Dict]) -> Optional[str]:
    """Comparable form of an output for consistency checks."""
    if out is None:
        return None
    e = out.get("energy")
    return json.dumps({"genre": out.get("genre"), "mood": out.get("mood"),
                       "energy": round(float(e), 1) if isinstance(e, (int, float)) else None,
                       "likes_acoustic": out.get("likes_acoustic")}, sort_keys=True)


def run_style(name: str, parse_fn, cases: List[Dict], known_moods: set, repeat: int) -> Dict:
    rows = []
    for case in cases:
        outputs, errors = [], []
        for _ in range(repeat):
            try:
                outputs.append(parse_fn(case["input"]))
            except Exception as exc:
                outputs.append(None)
                errors.append(f"{type(exc).__name__}: {str(exc)[:60]}")
        first = outputs[0]
        fields = score_fields(case["gold"], first, known_moods)
        graded = [v for v in fields.values() if v is not None]
        rows.append({
            "id": case["id"], "input": case["input"], "output": first, "fields": fields,
            "correct": sum(graded), "graded": len(graded), "all_correct": all(graded),
            "invented_mood": invented_mood(first, known_moods), "errors": errors,
            "consistent": len({normalize(o) for o in outputs}) == 1,
        })
    correct = sum(r["correct"] for r in rows)
    graded = sum(r["graded"] for r in rows)
    summary = {
        "style": name,
        "field_accuracy": round(correct / graded, 3) if graded else 0,
        "fields": f"{correct}/{graded}",
        "cases_all_correct": f"{sum(r['all_correct'] for r in rows)}/{len(rows)}",
        "invented_moods": sum(r["invented_mood"] for r in rows),
        "errors": sum(bool(r["errors"]) for r in rows),
        "consistency": f"{sum(r['consistent'] for r in rows)}/{len(rows)}" if repeat > 1 else "n/a",
        "per_field": {f: f"{sum(1 for r in rows if r['fields'][f])}/{sum(1 for r in rows if r['fields'][f] is not None)}"
                      for f in FIELDS},
    }
    return {"summary": summary, "rows": rows}


def short(out: Optional[Dict]) -> str:
    if out is None:
        return "error"
    e = out.get("energy")
    a = out.get("likes_acoustic")
    return (f"{out.get('genre') or '-'} / {out.get('mood') or '-'} / "
            f"{e if e is not None else '-'} / {'-' if a is None else ('ac' if a else 'no-ac')}")


def to_markdown(results: Dict[str, Dict], repeat: int) -> str:
    styles = list(results)
    lines = [
        "# Parser specialization: zero-shot vs few-shot Gemini",
        "",
        f"_Generated by `python -m src.eval_parser --repeat {repeat}` on {time.strftime('%Y-%m-%d')}. "
        f"16 held-out requests from `eval/parse_cases.json`; none are few-shot examples._",
        "",
        "## Summary",
        "",
        "| Metric | " + " | ".join(styles) + " |",
        "|---" * (len(styles) + 1) + "|",
    ]
    rows = [
        ("Field accuracy", lambda s: f"**{s['field_accuracy']:.0%}** ({s['fields']})"),
        ("Requests fully correct", lambda s: s["cases_all_correct"]),
        ("Invented (non-catalog) moods", lambda s: str(s["invented_moods"])),
        ("Run-to-run consistency", lambda s: s["consistency"]),
        ("API errors", lambda s: str(s["errors"])),
    ] + [(f"&nbsp;&nbsp;{f} correct", (lambda f: lambda s: s["per_field"][f])(f)) for f in FIELDS]
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join(fn(results[st]["summary"]) for st in styles) + " |")
    lines += [
        "",
        "## Per request (output shown as genre / mood / energy / acoustic; ✓ = every graded field correct)",
        "",
        "| ID | Input | " + " | ".join(styles) + " |",
        "|---|---|" + "---|" * len(styles),
    ]
    by_id = {st: {r["id"]: r for r in results[st]["rows"]} for st in styles}
    for case_id in by_id[styles[0]]:
        cells = []
        for st in styles:
            r = by_id[st][case_id]
            mark = "✓" if r["all_correct"] else f"✗ {r['correct']}/{r['graded']}"
            cells.append(f"{mark} `{short(r['output'])}`" + (" (invented mood)" if r["invented_mood"] else ""))
        lines.append(f"| {case_id} | \"{by_id[styles[0]][case_id]['input']}\" | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Compare zero-shot vs few-shot Gemini parsing")
    ap.add_argument("--repeat", type=int, default=2, help="calls per request per prompt (consistency)")
    args = ap.parse_args(argv)
    os.environ.setdefault("GEMINI_MIN_INTERVAL", "4.5")  # free tier: 15 requests/minute

    from src.llm_client import get_llm
    from src.parser import keyword_parse
    from src.retriever import Retriever

    llm = get_llm()
    if llm is None:
        raise SystemExit("This experiment needs GEMINI_API_KEY (Gemini is what is being compared).")
    retriever = Retriever()
    genres = sorted({d.name for d in retriever.docs if d.type == "genre"})
    moods = sorted({d.name for d in retriever.docs if d.type == "mood"})
    cases = json.loads((EVAL_DIR / "parse_cases.json").read_text())
    known_moods = set(moods)

    def keyword(text):
        r = keyword_parse(text, retriever)
        return {**r.profile, "confidence": r.confidence}

    results = {}
    for name, fn, rep in (
        ("zero-shot Gemini", lambda t: llm.parse_request(t, genres, moods, style="zeroshot"), args.repeat),
        ("few-shot Gemini", lambda t: llm.parse_request(t, genres, moods, style="fewshot"), args.repeat),
        ("keyword (offline)", keyword, 1),
    ):
        print(f"Running {name}...", flush=True)
        results[name] = run_style(name, fn, cases, known_moods, rep)

    (EVAL_DIR / "results_parser.json").write_text(json.dumps(results, indent=2, default=str))
    (EVAL_DIR / "report_parser.md").write_text(to_markdown(results, args.repeat))
    print()
    for name, res in results.items():
        s = res["summary"]
        print(f"{name:<18} field accuracy {s['field_accuracy']:.0%} ({s['fields']})  fully correct {s['cases_all_correct']}  "
              f"invented moods {s['invented_moods']}  consistency {s['consistency']}  errors {s['errors']}")
    print("\nReport: eval/report_parser.md")


if __name__ == "__main__":
    main()
