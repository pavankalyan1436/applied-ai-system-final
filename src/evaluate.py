"""
Reliability evaluation for VibeMatch.

Runs every labeled case in eval/cases.json through
  - the full agent (guardrails + RAG + verification), and
  - a baseline: the original Module 3 recommender (keyword-parsed profile,
    exact-match scoring, no retrieval, no guardrails, no verification),
then scores each against criteria that were written before the run.

    python -m src.evaluate                 # offline (deterministic, no API key)
    python -m src.evaluate --llm           # use Gemini for parsing/explanations
    python -m src.evaluate --llm --repeat 3  # also measure run-to-run consistency

Writes eval/results_<mode>.json and eval/report_<mode>.md (markdown tables).
"""

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"
LONG_INPUT = "pop " * 100  # stands in for "__LONG__" (400 chars, over the limit)


# ------------------------------------------------------------- criteria
def describe(expect: Dict) -> str:
    """Human-readable criteria for the report's Criteria column."""
    parts = []
    status = expect.get("status")
    parts.append("status " + (" or ".join(status) if isinstance(status, list) else status))
    if "top_genre_in" in expect:
        parts.append("#1 genre in {" + ", ".join(expect["top_genre_in"]) + "}")
    if "top3_genre_in" in expect:
        parts.append("top 3 include {" + ", ".join(expect["top3_genre_in"]) + "}")
    if "top_title" in expect:
        parts.append(f"#1 is {expect['top_title']}")
    if "top_energy_min" in expect:
        parts.append(f"#1 energy >= {expect['top_energy_min']}")
    if "top_energy_max" in expect:
        parts.append(f"#1 energy <= {expect['top_energy_max']}")
    if "top_acoustic_max" in expect:
        parts.append(f"#1 acousticness <= {expect['top_acoustic_max']}")
    if expect.get("all_checks_pass"):
        parts.append("all self-checks pass")
    if "flags_check" in expect:
        parts.append(f"flags '{expect['flags_check']}'")
    return "; ".join(parts)


def judge(expect: Dict, status: str, recs: List[Dict], checks: Dict[str, bool]) -> Tuple[bool, List[str]]:
    """Return (passed, list of failure reasons)."""
    fails = []
    allowed = expect["status"] if isinstance(expect["status"], list) else [expect["status"]]
    if status not in allowed:
        fails.append(f"status was {status}")
        return False, fails
    if status != "ok":
        return True, fails
    if not recs:
        return False, ["no recommendations"]
    top = recs[0]
    if "top_genre_in" in expect and top["genre"] not in expect["top_genre_in"]:
        fails.append(f"#1 genre was {top['genre']} ({top['title']})")
    if "top3_genre_in" in expect and not any(r["genre"] in expect["top3_genre_in"] for r in recs[:3]):
        fails.append("top 3 genres were " + ", ".join(r["genre"] for r in recs[:3]))
    if "top_title" in expect and top["title"] != expect["top_title"]:
        fails.append(f"#1 was {top['title']}")
    if "top_energy_min" in expect and top["energy"] < expect["top_energy_min"]:
        fails.append(f"#1 energy {top['energy']}")
    if "top_energy_max" in expect and top["energy"] > expect["top_energy_max"]:
        fails.append(f"#1 energy {top['energy']}")
    if "top_acoustic_max" in expect and top["acousticness"] > expect["top_acoustic_max"]:
        fails.append(f"#1 acousticness {top['acousticness']}")
    if expect.get("all_checks_pass") and not all(checks.values()):
        fails.append("failed checks: " + ", ".join(k for k, v in checks.items() if not v))
    if "flags_check" in expect and checks.get(expect["flags_check"], True):
        fails.append(f"did not flag {expect['flags_check']}")
    return not fails, fails


# -------------------------------------------------------------- runners
def run_baseline(text: str, songs, retriever) -> Tuple[str, List[Dict], Dict]:
    """Original Module 3 behavior: no guardrails, no retrieval, no checks."""
    from src.parser import keyword_parse
    from src.recommender import recommend_songs

    p = keyword_parse(text or "", retriever).profile
    prefs = {"genre": p["genre"] or "", "mood": p["mood"] or "",
             "energy": 0.5 if p["energy"] is None else p["energy"], "likes_acoustic": bool(p["likes_acoustic"])}
    recs = [{**s, "score": round(sc, 2)} for s, sc, _ in recommend_songs(prefs, songs, k=5)]
    return "ok", recs, {}


def load_cases(path: Path = EVAL_DIR / "cases.json") -> List[Dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    for c in cases:
        if c["input"] == "__LONG__":
            c["input"] = LONG_INPUT
    return cases


def evaluate(agent, cases: List[Dict], repeat: int = 1) -> Dict:
    rows = []
    for case in cases:
        runs = [agent.run(case["input"]) for _ in range(repeat)]
        r = runs[0]
        passed, fails = judge(case["expect"], r.status, r.recommendations, r.checks)
        b_status, b_recs, b_checks = run_baseline(case["input"], agent.songs, agent.retriever)
        b_passed, b_fails = judge(case["expect"], b_status, b_recs, b_checks)
        tops = [tuple(x["title"] for x in run.recommendations[:3]) for run in runs]
        statuses = {run.status for run in runs}
        rows.append({
            "id": case["id"], "category": case["category"], "input": case["input"],
            "criteria": describe(case["expect"]),
            "passed": passed, "fail_reasons": fails,
            "status": r.status, "top": r.recommendations[0]["title"] if r.recommendations else "",
            "confidence": r.confidence, "strategy": r.strategy, "parse_method": r.parse_method,
            "explanation_source": r.explanation_source,
            "retries": max(0, sum(1 for s in r.trace if s["step"] == "score_and_verify") - 1),
            "hallucination_rejected": any(s["step"] == "explanation_check" and s.get("hallucinated") for s in r.trace),
            "llm_errors": sum(1 for s in r.trace if (s["step"] == "explanation_check" and s.get("error"))
                              or (s["step"] == "parse" and any("fallback" in n for n in s["notes"]))),
            "consistent": len(statuses) == 1 and len(set(tops)) == 1,
            "baseline_passed": b_passed, "baseline_fail_reasons": b_fails,
            "baseline_top": b_recs[0]["title"] if b_recs else "",
        })
    return {"rows": rows, "summary": summarize(rows, repeat)}


def summarize(rows: List[Dict], repeat: int) -> Dict:
    def rate(subset, key="passed"):
        return f"{sum(r[key] for r in subset)}/{len(subset)}"

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    ok_rows = [r for r in rows if r["status"] == "ok"]
    conf_pass = [r["confidence"] for r in ok_rows if r["passed"]]
    conf_fail = [r["confidence"] for r in ok_rows if not r["passed"]]
    summary = {
        "agent_pass": rate(rows), "baseline_pass": rate(rows, "baseline_passed"),
        "by_category": {c: {"agent": rate(v), "baseline": rate(v, "baseline_passed")} for c, v in by_cat.items()},
        "avg_confidence_ok": round(statistics.mean([r["confidence"] for r in ok_rows]), 2) if ok_rows else None,
        "avg_confidence_passed": round(statistics.mean(conf_pass), 2) if conf_pass else None,
        "avg_confidence_failed": round(statistics.mean(conf_fail), 2) if conf_fail else None,
        "cases_with_retries": sum(r["retries"] > 0 for r in rows),
        "hallucinations_caught": sum(r["hallucination_rejected"] for r in rows),
        "llm_errors_recovered": sum(r["llm_errors"] for r in rows),
        "llm_parses": sum(r["parse_method"] == "llm" for r in rows),
        "keyword_parses": sum(r["parse_method"] == "keyword" for r in rows),
    }
    if repeat > 1:
        summary["consistency"] = f"{sum(r['consistent'] for r in rows)}/{len(rows)} identical across {repeat} runs"
    return summary


# --------------------------------------------------------------- report
def to_markdown(result: Dict, mode: str) -> str:
    s = result["summary"]
    short = lambda t: (t[:45] + "...") if len(t) > 48 else (t or "(empty)")
    lines = [
        f"# VibeMatch evaluation ({mode})",
        "",
        f"_Generated by `python -m src.evaluate{' --llm' if mode == 'gemini' else ''}` on {time.strftime('%Y-%m-%d')}._",
        "",
        "## Summary",
        "",
        "| Metric | Agent | Baseline (Module 3) |",
        "|---|---|---|",
        f"| Cases passed | {s['agent_pass']} | {s['baseline_pass']} |",
    ]
    for cat, v in s["by_category"].items():
        lines.append(f"| {cat} | {v['agent']} | {v['baseline']} |")
    lines += [
        "",
        f"- Average confidence on answered requests: {s['avg_confidence_ok']} "
        f"(passed cases {s['avg_confidence_passed']}, failed cases {s['avg_confidence_failed']})",
        f"- Cases where verification triggered a retry: {s['cases_with_retries']}",
        f"- LLM explanations rejected by the hallucination check: {s['hallucinations_caught']}",
        f"- LLM call failures recovered by fallbacks (first run of each case): {s['llm_errors_recovered']}",
        f"- Parser used: LLM {s['llm_parses']}, keyword {s['keyword_parses']} (guardrail-blocked cases are never parsed)",
    ]
    if "consistency" in s:
        lines.append(f"- Consistency: {s['consistency']}")
    lines += [
        "",
        "## Results",
        "",
        "| ID | Input | Evaluation criteria | Agent result | Conf. | Baseline |",
        "|---|---|---|---|---|---|",
    ]
    for r in result["rows"]:
        agent_cell = ("Pass" if r["passed"] else "Fail: " + "; ".join(r["fail_reasons"])) + \
            (f" ({r['top']})" if r["top"] else f" ({r['status']})")
        base_cell = "Pass" if r["baseline_passed"] else "Fail: " + "; ".join(r["baseline_fail_reasons"])
        lines.append(f"| {r['id']} | \"{short(r['input'])}\" | {r['criteria']} | {agent_cell} | "
                     f"{r['confidence']:.2f} | {base_cell} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Run the VibeMatch reliability evaluation")
    ap.add_argument("--llm", action="store_true", help="use Gemini (needs GEMINI_API_KEY); default is offline")
    ap.add_argument("--repeat", type=int, default=1, help="run each case N times to measure consistency")
    args = ap.parse_args(argv)

    if not args.llm:
        os.environ["VIBEMATCH_OFFLINE"] = "1"
    else:  # stay under the free tier's 15 requests/minute
        os.environ.setdefault("GEMINI_MIN_INTERVAL", "4.5")
    from src.agent import VibeMatchAgent  # imported after setting the env var

    agent = VibeMatchAgent()
    mode = "gemini" if agent.llm else "offline"
    if args.llm and not agent.llm:
        print("Warning: --llm requested but Gemini is unavailable; running offline.")

    result = evaluate(agent, load_cases(), repeat=max(1, args.repeat))
    EVAL_DIR.mkdir(exist_ok=True)
    (EVAL_DIR / f"results_{mode}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = to_markdown(result, mode)
    (EVAL_DIR / f"report_{mode}.md").write_text(report, encoding="utf-8")

    s = result["summary"]
    print(f"\nMode: {mode}")
    print(f"Agent passed {s['agent_pass']} cases; Module 3 baseline passed {s['baseline_pass']}.")
    for cat, v in s["by_category"].items():
        print(f"  {cat:<10} agent {v['agent']:>6}   baseline {v['baseline']:>6}")
    print(f"Avg confidence (answered): {s['avg_confidence_ok']}  passed={s['avg_confidence_passed']}  failed={s['avg_confidence_failed']}")
    print(f"Retries triggered: {s['cases_with_retries']}   Hallucinations caught: {s['hallucinations_caught']}"
          f"   LLM failures recovered: {s['llm_errors_recovered']}")
    if "consistency" in s:
        print(f"Consistency: {s['consistency']}")
    for r in result["rows"]:
        if not r["passed"]:
            print(f"  FAIL {r['id']} {r['input'][:40]!r}: {'; '.join(r['fail_reasons'])}")
    print(f"\nReport: eval/report_{mode}.md")


if __name__ == "__main__":
    main()
