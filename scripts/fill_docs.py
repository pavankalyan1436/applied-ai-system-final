"""
Copy evaluation numbers and tables from eval/ into model_card.md, so the docs
always match the latest generated reports.

    python scripts/fill_docs.py

Fills the <!-- eval-summary --> and <!-- eval-table --> blocks in model_card.md.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def fill(text: str, name: str, content: str) -> str:
    pattern = re.compile(rf"(<!-- {name}:start -->\n).*?(<!-- {name}:end -->)", re.S)
    if not pattern.search(text):
        raise SystemExit(f"marker {name} not found")
    return pattern.sub(lambda m: m.group(1) + content.strip() + "\n" + m.group(2), text)


def summary_table() -> str:
    runs = []
    for mode, label in (("offline", "Agent, offline (keyword parser)"), ("gemini", "Agent, Gemini")):
        path = ROOT / "eval" / f"results_{mode}.json"
        if path.exists():
            runs.append((label, json.loads(path.read_text())["summary"]))
    if not runs:
        return "_Run `python -m src.evaluate` first._"
    base = runs[0][1]
    cats = list(base["by_category"])
    head = "| Metric | " + " | ".join(label for label, _ in runs) + " | Module 3 baseline |"
    lines = [head, "|---" * (len(runs) + 2) + "|"]
    lines.append("| **Cases passed** | " + " | ".join(f"**{s['agent_pass']}**" for _, s in runs) + f" | **{base['baseline_pass']}** |")
    for c in cats:
        lines.append(f"| {c} | " + " | ".join(s["by_category"][c]["agent"] for _, s in runs) + f" | {base['by_category'][c]['baseline']} |")
    lines.append("| Avg confidence (answered) | " + " | ".join(str(s["avg_confidence_ok"]) for _, s in runs) + " | n/a |")
    lines.append("| Avg confidence: passed / failed cases | " + " | ".join(
        f"{s['avg_confidence_passed']} / {s['avg_confidence_failed'] or '-'}" for _, s in runs) + " | n/a |")
    lines.append("| Cases where verification retried | " + " | ".join(str(s["cases_with_retries"]) for _, s in runs) + " | n/a |")
    lines.append("| Hallucinated explanations caught | " + " | ".join(str(s["hallucinations_caught"]) for _, s in runs) + " | n/a |")
    lines.append("| LLM failures recovered by fallback | " + " | ".join(str(s["llm_errors_recovered"]) for _, s in runs) + " | n/a |")
    lines.append("| Run-to-run consistency | " + " | ".join(s.get("consistency", "n/a (deterministic)") for _, s in runs) + " | n/a |")
    return "\n".join(lines)


def results_table() -> str:
    report = (ROOT / "eval" / "report_offline.md").read_text()
    return report[report.index("| ID |"):]


def main() -> None:
    card = ROOT / "model_card.md"
    text = card.read_text()
    text = fill(text, "eval-summary", summary_table())
    text = fill(text, "eval-table", results_table())
    card.write_text(text)
    print("model_card.md updated")


if __name__ == "__main__":
    main()
