"""
Interactive VibeMatch assistant (command line).

    python -m src.cli                          # interactive, with feedback loop
    python -m src.cli "chill lofi for studying" # one request, then feedback
    python -m src.cli --offline --trace "..."   # no Gemini; show every agent step

Human-in-the-loop:
  - if the agent is unsure, it asks a clarifying question and you answer it
  - after results, rate songs: "1+" (like #1), "3-" (dislike #3); the agent re-runs
"""

import argparse
import logging
import os
import sys

from src.agent import VibeMatchAgent, apply_feedback

CHECK_LABELS = {
    "keeps_requested_label": "keeps your requested genre (or mood)",
    "all_songs_in_catalog": "every song exists in the catalog",
    "enough_results": "returned a full list",
    "energy_fits_majority": "most songs fit your energy",
    "top_pick_energy_fits": "top pick fits your energy",
    "top_score_strong": "top pick scores at least 3/6",
}


def show(result, show_trace: bool) -> None:
    print()
    if result.status == "blocked":
        print(f"[guardrail] {result.message}")
        return
    if result.status == "needs_clarification":
        print(f"[needs your input] {result.message}")
        return

    p = result.profile
    print(f"Understood ({result.parse_method}): genre={p['genre'] or '-'}, mood={p['mood'] or '-'}, "
          f"energy={p['energy']}, acoustic={'yes' if p['likes_acoustic'] else 'no'}")
    print("-" * 64)
    for i, r in enumerate(result.recommendations, start=1):
        print(f"{i}. {r['title']} by {r['artist']}  [{r['genre']}, {r['mood']}, energy {r['energy']}]  {r['score']:.2f}/6")
        for reason in r["reasons"].split("; "):
            print(f"     - {reason}")
    print("-" * 64)
    print(f"Why ({result.explanation_source}): {result.explanation}")
    print(f"\nConfidence: {result.confidence:.2f}   Strategy: {result.strategy}")
    for name, ok in result.checks.items():
        print(f"  [{'pass' if ok else 'FAIL'}] {CHECK_LABELS.get(name, name)}")
    if result.message:
        print(f"Note: {result.message} Your request may pull in different directions; try rephrasing or rate the songs below.")
    if show_trace:
        print("\nAgent trace:")
        for step in result.trace:
            print("  ", step)


def session(agent: VibeMatchAgent, request: str, show_trace: bool, interactive: bool) -> None:
    result = agent.run(request)
    show(result, show_trace)

    # Clarification loop (at most 2 questions, then give up politely)
    for _ in range(2):
        if result.status != "needs_clarification" or not interactive:
            break
        answer = input("> ").strip()
        if not answer:
            return
        request = f"{request} {answer}"
        result = agent.run(request)
        show(result, show_trace)

    if result.status != "ok" or not interactive:
        return

    profile, excluded = dict(result.profile), set()
    while True:
        fb = input("\nRate a song ('1+' like, '3-' dislike) or press Enter to finish: ").strip()
        if not fb:
            return
        try:
            idx, sign = int(fb[:-1]) - 1, fb[-1]
            song = result.recommendations[idx]
            assert sign in "+-"
        except (ValueError, IndexError, AssertionError):
            print("Please type a number followed by + or -, e.g. 2+")
            continue
        if sign == "+":
            profile = apply_feedback(profile, song, liked=True)
            print(f"Noted: more like {song['title']} (target energy now {profile['energy']}).")
        else:
            excluded.add(song["title"])
            print(f"Noted: {song['title']} removed.")
        result = agent.run(request, profile_overrides=profile, exclude=excluded)
        show(result, show_trace)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="VibeMatch: an AI music recommender agent")
    ap.add_argument("request", nargs="*", help="what you want to listen to")
    ap.add_argument("--offline", action="store_true", help="skip Gemini; use deterministic fallbacks")
    ap.add_argument("--trace", action="store_true", help="print every agent step")
    ap.add_argument("--no-feedback", action="store_true", help="do not prompt for clarification or ratings")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if args.offline:
        os.environ["VIBEMATCH_OFFLINE"] = "1"
    agent = VibeMatchAgent()
    print(f"VibeMatch ready ({'Gemini' if agent.llm else 'offline mode'}). Logs: logs/vibematch.jsonl")

    interactive = sys.stdin.isatty() and not args.no_feedback
    if args.request:
        session(agent, " ".join(args.request), args.trace, interactive)
        return
    while True:
        try:
            request = input("\nWhat do you want to listen to? (Enter to quit) ").strip()
        except EOFError:
            return
        if not request:
            return
        session(agent, request, args.trace, interactive=True)


if __name__ == "__main__":
    main()
