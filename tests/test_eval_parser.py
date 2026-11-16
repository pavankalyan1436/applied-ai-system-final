"""Tests for the few-shot specialization: prompt selection, fairness, and scoring."""

import json
from pathlib import Path

from src.eval_parser import invented_mood, normalize, score_fields
from src.llm_client import FEWSHOT_EXAMPLES, GeminiClient

ROOT = Path(__file__).resolve().parent.parent
MOODS = {"happy", "chill", "sad", "intense"}


def _client(style="fewshot"):
    """GeminiClient that records prompts instead of calling the API."""
    c = GeminiClient.__new__(GeminiClient)
    c.prompt_style, c.prompts = style, []

    def fake_generate(prompt, json_mode=False):
        c.prompts.append(prompt)
        return '{"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": null, "confidence": 0.9}'

    c._generate = fake_generate
    return c


def test_fewshot_prompt_includes_rules_and_examples():
    c = _client("fewshot")
    c.parse_request("happy pop", ["pop"], ["happy"])
    assert "Examples:" in c.prompts[0]
    assert "MUST be exactly one of the catalog moods" in c.prompts[0]
    assert all(req in c.prompts[0] for req, _ in FEWSHOT_EXAMPLES)


def test_zeroshot_prompt_has_no_examples():
    c = _client("zeroshot")
    c.parse_request("happy pop", ["pop"], ["happy"])
    assert "Examples:" not in c.prompts[0]


def test_style_argument_overrides_default():
    c = _client("fewshot")
    c.parse_request("happy pop", ["pop"], ["happy"], style="zeroshot")
    assert "Examples:" not in c.prompts[0]


def test_fewshot_examples_are_not_test_requests():
    """Fairness: the examples must not leak into either evaluation set."""
    examples = {req.lower() for req, _ in FEWSHOT_EXAMPLES}
    for name in ("cases.json", "parse_cases.json"):
        inputs = {c["input"].lower() for c in json.loads((ROOT / "eval" / name).read_text())}
        assert not examples & inputs, f"few-shot example leaked into eval/{name}"


def test_fewshot_example_moods_are_catalog_moods():
    """The examples must follow the prompt's own rule (no invented moods)."""
    from src.retriever import Retriever

    known = {d.name for d in Retriever().docs if d.type == "mood"}
    assert all(out["mood"] is None or out["mood"] in known for _, out in FEWSHOT_EXAMPLES)


def test_score_fields_accepts_any_listed_answer_and_null():
    gold = {"genre": ["folk", None], "mood": ["chill", "relaxed"], "energy": [0.0, 0.45], "likes_acoustic": True}
    out = {"genre": None, "mood": "Chill", "energy": 0.3, "likes_acoustic": True}
    assert score_fields(gold, out, MOODS) == {"genre": True, "mood": True, "energy": True,
                                               "likes_acoustic": True, "confidence": None}


def test_score_fields_marks_wrong_fields():
    gold = {"genre": ["jazz"], "energy": [0.0, 0.5]}
    out = {"genre": "pop", "mood": "happy", "energy": 0.9}
    res = score_fields(gold, out, MOODS)
    assert res["genre"] is False and res["energy"] is False and res["mood"] is None


def test_score_fields_vague_request_needs_low_confidence():
    gold = {"genre": [None], "mood": [None], "max_confidence": 0.4}
    assert score_fields(gold, {"genre": None, "mood": None, "confidence": 0.2}, MOODS)["confidence"] is True
    assert score_fields(gold, {"genre": None, "mood": None, "confidence": 0.9}, MOODS)["confidence"] is False


def test_score_fields_counts_errors_as_wrong():
    res = score_fields({"genre": ["jazz"], "mood": ["chill"]}, None, MOODS)
    assert res["genre"] is False and res["mood"] is False and res["energy"] is None


def test_invented_mood_detection():
    assert invented_mood({"mood": "nice"}, MOODS)
    assert not invented_mood({"mood": "happy"}, MOODS)
    assert not invented_mood({"mood": None}, MOODS)


def test_normalize_ignores_tiny_energy_differences():
    a = {"genre": "pop", "mood": "happy", "energy": 0.81, "likes_acoustic": None}
    b = {"genre": "pop", "mood": "happy", "energy": 0.79, "likes_acoustic": None}
    assert normalize(a) == normalize(b)
