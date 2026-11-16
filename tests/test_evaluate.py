"""Tests for the evaluation harness and the Gemini client's reliability guards."""

import pytest

import src.agent as agent_module
import src.llm_client as llm_module
from src.agent import VibeMatchAgent
from src.evaluate import describe, evaluate, judge, load_cases, to_markdown
from src.llm_client import GeminiClient

SONG = {"title": "Storm Runner", "genre": "rock", "energy": 0.91, "acousticness": 0.10}


@pytest.fixture(autouse=True)
def temp_log(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_module, "LOG_PATH", tmp_path / "runs.jsonl")


def test_judge_passes_matching_result():
    ok, fails = judge({"status": "ok", "top_genre_in": ["rock"], "top_energy_min": 0.8}, "ok", [SONG], {})
    assert ok and fails == []


def test_judge_reports_every_failed_criterion():
    ok, fails = judge({"status": "ok", "top_genre_in": ["jazz"], "top_energy_max": 0.5}, "ok", [SONG], {})
    assert not ok and len(fails) == 2


def test_judge_status_mismatch_short_circuits():
    ok, fails = judge({"status": "blocked"}, "ok", [SONG], {})
    assert not ok and fails == ["status was ok"]


def test_judge_flags_check_requires_failed_check():
    expect = {"status": "ok", "flags_check": "top_pick_energy_fits"}
    assert judge(expect, "ok", [SONG], {"top_pick_energy_fits": False})[0]
    assert not judge(expect, "ok", [SONG], {"top_pick_energy_fits": True})[0]


def test_describe_is_readable():
    assert describe({"status": ["ok", "needs_clarification"], "top_genre_in": ["lofi"]}) == \
        "status ok or needs_clarification; #1 genre in {lofi}"


def test_cases_file_is_valid():
    cases = load_cases()
    assert len(cases) >= 20
    assert len({c["id"] for c in cases}) == len(cases)
    assert {c["category"] for c in cases} == {"standard", "edge", "vague", "guardrail"}


def test_offline_evaluation_beats_baseline_and_writes_table():
    result = evaluate(VibeMatchAgent(llm=None), load_cases())
    s = result["summary"]
    agent_passed = int(s["agent_pass"].split("/")[0])
    baseline_passed = int(s["baseline_pass"].split("/")[0])
    assert agent_passed > baseline_passed
    assert s["by_category"]["guardrail"]["agent"] == "5/5"
    md = to_markdown(result, "offline")
    assert "| ID | Input | Evaluation criteria |" in md and md.count("\n| ") >= 25


class _RateLimited(Exception):
    code = 429


class _FakeModels:
    def __init__(self, failures):
        self.failures, self.calls = failures, 0

    def generate_content(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise _RateLimited("429 RESOURCE_EXHAUSTED. Please retry in 1.5s.")
        return type("R", (), {"text": ' {"ok": true} '})()


def _client(failures):
    c = GeminiClient.__new__(GeminiClient)  # skip real network setup
    c.client = type("C", (), {"models": _FakeModels(failures)})()
    c.model, c.min_interval, c._last_call = "test", 0.0, 0.0
    return c


def test_gemini_client_retries_once_on_rate_limit(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm_module.time, "sleep", sleeps.append)
    c = _client(failures=1)
    assert c._generate("hi") == '{"ok": true}'
    assert c.client.models.calls == 2 and sleeps == [2.0]  # asked 1.5s + 0.5s margin


def test_gemini_client_gives_up_after_second_rate_limit(monkeypatch):
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)
    with pytest.raises(_RateLimited):
        _client(failures=2)._generate("hi")
