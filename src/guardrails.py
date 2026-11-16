"""
Guardrails for VibeMatch: checks on what goes into the AI and what comes out.

Input guardrail  - runs before any model call; rejects empty, oversized, or
                   prompt-injection requests and strips control characters.
Profile check    - validates and repairs whatever profile the parser (LLM or
                   keyword) produced, so the scorer never sees bad values.
Output checks    - used by the agent's verification step: recommendations must
                   be real catalog songs, and an LLM explanation may only name
                   songs that were actually recommended (no hallucinations).
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

MAX_REQUEST_CHARS = 300

INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above) (instructions|rules|prompts?)",
    r"disregard (the |your )?(instructions|rules|system)",
    r"system prompt",
    r"you are now",
    r"reveal (your|the) (prompt|instructions|api key)",
    r"api[_ ]?key",
    r"<\s*/?\s*(system|script)",
]

# Requests that are clearly not about music (the agent declines instead of guessing)
OFF_TOPIC_PATTERNS = [
    r"\b(hack|password|credit card|bomb|weapon|kill)\b",
    r"\b(write|do) my (essay|homework)\b",
]


@dataclass
class GuardrailResult:
    ok: bool
    cleaned: str = ""
    reason: str = ""


def check_input(text: Optional[str]) -> GuardrailResult:
    """Validate a raw user request before it reaches any model."""
    if text is None or not str(text).strip():
        return GuardrailResult(False, reason="Empty request. Describe the music you want, e.g. 'chill songs for studying'.")
    cleaned = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", str(text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > MAX_REQUEST_CHARS:
        return GuardrailResult(False, reason=f"Request is too long ({len(cleaned)} chars). Keep it under {MAX_REQUEST_CHARS}.")
    lowered = cleaned.lower()
    for pat in INJECTION_PATTERNS:
        if re.search(pat, lowered):
            return GuardrailResult(False, reason="Request looks like an attempt to change the assistant's instructions, so it was blocked.")
    for pat in OFF_TOPIC_PATTERNS:
        if re.search(pat, lowered):
            return GuardrailResult(False, reason="VibeMatch only recommends music; that request is out of scope.")
    return GuardrailResult(True, cleaned=cleaned)


@dataclass
class ProfileCheck:
    profile: Dict
    fixes: List[str] = field(default_factory=list)


def validate_profile(raw: Dict, known_genres: set, known_moods: set) -> ProfileCheck:
    """
    Coerce a parsed profile into safe values. Anything invalid is repaired and
    recorded in `fixes` so the trace shows what the guardrail changed.
    """
    fixes: List[str] = []
    raw = raw if isinstance(raw, dict) else {}

    def label(key: str, known: set) -> str:
        value = raw.get(key)
        if value is None or value == "":
            return ""
        value = str(value).strip().lower()[:30]
        if value not in known:
            fixes.append(f"{key} '{value}' is not in the knowledge base; kept for retrieval only")
        return value

    genre = label("genre", known_genres)
    mood = label("mood", known_moods)

    energy = raw.get("energy")
    try:
        energy = float(energy)
        if not 0.0 <= energy <= 1.0:
            fixes.append(f"energy {energy} clamped to 0-1")
            energy = min(1.0, max(0.0, energy))
    except (TypeError, ValueError):
        if energy is not None:
            fixes.append(f"energy '{energy}' was not a number; using default")
        energy = None

    acoustic = raw.get("likes_acoustic")
    if isinstance(acoustic, str):
        acoustic = acoustic.strip().lower() in {"true", "yes", "1"}
    elif acoustic is not None and not isinstance(acoustic, bool):
        fixes.append("likes_acoustic was not true/false; using default")
        acoustic = None

    return ProfileCheck(
        profile={"genre": genre, "mood": mood, "energy": energy, "likes_acoustic": acoustic},
        fixes=fixes,
    )


def unknown_titles(results: List[Dict], catalog: List[Dict]) -> List[str]:
    """Titles in results that do not exist in the catalog (should always be empty)."""
    titles = {s["title"] for s in catalog}
    return [r["title"] for r in results if r["title"] not in titles]


def hallucinated_titles(explanation: str, recommended: List[Dict], catalog: List[Dict]) -> List[str]:
    """Catalog titles the explanation mentions that were NOT recommended."""
    allowed = {s["title"] for s in recommended}
    text = explanation.lower()
    return [s["title"] for s in catalog if s["title"] not in allowed and s["title"].lower() in text]
