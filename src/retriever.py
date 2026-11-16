"""
Retrieval layer (the "R" in RAG) for VibeMatch.

Loads the markdown notes in knowledge/, splits them into one document per
"## heading" section, and ranks those documents against a free-text request
with TF-IDF + cosine similarity (pure Python, no extra dependencies).

The retrieved documents are turned into a RetrievalContext that the scorer
uses directly: related genres and moods earn partial credit, and activity
notes can supply an energy target. Every hint keeps the source it came from,
so explanations can cite where a decision came from.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"

# Keeps hyphenated words and "r&b" whole: "k-pop", "lo-fi", "r&b"
TOKEN_RE = re.compile(r"[a-z0-9&]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> List[str]:
    """Lowercase and split text into word tokens."""
    return TOKEN_RE.findall(text.lower())


@dataclass
class Doc:
    """One knowledge-base section, e.g. the '## pop' notes in genres.md."""

    name: str
    type: str  # "genre", "mood", or "activity"
    related: List[str]
    text: str
    source: str  # e.g. "genres.md#pop"
    energy: Optional[float] = None
    acoustic: Optional[bool] = None


def parse_knowledge_file(path: Path) -> List[Doc]:
    """Split one markdown file into Docs, one per '## ' section."""
    docs = []
    for chunk in path.read_text(encoding="utf-8").split("\n## ")[1:]:
        lines = chunk.strip().splitlines()
        name = lines[0].strip().lower()
        fields: Dict[str, str] = {}
        body: List[str] = []
        for line in lines[1:]:
            key, sep, value = line.partition(":")
            if sep and key.strip() in {"type", "related", "keywords", "energy", "acoustic"}:
                fields[key.strip()] = value.strip()
            elif line.strip():
                body.append(line.strip())
        related = [r.strip().lower() for r in fields.get("related", "").split(",") if r.strip()]
        acoustic = fields.get("acoustic")
        docs.append(
            Doc(
                name=name,
                type=fields.get("type", "unknown"),
                related=related,
                # Name is repeated so a direct mention ("k-pop") outweighs a keyword hit
                text=" ".join([name, name, fields.get("keywords", "")] + body),
                source=f"{path.name}#{name}",
                energy=float(fields["energy"]) if "energy" in fields else None,
                acoustic=None if acoustic is None else acoustic.lower() == "yes",
            )
        )
    return docs


def load_knowledge(directory: Path = KNOWLEDGE_DIR) -> List[Doc]:
    """Load every .md file in the knowledge directory."""
    docs: List[Doc] = []
    for path in sorted(Path(directory).glob("*.md")):
        docs.extend(parse_knowledge_file(path))
    return docs


class Retriever:
    """TF-IDF retriever over knowledge-base documents."""

    def __init__(self, docs: Optional[List[Doc]] = None):
        self.docs = docs if docs is not None else load_knowledge()
        self.by_name = {(d.type, d.name): d for d in self.docs}
        doc_tokens = [tokenize(d.text) for d in self.docs]
        n = len(self.docs)
        df = Counter(tok for toks in doc_tokens for tok in set(toks))
        self.idf = {tok: math.log((n + 1) / (count + 1)) + 1 for tok, count in df.items()}
        self.vectors = [self._vectorize(toks) for toks in doc_tokens]

    def _vectorize(self, tokens: List[str]) -> Dict[str, float]:
        counts = Counter(tokens)
        vec = {t: c * self.idf.get(t, 0.0) for t, c in counts.items() if t in self.idf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def retrieve(self, query: str, k: int = 3, min_score: float = 0.15) -> List[Tuple[Doc, float]]:
        """Return up to k (doc, similarity) pairs above min_score, best first."""
        qvec = self._vectorize(tokenize(query))
        scored = []
        for doc, dvec in zip(self.docs, self.vectors):
            sim = sum(w * dvec.get(t, 0.0) for t, w in qvec.items())
            if sim >= min_score:
                scored.append((doc, sim))
        return sorted(scored, key=lambda x: x[1], reverse=True)[:k]

    def lookup(self, doc_type: str, name: str) -> Optional[Doc]:
        """Exact lookup, e.g. lookup('genre', 'pop')."""
        return self.by_name.get((doc_type, (name or "").lower()))


@dataclass
class RetrievalContext:
    """What retrieval contributes to scoring, with a source for every hint."""

    related_genres: Dict[str, str] = field(default_factory=dict)  # catalog genre -> source
    related_moods: Dict[str, str] = field(default_factory=dict)  # catalog mood -> source
    energy_hint: Optional[float] = None
    acoustic_hint: Optional[bool] = None
    sources: List[str] = field(default_factory=list)
    retrieved: List[Tuple[str, float]] = field(default_factory=list)  # (source, similarity)


def build_context(retriever: Retriever, prefs: Dict, query: str = "") -> RetrievalContext:
    """
    Gather related genres/moods and hints for a profile (and optional free text).

    1. Exact lookup of the profile's genre and mood notes.
    2. TF-IDF retrieval over the free-text request, which can surface notes
       the structured profile missed (e.g. "gym" -> the workout activity).
    """
    ctx = RetrievalContext()
    docs: List[Doc] = []
    for doc_type, key in (("genre", "genre"), ("mood", "mood")):
        doc = retriever.lookup(doc_type, prefs.get(key, ""))
        if doc:
            docs.append(doc)
    if query:
        for doc, sim in retriever.retrieve(query):
            ctx.retrieved.append((doc.source, round(sim, 3)))
            if doc not in docs:
                docs.append(doc)

    for doc in docs:
        ctx.sources.append(doc.source)
        if doc.type == "genre":
            targets = ctx.related_genres
            candidates = [doc.name] + doc.related
        else:  # moods and activities both point at moods
            targets = ctx.related_moods
            candidates = ([doc.name] if doc.type == "mood" else []) + doc.related
            if doc.type == "activity":
                if ctx.energy_hint is None:
                    ctx.energy_hint = doc.energy
                if ctx.acoustic_hint is None:
                    ctx.acoustic_hint = doc.acoustic
        for name in candidates:
            targets.setdefault(name, doc.source)

    # The user's own labels score as exact matches, not as "related"
    ctx.related_genres.pop(prefs.get("genre"), None)
    ctx.related_moods.pop(prefs.get("mood"), None)
    return ctx
