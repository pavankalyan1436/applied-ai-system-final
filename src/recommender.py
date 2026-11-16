import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import asdict, dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song], weights: Optional[Dict] = None):
        self.songs = songs
        self.weights = weights

    @staticmethod
    def _prefs(user: UserProfile) -> Dict:
        """Translate a UserProfile into the dict format score_song expects."""
        return {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }

    def recommend(self, user: UserProfile, k: int = 5, context=None) -> List[Song]:
        """Return the top k Song objects, highest score first."""
        prefs = self._prefs(user)
        ranked = sorted(
            self.songs,
            key=lambda s: score_song(prefs, asdict(s), self.weights, context)[0],
            reverse=True,
        )
        return ranked[:k]

    def explain_recommendation(self, user: UserProfile, song: Song, context=None) -> str:
        """Return a human-readable explanation of how a song earned its score."""
        score, reasons = score_song(self._prefs(user), asdict(song), self.weights, context)
        return f"{song.title} scored {score:.2f}/6.00 because: " + "; ".join(reasons)

def load_songs(csv_path: str) -> List[Dict]:
    """Read songs.csv into a list of dicts, converting numeric columns to int/float."""
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["id"] = int(row["id"])
            for key in ("energy", "tempo_bpm", "valence", "danceability", "acousticness"):
                row[key] = float(row[key])
            songs.append(row)
    print(f"Loaded songs: {len(songs)}")
    return songs

# Points available for each part of the scoring recipe (max total = 6.0)
DEFAULT_WEIGHTS = {"genre": 2.0, "mood": 1.5, "energy": 1.5, "acoustic": 1.0}

# Share of the genre/mood points a *related* label earns (exact match = 1.0)
RELATED_CREDIT = 0.5


def score_song(
    user_prefs: Dict, song: Dict, weights: Optional[Dict] = None, context=None
) -> Tuple[float, List[str]]:
    """
    Score one song using the weighted recipe; return (score, reasons).

    `context` is an optional RetrievalContext (see src/retriever.py). When given,
    genres and moods that the retrieved notes list as related earn partial
    credit, and the reason cites the note it came from. With no context the
    scoring is exactly the original Module 3 recipe.
    """
    w = weights or DEFAULT_WEIGHTS
    score = 0.0
    reasons = []
    related_genres = getattr(context, "related_genres", {}) or {}
    related_moods = getattr(context, "related_moods", {}) or {}

    if song["genre"] == user_prefs["genre"]:
        score += w["genre"]
        reasons.append(f"genre match (+{w['genre']:.1f})")
    elif song["genre"] in related_genres:
        pts = w["genre"] * RELATED_CREDIT
        score += pts
        reasons.append(f"related genre '{song['genre']}' (+{pts:.1f}) [source: {related_genres[song['genre']]}]")

    if song["mood"] == user_prefs["mood"]:
        score += w["mood"]
        reasons.append(f"mood match (+{w['mood']:.1f})")
    elif song["mood"] in related_moods:
        pts = w["mood"] * RELATED_CREDIT
        score += pts
        reasons.append(f"related mood '{song['mood']}' (+{pts:.2f}) [source: {related_moods[song['mood']]}]")

    energy_points = w["energy"] * (1 - abs(song["energy"] - user_prefs["energy"]))
    score += energy_points
    reasons.append(f"energy closeness (+{energy_points:.2f})")

    if user_prefs.get("likes_acoustic", False):
        acoustic_points = w["acoustic"] * song["acousticness"]
    else:
        acoustic_points = w["acoustic"] * (1 - song["acousticness"])
    score += acoustic_points
    reasons.append(f"acoustic fit (+{acoustic_points:.2f})")

    return score, reasons

def recommend_songs(
    user_prefs: Dict, songs: List[Dict], k: int = 5, weights: Optional[Dict] = None, context=None
) -> List[Tuple[Dict, float, str]]:
    """Score every song with score_song, then return the top k as (song, score, explanation)."""
    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song, weights, context)
        scored.append((song, score, "; ".join(reasons)))
    return sorted(scored, key=lambda item: item[1], reverse=True)[:k]
