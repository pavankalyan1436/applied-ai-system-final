"""Tests for the knowledge base, retriever, and retrieval-aware scoring."""

from src.recommender import Recommender, Song, UserProfile, load_songs, recommend_songs, score_song
from src.retriever import RetrievalContext, Retriever, build_context, tokenize

RETRIEVER = Retriever()
SONGS = load_songs("data/songs.csv")


def test_tokenize_keeps_hyphenated_and_ampersand_words():
    assert tokenize("Upbeat K-Pop and R&B, lo-fi!") == ["upbeat", "k-pop", "and", "r&b", "lo-fi"]


def test_knowledge_base_covers_every_catalog_genre_and_mood():
    for song in SONGS:
        assert RETRIEVER.lookup("genre", song["genre"]), f"no notes for genre {song['genre']}"
        assert RETRIEVER.lookup("mood", song["mood"]), f"no notes for mood {song['mood']}"


def test_related_labels_point_at_known_documents():
    names = {d.name for d in RETRIEVER.docs}
    for doc in RETRIEVER.docs:
        for rel in doc.related:
            assert rel in names, f"{doc.source} lists unknown related label '{rel}'"


def test_retrieve_finds_genre_by_name():
    top_doc, _ = RETRIEVER.retrieve("some k-pop please")[0]
    assert top_doc.source == "genres.md#k-pop"


def test_retrieve_maps_activity_words_to_activity_notes():
    sources = [d.source for d, _ in RETRIEVER.retrieve("music for the gym")]
    assert "activities.md#workout" in sources


def test_retrieve_returns_nothing_for_gibberish():
    assert RETRIEVER.retrieve("qwzx plorf") == []


def test_build_context_excludes_users_own_labels_and_keeps_sources():
    ctx = build_context(RETRIEVER, {"genre": "pop", "mood": "happy"})
    assert "pop" not in ctx.related_genres
    assert "happy" not in ctx.related_moods
    assert ctx.related_genres["indie pop"] == "genres.md#pop"


def test_build_context_takes_energy_hint_from_activity():
    ctx = build_context(RETRIEVER, {"genre": "", "mood": ""}, "songs to study to")
    assert ctx.energy_hint == 0.40
    assert ctx.acoustic_hint is True


def test_scoring_without_context_matches_original_recipe():
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False}
    sunrise = SONGS[0]
    score, _ = score_song(prefs, sunrise)
    assert round(score, 2) == 5.79  # value documented in the Module 3 README


def test_related_genre_earns_half_credit_and_cites_source():
    prefs = {"genre": "k-pop", "mood": "none", "energy": 0.8, "likes_acoustic": False}
    sunrise = SONGS[0]  # pop
    base, _ = score_song(prefs, sunrise)
    ctx = build_context(RETRIEVER, prefs)
    boosted, reasons = score_song(prefs, sunrise, context=ctx)
    assert round(boosted - base, 2) == 1.0
    assert any("genres.md#k-pop" in r for r in reasons)


def test_retrieval_moves_genre_neighbours_up_for_unknown_genre():
    prefs = {"genre": "k-pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False}
    ctx = build_context(RETRIEVER, prefs)
    top = [s["genre"] for s, _, _ in recommend_songs(prefs, SONGS, k=5, context=ctx)]
    assert all(g in {"pop", "indie pop", "edm", "hip hop"} for g in top)


def test_recommender_class_ranks_by_score_with_context():
    songs = [
        Song(1, "Calm", "A", "ambient", "chill", 0.2, 60, 0.5, 0.3, 0.9),
        Song(2, "Bright", "B", "indie pop", "happy", 0.8, 120, 0.8, 0.8, 0.2),
    ]
    user = UserProfile("k-pop", "happy", 0.8, False)
    rec = Recommender(songs)
    ctx = build_context(RETRIEVER, Recommender._prefs(user))
    assert rec.recommend(user, k=1, context=ctx)[0].title == "Bright"
    assert "genres.md#k-pop" in rec.explain_recommendation(user, songs[1], context=ctx)


def test_empty_context_behaves_like_no_context():
    prefs = {"genre": "rock", "mood": "intense", "energy": 0.9, "likes_acoustic": False}
    assert score_song(prefs, SONGS[2]) == score_song(prefs, SONGS[2], context=RetrievalContext())
