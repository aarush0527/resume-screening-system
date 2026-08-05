"""Unit tests for the deterministic scoring engine.

Run with: pytest tests/ -v

semantic_similarity's cosine-similarity math is tested by monkeypatching the
embedding model with fixed vectors -- the real sentence-transformers model
needs to download weights from huggingface.co on first use, which requires
normal internet access (works fine on a real machine; not available in
every sandboxed CI environment).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import scorer
from src.models import CandidateProfile, JobRequirements, Role
from src.scorer import ScoreWeights, experience_fit, score_candidate, skill_overlap


# ---------- skill_overlap ----------

def test_skill_overlap_full_match_case_insensitive():
    score, matched, missing = skill_overlap(["Python", "SQL", "AWS"], ["python", "sql"])
    assert score == 1.0
    assert missing == []
    assert set(matched) == {"python", "sql"}


def test_skill_overlap_partial_match():
    score, matched, missing = skill_overlap(["Python"], ["Python", "Kubernetes"])
    assert score == 0.5
    assert "Kubernetes" in missing


def test_skill_overlap_no_requirements_scores_full():
    score, matched, missing = skill_overlap(["Python"], [])
    assert score == 1.0
    assert matched == [] and missing == []


def test_skill_overlap_no_candidate_skills():
    score, matched, missing = skill_overlap([], ["Python", "SQL"])
    assert score == 0.0
    assert missing == ["Python", "SQL"]


# ---------- experience_fit ----------

def test_experience_fit_meets_requirement():
    assert experience_fit(5, 3) == 1.0


def test_experience_fit_exactly_meets_requirement():
    assert experience_fit(3, 3) == 1.0


def test_experience_fit_under_requirement_scales_linearly():
    assert experience_fit(1, 4) == 0.25


def test_experience_fit_no_requirement_scores_full():
    assert experience_fit(0, 0) == 1.0


def test_experience_fit_zero_years_against_requirement():
    assert experience_fit(0, 3) == 0.0


# ---------- semantic_similarity (mocked embeddings) ----------

class _FakeModel:
    """Returns a fixed pair of vectors regardless of input text, so we can
    test the cosine-similarity/clamping math in isolation."""
    def __init__(self, vec_a, vec_b):
        self._vecs = [np.array(vec_a), np.array(vec_b)]

    def encode(self, texts):
        return self._vecs


def test_semantic_similarity_identical_vectors_scores_near_one(monkeypatch):
    monkeypatch.setattr(scorer, "_get_embedding_model", lambda: _FakeModel([1, 0, 0], [1, 0, 0]))
    assert scorer.semantic_similarity("a", "b") == pytest.approx(1.0, abs=1e-6)


def test_semantic_similarity_orthogonal_vectors_scores_half(monkeypatch):
    monkeypatch.setattr(scorer, "_get_embedding_model", lambda: _FakeModel([1, 0], [0, 1]))
    assert scorer.semantic_similarity("a", "b") == pytest.approx(0.5, abs=1e-6)


def test_semantic_similarity_opposite_vectors_scores_near_zero(monkeypatch):
    monkeypatch.setattr(scorer, "_get_embedding_model", lambda: _FakeModel([1, 0], [-1, 0]))
    assert scorer.semantic_similarity("a", "b") == pytest.approx(0.0, abs=1e-6)


def test_semantic_similarity_empty_text_short_circuits():
    # Should return 0.0 without even touching the embedding model
    assert scorer.semantic_similarity("", "something") == 0.0


# ---------- score_candidate weighting + a real sanity-ranking case ----------

def _jd(required_skills, min_years=3):
    return JobRequirements(
        title="Backend Software Engineer",
        raw_text="dummy jd text",
        required_skills=required_skills,
        min_years_experience=min_years,
    )


def _candidate(name, skills, years, role_desc="dummy scoring text"):
    return CandidateProfile(
        file_name=f"{name}.txt", name=name, skills=skills, years_experience=years,
        roles=[Role("Engineer", "Co", "1y", role_desc)],
    )


def test_score_candidate_weights_sum_and_combine_correctly(monkeypatch):
    monkeypatch.setattr(scorer, "_get_embedding_model", lambda: _FakeModel([1, 0], [1, 0]))  # semantic=1.0
    jd = _jd(["Python", "SQL"], min_years=3)
    candidate = _candidate("Test", ["Python", "SQL"], years=5)  # skill=1.0, exp=1.0
    final, breakdown, matched, missing = score_candidate(candidate, jd, ScoreWeights())
    assert final == pytest.approx(1.0, abs=1e-6)
    assert breakdown.semantic_similarity == 1.0
    assert breakdown.skill_overlap == 1.0
    assert breakdown.experience_fit == 1.0


def test_strong_fit_outranks_wrong_field_candidate(monkeypatch):
    """The core sanity check: a candidate whose skills/experience genuinely
    match the JD should outrank one from an unrelated field, even before
    any LLM call happens."""
    monkeypatch.setattr(scorer, "_get_embedding_model", lambda: _FakeModel([1, 0], [0.9, 0.1]))
    jd = _jd(["Python", "PostgreSQL", "AWS", "REST APIs"], min_years=3)

    strong_fit = _candidate("StrongFit", ["Python", "PostgreSQL", "AWS", "REST APIs"], years=5)
    wrong_field = _candidate("WrongField", ["Excel", "SQL"], years=4)

    strong_score, *_ = score_candidate(strong_fit, jd, ScoreWeights())
    weak_score, *_ = score_candidate(wrong_field, jd, ScoreWeights())

    assert strong_score > weak_score


def test_low_evidence_specificity_dampens_skill_overlap_but_not_to_zero(monkeypatch):
    """The fix for the keyword-stuffing finding: a candidate who lists every
    required skill but with generic, unsubstantiated descriptions should
    score lower than an equally-skilled candidate with concrete evidence --
    but not be crushed to zero just for being vague."""
    monkeypatch.setattr(scorer, "_get_embedding_model", lambda: _FakeModel([1, 0], [1, 0]))
    jd = _jd(["Python", "AWS", "PostgreSQL"], min_years=3)

    specific = _candidate("Specific", ["Python", "AWS", "PostgreSQL"], years=4)
    specific.evidence_specificity = 1.0
    vague = _candidate("Vague", ["Python", "AWS", "PostgreSQL"], years=4)
    vague.evidence_specificity = 0.2

    specific_score, specific_bd, *_ = score_candidate(specific, jd, ScoreWeights())
    vague_score, vague_bd, *_ = score_candidate(vague, jd, ScoreWeights())

    assert specific_score > vague_score
    # raw skill_overlap is identical (both listed everything) --
    # the dampening shows up in final_score, not in the raw breakdown number
    assert specific_bd.skill_overlap == vague_bd.skill_overlap == 1.0
    # floor is 0.5x, never fully zeroed for being vague
    assert vague_score > 0


def test_keyword_stuffed_candidate_does_not_beat_genuine_experience_match(monkeypatch):
    """Two candidates list the same skills, but one is well under the
    experience bar. Skill overlap alone shouldn't be enough to win --
    experience_fit should pull the underqualified one down."""
    monkeypatch.setattr(scorer, "_get_embedding_model", lambda: _FakeModel([1, 0], [1, 0]))
    jd = _jd(["Python", "AWS", "Kubernetes"], min_years=5)

    seasoned = _candidate("Seasoned", ["Python", "AWS", "Kubernetes"], years=6)
    keyword_stuffed_junior = _candidate("KeywordStuffed", ["Python", "AWS", "Kubernetes"], years=1)

    seasoned_score, *_ = score_candidate(seasoned, jd, ScoreWeights())
    junior_score, *_ = score_candidate(keyword_stuffed_junior, jd, ScoreWeights())

    assert seasoned_score > junior_score
