"""Deterministic hybrid scoring: semantic similarity + skill overlap + experience fit.

Design decision (see README "Approach" section for the full writeup): the
LLM is never asked to output the fit score itself. Asking an LLM for "a
number 1-10" is the common shallow version of this agent -- it's
non-deterministic across runs, expensive to batch, and hard to justify to
a candidate or hiring manager who asks "why did I score a 6?". Here, the
score is three auditable, recomputable numbers combined with fixed
weights; the LLM's only job (in ranker.py) is to explain a score that has
already been computed, not to invent one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import CandidateProfile, JobRequirements, ScoreBreakdown

_embedding_model = None  # lazy-loaded + cached so we only pay the load cost once per run


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        # all-MiniLM-L6-v2: small (~80MB), fast, local, free -- no per-call
        # API cost or rate limit at batch scale. Swap to all-mpnet-base-v2
        # if quality matters more than speed (see README tradeoffs).
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def semantic_similarity(candidate_text: str, jd_text: str) -> float:
    """Cosine similarity between embedded candidate text and embedded JD
    text, rescaled from [-1, 1] to [0, 1]."""
    if not candidate_text.strip() or not jd_text.strip():
        return 0.0
    model = _get_embedding_model()
    embeddings = model.encode([candidate_text, jd_text])
    a, b = embeddings[0], embeddings[1]
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    cos = float(np.dot(a, b) / denom)
    return max(0.0, min(1.0, (cos + 1) / 2))


def skill_overlap(candidate_skills: list[str], required_skills: list[str]) -> tuple[float, list[str], list[str]]:
    """Case-insensitive, substring-tolerant overlap between candidate skills
    and JD-required skills. Returns (score, matched_skills, missing_skills).

    V1 uses simple normalized string matching rather than fuzzy/embedding-
    based skill matching -- it ships faster and is easy to audit, at the
    cost of missing some synonyms (e.g. "JS" vs "JavaScript"). See README
    tradeoffs for the V2 upgrade path.
    """
    if not required_skills:
        return 1.0, [], []
    cand_norm = {s.lower().strip() for s in candidate_skills if s.strip()}
    matched, missing = [], []
    for req in required_skills:
        req_norm = req.lower().strip()
        hit = any(req_norm == c or req_norm in c or c in req_norm for c in cand_norm)
        (matched if hit else missing).append(req)
    return len(matched) / len(required_skills), matched, missing


def experience_fit(candidate_years: float, required_years: float) -> float:
    """1.0 if the candidate meets or exceeds the requirement (no bonus for
    over-qualification -- that's a separate judgment call for a human, not
    something this score should silently reward or punish). Linear falloff
    below the requirement."""
    if required_years <= 0:
        return 1.0
    if candidate_years >= required_years:
        return 1.0
    return max(0.0, candidate_years / required_years)


@dataclass
class ScoreWeights:
    semantic: float = 0.4
    skill: float = 0.4
    experience: float = 0.2


def score_candidate(
    candidate: CandidateProfile,
    jd: JobRequirements,
    weights: ScoreWeights = ScoreWeights(),
) -> tuple[float, ScoreBreakdown, list[str], list[str]]:
    """Returns (final_score, breakdown, matched_skills, missing_skills).

    candidate.scoring_text() intentionally excludes name/contact info --
    see models.py. Only role/skill/education content is embedded or scored.

    evidence_specificity dampens skill_overlap before it's weighted in.
    This exists because of an adversarial test case: a resume that lists
    every required keyword with only generic, unsubstantiated descriptions
    ("assisted with various projects involving X, Y, Z") scored
    competitively with genuinely strong candidates on skill_overlap alone,
    since that component only checks whether a skill is *listed*, not
    whether it's *demonstrated*. The floor is 0.5x (not 0x) deliberately --
    a concise resume shouldn't be crushed just for being concise, only a
    demonstrably vague one.
    """
    sem = semantic_similarity(candidate.scoring_text(), jd.raw_text)
    skill_score, matched, missing = skill_overlap(candidate.skills, jd.required_skills)
    exp = experience_fit(candidate.years_experience, jd.min_years_experience)
    specificity = max(0.0, min(1.0, candidate.evidence_specificity))
    effective_skill_score = skill_score * (0.5 + 0.5 * specificity)

    final = weights.semantic * sem + weights.skill * effective_skill_score + weights.experience * exp
    breakdown = ScoreBreakdown(
        semantic_similarity=round(sem, 3),
        skill_overlap=round(skill_score, 3),
        experience_fit=round(exp, 3),
        evidence_specificity=round(specificity, 3),
    )
    return round(final, 3), breakdown, matched, missing
