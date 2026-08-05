"""Orchestrates scoring across all candidates and generates evidence-based
rationales for the already-computed scores."""
from __future__ import annotations

from .llm_client import LLMClient
from .models import CandidateProfile, JobRequirements, ScoredCandidate
from .scorer import ScoreWeights, score_candidate

RATIONALE_SYSTEM_PROMPT = """You write short, evidence-based hiring rationales for a resume-screening \
tool. You are given a candidate's matched skills, missing skills, years of experience, a short \
background summary, and a pre-computed fit score. You do NOT decide the score -- it has already been \
computed -- your only job is to explain it in 2-3 concrete sentences that reference specific evidence. \
Do not restate the raw score number. Be specific, not generic."""


def rank_candidates(
    llm: LLMClient,
    candidates: list[CandidateProfile],
    jd: JobRequirements,
    weights: ScoreWeights = ScoreWeights(),
    generate_rationale: bool = True,
) -> list[ScoredCandidate]:
    scored: list[ScoredCandidate] = []
    for c in candidates:
        final_score, breakdown, matched, missing = score_candidate(c, jd, weights)
        rationale = ""
        if generate_rationale:
            rationale = _generate_rationale(llm, c, jd, final_score, matched, missing)
        scored.append(
            ScoredCandidate(
                rank=0,  # assigned below, after sorting
                file_name=c.file_name,
                name=c.name,
                final_score=final_score,
                score_breakdown=breakdown,
                matched_skills=matched,
                missing_skills=missing,
                years_experience=c.years_experience,
                rationale=rationale,
            )
        )

    scored.sort(key=lambda sc: sc.final_score, reverse=True)
    for i, sc in enumerate(scored, start=1):
        sc.rank = i
    return scored


def _generate_rationale(
    llm: LLMClient,
    candidate: CandidateProfile,
    jd: JobRequirements,
    final_score: float,
    matched: list[str],
    missing: list[str],
) -> str:
    prompt = (
        f"Job title: {jd.title}\n"
        f"Candidate years of experience: {candidate.years_experience}\n"
        f"Fit score: {final_score:.2f} on a 0-1 scale\n"
        f"Matched required skills: {', '.join(matched) or 'none'}\n"
        f"Missing required skills: {', '.join(missing) or 'none'}\n"
        f"Candidate background summary: {candidate.summary or 'not available'}\n\n"
        "Write the rationale now."
    )
    try:
        return llm.complete(RATIONALE_SYSTEM_PROMPT, prompt).strip()
    except Exception as e:
        # A failed rationale call shouldn't take down the whole ranked
        return f"[rationale unavailable: {e}]"
