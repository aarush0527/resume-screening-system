"""Data models shared across the pipeline.

CandidateProfile.scoring_text() deliberately excludes name/contact info --
see README "Design decisions" for why. name/contact exist only for display
in the final ranked output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Role:
    title: str = ""
    company: str = ""
    duration: str = ""
    description: str = ""


@dataclass
class Education:
    degree: str = ""
    field: str = ""
    institution: str = ""


@dataclass
class CandidateProfile:
    file_name: str
    name: str = "Unknown"
    contact: str = ""
    raw_text: str = ""
    skills: list[str] = field(default_factory=list)
    years_experience: float = 0.0
    education: list[Education] = field(default_factory=list)
    roles: list[Role] = field(default_factory=list)
    summary: str = ""
    # 0-1, how much of the candidate's skill/experience claims are backed by
    # concrete, verifiable detail (metrics, named systems used in context)
    # versus generic buzzword-listing. Defaults to 1.0 (no penalty) so the
    # scorer degrades gracefully if this was never set. See scorer.py and
    # README "Approach" -- added after an adversarial test resume that lists
    # every required keyword with no substantiating detail scored
    # competitively with genuinely strong candidates on skill_overlap alone.
    evidence_specificity: float = 1.0

    def scoring_text(self) -> str:
        """Text handed to the embedding model and the rationale LLM call.
        No name, no contact info -- only role/skill/education content.
        """
        parts = [self.summary]
        parts += [f"{r.title} at {r.company}: {r.description}".strip(": ") for r in self.roles]
        if self.skills:
            parts.append("Skills: " + ", ".join(self.skills))
        parts += [f"{e.degree} in {e.field}, {e.institution}".strip(", ") for e in self.education]
        return "\n".join(p for p in parts if p and p.strip())


@dataclass
class JobRequirements:
    title: str = ""
    raw_text: str = ""
    required_skills: list[str] = field(default_factory=list)
    min_years_experience: float = 0.0
    required_education: Optional[str] = None
    key_responsibilities: list[str] = field(default_factory=list)


@dataclass
class ScoreBreakdown:
    semantic_similarity: float
    skill_overlap: float          # raw overlap, unadjusted -- kept for auditability
    experience_fit: float
    evidence_specificity: float   # 0-1 dampener actually applied to skill_overlap in final_score


@dataclass
class ScoredCandidate:
    rank: int
    file_name: str
    name: str
    final_score: float
    score_breakdown: ScoreBreakdown
    matched_skills: list[str]
    missing_skills: list[str]
    years_experience: float
    rationale: str = ""
