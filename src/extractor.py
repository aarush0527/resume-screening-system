"""Turns raw resume/JD text into structured objects via a single LLM call each.

This is the one place the LLM's judgment is trusted "as data" -- it never
decides the fit score (see scorer.py / ranker.py for why that's kept
separate and deterministic).
"""
from __future__ import annotations

from .llm_client import LLMClient
from .models import CandidateProfile, Education, JobRequirements, Role

RESUME_SYSTEM_PROMPT = """You are an information extraction system. Given raw resume text, extract \
structured fields and return ONLY a valid JSON object matching this schema:

{
  "name": "candidate full name, or \\"Unknown\\" if not found",
  "contact": "email and/or phone if present, else empty string",
  "skills": ["list", "of", "technical and professional skills, normalized (e.g. \\"JavaScript\\" not \\"JS\\")"],
  "years_experience": <number, total years of professional experience -- estimate from role durations if not stated explicitly>,
  "education": [{"degree": "...", "field": "...", "institution": "..."}],
  "roles": [{"title": "...", "company": "...", "duration": "...", "description": "1-2 sentence summary of responsibilities and impact"}],
  "summary": "2-3 sentence neutral factual summary of the candidate's professional background",
  "evidence_specificity": <number 0-1: how much of the candidate's listed skills and experience are backed by \
concrete, verifiable detail -- quantified outcomes ("served 2M+ daily requests"), named systems/technologies \
used in a specific context, clear ownership of a real deliverable -- versus generic buzzword-listing with \
vague phrasing like "assisted with various projects involving X, Y, Z". 1.0 = highly specific and verifiable \
throughout. 0.3-0.5 = mostly generic phrasing padded with keywords. Judge the writing itself, not how \
impressive the claimed skills sound.>
}

Extract only what is actually present or reasonably inferable from the text. Do not invent employers, \
titles, or skills that aren't there. Return ONLY the JSON object -- no markdown fences, no commentary."""

JD_SYSTEM_PROMPT = """You are an information extraction system. Given a raw job description, extract \
structured requirements and return ONLY a valid JSON object matching this schema:

{
  "title": "job title",
  "required_skills": ["list", "of", "required or clearly important skills/technologies, normalized"],
  "min_years_experience": <number, minimum years of experience required, 0 if not specified>,
  "required_education": "required education level/field as a short string, or null if not specified",
  "key_responsibilities": ["list", "of", "main responsibilities, one short phrase each"]
}

Return ONLY the JSON object -- no markdown fences, no commentary."""


def extract_candidate(llm: LLMClient, file_name: str, raw_text: str) -> CandidateProfile:
    # Cap input length defensively -- long resumes shouldn't blow the context
    # window or the per-call cost budget for what's a single-page-ish document.
    data = llm.complete_json(RESUME_SYSTEM_PROMPT, raw_text[:12000])
    return CandidateProfile(
        file_name=file_name,
        name=(data.get("name") or "Unknown").strip() or "Unknown",
        contact=(data.get("contact") or "").strip(),
        raw_text=raw_text,
        skills=[s.strip() for s in data.get("skills", []) if isinstance(s, str) and s.strip()],
        years_experience=_safe_float(data.get("years_experience", 0)),
        education=[
            Education(e.get("degree", ""), e.get("field", ""), e.get("institution", ""))
            for e in data.get("education", []) if isinstance(e, dict)
        ],
        roles=[
            Role(r.get("title", ""), r.get("company", ""), r.get("duration", ""), r.get("description", ""))
            for r in data.get("roles", []) if isinstance(r, dict)
        ],
        summary=(data.get("summary") or "").strip(),
        evidence_specificity=_clamp01(_safe_float(data.get("evidence_specificity", 1.0))),
    )


def extract_job_requirements(llm: LLMClient, raw_text: str) -> JobRequirements:
    data = llm.complete_json(JD_SYSTEM_PROMPT, raw_text[:8000])
    return JobRequirements(
        title=(data.get("title") or "").strip(),
        raw_text=raw_text,
        required_skills=[s.strip() for s in data.get("required_skills", []) if isinstance(s, str) and s.strip()],
        min_years_experience=_safe_float(data.get("min_years_experience", 0)),
        required_education=(data.get("required_education") or None),
        key_responsibilities=[
            r.strip() for r in data.get("key_responsibilities", []) if isinstance(r, str) and r.strip()
        ],
    )


def _safe_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))
