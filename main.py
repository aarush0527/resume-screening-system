#!/usr/bin/env python3
"""Resume Screening Agent -- CLI entry point.

Usage:
    python main.py --jd data/job_description.txt --resumes data/resumes/

Writes output/ranked_candidates.json and output/ranked_candidates.csv by
default (override with --out).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.extractor import extract_candidate, extract_job_requirements
from src.llm_client import LLMClient, LLMError
from src.parser import ParseError, find_resume_files, parse_file
from src.ranker import rank_candidates
from src.scorer import ScoreWeights

load_dotenv()


def main() -> None:
    args = _parse_args()

    if not args.jd.exists():
        sys.exit(f"Job description not found: {args.jd}")
    if not args.resumes.is_dir():
        sys.exit(f"Resumes folder not found: {args.resumes}")

    try:
        llm = LLMClient()
    except LLMError as e:
        sys.exit(f"LLM setup failed: {e}\n\nSee .env.example for how to configure a provider.")

    print(f"Using LLM provider: {llm.provider} (model: {llm.model})")

    jd_text = args.jd.read_text(encoding="utf-8")
    print("Extracting job requirements from JD...")
    try:
        jd = extract_job_requirements(llm, jd_text)
    except LLMError as e:
        sys.exit(f"Failed to extract job requirements: {e}")
    print(f"  -> role: {jd.title or '(untitled)'}, "
          f"{len(jd.required_skills)} required skills, "
          f"{jd.min_years_experience}+ yrs experience")

    resume_files = find_resume_files(args.resumes)
    if not resume_files:
        sys.exit(f"No .pdf/.docx/.txt resumes found in {args.resumes}")
    print(f"Found {len(resume_files)} resume file(s)")

    candidates = []
    failures = []
    for path in resume_files:
        print(f"  Processing: {path.name}")
        try:
            raw_text = parse_file(path)
            candidates.append(extract_candidate(llm, path.name, raw_text))
        except ParseError as e:
            print(f"    SKIPPED (parse failed): {e}")
            failures.append({"file": path.name, "reason": str(e)})
        except LLMError as e:
            print(f"    SKIPPED (extraction failed): {e}")
            failures.append({"file": path.name, "reason": str(e)})

    if not candidates:
        sys.exit("No resumes could be processed successfully -- nothing to rank.")

    print(f"\nScoring and ranking {len(candidates)} candidate(s)"
          f"{' (rationale generation on)' if not args.no_rationale else ' (rationale generation off)'}...")
    ranked = rank_candidates(
        llm, candidates, jd, ScoreWeights(), generate_rationale=not args.no_rationale
    )

    _write_output(args.out, jd, ranked, failures, llm)
    print(f"\nWrote {args.out.with_suffix('.json')} and {args.out.with_suffix('.csv')}")
    if failures:
        print(f"({len(failures)} file(s) skipped -- see failed_files in the JSON output)")
    _print_summary(ranked)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank resumes against a job description.")
    parser.add_argument("--jd", required=True, type=Path, help="Path to job description text file")
    parser.add_argument("--resumes", required=True, type=Path, help="Folder of resumes (PDF/DOCX/TXT)")
    parser.add_argument(
        "--out", default=Path("output/ranked_candidates"), type=Path,
        help="Output path prefix -- writes <prefix>.json and <prefix>.csv (default: output/ranked_candidates)",
    )
    parser.add_argument(
        "--no-rationale", action="store_true",
        help="Skip per-candidate LLM rationale generation (faster, cheaper, scores only)",
    )
    return parser.parse_args()


def _write_output(out_prefix: Path, jd, ranked, failures, llm) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "job_title": jd.title,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "llm_provider": llm.provider,
        "llm_model": llm.model,
        "embedding_model": "all-MiniLM-L6-v2",
        "candidates": [asdict(c) for c in ranked],
        "failed_files": failures,
    }
    out_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with out_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "file_name", "name", "final_score", "semantic_similarity",
            "skill_overlap", "evidence_specificity", "experience_fit", "years_experience",
            "matched_skills", "missing_skills", "rationale",
        ])
        for c in ranked:
            writer.writerow([
                c.rank, c.file_name, c.name, c.final_score,
                c.score_breakdown.semantic_similarity, c.score_breakdown.skill_overlap,
                c.score_breakdown.evidence_specificity, c.score_breakdown.experience_fit,
                c.years_experience,
                "; ".join(c.matched_skills), "; ".join(c.missing_skills), c.rationale,
            ])


def _print_summary(ranked) -> None:
    print("\nRanked shortlist:")
    for c in ranked:
        print(f"  #{c.rank:<3} {c.name:30s} score={c.final_score:.2f}  ({c.file_name})")


if __name__ == "__main__":
    main()
