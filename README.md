# Resume Screening System

> An end-to-end AI-powered resume screening system that parses resumes, extracts structured candidate information, computes semantic relevance against a job description, and generates an explainable ranked shortlist.

Built for the **Rooman Technologies 24-Hour AI Challenge**.

---

## Overview

Recruiters often spend significant time manually reviewing resumes before identifying candidates worth interviewing. This project automates the initial screening process by combining Large Language Models with deterministic scoring techniques to produce transparent and reproducible candidate rankings.

Unlike many AI-powered resume screeners that ask an LLM to directly assign a score, this system separates **information extraction** from **decision making**.

The LLM is responsible for understanding unstructured resume content and converting it into structured candidate information. The final ranking is then computed using a deterministic hybrid scoring algorithm that combines semantic similarity, skill matching, and experience matching.

This design produces rankings that are more consistent, explainable, and easier to audit than relying solely on an LLM-generated score.

---

## Problem Statement

Given:

- A job description
- A folder containing multiple resumes

The system automatically:

- Parses resumes in **PDF**, **DOCX**, and **TXT** formats
- Extracts candidate information using an LLM
- Computes semantic similarity between each resume and the job description
- Evaluates skill overlap and experience fit
- Produces a ranked shortlist with score breakdowns
- Generates concise recruiter-friendly reasoning for every candidate
- Exports results as both **JSON** and **CSV**

---

## Key Features

- Resume parsing for PDF, DOCX, and TXT files
- LLM-powered structured information extraction
- Semantic similarity using Sentence Transformers
- Hybrid deterministic scoring pipeline
- Explainable candidate rankings
- Batch processing of multiple resumes in a single execution
- JSON and CSV export
- Modular architecture with separated parsing, extraction, scoring, and ranking components
- Configurable LLM provider through environment variables

---

## Project Workflow

```text
                    Job Description
                           │
                           ▼
              Extract Job Requirements
                           │
                           ▼
               Resume Parsing Pipeline
      (PDF / DOCX / TXT → Raw Resume Text)
                           │
                           ▼
          LLM Information Extraction
                           │
                           ▼
              Structured Candidate Profile
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
Semantic Similarity   Skill Matching   Experience Fit
          │                │                │
          └────────────────┼────────────────┘
                           ▼
               Hybrid Score Calculation
                           │
                           ▼
               Candidate Ranking Engine
                           │
                           ▼
         LLM Generated Candidate Explanation
                           │
                           ▼
            JSON + CSV Ranked Shortlist
```

---

## Example Use Case

A recruiter wants to hire a Backend Software Engineer.

Instead of manually reviewing twenty resumes, the recruiter provides:

- a job description
- a folder containing candidate resumes

The system automatically analyzes every resume, evaluates how well each candidate matches the role, ranks them by relevance, and generates a concise explanation describing the strengths and gaps of each candidate.

---

# Repository Structure

```text
resume-screening-system/
│
├── data/
│   ├── job_description.txt
│   └── resumes/
│       ├── sample_resume_1.pdf
│       ├── sample_resume_2.docx
│       ├── ...
│
├── output/
│   ├── ranked_candidates.json
│   ├── ranked_candidates.csv
│   └── sample_resume_screening_report.pdf
│
├── src/
│   ├── extractor.py
│   ├── llm_client.py
│   ├── models.py
│   ├── parser.py
│   ├── ranker.py
│   └── scorer.py
│
├── tests/
│
├── main.py
├── requirements.txt
├── .env.example
└── README.md
```

---

# Technology Stack

| Category | Technology |
|-----------|------------|
| Language | Python |
| LLM | Groq / OpenAI / Anthropic |
| NLP Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Similarity Metric | Cosine Similarity |
| Resume Parsing | pdfplumber, python-docx |
| Data Models | Dataclasses |
| Output | JSON, CSV |
| Testing | Pytest |

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/<your-username>/resume-screening-system.git

cd resume-screening-system
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure API Keys

Create a `.env` file.

You may use **any one** of the supported providers.

```text
GROQ_API_KEY=your_key_here

OPENAI_API_KEY=your_key_here

ANTHROPIC_API_KEY=your_key_here
```

Only one provider is required.

If multiple providers are configured, the project automatically selects one based on the implemented provider detection logic.

---

# Running the Project

Place:

- your job description inside `data/job_description.txt`
- your resumes inside `data/resumes/`

Then run:

```bash
python main.py \
    --jd data/job_description.txt \
    --resumes data/resumes/
```

After execution, ranked results are generated inside the `output/` directory.

```
output/

ranked_candidates.json

ranked_candidates.csv
```

Optionally, rationale generation may be disabled for faster execution using the available command-line flag implemented by the project.

---

# Sample Dataset

This repository includes sample resumes for demonstration purposes.

The resumes were collected from the publicly available **Resume Dataset** on Kaggle and are included solely to demonstrate the screening pipeline.

The repository also includes:

- sample job description
- sample ranked JSON output
- sample CSV ranking output
- a PDF resume screening report generated by the system

These artifacts allow reviewers to quickly understand the expected inputs and outputs before running the project themselves.

# How the System Works

The resume screening pipeline consists of five independent stages.

Each stage performs one well-defined task, making the overall system easier to maintain, debug, and extend.

## Step 1: Resume Parsing

The pipeline begins by reading every resume from the input directory.

Supported formats include:

- PDF
- DOCX
- TXT

Each parser extracts raw textual content while abstracting away the underlying document format.

After parsing, every resume is represented as plain text, allowing the remainder of the pipeline to operate independently of the original file type.

---

## Step 2: Structured Information Extraction

Raw resume text is difficult to compare directly because every candidate formats their resume differently.

To solve this, the project uses a Large Language Model to convert unstructured resumes into a standardized representation.

For each resume, the LLM extracts information such as:

- Candidate name
- Technical skills
- Years of experience
- Previous job roles
- Educational background
- Professional summary

Similarly, the job description is converted into a structured representation containing:

- Required skills
- Minimum experience
- Preferred qualifications
- Target role

After this stage, both resumes and the job description share a common structured format that can be evaluated programmatically.

---

## Step 3: Candidate Scoring

Rather than asking the LLM to decide which candidate is best, the project computes scores using deterministic algorithms implemented in Python.

Three independent scores are calculated.

### 1. Semantic Similarity

The project uses the Sentence Transformer model:

```
all-MiniLM-L6-v2
```

Both the candidate profile and job description are converted into embedding vectors.

Cosine similarity between these vectors measures how closely the resume matches the overall meaning of the job description rather than relying only on exact keyword matches.

This helps identify resumes that describe relevant experience using different wording.

---

### 2. Skill Match

Required skills extracted from the job description are compared against the candidate's extracted skills.

The score reflects the proportion of required skills matched by the candidate.

This ensures that candidates possessing core technical requirements receive appropriate credit.

---

### 3. Experience Match

The candidate's years of experience are compared against the minimum experience specified in the job description.

Candidates meeting or exceeding the requirement receive a higher experience score.

---

## Final Score

The overall candidate score is calculated using a weighted hybrid scoring strategy.

| Component | Weight |
|-----------|---------|
| Semantic Similarity | 40% |
| Skill Match | 40% |
| Experience Match | 20% |

This weighting prioritizes technical relevance and demonstrated skills while still considering professional experience.

The final score is computed deterministically, ensuring identical inputs always produce identical rankings.

---

## Step 4: Candidate Ranking

After every candidate receives a final score, candidates are sorted in descending order.

The ranking process is completely deterministic and independent of the LLM.

This guarantees consistent rankings across repeated executions.

---

## Step 5: Explanation Generation

Once ranking is complete, the LLM generates a concise explanation for each candidate.

Importantly, the explanation is generated **after** the score has already been calculated.

The model does not decide the ranking.

Instead, it explains:

- why the candidate scored highly
- which skills matched
- where experience aligned
- which important requirements were missing

This separation improves both transparency and reproducibility.

---

# Why Hybrid Scoring?

Many resume screening systems ask an LLM to directly assign a score.

For example:

> "Rate this resume from 1 to 10."

While simple to implement, this approach has several disadvantages:

- Results vary between executions.
- Scores are difficult to justify.
- Rankings become difficult to reproduce.
- Small prompt changes can significantly alter outcomes.

This project instead separates understanding from decision making.

The LLM is responsible only for extracting structured information from unstructured text and generating human-readable explanations.

All candidate ranking decisions are computed using deterministic algorithms implemented in Python.

This architecture provides:

- Consistent rankings
- Reproducible results
- Explainable score breakdowns
- Lower dependence on LLM behavior

---

# Why Sentence Transformers?

Semantic similarity is computed using the Sentence Transformers model:

```
all-MiniLM-L6-v2
```

This model was selected because it offers a practical balance between quality, speed, and computational cost.

Advantages include:

- Runs entirely locally
- No API cost
- No rate limits
- Fast CPU inference
- Strong semantic search performance
- Widely used in information retrieval applications

Although larger embedding models can achieve slightly better semantic accuracy, `all-MiniLM-L6-v2` provides excellent performance for resume matching while remaining lightweight enough for local execution.

---

# Project Architecture

The project follows a modular architecture where each component has a single responsibility.

```
                main.py
                   │
                   ▼
             Resume Parser
                   │
                   ▼
         Structured Extraction
                   │
                   ▼
            Hybrid Scoring
                   │
                   ▼
             Candidate Ranking
                   │
                   ▼
         Explanation Generation
                   │
                   ▼
            JSON / CSV Output
```

Each stage is isolated from the others, making it straightforward to replace individual components without affecting the remainder of the pipeline.

For example:

- a different embedding model can replace Sentence Transformers
- another LLM provider can replace the current provider
- additional scoring metrics can be introduced without modifying the parser

---

# Repository Components

## `main.py`

Entry point of the application.

Responsible for:

- loading inputs
- orchestrating the screening pipeline
- exporting final results

---

## `parser.py`

Extracts textual content from PDF, DOCX, and TXT resumes.

Acts as the document ingestion layer.

---

## `extractor.py`

Uses the configured LLM to convert raw text into structured candidate and job description objects.

---

## `scorer.py`

Implements the deterministic hybrid scoring algorithm.

Responsible for:

- semantic similarity
- skill matching
- experience scoring
- weighted final score calculation

---

## `ranker.py`

Coordinates candidate scoring, sorting, and explanation generation.

Produces the final ranked shortlist.

---

## `models.py`

Defines the project's core data models used throughout the pipeline.

---

## `llm_client.py`

Provides a unified interface for interacting with supported LLM providers.

This abstraction allows the remainder of the project to remain independent of the underlying provider implementation.

# Sample Data

This repository includes sample data to demonstrate the complete resume screening workflow.

## Job Description

The repository contains a sample job description located in:

```text
data/job_description.txt
```

This file represents the target role against which all candidate resumes are evaluated.

---

## Sample Resumes

The `data/resumes/` directory contains approximately 15–20 sample resumes collected from the publicly available **Resume Dataset** on Kaggle.

Dataset:

https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

These resumes are included solely for demonstration and testing purposes.

They allow reviewers to immediately execute the project without collecting their own sample data.

The screening pipeline supports:

- PDF
- DOCX
- TXT

Any additional resumes following these formats can be added to the directory and processed without modifying the source code.

---

## Sample Outputs

The repository also includes example outputs generated by the system.

Located in:

```text
output/
```

Included outputs:

- Ranked candidate list (JSON)
- Ranked candidate list (CSV)
- Resume screening report (PDF)

These files demonstrate the expected structure and format produced by the application after successfully processing a batch of resumes.

---

# Testing

The project includes automated tests covering the deterministic components of the ranking pipeline.

Run the test suite using:

```bash
pytest
```

The tests validate the scoring logic independently from external LLM providers, helping ensure that the ranking algorithm remains reproducible and reliable.

---

# Design Decisions

Several design choices were made to improve transparency, reproducibility, and maintainability.

## 1. Deterministic Scoring

The project intentionally avoids allowing the LLM to directly assign candidate scores.

Instead, the LLM is used only for:

- extracting structured information
- generating recruiter-friendly explanations

Final rankings are computed using deterministic Python logic.

This guarantees that identical inputs always produce identical rankings.

---

## 2. Modular Architecture

The system is divided into independent modules responsible for:

- parsing
- extraction
- scoring
- ranking
- provider communication

This separation makes the project easier to test, maintain, and extend.

---

## 3. Local Embeddings

Semantic similarity is computed locally using Sentence Transformers.

Advantages include:

- no embedding API cost
- no embedding rate limits
- low latency
- offline inference after the initial model download

---

## 4. Explainable Results

Instead of returning only a numerical score, the system generates concise explanations describing why each candidate ranked where they did.

This improves transparency for recruiters and makes ranking decisions easier to interpret.

---

# Limitations

Although the project provides a complete end-to-end screening pipeline, several limitations remain.

### OCR Support

Scanned or image-based resumes are not currently supported.

Future versions could integrate OCR or multimodal vision models for document understanding.

---

### Skill Matching

Skill matching currently relies primarily on extracted skill comparison.

Future improvements could incorporate embedding-based skill matching or ontology-driven normalization to better recognize synonymous technologies.

For example:

- JS ↔ JavaScript
- PyTorch ↔ Torch
- PostgreSQL ↔ Postgres

---

### Fixed Scoring Weights

The hybrid scoring weights are manually selected.

A production system could learn optimal weights from historical hiring decisions using supervised learning.

---

### LLM Availability

The extraction stage depends on an external LLM provider.

API outages or rate limits may temporarily interrupt resume extraction.

Potential improvements include:

- retry mechanisms
- exponential backoff
- response caching
- local fallback extraction

---

# Future Improvements

Potential future enhancements include:

- Streamlit or React-based user interface
- OCR support for scanned resumes
- Embedding-based skill normalization
- Resume embedding cache for faster repeated evaluations
- Batch processing optimizations
- Additional export formats
- Recruiter dashboard with interactive candidate filtering
- Configurable scoring weights
- Fine-tuned domain-specific extraction models
- Bias evaluation and fairness auditing

---

# Evaluation Requirements

This project satisfies the expected capabilities for the **Resume Screening System** challenge.

| Requirement | Status |
|------------|--------|
| Parse PDF, DOCX and TXT resumes | ✅ |
| Extract skills, experience and education | ✅ |
| Compute NLP similarity against the job description | ✅ |
| Rank candidates | ✅ |
| Generate reasoning | ✅ |
| Process multiple resumes in one execution | ✅ |
| Output JSON and CSV | ✅ |
| Include sample resumes | ✅ |
| Include sample job description | ✅ |
| Explain scoring methodology | ✅ |

---

# Conclusion

This project demonstrates an end-to-end AI-powered resume screening pipeline that combines Large Language Models with deterministic ranking techniques.

Rather than relying entirely on an LLM to make hiring decisions, the system separates language understanding from scoring.

This hybrid approach produces rankings that are transparent, reproducible, and easier to explain while maintaining the flexibility of modern language models for extracting information from unstructured resumes.

The modular architecture also makes the project straightforward to extend with additional scoring metrics, embedding models, LLM providers, or user interfaces.

---
