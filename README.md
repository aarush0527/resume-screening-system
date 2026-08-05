# Resume Screening System

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/SentenceTransformers-all--MiniLM--L6--v2-orange)
![Pydantic](https://img.shields.io/badge/Pydantic-Validation-E92063?logo=pydantic&logoColor=white)
![Pytest](https://img.shields.io/badge/Tested%20with-Pytest-0A9EDC?logo=pytest)
![LLM Ready](https://img.shields.io/badge/LLM-Groq%20%7C%20OpenAI%20%7C%20Anthropic-7B3FE4)

An end-to-end AI-powered resume screening system that automatically parses resumes, extracts structured candidate information, measures semantic relevance against a job description, and produces an explainable ranked shortlist using a hybrid scoring pipeline.

Instead of relying on an LLM to directly assign candidate scores, this project combines deterministic scoring with language models to create a ranking process that is both reproducible and transparent.

The system supports PDF, DOCX, and TXT resumes, processes multiple candidates in a single run, and exports detailed ranking results in both JSON and CSV formats.

---

## Overview

Recruiters often receive dozens or even hundreds of resumes for a single position. Manually reviewing every application is time-consuming, inconsistent, and difficult to scale.

This project automates the initial screening process by:

* Parsing resumes from multiple document formats
* Extracting structured candidate information using an LLM
* Measuring semantic similarity between each resume and the target job description
* Evaluating skills and professional experience using deterministic scoring
* Ranking all candidates based on a weighted hybrid score
* Generating concise, evidence-based explanations for every ranking decision

The goal is not to replace human decision-making, but to provide a consistent, explainable first-pass screening system that significantly reduces manual effort.

---

## Features

### Resume Parsing

Supports multiple resume formats out of the box.

* PDF
* DOCX
* TXT

Each document is converted into plain text before entering the extraction pipeline.

---

### Structured Candidate Extraction

Raw resume text is converted into structured candidate profiles using an LLM.

The extracted information includes:

* Candidate name
* Skills
* Professional experience
* Education
* Previous roles
* Resume summary
* Evidence specificity score

Similarly, the job description is converted into a structured representation containing:

* Role title
* Required skills
* Preferred qualifications
* Minimum experience

---

### Hybrid Candidate Scoring

Candidate ranking is intentionally **not** performed by the language model.

Instead, each resume receives three independent scores:

* Semantic similarity
* Skill overlap
* Experience fit

These scores are combined using fixed weights to produce a reproducible final ranking.

This design ensures that identical inputs always produce identical rankings.

---

### Semantic Resume Matching

The project uses Sentence Transformers to compare the meaning of resumes and job descriptions instead of relying solely on keyword matching.

Embedding model:

```
sentence-transformers/all-MiniLM-L6-v2
```

Semantic similarity is calculated using cosine similarity between resume and job description embeddings.

---

### Explainable Rankings

Every ranked candidate includes a concise explanation describing:

* strengths
* missing requirements
* relevant skills
* overall fit for the position

Importantly, the LLM explains the computed score rather than generating the score itself.

---

### Batch Processing

The system processes an entire directory of resumes in one execution.

Current repository includes:

* Sample Job Description
* 15+ sample resumes
* Example ranking outputs
* JSON output
* PDF report

---

### Export Formats

Results are automatically exported as:

* JSON (complete structured output)
* CSV (easy recruiter review)

---

## Architecture

```text
                    Job Description
                           │
                           ▼
                 Requirement Extraction
                      (LLM Processing)
                           │
                           ▼

 Resume Folder ─────► Resume Parsing
                  PDF • DOCX • TXT
                           │
                           ▼
                Candidate Information
                   Extraction (LLM)
                           │
                           ▼
                 Hybrid Scoring Engine

             • Semantic Similarity
             • Skill Matching
             • Experience Matching

                           │
                           ▼
                  Candidate Ranking
                           │
                           ▼
                Explanation Generation
                           │
                           ▼
              JSON + CSV Ranking Output
```

---

## Technology Stack

| Category | Technologies |
|-----------|--------------|
| Language | Python |
| LLM Providers | Groq, OpenAI, Anthropic |
| NLP | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Similarity Metric | Cosine Similarity |
| PDF Parsing | pdfplumber |
| DOCX Parsing | python-docx |
| Data Validation | Pydantic |
| Configuration | python-dotenv |
| Testing | pytest |

---

## Repository Structure

```
resume-screening-system/
│
├── data/
│   ├── resumes/
│   └── job_description.txt
│
├── output/
│   ├── ranked_candidates_example.json
│   └── ranked_candidates_example.csv
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
└── README.md
```

Each module is responsible for a single stage of the screening pipeline, making the system modular, maintainable, and easy to extend.

## Getting Started

### Prerequisites

Before running the project, ensure you have the following installed:

* Python 3.10 or later
* pip
* Git

An API key from any one of the supported providers:

* Groq
* OpenAI
* Anthropic

> **Note**
> Only one provider is required. The application automatically detects which provider to use based on the available API key.

---

## Installation

Clone the repository.

```bash
git clone https://github.com/<your-username>/resume-screening-system.git
```

Move into the project directory.

```bash
cd resume-screening-system
```

Create a virtual environment.

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

Install all dependencies.

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root.

Example:

```env
GROQ_API_KEY=your_groq_api_key
```

or

```env
OPENAI_API_KEY=your_openai_api_key
```

or

```env
ANTHROPIC_API_KEY=your_anthropic_api_key
```

Only one provider is necessary.

During startup, the application automatically detects the available provider and initializes the corresponding client.

---

## Running the Project

The repository already contains:

* a sample job description
* multiple sample resumes
* example outputs

Running the complete pipeline requires only a single command.

```bash
python main.py \
    --jd data/job_description.txt \
    --resumes data/resumes
```

If everything is configured correctly, the application will:

1. Read the job description
2. Parse every resume
3. Extract structured information
4. Compute candidate scores
5. Rank every applicant
6. Generate explanations
7. Export the final results

---

## Optional Arguments

| Argument | Description |
|-----------|-------------|
| `--jd` | Path to the job description |
| `--resumes` | Directory containing resumes |
| `--out` | Output file prefix |
| `--no-rationale` | Skip explanation generation for faster execution |

Example:

```bash
python main.py \
    --jd custom_jd.txt \
    --resumes ./candidate_folder \
    --out ./results/backend_hiring
```

---

## Example Workflow

```
Job Description
        │
        ▼

Read Job Requirements
        │
        ▼

Load Resume Folder
        │
        ▼

Parse Documents
(PDF / DOCX / TXT)
        │
        ▼

Extract Structured Profiles
        │
        ▼

Compute Similarity Scores
        │
        ▼

Rank Candidates
        │
        ▼

Generate Explanations
        │
        ▼

Export JSON + CSV
```

---

## Sample Data

The repository includes everything required to execute the project without preparing additional data.

### Job Description

```
data/job_description.txt
```

Contains a complete backend software engineering job description used during development and testing.

---

### Sample Resumes

```
data/resumes/
```

The repository includes a collection of sample resumes covering a broad range of candidate profiles.

These resumes originate from the public Kaggle Resume Dataset and are included solely for demonstrating and evaluating the screening pipeline.

The collection intentionally contains candidates with varying levels of experience, technical backgrounds, and overall relevance to the provided job description. This allows the ranking system to be evaluated across strong, moderate, and weak matches.

Supported formats include:

* PDF
* DOCX
* TXT

---

### Sample Outputs

```
output/
```

The repository also contains example outputs generated by the screening pipeline.

Included examples demonstrate:

* ranked candidate list
* score breakdown
* generated reasoning
* structured JSON output
* PDF report

These files allow the complete output format to be inspected without executing the application.

---

## Output

### JSON

The JSON output contains complete information for every screened candidate.

Each record includes:

* candidate profile
* semantic similarity score
* skill score
* experience score
* final weighted score
* ranking
* explanation

This format is intended for downstream applications or further analysis.

---

### CSV

The CSV export provides a recruiter-friendly overview containing the highest priority information.

Typical columns include:

* Rank
* Candidate
* Final Score
* Semantic Score
* Skill Score
* Experience Score
* Summary

The CSV is designed for quick review inside spreadsheet software.

---

## Performance Notes

The embedding model is downloaded automatically the first time the project is executed.

This is a one-time download.

Subsequent executions reuse the locally cached model.

Because embeddings are generated locally, semantic similarity computation incurs no API cost and is unaffected by external rate limits.

Only structured information extraction and explanation generation require LLM API calls.

# Design Decisions

The system is designed around a simple principle:

> **Language models should understand information, not make hiring decisions.**

Large language models excel at interpreting unstructured text, but allowing them to directly score candidates introduces several practical problems.

* Scores become non-deterministic across runs.
* Rankings become difficult to justify.
* Small prompt changes can significantly alter results.
* The decision-making process becomes difficult to audit.

Instead of asking an LLM to answer questions such as:

> "How suitable is this candidate for the role?"

the language model is only responsible for transforming unstructured documents into structured information.

Once candidate information has been extracted, all ranking decisions are performed using deterministic scoring implemented in Python.

This separation provides consistent results, improves transparency, and makes the scoring process reproducible.

---

# Scoring Methodology

Each candidate receives three independent scores.

## 1. Semantic Similarity

The semantic similarity score measures how closely the overall resume aligns with the job description.

Unlike traditional keyword matching, semantic embeddings capture contextual meaning.

For example:

```
Backend REST API Development
```

and

```
Designed scalable backend services exposing REST endpoints
```

may not share many identical words, but describe highly similar concepts.

To capture this relationship, both the resume and the job description are converted into dense vector representations using Sentence Transformers.

Embedding Model

```
sentence-transformers/all-MiniLM-L6-v2
```

Similarity is computed using cosine similarity between the two vectors.

This score reflects the overall contextual alignment between the candidate and the role.

---

## 2. Skill Matching

Semantic similarity alone is not sufficient.

Two resumes may discuss similar topics while possessing very different technical skills.

The second component explicitly compares candidate skills against the required skills extracted from the job description.

Examples include:

* Python
* PostgreSQL
* REST APIs
* AWS
* Docker
* Kubernetes

The overlap is normalized into a score between 0 and 1 before contributing to the final ranking.

This ensures candidates possessing the required technologies are appropriately rewarded.

---

## 3. Experience Fit

Experience requirements are evaluated independently from semantic similarity and skill matching.

The candidate's years of professional experience are compared against the minimum experience specified within the job description.

Candidates meeting or exceeding the requirement receive the highest score, while candidates with less experience receive proportionally lower scores.

Separating experience into its own component prevents highly relevant junior candidates from being ranked above significantly more experienced applicants solely due to textual similarity.

---

## Final Score

The three independent scores are combined using fixed weights.

| Component | Weight |
|-----------|--------|
| Semantic Similarity | 40% |
| Skill Match | 40% |
| Experience Fit | 20% |

The weighted combination produces a final score between 0 and 1.

This score is entirely deterministic.

Running the project multiple times on identical inputs will always produce the same ranking.

---

# Why Hybrid Scoring?

A purely embedding-based ranking often struggles with explicit hiring requirements.

For example, two candidates may have highly similar resumes while only one possesses a mandatory technology such as PostgreSQL or Kubernetes.

Conversely, simple keyword matching ignores context and often rewards keyword stuffing.

Combining semantic similarity with structured rule-based evaluation provides a more balanced ranking.

Semantic similarity captures contextual relevance.

Skill matching validates required technologies.

Experience scoring ensures minimum professional requirements are considered.

The result is a ranking process that is significantly more robust than relying on any individual method.

---

# Explainability

After the ranking has been computed, the language model generates a concise explanation for each candidate.

Importantly, these explanations do not determine the ranking.

Instead, they summarize the computed results by highlighting:

* matching skills
* relevant experience
* missing qualifications
* overall strengths
* potential gaps

This keeps the decision-making process transparent while still providing human-readable feedback.

---

# Why Sentence Transformers?

Several approaches were considered for measuring resume similarity.

Traditional keyword matching lacks contextual understanding and fails when different terminology describes the same concept.

Commercial embedding APIs provide excellent quality but introduce recurring API costs and rate limits.

Sentence Transformers offers an effective balance between accuracy, speed, and cost.

The chosen embedding model,

```
all-MiniLM-L6-v2
```

runs entirely on the local machine, requires no external inference service, and performs well for semantic similarity tasks while remaining lightweight enough for CPU execution.

This makes it particularly suitable for local batch processing.

---

# Why Local Embeddings?

Embedding generation is performed locally rather than through an external API.

Advantages include:

* no embedding API cost
* no rate limits
* lower latency
* offline embedding generation after the initial model download
* consistent performance regardless of batch size

Only information extraction and explanation generation require LLM inference.

Semantic similarity remains entirely local.

---

# Modular Architecture

Each stage of the pipeline is intentionally isolated into its own module.

| Module | Responsibility |
|---------|----------------|
| `parser.py` | Document parsing |
| `extractor.py` | Structured information extraction |
| `scorer.py` | Hybrid candidate scoring |
| `ranker.py` | Candidate ranking and explanation generation |
| `llm_client.py` | Provider abstraction |
| `models.py` | Shared data models |

Keeping each responsibility independent makes the project easier to test, maintain, and extend.

For example, replacing the embedding model or introducing a new document parser requires minimal changes to the remainder of the codebase.

# Testing

The deterministic components of the ranking pipeline are covered using automated unit tests.

Tests validate the correctness of:

* Semantic similarity computation
* Skill matching logic
* Experience scoring
* Final weighted score calculation
* Candidate ranking behavior

Run the test suite using:

```bash
pytest tests/ -v
```

Because the scoring engine is deterministic, identical inputs always produce identical outputs, making the system straightforward to test and validate.

---

# Performance Considerations

The project is designed primarily for batch resume screening rather than real-time inference.

Some implementation choices were made with efficiency in mind:

### Local Embeddings

Semantic embeddings are generated locally using Sentence Transformers.

The embedding model is downloaded only once and reused for future executions.

This avoids repeated API calls while providing fast similarity computation for large resume batches.

---

### Lightweight Document Parsing

Resume parsing uses lightweight libraries specialized for each supported format:

* `pdfplumber`
* `python-docx`
* native text loading

Only the extracted text proceeds through the remainder of the pipeline.

---

### Modular Processing

Parsing, extraction, scoring, ranking, and explanation generation are separated into independent stages.

This makes future optimizations, such as parallel resume processing or embedding caching, straightforward to introduce without redesigning the entire system.

---

# Current Limitations

Although the project provides a complete end-to-end screening workflow, there are several areas where production systems would typically go further.

### Scanned PDFs

The parser currently supports text-based PDF documents.

Image-only or scanned resumes require OCR before text extraction and are outside the current scope.

---

### Exact Skill Matching

Skill comparison primarily relies on normalized string matching.

While this performs well for standard terminology, it may not recognize synonymous technologies.

For example:

```
JS
```

and

```
JavaScript
```

may not always be treated as equivalent.

A semantic skill matching approach would improve robustness.

---

### Static Score Weights

The hybrid scoring weights are fixed.

```
Semantic Similarity : 40%
Skill Match         : 40%
Experience Fit      : 20%
```

These values were chosen to provide balanced rankings across the included sample data.

In a production environment, these weights could be learned directly from historical hiring outcomes or adjusted for different job families.

---

### LLM Availability

Structured information extraction depends on access to an external language model.

If the selected provider is unavailable or rate-limited, extraction cannot continue until the service becomes available.

Introducing automatic retries, provider failover, and extraction caching would improve resilience.

---

### Bias Considerations

Personal identifiers are intentionally excluded from the scoring pipeline wherever practical.

However, no automated hiring system can completely eliminate potential sources of bias.

The generated rankings should therefore be viewed as decision support rather than fully autonomous hiring decisions.

Human review remains an essential part of the recruitment process.

---

# Future Improvements

Several enhancements could further improve both performance and production readiness.

## Intelligent Skill Matching

Replace exact string comparison with embedding-based skill similarity or ontology-driven matching.

This would improve recognition of equivalent technologies and related terminology.

---

## Embedding Cache

Persist embeddings between executions to avoid recomputing representations for resumes that have already been processed.

---

## OCR Support

Add an OCR pipeline to support scanned resumes and image-based PDF documents.

---

## Parallel Processing

Process multiple resumes concurrently to improve throughput for larger candidate pools.

---

## Automatic Retry Logic

Implement retry strategies with exponential backoff for temporary LLM failures and API rate limits.

---

## Multi-Role Screening

Support evaluating candidates against multiple job descriptions within a single execution.

---

## Recruiter Feedback Loop

Allow recruiters to provide feedback on ranking quality and use that information to continuously improve future recommendations.

---

# Example Use Cases

The system can be adapted to a variety of recruitment workflows, including:

* Technical hiring
* Graduate recruitment
* Internship screening
* Internal candidate evaluation
* Resume database search
* Recruitment process automation

Although demonstrated using a backend engineering role, the pipeline is designed to operate with any textual job description.

---

# Project Highlights

✔ Supports PDF, DOCX, and TXT resumes

✔ Processes multiple resumes in a single execution

✔ Uses semantic embeddings for contextual matching

✔ Hybrid deterministic scoring

✔ Explainable ranking output

✔ JSON and CSV exports

✔ Modular architecture

✔ Automated unit tests

✔ Provider-agnostic LLM support

✔ Easily extensible design

---

# Acknowledgements

Sample resumes included in the repository are derived from the public **Kaggle Resume Dataset** and are used exclusively for demonstrating the resume screening pipeline.



