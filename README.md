# Internship Finder

A CLI tool that searches LinkedIn, Handshake, and Simplify for recently posted internship and co-op roles, then scores each one against your resume using an LLM and a two-layer ATS analyzer. Outputs a ranked Markdown report with apply links for manual application.

---

## How it works

Each run does three things in sequence:

**1. Scrape.** Three Selenium scrapers run in parallel — one browser per job site — pulling recently posted listings that match your configured positions, locations, and filters. Results are deduplicated before scoring.

**2. Score fit.** Each job is sent to [Groq](https://console.groq.com) (free tier) running Llama 3.3 70B. The model reads your resume and the job description and returns a 1–10 fit score with plain-English reasoning. A smart extraction step prioritizes the requirements section of the JD over company preamble, since most postings bury technical requirements after ~1,000 characters of marketing copy. Jobs below the score threshold (default: 7) are dropped.

**3. Score ATS.** Matched jobs get a two-layer ATS analysis at zero API cost:
- **Keyword layer** — skills are extracted from the JD using `jjzha/jobbert_skill_extraction` (an ML-based NER model trained on job postings), supplemented by a curated list of ~130 known tech terms. Each extracted skill is checked against your resume, with synonym matching (k8s ↔ kubernetes, ml ↔ machine learning, etc.)
- **Semantic layer** — your resume and the JD are encoded with `all-MiniLM-L6-v2` (sentence-transformers) and compared by cosine similarity, catching related phrasing that keyword matching misses
- Combined ATS score = 60% keyword + 40% semantic

The resume embedding is computed once at startup and reused across all jobs.

---

## Output

Each run writes a Markdown report and a JSON archive to `data_folder/output/reports/`:

```
# Job Matches — May 23, 2026

Searched 50 recently posted jobs across LinkedIn, Handshake, and Simplify.
38 jobs matched your resume (fit score ≥ 7/10).

---

## Fit 9/10 | ATS 64% — Data Engineer Intern | TikTok

| | |
|---|---|
| Location        | San Jose, CA |
| Source          | Simplify |
| Keyword match   | #######--- 67% |
| Semantic align  | ######---- 60% |
| Keywords you have | python, sql |
| Critical gaps   | express |

Why you match: ...

Quick ATS fix: Add "express" to your resume — likely ATS filter keyword.

→ Apply here: https://...
```

---

## Setup

### Requirements
- Python 3.11+
- Google Chrome installed
- A free [Groq API key](https://console.groq.com) (Llama 3.3 70B, free tier)
- LinkedIn and/or Handshake account credentials

### Install

```bash
git clone https://github.com/yaelmendez/internship-finder
cd internship-finder
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

The ATS ML models (`jobbert` + `all-MiniLM-L6-v2`) download automatically on first run (~500 MB total). If they fail to load, the scorer falls back to static keyword matching with no interruption.

### Configure

**`data_folder/secrets.yaml`** — API keys and site credentials:
```yaml
groq_api_key: "gsk_..."
linkedin_email: "you@example.com"
linkedin_password: "..."
handshake_email: "you@example.com"
handshake_password: "..."
```

**`data_folder/work_preferences.yaml`** — what to search for:
```yaml
remote: true
experience_level:
  internship: true
  entry: true
positions:
  - Data Engineer Intern
  - Machine Learning Engineer Intern
  - Software Engineer Intern
locations:
  - "United States"
date:
  month: true        # posted within the last month
max_jobs_to_score: 50
```

**`data_folder/plain_text_resume.yaml`** — your resume in plain text (YAML format). The LLM scorer and ATS analyzer both read from this file.

### Run

```bash
python main.py
```

The scraper opens browser windows, collects listings, then closes them. Scoring runs automatically. The terminal shows live progress; the final report path is printed when done.

---

## Architecture

```
main.py
├── ConfigValidator          reads and validates work_preferences.yaml
├── FileManager              manages output paths and report writing
└── search_and_score_jobs()
    ├── JobSearchManager     parallel Selenium scrapers (LinkedIn, Handshake, Simplify)
    ├── JobFitScorer         Groq / Llama 3.3 70B — resume × JD → score 1-10
    └── ATSScorer            jobbert NER + sentence-transformers cosine sim → ATS score 0-100
```

| File | Purpose |
|---|---|
| `main.py` | Entry point, config validation, report writer |
| `src/job_searcher.py` | Selenium scrapers + `JobSearchManager` |
| `src/job_fit_scorer.py` | LLM scoring via Groq API with retry/backoff |
| `src/ats_scorer.py` | Two-layer ATS analysis (jobbert + sentence-transformers) |
| `src/job.py` | `Job` dataclass |
| `src/utils/chrome_utils.py` | Browser initialization (standard + stealth mode for LinkedIn) |
| `config.py` | Score threshold, log settings |

---

## Configuration reference

| Setting | Default | Description |
|---|---|---|
| `JOB_SUITABILITY_SCORE` | `7` | Minimum Groq fit score (1–10) to include a job in the report |
| `max_jobs_to_score` | `50` | Cap on LLM calls per run (Groq free tier has a daily token limit) |
| `date.month` | `true` | Search within the last month |

---

## API

The scoring pipeline is also available as a deployed HTTP API.

### Endpoints

**`GET /health`** — liveness check

```json
{
  "status": "ok",
  "groq_ready": true,
  "ats_models_loaded": true
}
```

**`POST /score`** — score a job description against a resume; automatically stores the result in ChromaDB

```bash
curl -X POST https://internship-finder-h6sq.onrender.com/score \
  -H "Content-Type: application/json" \
  -d '{
    "resume": "Python, SQL, Apache Airflow, dbt...",
    "job_description": "We are looking for a Data Engineer intern...",
    "role": "Data Engineer Intern",
    "company": "Acme Corp"
  }'
```

```json
{
  "fit_score": 8,
  "fit_reasoning": "Strong match — candidate has Python, SQL, and pipeline experience...",
  "is_match": true,
  "ats_score": 74,
  "keyword_score": 80,
  "semantic_score": 65,
  "matched_keywords": ["python", "sql", "etl", "airflow"],
  "missing_keywords": ["snowflake", "dbt"],
  "critical_missing": ["snowflake"],
  "ats_tip": "Add \"snowflake\" to your resume — likely ATS filter keyword.",
  "stored": true
}
```

**`POST /search`** — semantic search over all previously scored jobs

```bash
curl -X POST https://internship-finder-h6sq.onrender.com/search \
  -H "Content-Type: application/json" \
  -d '{"query": "remote data engineering Python Airflow", "top_k": 5}'
```

```json
{
  "results": [
    {
      "role": "Data Engineer Intern",
      "company": "Rocket Lawyer",
      "location": "Remote",
      "fit_score": 9,
      "similarity": 0.91,
      "excerpt": "Role: Data Engineer Intern\nCompany: Rocket Lawyer..."
    }
  ],
  "total_stored": 12
}
```

**`POST /ask`** — RAG: retrieve relevant jobs, synthesize an answer via Groq

```bash
curl -X POST https://internship-finder-h6sq.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which stored jobs are fully remote and mention Airflow?", "top_k": 5}'
```

```json
{
  "answer": "Based on the stored listings, two remote positions mention Airflow: Data Engineer Intern at Rocket Lawyer and Data Engineering Intern at Teacher Retirement System of Texas...",
  "sources": [
    {"role": "Data Engineer Intern", "company": "Rocket Lawyer", "fit_score": 9, "similarity": 0.91}
  ],
  "total_stored": 12
}
```

### Deploy to Render

1. Push this repo to GitHub
2. Create a new Web Service on [Render](https://render.com), connect the repo, select **Docker** runtime
3. Set environment variables in the Render dashboard:
   - `GROQ_API_KEY` — your [Groq API key](https://console.groq.com) (free tier works)
   - `API_KEY` — optional, any string; if set, all `/score` requests must include `X-Api-Key: <value>`
4. Deploy — the `all-MiniLM-L6-v2` embedding model is baked into the image at build time so startup is instant

To run locally with Docker:

```bash
docker build -t internship-fit-scorer .
docker run -e GROQ_API_KEY=gsk_... -p 8000:8000 internship-fit-scorer
```

---

## Tech stack

- **LLM** — Groq API, Llama 3.3 70B (free tier)
- **Embeddings** — `sentence-transformers`, `all-MiniLM-L6-v2`
- **NER** — HuggingFace Transformers, `jjzha/jobbert_skill_extraction`
- **Scraping** — Selenium, `undetected-chromedriver`
- **Config** — PyYAML, Pydantic
- **Logging** — Loguru
- **Tests** — pytest
