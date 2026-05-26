"""
FastAPI service wrapping the job scoring pipeline.

Endpoints:
  GET  /health  — liveness check
  POST /score   — score a job description against a resume
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from src.ats_scorer import ATSScorer
from src.job import Job
from src.job_fit_scorer import JobFitScorer

_fit_scorer: Optional[JobFitScorer] = None
_ats_scorer: Optional[ATSScorer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _fit_scorer, _ats_scorer

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        _fit_scorer = JobFitScorer(groq_api_key=groq_key)

    # Initialise with a placeholder resume to trigger model loading.
    # analyze() computes a fresh embedding per request, so this value
    # doesn't affect scoring results.
    _ats_scorer = ATSScorer(resume_text="python sql machine learning data engineering")

    yield


app = FastAPI(
    title="Internship Fit Scorer",
    description=(
        "Scores job descriptions against a resume using Groq (Llama 3.3 70B) "
        "for fit reasoning and sentence-transformers cosine similarity for ATS analysis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

_API_KEY = os.environ.get("API_KEY", "")


def _check_auth(x_api_key: Optional[str]) -> None:
    if _API_KEY and x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Request / response models ─────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    resume: str
    job_description: str
    role: str = ""
    company: str = ""
    location: str = ""


class ScoreResponse(BaseModel):
    fit_score: int
    fit_reasoning: str
    is_match: bool
    ats_score: int
    keyword_score: int
    semantic_score: int
    matched_keywords: list[str]
    missing_keywords: list[str]
    critical_missing: list[str]
    ats_tip: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "groq_ready": _fit_scorer is not None,
        "ats_models_loaded": _ats_scorer is not None and _ats_scorer._ml_ready,
    }


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest, x_api_key: Optional[str] = Header(default=None)):
    _check_auth(x_api_key)

    if not _fit_scorer:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY not configured")
    if not _ats_scorer:
        raise HTTPException(status_code=503, detail="ATS scorer not initialised")

    job = Job(
        role=req.role or "Unknown Role",
        company=req.company or "Unknown Company",
        location=req.location,
        description=req.job_description,
        apply_method="api",
    )

    fit = _fit_scorer.score(resume_yaml=req.resume, job=job)

    ats = _ats_scorer.analyze(resume_text=req.resume, job=job)

    return ScoreResponse(
        fit_score=fit["score"],
        fit_reasoning=fit["reasoning"],
        is_match=fit["is_match"],
        ats_score=ats["ats_score"],
        keyword_score=ats["keyword_score"],
        semantic_score=ats["semantic_score"],
        matched_keywords=ats["matched_keywords"],
        missing_keywords=ats["missing_keywords"],
        critical_missing=ats["critical_missing"],
        ats_tip=ats["tip"],
    )
