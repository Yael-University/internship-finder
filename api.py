"""
FastAPI service wrapping the job scoring pipeline.

Endpoints:
  GET  /health  — liveness check, shows model + store status
  POST /score   — score a job description against a resume; stores result in ChromaDB
  POST /search  — semantic search over all stored jobs
  POST /ask     — RAG: retrieve relevant jobs, synthesize answer via Groq
  GET  /eval    — run offline eval against data_folder/eval_labels.json
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from src.ats_scorer import ATSScorer
from src.job import Job
from src.job_fit_scorer import JobFitScorer
from src.job_store import JobStore

_fit_scorer: Optional[JobFitScorer] = None
_ats_scorer: Optional[ATSScorer] = None
_job_store:  Optional[JobStore]   = None


def _init_ats_scorer() -> ATSScorer:
    return ATSScorer(resume_text="python sql machine learning data engineering")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _fit_scorer, _ats_scorer, _job_store

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        _fit_scorer = JobFitScorer(groq_api_key=groq_key)

    _job_store = JobStore()

    # Load sentence-transformers in a thread so the event loop stays free
    # and Render's health-check can reach /health immediately on startup.
    async def _bg_load():
        global _ats_scorer
        loop = asyncio.get_event_loop()
        _ats_scorer = await loop.run_in_executor(None, _init_ats_scorer)

    asyncio.create_task(_bg_load())

    yield


app = FastAPI(
    title="Internship Fit Scorer",
    description=(
        "Scores job descriptions against a resume using Groq (Llama 3.3 70B) "
        "for fit reasoning and sentence-transformers cosine similarity for ATS analysis. "
        "Stores results in ChromaDB for semantic search and RAG."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

_API_KEY = os.environ.get("API_KEY", "")


def _check_auth(x_api_key: Optional[str]) -> None:
    if _API_KEY and x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _embed(text: str) -> list[float]:
    """Embed text using the loaded sentence-transformers model."""
    if _ats_scorer and _ats_scorer._sem_model is not None:
        return _ats_scorer._sem_model.encode(text, convert_to_tensor=False).tolist()
    return []


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
    stored: bool


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    role: str
    company: str
    location: str
    fit_score: int
    similarity: float
    excerpt: str


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total_stored: int


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    total_stored: int


class EvalResponse(BaseModel):
    fit_scores: dict
    retrieval: dict


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {
        "status":            "ok",
        "groq_ready":        _fit_scorer is not None,
        "ats_models_loaded": _ats_scorer is not None and _ats_scorer._ml_ready,
        "jobs_stored":       _job_store.count() if _job_store else 0,
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

    # Store in vector DB if embedding model is ready
    stored = False
    if _job_store and _ats_scorer._ml_ready:
        embedding = _embed(req.job_description)
        if embedding:
            _job_store.upsert(
                company=job.company,
                role=job.role,
                location=job.location,
                description=req.job_description,
                fit_score=fit["score"],
                embedding=embedding,
            )
            stored = True

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
        stored=stored,
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, x_api_key: Optional[str] = Header(default=None)):
    _check_auth(x_api_key)

    if not _job_store:
        raise HTTPException(status_code=503, detail="Job store not initialised")
    if not _ats_scorer or not _ats_scorer._ml_ready:
        raise HTTPException(status_code=503, detail="Embedding model not ready")

    embedding = _embed(req.query)
    if not embedding:
        raise HTTPException(status_code=503, detail="Failed to embed query")

    hits = _job_store.search(query_embedding=embedding, n=req.top_k)

    results = [
        SearchResult(
            role=h["metadata"].get("role", ""),
            company=h["metadata"].get("company", ""),
            location=h["metadata"].get("location", ""),
            fit_score=h["metadata"].get("fit_score", 0),
            similarity=h["similarity"],
            excerpt=h["document"][:300],
        )
        for h in hits
    ]

    return SearchResponse(results=results, total_stored=_job_store.count())


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, x_api_key: Optional[str] = Header(default=None)):
    _check_auth(x_api_key)

    if not _fit_scorer:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY not configured")
    if not _job_store:
        raise HTTPException(status_code=503, detail="Job store not initialised")
    if not _ats_scorer or not _ats_scorer._ml_ready:
        raise HTTPException(status_code=503, detail="Embedding model not ready")

    total = _job_store.count()
    if total == 0:
        return AskResponse(
            answer="No jobs stored yet. Score some jobs first via POST /score.",
            sources=[],
            total_stored=0,
        )

    # Retrieve
    embedding = _embed(req.question)
    hits = _job_store.search(query_embedding=embedding, n=req.top_k)

    # Build context
    context_parts = []
    for i, h in enumerate(hits, 1):
        meta = h["metadata"]
        context_parts.append(
            f"[Job {i}] {meta.get('role', '')} at {meta.get('company', '')} "
            f"({meta.get('location', '')}) — Fit score: {meta.get('fit_score', '?')}/10\n"
            f"{h['document'][h['document'].find(chr(10)*2)+2:][:400]}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # Generate
    response = _fit_scorer._client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant answering questions about job listings. "
                    "Answer using only the provided listings. Be concise and specific."
                ),
            },
            {
                "role": "user",
                "content": f"Job listings:\n\n{context}\n\nQuestion: {req.question}",
            },
        ],
        temperature=0.3,
        max_tokens=400,
    )

    answer = response.choices[0].message.content or ""

    sources = [
        {
            "role":       h["metadata"].get("role", ""),
            "company":    h["metadata"].get("company", ""),
            "fit_score":  h["metadata"].get("fit_score", 0),
            "similarity": h["similarity"],
        }
        for h in hits
    ]

    return AskResponse(answer=answer, sources=sources, total_stored=total)
