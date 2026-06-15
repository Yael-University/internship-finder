"""
Tests for the FastAPI service in api.py using Starlette's TestClient.

JobFitScorer, ATSScorer, and JobStore are replaced with in-memory fakes so the
endpoints are exercised with zero network calls, no Groq API key, and no ML
model downloads. Each fake mirrors only the surface the API actually touches.

Run with: pytest tests/test_api.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ── Fakes ──────────────────────────────────────────────────────────────────

class FakeScorer:
    """Stand-in for JobFitScorer — no Groq client constructed."""
    def __init__(self, groq_api_key: str = "test"):
        self.threshold = 7

    def score(self, resume_yaml: str, job) -> dict:
        return {"score": 8, "reasoning": "Strong overlap with data engineering.", "is_match": True}

    def chat(self, system: str, user: str, **kwargs) -> str:
        return "Based on the listings, the best match is the Data Engineer role."


class _FakeSemModel:
    """Deterministic stand-in embedding model (8-dim vectors)."""
    _DIM = 8

    def _vec(self, text: str) -> list[float]:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.random(self._DIM).tolist()

    def encode(self, text, convert_to_tensor: bool = False):
        if isinstance(text, list):
            return np.array([self._vec(t) for t in text])
        return np.array(self._vec(text))


class FakeATS:
    """Stand-in for ATSScorer with ML marked ready."""
    def __init__(self):
        self._ml_ready = True
        self._sem_model = _FakeSemModel()

    def analyze(self, resume_text: str, job) -> dict:
        return {
            "ats_score":        72,
            "keyword_score":    80,
            "semantic_score":   60,
            "matched_keywords": ["python", "sql"],
            "missing_keywords": ["airflow"],
            "critical_missing": ["airflow"],
            "tip":              "Add \"airflow\" to your resume.",
        }


class FakeStore:
    """In-memory stand-in for the ChromaDB-backed JobStore."""
    def __init__(self):
        self._items: list[dict] = []

    def count(self) -> int:
        return len(self._items)

    def upsert(self, company, role, location, description, fit_score, embedding) -> str:
        job_id = f"job-{len(self._items)}"
        self._items.append({
            "job_id": job_id,
            "document": f"Role: {role}\nCompany: {company}\nLocation: {location}\n\n{description[:600]}",
            "metadata": {"role": role, "company": company, "location": location, "fit_score": fit_score},
        })
        return job_id

    def search(self, query_embedding, n: int = 5) -> list[dict]:
        return [
            {**item, "similarity": 0.9}
            for item in self._items[:n]
        ]


# ── Fixture ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client(monkeypatch):
    import api

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(api, "JobFitScorer", FakeScorer)
    monkeypatch.setattr(api, "JobStore", FakeStore)
    monkeypatch.setattr(api, "_init_ats_scorer", lambda: FakeATS())
    # Auth is disabled unless API_KEY is set
    monkeypatch.setattr(api, "_API_KEY", "")

    with TestClient(api.app) as c:
        # The background loader runs asynchronously; set it deterministically so
        # endpoints that require the embedding model don't race startup.
        api._ats_scorer = FakeATS()
        yield c


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["groq_ready"] is True
    assert body["ats_models_loaded"] is True


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (307, 308)
    assert "/docs" in r.headers["location"]


# ── /score ────────────────────────────────────────────────────────────────────

def test_score_returns_fit_and_ats(client):
    r = client.post("/score", json={
        "resume": "python sql data engineering",
        "job_description": "We need a data engineer with python, sql, and airflow.",
        "role": "Data Engineer Intern",
        "company": "Acme",
        "location": "Remote",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["fit_score"] == 8
    assert body["is_match"] is True
    assert body["ats_score"] == 72
    assert body["matched_keywords"] == ["python", "sql"]
    assert body["stored"] is True


def test_score_stores_job_in_vector_db(client):
    payload = {
        "resume": "python sql",
        "job_description": "Data engineer needed.",
        "role": "DE", "company": "Acme", "location": "Remote",
    }
    client.post("/score", json=payload)
    r = client.get("/health")
    assert r.json()["jobs_stored"] == 1


# ── /search ────────────────────────────────────────────────────────────────────

def test_search_returns_stored_jobs(client):
    client.post("/score", json={
        "resume": "python sql",
        "job_description": "Data engineer needed at Acme.",
        "role": "Data Engineer", "company": "Acme", "location": "Remote",
    })
    r = client.post("/search", json={"query": "data engineering", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["total_stored"] == 1
    assert len(body["results"]) == 1
    assert body["results"][0]["company"] == "Acme"


def test_search_empty_store_returns_no_results(client):
    r = client.post("/search", json={"query": "anything", "top_k": 5})
    assert r.status_code == 200
    assert r.json()["results"] == []


# ── /ask ───────────────────────────────────────────────────────────────────────

def test_ask_with_no_jobs_prompts_to_score_first(client):
    r = client.post("/ask", json={"question": "What roles are available?"})
    assert r.status_code == 200
    body = r.json()
    assert body["total_stored"] == 0
    assert "score some jobs" in body["answer"].lower()


def test_ask_synthesizes_answer_from_store(client):
    client.post("/score", json={
        "resume": "python",
        "job_description": "Data engineer at Acme.",
        "role": "Data Engineer", "company": "Acme", "location": "Remote",
    })
    r = client.post("/ask", json={"question": "Which company is hiring?"})
    assert r.status_code == 200
    body = r.json()
    assert body["total_stored"] == 1
    assert isinstance(body["answer"], str) and body["answer"]
    assert len(body["sources"]) == 1


# ── /eval ──────────────────────────────────────────────────────────────────────

def test_eval_returns_fit_scores(client):
    r = client.get("/eval")
    assert r.status_code == 200
    body = r.json()
    assert "fit_scores" in body
    assert "retrieval" in body
    # The hand-labeled eval set yields a precision/recall report
    assert "precision" in body["fit_scores"] or "error" in body["fit_scores"]


# ── auth ───────────────────────────────────────────────────────────────────────

def test_score_rejects_bad_api_key(client, monkeypatch):
    import api
    monkeypatch.setattr(api, "_API_KEY", "secret")
    r = client.post("/score", json={
        "resume": "x", "job_description": "y", "role": "r", "company": "c", "location": "l",
    }, headers={"x-api-key": "wrong"})
    assert r.status_code == 401
