"""
Vector store for job listings using ChromaDB.

Jobs are embedded with sentence-transformers (all-MiniLM-L6-v2) and stored
for semantic search and RAG. The calling layer provides pre-computed embeddings
so this module has no model dependency of its own.

Storage mode:
  - CHROMA_IN_MEMORY=1 (default) — ephemeral, resets on restart; fine for deployment
  - CHROMA_IN_MEMORY=0            — persistent to ./chroma_db; used locally
"""
from __future__ import annotations

import hashlib
import os

import chromadb


def _job_id(company: str, role: str, description: str) -> str:
    key = f"{company.lower()}|{role.lower()}|{description[:80]}"
    return hashlib.md5(key.encode()).hexdigest()


class JobStore:
    def __init__(self) -> None:
        if os.environ.get("CHROMA_IN_MEMORY", "1") == "1":
            self._client = chromadb.EphemeralClient()
        else:
            self._client = chromadb.PersistentClient(path="./chroma_db")

        # No embedding function — embeddings are passed in from sentence-transformers
        self._col = self._client.get_or_create_collection(
            name="jobs",
            metadata={"hnsw:space": "cosine"},
        )

    # ── write ─────────────────────────────────────────────────────────────────

    def upsert(
        self,
        company: str,
        role: str,
        location: str,
        description: str,
        fit_score: int,
        embedding: list[float],
    ) -> str:
        job_id = _job_id(company, role, description)
        # Store a concise document for RAG context
        document = (
            f"Role: {role}\nCompany: {company}\nLocation: {location}\n\n"
            f"{description[:600]}"
        )
        self._col.upsert(
            ids=[job_id],
            documents=[document],
            metadatas=[{
                "role":      role,
                "company":   company,
                "location":  location,
                "fit_score": fit_score,
            }],
            embeddings=[embedding],
        )
        return job_id

    # ── read ──────────────────────────────────────────────────────────────────

    def search(self, query_embedding: list[float], n: int = 5) -> list[dict]:
        count = self._col.count()
        if count == 0:
            return []
        results = self._col.query(
            query_embeddings=[query_embedding],
            n_results=min(n, count),
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for i in range(len(results["ids"][0])):
            out.append({
                "job_id":     results["ids"][0][i],
                "document":   results["documents"][0][i],
                "metadata":   results["metadatas"][0][i],
                "similarity": round(1 - results["distances"][0][i], 3),
            })
        return out

    def count(self) -> int:
        return self._col.count()
