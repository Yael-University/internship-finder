"""
Job fit scorer: uses Groq (free API, Llama 3.3 70B) to evaluate how well the
user's resume matches each scraped job description and return a 1-10 score with
plain-English reasoning.

Get a free Groq API key at: https://console.groq.com
"""
from __future__ import annotations

import re
import time
import random

from groq import Groq

import config as cfg
from src.job import Job
from src.logging import logger
from src.preferences import build_system_prompt

_RETRY_EXCEPTIONS = ("rate_limit", "RateLimitError", "429", "too many requests")

_MODEL = "llama-3.3-70b-versatile"   # best free model on Groq

# Job-description slicing for the scoring prompt
_JD_CHAR_LIMIT = 2000   # max chars of JD sent to the LLM
_JD_INTRO_CHARS = 300   # leading context kept before the requirements section

_REQ_HEADERS = (
    "requirements", "qualifications", "what you'll need", "what we're looking for",
    "what you need", "minimum qualifications", "basic qualifications",
    "preferred qualifications", "you will need", "who you are",
    "what you bring", "skills required", "technical requirements",
)


def _extract_jd_for_scoring(desc: str, char_limit: int = _JD_CHAR_LIMIT) -> str:
    """
    Prioritize the requirements section of a job description over the company
    intro. Most postings bury technical requirements after ~1000 chars of preamble,
    so a naive [:1500] slice misses them entirely. If a recognizable section header
    is found, return 300 chars of intro context + the requirements section up to
    char_limit. Falls back to [:char_limit] if no header is found.
    """
    lower = desc.lower()
    earliest = len(desc)
    for header in _REQ_HEADERS:
        pos = lower.find(header)
        if 0 < pos < earliest:
            earliest = pos

    if earliest < len(desc) - 100:
        intro = desc[:_JD_INTRO_CHARS]
        reqs  = desc[earliest:earliest + (char_limit - _JD_INTRO_CHARS)]
        return (intro + "\n\n" + reqs)[:char_limit]
    return desc[:char_limit]


class JobFitScorer:
    def __init__(self, groq_api_key: str, *, system_prompt: str | None = None):
        self._client = Groq(api_key=groq_api_key)
        self.threshold: int = cfg.JOB_SUITABILITY_SCORE
        # When no prompt is supplied (e.g. the generic API path), fall back to the
        # default candidate profile. main.py passes a prompt built from
        # work_preferences.yaml so the tool is reusable by anyone.
        self.system_prompt: str = system_prompt or build_system_prompt()

    def chat(
        self,
        system: str,
        user: str,
        *,
        model: str = _MODEL,
        temperature: float = 0.3,
        max_tokens: int = 400,
    ) -> str:
        """
        Single-turn chat completion against Groq. Exposes the underlying client
        so callers (e.g. the RAG /ask endpoint) don't reach into private state.
        Returns the assistant message content (empty string if none).
        """
        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def score(self, resume_yaml: str, job: Job) -> dict:
        """
        Returns:
            {
                "score":     int  1-10,
                "reasoning": str,
                "is_match":  bool,
            }
        """
        if not job.description:
            return {"score": 0, "reasoning": "No job description available.", "is_match": False}

        user_content = (
            f"<resume>\n{resume_yaml}\n</resume>\n\n"
            f"<job_description>\n{_extract_jd_for_scoring(job.description)}\n</job_description>"
        )

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                response = self._client.chat.completions.create(
                    model=_MODEL,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user",   "content": user_content},
                    ],
                    temperature=0.2,
                    max_tokens=150,
                )
                raw = response.choices[0].message.content or ""

                score_m  = re.search(r"Score:\s*(\d+)", raw, re.IGNORECASE)
                reason_m = re.search(r"Reasoning:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)

                score     = min(10, max(1, int(score_m.group(1)))) if score_m else 5
                reasoning = reason_m.group(1).strip() if reason_m else raw.strip()

                logger.info(f"[Scorer] {job.role} @ {job.company} → {score}/10")
                return {
                    "score":     score,
                    "reasoning": reasoning,
                    "is_match":  score >= self.threshold,
                }

            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = any(k.lower() in err_str for k in _RETRY_EXCEPTIONS)
                is_daily_limit = "tokens per day" in err_str or "tpd" in err_str

                if is_daily_limit:
                    logger.warning(
                        f"[Scorer] Daily token quota exhausted — skipping remaining jobs. "
                        f"Resets at midnight UTC."
                    )
                    raise

                if is_rate_limit and attempt < max_attempts:
                    wait = min(2 ** attempt + random.uniform(0, 1), 60)
                    logger.warning(
                        f"[Scorer] Rate limited — retrying in {wait:.1f}s "
                        f"(attempt {attempt}/{max_attempts})"
                    )
                    time.sleep(wait)
                    continue

                logger.error(f"[Scorer] Failed for '{job.role}' at '{job.company}': {e}")
                return {"score": 0, "reasoning": f"Scoring error: {e}", "is_match": False}
