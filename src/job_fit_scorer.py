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

_RETRY_EXCEPTIONS = ("rate_limit", "RateLimitError", "429", "too many requests")

_MODEL = "llama-3.3-70b-versatile"   # best free model on Groq

_REQ_HEADERS = (
    "requirements", "qualifications", "what you'll need", "what we're looking for",
    "what you need", "minimum qualifications", "basic qualifications",
    "preferred qualifications", "you will need", "who you are",
    "what you bring", "skills required", "technical requirements",
)


def _extract_jd_for_scoring(desc: str, char_limit: int = 2000) -> str:
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
        intro = desc[:300]
        reqs  = desc[earliest:earliest + (char_limit - 300)]
        return (intro + "\n\n" + reqs)[:char_limit]
    return desc[:char_limit]

_SYSTEM_PROMPT = """\
You are an expert technical recruiter evaluating candidate fit for internship and co-op positions.

The candidate is a first-year master's student in Data Science at Northeastern University (starting Fall 2026).
Their target roles in priority order are:
  PRIORITY 1 — Data Engineering
  PRIORITY 2 — Data Science
  PRIORITY 3 — Machine Learning Engineering
  PRIORITY 4 — Software Engineering

Location preferences by season:
  - Fall and spring internships/co-ops: REMOTE strongly preferred (onsite-only hurts the score)
  - Summer internships: onsite or hybrid is fully acceptable

Score how well this candidate matches the job on a scale of 1–10:
  1–3  Poor match — wrong role type, requires PhD or senior experience, or onsite-only for a non-summer term
  4–6  Partial match — lower-priority role or notable skill gaps
  7–8  Good match — high-priority role, meets most requirements
  9–10 Excellent match — top-priority role targeting master's/graduate students, strong skill alignment

Reply in EXACTLY this format with nothing else:
Score: <integer>
Reasoning: <2–3 sentences covering matching skills, role priority fit, and location/season alignment>"""


class JobFitScorer:
    def __init__(self, groq_api_key: str):
        self._client = Groq(api_key=groq_api_key)
        self.threshold: int = cfg.JOB_SUITABILITY_SCORE

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
                        {"role": "system", "content": _SYSTEM_PROMPT},
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
