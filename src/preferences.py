"""
Candidate-profile preferences shared by the fit scorer and the role-priority
ranking.

Both the Groq system prompt (src/job_fit_scorer.py) and the result ordering
(main.py) are derived from a single ``candidate_profile`` block in
work_preferences.yaml so the tool can be reused by anyone without editing code.
When no profile is configured, DEFAULT_PROFILE is used, which reproduces the
original hardcoded behavior.
"""
from __future__ import annotations

from typing import Optional

# The default profile reproduces the project author's original hardcoded prompt
# and role priorities. Override any field via a ``candidate_profile`` block in
# work_preferences.yaml.
DEFAULT_PROFILE: dict = {
    "summary": (
        "a first-year master's student in Data Science at Northeastern "
        "University (starting Fall 2026)"
    ),
    "role_priorities": [
        {"name": "Data Engineering",            "keywords": ["data engineer", "data engineering"]},
        {"name": "Data Science",                "keywords": ["data scientist", "data science"]},
        {"name": "Machine Learning Engineering", "keywords": ["machine learning", "ml engineer"]},
        {"name": "Software Engineering",         "keywords": ["software engineer", "software engineering"]},
    ],
    "location_preference": [
        "Fall and spring internships/co-ops: REMOTE strongly preferred (onsite-only hurts the score)",
        "Summer internships: onsite or hybrid is fully acceptable",
    ],
}

_UNRANKED = 99  # priority for roles that match no configured category


def get_profile(config: Optional[dict]) -> dict:
    """Return the candidate profile from config, falling back to DEFAULT_PROFILE.

    Missing individual fields are backfilled from the default so a partial
    ``candidate_profile`` block still produces a complete prompt.
    """
    configured = (config or {}).get("candidate_profile") or {}
    if not configured:
        return DEFAULT_PROFILE
    merged = dict(DEFAULT_PROFILE)
    for key in ("summary", "role_priorities", "location_preference"):
        if configured.get(key):
            merged[key] = configured[key]
    return merged


def _role_priority_names(profile: dict) -> list[str]:
    names: list[str] = []
    for rp in profile.get("role_priorities", []):
        names.append(rp["name"] if isinstance(rp, dict) else str(rp))
    return names


def role_priority_patterns(profile: Optional[dict] = None) -> list[tuple[str, int]]:
    """Flatten role_priorities into (keyword, priority_index) pairs.

    Lower index = higher priority. Roles are matched by lowercased substring,
    matching the prompt's PRIORITY ordering exactly.
    """
    profile = profile or DEFAULT_PROFILE
    patterns: list[tuple[str, int]] = []
    for idx, rp in enumerate(profile.get("role_priorities", [])):
        keywords = rp.get("keywords", []) if isinstance(rp, dict) else [rp]
        for kw in keywords:
            if kw:
                patterns.append((str(kw).lower(), idx))
    return patterns


def role_priority(role: str, profile: Optional[dict] = None) -> int:
    """Priority index for a role title (lower = higher priority, 99 = unranked)."""
    r = (role or "").lower()
    for pattern, pri in role_priority_patterns(profile):
        if pattern in r:
            return pri
    return _UNRANKED


def build_system_prompt(profile: Optional[dict] = None) -> str:
    """Render the Groq fit-scoring system prompt from a candidate profile."""
    profile = profile or DEFAULT_PROFILE

    priorities = "\n".join(
        f"  PRIORITY {i} — {name}"
        for i, name in enumerate(_role_priority_names(profile), 1)
    )

    location_pref = profile.get("location_preference", [])
    if isinstance(location_pref, str):
        location_block = location_pref
    else:
        location_block = "\n".join(f"  - {item}" for item in location_pref)

    return (
        "You are an expert technical recruiter evaluating candidate fit for "
        "internship and co-op positions.\n\n"
        f"The candidate is {profile.get('summary', '')}.\n"
        "Their target roles in priority order are:\n"
        f"{priorities}\n\n"
        "Location preferences by season:\n"
        f"{location_block}\n\n"
        "Score how well this candidate matches the job on a scale of 1\u201310:\n"
        "  1\u20133  Poor match \u2014 wrong role type, requires PhD or senior experience, "
        "or onsite-only for a non-summer term\n"
        "  4\u20136  Partial match \u2014 lower-priority role or notable skill gaps\n"
        "  7\u20138  Good match \u2014 high-priority role, meets most requirements\n"
        "  9\u201310 Excellent match \u2014 top-priority role targeting master's/graduate "
        "students, strong skill alignment\n\n"
        "Reply in EXACTLY this format with nothing else:\n"
        "Score: <integer>\n"
        "Reasoning: <2\u20133 sentences covering matching skills, role priority fit, "
        "and location/season alignment>"
    )
