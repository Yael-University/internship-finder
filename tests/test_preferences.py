"""
Tests for src.preferences — the candidate-profile-driven prompt and ranking.

Pure functions, no network: profile resolution, role-priority ranking, and
system-prompt rendering.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.preferences import (
    DEFAULT_PROFILE,
    build_system_prompt,
    get_profile,
    role_priority,
    role_priority_patterns,
)


# ── get_profile ──────────────────────────────────────────────────────────────

class TestGetProfile:
    def test_none_config_returns_default(self):
        assert get_profile(None) is DEFAULT_PROFILE

    def test_missing_block_returns_default(self):
        assert get_profile({"positions": []}) is DEFAULT_PROFILE

    def test_partial_block_backfills_from_default(self):
        prof = get_profile({"candidate_profile": {"summary": "a CS undergrad"}})
        assert prof["summary"] == "a CS undergrad"
        # role_priorities not provided → backfilled from default
        assert prof["role_priorities"] == DEFAULT_PROFILE["role_priorities"]

    def test_full_override(self):
        custom = {
            "summary": "a bootcamp grad",
            "role_priorities": [{"name": "Frontend", "keywords": ["frontend", "react"]}],
            "location_preference": ["Remote only"],
        }
        prof = get_profile({"candidate_profile": custom})
        assert prof["summary"] == "a bootcamp grad"
        assert prof["role_priorities"] == custom["role_priorities"]


# ── role_priority ────────────────────────────────────────────────────────────

class TestRolePriority:
    def test_default_ranking_matches_legacy(self):
        assert role_priority("Data Engineer Intern") == 0
        assert role_priority("Data Scientist") == 1
        assert role_priority("Machine Learning Engineer") == 2
        assert role_priority("Software Engineering Co-op") == 3
        assert role_priority("Marketing Analyst") == 99

    def test_case_insensitive(self):
        assert role_priority("DATA ENGINEER") == 0

    def test_custom_profile_ranking(self):
        prof = {
            "role_priorities": [
                {"name": "Frontend", "keywords": ["frontend", "react"]},
                {"name": "Backend", "keywords": ["backend", "api"]},
            ],
        }
        assert role_priority("Senior React Developer", prof) == 0
        assert role_priority("Backend API Engineer", prof) == 1
        assert role_priority("Data Engineer", prof) == 99

    def test_patterns_flatten_in_order(self):
        patterns = role_priority_patterns()
        assert ("data engineer", 0) in patterns
        assert ("software engineering", 3) in patterns


# ── build_system_prompt ──────────────────────────────────────────────────────

class TestBuildSystemPrompt:
    def test_default_prompt_contains_expected_profile(self):
        p = build_system_prompt()
        assert "first-year master's student in Data Science at Northeastern" in p
        assert "PRIORITY 1 \u2014 Data Engineering" in p
        assert "PRIORITY 4 \u2014 Software Engineering" in p
        assert "REMOTE strongly preferred" in p
        # The output contract must be preserved verbatim.
        assert "Reply in EXACTLY this format with nothing else:" in p
        assert "Score: <integer>" in p

    def test_custom_profile_renders_in_prompt(self):
        prof = {
            "summary": "a self-taught developer",
            "role_priorities": [{"name": "Frontend", "keywords": ["frontend"]}],
            "location_preference": ["Remote only, all year"],
        }
        p = build_system_prompt(prof)
        assert "The candidate is a self-taught developer." in p
        assert "PRIORITY 1 \u2014 Frontend" in p
        assert "  - Remote only, all year" in p
        # Only one priority configured → no PRIORITY 2 line.
        assert "PRIORITY 2" not in p

    def test_string_location_preference_is_used_verbatim(self):
        prof = {"summary": "x", "role_priorities": [{"name": "A", "keywords": ["a"]}],
                "location_preference": "Anywhere is fine"}
        assert "Anywhere is fine" in build_system_prompt(prof)
