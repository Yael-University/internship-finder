"""
Unit tests for the pure helper in src/job_fit_scorer.py.

_extract_jd_for_scoring() prioritizes the requirements section of a job
description over the company intro before it's sent to the LLM. No network
calls are made (the Groq client is never constructed).

Run with: pytest tests/test_job_fit_scorer.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.job_fit_scorer import _JD_CHAR_LIMIT, _JD_INTRO_CHARS, _extract_jd_for_scoring


class TestExtractJdForScoring:
    def test_short_description_returned_as_is(self):
        desc = "We need a Python developer."
        assert _extract_jd_for_scoring(desc) == desc

    def test_no_header_falls_back_to_prefix_slice(self):
        desc = "x" * 5000
        out = _extract_jd_for_scoring(desc, char_limit=2000)
        assert out == "x" * 2000

    def test_respects_char_limit(self):
        desc = "intro " * 200 + "Requirements: " + "python " * 500
        out = _extract_jd_for_scoring(desc, char_limit=1000)
        assert len(out) <= 1000

    def test_pulls_requirements_section_forward(self):
        # Bury the requirements far past a naive prefix slice, with trailing
        # detail so the header isn't within the final 100 chars (which would
        # disable the requirements-forwarding branch).
        intro = "About our company. " * 100  # ~1900 chars of preamble
        reqs = "Requirements: must know Kubernetes and Terraform. " + ("more detail " * 50)
        desc = intro + reqs
        out = _extract_jd_for_scoring(desc, char_limit=1500)
        # The requirements keywords should survive even though they sit
        # well past the char_limit in the raw description.
        assert "Kubernetes" in out
        assert "Terraform" in out

    def test_keeps_intro_context(self):
        intro = "UniqueIntroToken. " + ("filler " * 100)
        reqs = "Qualifications: SQL and Python. " + ("detail " * 50)
        desc = intro + reqs
        out = _extract_jd_for_scoring(desc, char_limit=1500)
        assert "UniqueIntroToken" in out

    def test_intro_slice_bounded_by_constant(self):
        intro = "A" * 400
        reqs = "Requirements: Python. " + ("z" * 200)
        desc = intro + reqs
        out = _extract_jd_for_scoring(desc, char_limit=1500)
        # Only the first _JD_INTRO_CHARS of the intro are retained before the
        # requirements section is appended.
        assert out.startswith("A" * _JD_INTRO_CHARS)
        assert "A" * (_JD_INTRO_CHARS + 1) not in out

    def test_default_char_limit_constant_applied(self):
        desc = "y" * 5000
        out = _extract_jd_for_scoring(desc)
        assert len(out) == _JD_CHAR_LIMIT
