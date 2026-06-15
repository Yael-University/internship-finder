"""
Unit tests for the pure helpers in main.py.

These functions have no network or Selenium dependencies, so they're fully
testable in isolation: resume compression, role-priority ranking, and the
score-cache get/set/load/save round-trip.

Run with: pytest tests/test_main_helpers.py -v
"""
import json
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import (
    DEFAULT_FIXTURES,
    _CACHE_TTL_DAYS,
    _cache_get,
    _cache_set,
    _compress_resume,
    _load_score_cache,
    _parse_args,
    _role_priority,
    _save_score_cache,
    load_jobs_from_fixtures,
)


# ── _role_priority ──────────────────────────────────────────────────────────

class TestRolePriority:
    def test_data_engineer_is_top_priority(self):
        assert _role_priority("Data Engineer Intern") == 0
        assert _role_priority("Data Engineering Co-op") == 0

    def test_data_scientist_second(self):
        assert _role_priority("Data Scientist") == 1
        assert _role_priority("Junior Data Science Intern") == 1

    def test_machine_learning_third(self):
        assert _role_priority("Machine Learning Engineer") == 2
        assert _role_priority("ML Engineer Intern") == 2

    def test_software_engineer_fourth(self):
        assert _role_priority("Software Engineer") == 3
        assert _role_priority("Software Engineering Co-op") == 3

    def test_unknown_role_gets_lowest_priority(self):
        assert _role_priority("Marketing Analyst") == 99

    def test_case_insensitive(self):
        assert _role_priority("DATA ENGINEER") == 0

    def test_ordering_reflects_priority(self):
        roles = ["Software Engineer", "Data Engineer", "ML Engineer", "Data Scientist"]
        ordered = sorted(roles, key=_role_priority)
        assert ordered == ["Data Engineer", "Data Scientist", "ML Engineer", "Software Engineer"]


# ── _compress_resume ──────────────────────────────────────────────────────────

_SAMPLE_RESUME = """\
personal_information:
  name: Ada
  surname: Lovelace
  city: Boston
  linkedin: https://linkedin.com/in/ada
education_details:
  - education_level: Master's
    field_of_study: Data Science
    institution: Northeastern University
    start_date: 2026
    year_of_completion: 2028
    final_evaluation_grade: 3.9
experience_details:
  - position: Data Engineering Intern
    company: Acme Corp
    employment_period: Summer 2025
    location: Remote
    industry: Tech
    key_responsibilities:
      - responsibility: Built ETL pipelines in Python and Spark
    skills_acquired:
      - Airflow
      - dbt
projects:
  - name: Pipeline Monitor
    description: Real-time data quality dashboard
    link: https://github.com/ada/monitor
certifications:
  - AWS Certified
languages:
  - language: English
    proficiency: Native
self_identification:
  gender: Female
salary_expectations:
  salary_range_usd: 90000
"""


class TestCompressResume:
    def test_includes_core_signal(self):
        out = _compress_resume(_SAMPLE_RESUME)
        assert "Ada Lovelace" in out
        assert "Data Science" in out
        assert "Northeastern University" in out
        assert "Data Engineering Intern" in out
        assert "Acme Corp" in out
        assert "ETL pipelines" in out
        assert "Airflow" in out
        assert "Pipeline Monitor" in out
        assert "AWS Certified" in out
        assert "English" in out

    def test_drops_irrelevant_sections(self):
        out = _compress_resume(_SAMPLE_RESUME)
        # self_identification and salary_expectations carry no scoring signal
        assert "Female" not in out
        assert "90000" not in out

    def test_is_more_compact_than_raw_yaml(self):
        out = _compress_resume(_SAMPLE_RESUME)
        assert len(out) < len(_SAMPLE_RESUME)

    def test_empty_yaml_has_no_personal_data(self):
        # Empty input yields only the static section skeleton, no real content.
        out = _compress_resume("")
        assert "Ada" not in out
        assert "Acme" not in out

    def test_handles_missing_sections_gracefully(self):
        minimal = "personal_information:\n  name: Solo\n"
        out = _compress_resume(minimal)
        assert "Solo" in out


# ── score cache: get / set ──────────────────────────────────────────────────

class TestCacheGetSet:
    def test_set_then_get_roundtrip(self):
        cache: dict = {}
        result = {"score": 8, "reasoning": "good fit", "is_match": True}
        _cache_set(cache, "https://job/1", result)
        assert _cache_get(cache, "https://job/1") == result

    def test_get_missing_returns_none(self):
        assert _cache_get({}, "https://job/missing") is None

    def test_expired_entry_returns_none(self):
        stale = (datetime.now() - timedelta(days=_CACHE_TTL_DAYS + 1)).isoformat()
        cache = {"https://job/old": {"cached_at": stale, "result": {"score": 5}}}
        assert _cache_get(cache, "https://job/old") is None

    def test_fresh_entry_within_ttl_returned(self):
        recent = (datetime.now() - timedelta(days=1)).isoformat()
        cache = {"https://job/new": {"cached_at": recent, "result": {"score": 9}}}
        assert _cache_get(cache, "https://job/new") == {"score": 9}

    def test_malformed_timestamp_returns_none(self):
        cache = {"https://job/bad": {"cached_at": "not-a-date", "result": {"score": 1}}}
        assert _cache_get(cache, "https://job/bad") is None


# ── score cache: load / save ────────────────────────────────────────────────

class TestCacheLoadSave:
    def test_save_then_load_roundtrip(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache: dict = {}
        _cache_set(cache, "https://job/1", {"score": 7})
        _save_score_cache(output_dir, cache)

        # cache file is written one level above output_dir
        assert (tmp_path / "score_cache.json").exists()
        loaded = _load_score_cache(output_dir)
        assert "https://job/1" in loaded
        assert loaded["https://job/1"]["result"] == {"score": 7}

    def test_load_missing_file_returns_empty(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        assert _load_score_cache(output_dir) == {}

    def test_load_drops_expired_entries(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        stale = (datetime.now() - timedelta(days=_CACHE_TTL_DAYS + 5)).isoformat()
        fresh = (datetime.now() - timedelta(days=1)).isoformat()
        raw = {
            "https://job/old": {"cached_at": stale, "result": {"score": 3}},
            "https://job/new": {"cached_at": fresh, "result": {"score": 8}},
        }
        (tmp_path / "score_cache.json").write_text(json.dumps(raw), encoding="utf-8")

        loaded = _load_score_cache(output_dir)
        assert "https://job/new" in loaded
        assert "https://job/old" not in loaded


# ── dry-run fixture loading ─────────────────────────────────────────────────

class TestLoadJobsFromFixtures:
    def test_loads_list_form(self, tmp_path):
        f = tmp_path / "jobs.json"
        f.write_text(json.dumps([
            {"role": "DE", "company": "Acme", "location": "Remote",
             "link": "https://x/1", "source": "fixture", "description": "desc"},
        ]), encoding="utf-8")
        jobs = load_jobs_from_fixtures(f)
        assert len(jobs) == 1
        assert jobs[0].role == "DE"
        assert jobs[0].apply_method == "fixture"
        assert jobs[0].description == "desc"

    def test_loads_object_with_jobs_key(self, tmp_path):
        f = tmp_path / "jobs.json"
        f.write_text(json.dumps({"jobs": [
            {"role": "DS", "company": "N", "apply_method": "handshake"},
        ]}), encoding="utf-8")
        jobs = load_jobs_from_fixtures(f)
        assert len(jobs) == 1
        assert jobs[0].apply_method == "handshake"
        # Missing fields fall back to empty strings.
        assert jobs[0].location == ""

    def test_source_aliases_apply_method(self, tmp_path):
        f = tmp_path / "jobs.json"
        f.write_text(json.dumps([{"role": "r", "company": "c", "source": "simplify"}]),
                     encoding="utf-8")
        assert load_jobs_from_fixtures(f)[0].apply_method == "simplify"

    def test_bundled_sample_fixture_is_valid(self):
        # The committed sample fixture should always load cleanly.
        jobs = load_jobs_from_fixtures(DEFAULT_FIXTURES)
        assert len(jobs) >= 1
        assert all(j.role and j.description for j in jobs)


# ── CLI argument parsing ────────────────────────────────────────────────────

class TestParseArgs:
    def test_default_is_live_run(self):
        assert _parse_args([]).dry_run is None

    def test_dry_run_flag_uses_default_fixture(self):
        assert _parse_args(["--dry-run"]).dry_run == str(DEFAULT_FIXTURES)

    def test_dry_run_accepts_explicit_path(self):
        assert _parse_args(["--dry-run", "/tmp/jobs.json"]).dry_run == "/tmp/jobs.json"
