"""
Unit tests for src/ats_scorer.py.

ATSScorer.analyze() is pure Python with zero external calls — fully testable.
Run with: pytest tests/test_ats_scorer.py -v
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from src.ats_scorer import ATSScorer, _keyword_present, _extract_keywords_static, _ALWAYS_EXTRACT
from src.job import Job


def _make_job(description: str = "", role: str = "SWE", company: str = "ACME") -> Job:
    j = Job()
    j.role = role
    j.company = company
    j.description = description
    j.location = ""
    j.link = ""
    j.apply_method = "test"
    return j


# ── _keyword_present ──────────────────────────────────────────────────────────

class TestKeywordPresent:
    def test_exact_match(self):
        assert _keyword_present("python", "i know python and java") is True

    def test_no_match(self):
        assert _keyword_present("golang", "i know python and java") is False

    def test_multiword_exact(self):
        assert _keyword_present("machine learning", "experience with machine learning models") is True

    def test_synonym_k8s_to_kubernetes(self):
        assert _keyword_present("k8s", "we use kubernetes in prod") is True

    def test_synonym_kubernetes_to_k8s(self):
        assert _keyword_present("kubernetes", "must know k8s") is True

    def test_synonym_ml_to_machine_learning(self):
        assert _keyword_present("ml", "experience with machine learning") is True

    def test_synonym_machine_learning_to_ml(self):
        assert _keyword_present("machine learning", "strong ml background") is True

    def test_synonym_js_to_javascript(self):
        assert _keyword_present("js", "built with javascript and react") is True

    def test_synonym_typescript_to_ts(self):
        assert _keyword_present("typescript", "ts and react project") is True

    def test_synonym_postgres_to_postgresql(self):
        assert _keyword_present("postgres", "database: postgresql") is True

    def test_synonym_golang_to_go(self):
        assert _keyword_present("golang", "written in go language") is True

    def test_synonym_node_to_nodejs(self):
        assert _keyword_present("node", "nodejs backend") is True


# ── _extract_keywords_static ────────────────────────────────────────────────────

class TestExtractKeywords:
    def test_empty_text_returns_empty(self):
        assert _extract_keywords_static("") == []

    def test_tech_single_word_extracted(self):
        kws = _extract_keywords_static("strong python and sql skills required")
        assert "python" in kws
        assert "sql" in kws

    def test_bigram_tech_term_extracted(self):
        kws = _extract_keywords_static("experience with machine learning and deep learning")
        assert "machine learning" in kws

    def test_bigram_before_single_word(self):
        text = "machine learning engineer with python skills"
        kws = _extract_keywords_static(text)
        ml_pos = kws.index("machine learning") if "machine learning" in kws else 999
        py_pos = kws.index("python") if "python" in kws else 999
        # multi-word terms should appear earlier in the list
        assert ml_pos < py_pos

    def test_stop_words_excluded(self):
        kws = _extract_keywords_static("the and or but with for from is are")
        assert kws == []

    def test_top_n_limit(self):
        text = " ".join(["python", "java", "sql", "aws", "docker", "kubernetes",
                         "react", "angular", "redis", "kafka", "spark", "tensorflow",
                         "pytorch", "golang", "rust", "scala", "mongodb", "postgres",
                         "elasticsearch", "airflow", "terraform", "jenkins"])
        kws = _extract_keywords_static(text, top_n=10)
        assert len(kws) <= 10


# ── ATSScorer.analyze ─────────────────────────────────────────────────────────

class TestATSScorer:
    scorer = ATSScorer()

    def test_empty_description_returns_zero(self):
        result = self.scorer.analyze("python developer with sql skills", _make_job(""))
        assert result["ats_score"] == 0
        assert result["matched_keywords"] == []
        assert result["missing_keywords"] == []

    def test_perfect_match_scores_100(self):
        # Use only tech terms so _extract_keywords_static picks up nothing extra
        jd = "python sql aws"
        resume = "python sql aws"
        result = self.scorer.analyze(resume, _make_job(jd))
        assert result["ats_score"] == 100

    def test_zero_match_scores_0(self):
        jd = "requires kubernetes terraform aws"
        resume = "excel powerpoint word office"
        result = self.scorer.analyze(resume, _make_job(jd))
        assert result["ats_score"] == 0

    def test_score_calculation_matches_ratio(self):
        resume = "python java sql"
        jd = "python java sql javascript c++"
        result = self.scorer.analyze(resume, _make_job(jd))
        matched = len(result["matched_keywords"])
        total = matched + len(result["missing_keywords"])
        assert total > 0
        assert result["ats_score"] == round((matched / total) * 100)

    def test_critical_missing_subset_of_missing(self):
        resume = "general developer"
        jd = "requires kubernetes terraform python and excellent teamwork"
        result = self.scorer.analyze(resume, _make_job(jd))
        for kw in result["critical_missing"]:
            assert kw in result["missing_keywords"]

    def test_critical_missing_are_is_critical(self):
        from src.ats_scorer import _is_critical
        resume = "general developer"
        jd = "requires kubernetes terraform python"
        result = self.scorer.analyze(resume, _make_job(jd))
        # Every critical_missing keyword must satisfy _is_critical()
        for kw in result["critical_missing"]:
            assert _is_critical(kw), f"{kw!r} is in critical_missing but _is_critical() is False"
        # Known _ALWAYS_EXTRACT terms must end up in critical_missing
        for kw in ("kubernetes", "terraform", "python"):
            if kw in result["missing_keywords"]:
                assert kw in result["critical_missing"]

    def test_synonym_k8s_matches_kubernetes_in_resume(self):
        resume = "deployed with kubernetes on gke"
        jd = "must know k8s and docker"
        result = self.scorer.analyze(resume, _make_job(jd))
        # k8s is in JD keywords; kubernetes is in resume — synonym should match
        assert "k8s" in result["matched_keywords"]

    def test_tip_present(self):
        resume = "python developer"
        jd = "python kubernetes aws required"
        result = self.scorer.analyze(resume, _make_job(jd))
        assert isinstance(result["tip"], str)
        assert len(result["tip"]) > 0

    def test_tip_positive_when_full_match(self):
        resume = "python sql aws developer"
        jd = "python sql aws"
        result = self.scorer.analyze(resume, _make_job(jd))
        assert "great" in result["tip"].lower() or "well" in result["tip"].lower()
