"""
ATS scorer: two-layer analysis per matched job.

  Layer 1 — Keyword coverage (what real ATS systems actually do)
    Skills extracted from the JD by jobbert (ML-based NER trained on job postings),
    supplemented by a static list of known tech terms. Checks which appear in the resume.

  Layer 2 — Semantic alignment (sentence-transformers cosine similarity)
    Measures how conceptually close the resume and JD are, catching synonyms and
    related phrasing that keyword matching misses.

Both models download once (~500 MB total) and run locally — no API calls, no cost.
Falls back to the original static keyword approach if models are unavailable.

Combined ATS score = 60 % keyword + 40 % semantic.
"""
from __future__ import annotations

import re
from collections import Counter

from src.job import Job
from src.logging import logger

# ── Optional ML imports — graceful fallback if not installed ───────────────────
try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import util as st_util
    import torch
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    from transformers import pipeline as hf_pipeline
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

_ML_AVAILABLE = _ST_AVAILABLE and _HF_AVAILABLE

_SKILL_MODEL_ID    = "jjzha/jobbert_skill_extraction"
_SEMANTIC_MODEL_ID = "all-MiniLM-L6-v2"

# ── Stop words ─────────────────────────────────────────────────────────────────
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "must",
    "that", "this", "these", "those", "it", "its", "we", "you", "they",
    "our", "your", "their", "as", "if", "not", "no", "so", "up", "out",
    "about", "into", "than", "then", "when", "who", "which", "what", "how",
    "all", "any", "both", "each", "more", "most", "other", "some", "such",
    "also", "well", "just", "like", "use", "used", "using", "work", "team",
    "strong", "experience", "ability", "skills", "knowledge", "understanding",
    "responsibilities", "qualifications", "requirements", "position", "role",
    "job", "company", "candidate", "looking", "seeking", "please", "apply",
    "include", "including", "plus", "bonus", "preferred", "required",
    "equivalent", "years", "year", "month", "day", "time", "full", "part",
    "new", "good", "great", "excellent", "highly", "within", "across",
    "ensure", "support", "help", "provide", "make", "build", "create",
    "develop", "design", "implement", "manage", "lead", "drive", "deliver",
}

# ── Known tech terms (static supplement — catches terms jobbert may miss) ──────
_ALWAYS_EXTRACT = {
    # Languages
    "python", "java", "javascript", "typescript", "golang", "rust",
    "c++", "c#", "scala", "kotlin", "swift", "ruby", "r", "matlab", "php",
    "bash", "shell", "sql", "nosql",
    # Frameworks / libs
    "react", "vue", "angular", "node", "nodejs", "flask", "django", "fastapi",
    "express", "next.js", "nextjs", "graphql", "rest", "grpc",
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "pandas",
    "numpy", "spark", "pyspark", "kafka", "flink", "airflow", "dbt", "luigi",
    "spring boot", "spring framework",
    "sqlalchemy", "pydantic", "pytest", "jupyter", "streamlit", "gradio",
    "matplotlib", "seaborn", "plotly", "scipy", "statsmodels", "dask", "ray",
    "xgboost", "lightgbm", "catboost",
    # Cloud / infra
    "aws", "gcp", "azure", "s3", "ec2", "lambda", "cloudformation", "terraform",
    "kubernetes", "k8s", "docker", "helm", "ci/cd", "github actions", "jenkins",
    "ansible", "prometheus", "grafana", "datadog", "snowflake", "databricks",
    "bigquery", "redshift", "elasticsearch", "cassandra", "mongodb", "postgres",
    "postgresql", "mysql", "redis", "dynamodb",
    "sagemaker", "vertex ai", "glue", "athena", "emr", "kinesis",
    "dataflow", "pub/sub", "pubsub",
    # Data engineering
    "prefect", "dagster", "mage", "apache beam", "fivetran", "airbyte", "stitch",
    "delta lake", "apache iceberg", "iceberg", "hudi", "parquet", "avro", "orc",
    "data modeling", "dimensional modeling", "star schema",
    "data quality", "data governance", "data catalog", "data lakehouse",
    "great expectations", "dbt cloud",
    # Data / ML
    "machine learning", "deep learning", "nlp", "llm", "etl", "elt", "data pipeline",
    "data warehouse", "data lake", "mlops", "feature engineering", "a/b testing",
    "gradient boosting", "random forest", "neural network", "transformer",
    "mlflow", "wandb", "kubeflow", "rag", "vector database",
    "langchain", "hugging face", "huggingface", "openai", "llama",
    # BI / visualization
    "tableau", "power bi", "looker", "superset", "metabase",
    # Practices
    "agile", "scrum", "tdd", "microservices", "api", "sdk", "oop",
    "data structures", "algorithms", "system design", "distributed systems",
    "git", "github", "jira", "confluence",
}

_SYNONYMS: dict[str, list[str]] = {
    "ml":               ["machine learning"],
    "machine learning": ["ml"],
    "k8s":              ["kubernetes"],
    "kubernetes":       ["k8s"],
    "js":               ["javascript"],
    "javascript":       ["js"],
    "ts":               ["typescript"],
    "typescript":       ["ts"],
    "golang":           ["go"],
    "node":             ["node.js", "nodejs"],
    "node.js":          ["node", "nodejs"],
    "nodejs":           ["node", "node.js"],
    "postgres":         ["postgresql"],
    "postgresql":       ["postgres"],
}


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s\-\+\./]", " ", text)
    return [t.strip("-./") for t in text.split() if t.strip("-./")]


def _extract_ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def _keyword_present(keyword: str, resume_lower: str) -> bool:
    """
    Check if a keyword (or any synonym) is in the resume.
    Short purely-alphabetic terms use word-boundary matching to avoid false positives
    (e.g. 'r' inside 'requirements', 'go' inside 'algorithms').
    """
    def _match(kw: str) -> bool:
        if " " not in kw and kw.isalpha() and len(kw) <= 3:
            return bool(re.search(r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])", resume_lower))
        return kw in resume_lower

    if _match(keyword):
        return True
    for synonym in _SYNONYMS.get(keyword, []):
        if _match(synonym):
            return True
    return False


def _is_critical(keyword: str) -> bool:
    return keyword in _ALWAYS_EXTRACT


def _extract_always_extract_terms(text: str) -> list[str]:
    """Return only _ALWAYS_EXTRACT terms that appear in the text (no frequency pass)."""
    tokens   = _tokenize(text)
    bigrams  = _extract_ngrams(tokens, 2)
    trigrams = _extract_ngrams(tokens, 3)
    found: list[str] = []
    for phrase in trigrams + bigrams:
        if phrase in _ALWAYS_EXTRACT and phrase not in found:
            found.append(phrase)
    for token in tokens:
        if token in _ALWAYS_EXTRACT and token not in found:
            found.append(token)
    return found


def _extract_keywords_static(text: str, top_n: int = 20) -> list[str]:
    """Full static extraction — always-extract terms plus high-frequency words.
    Used as the sole extractor when ML models are unavailable."""
    tokens   = _tokenize(text)
    bigrams  = _extract_ngrams(tokens, 2)
    trigrams = _extract_ngrams(tokens, 3)
    found: list[str] = []

    for phrase in trigrams + bigrams:
        if phrase in _ALWAYS_EXTRACT and phrase not in found:
            found.append(phrase)
    for token in tokens:
        if token in _ALWAYS_EXTRACT and token not in found:
            found.append(token)

    freq = Counter(
        t for t in tokens
        if t not in _STOP_WORDS and len(t) > 3 and not t.isdigit() and t not in found
    )
    for word, _ in freq.most_common(top_n):
        if len(found) >= top_n:
            break
        found.append(word)

    return found[:top_n]


def _empty(reason: str) -> dict:
    return {
        "ats_score":        0,
        "keyword_score":    0,
        "semantic_score":   0,
        "matched_keywords": [],
        "missing_keywords": [],
        "critical_missing": [],
        "tip":              reason,
    }


class ATSScorer:
    """
    Two-layer ATS scorer: jobbert skill extraction + sentence-transformers
    semantic similarity. Initialise once per run, passing the resume text so
    the embedding is computed once and reused across all jobs.
    Falls back to static keyword matching if ML models are unavailable.
    """

    def __init__(self, resume_text: str = ""):
        self._resume_text  = resume_text
        self._resume_embed = None
        self._skill_pipe   = None
        self._sem_model    = None
        self._ml_ready     = False

        if resume_text:
            self._load_models()

    def _load_models(self) -> None:
        # jobbert is optional — load it if available, fall back to static extraction if not
        if _HF_AVAILABLE:
            try:
                logger.info("[ATS] Loading jobbert skill extractor…")
                self._skill_pipe = hf_pipeline(
                    "token-classification",
                    model=_SKILL_MODEL_ID,
                    aggregation_strategy="simple",
                )
                logger.info("[ATS] jobbert loaded")
            except Exception as e:
                logger.warning(f"[ATS] jobbert unavailable ({e}) — keyword layer uses static extraction")
                self._skill_pipe = None

        # sentence-transformers is loaded independently so jobbert failure doesn't block it
        if _ST_AVAILABLE:
            try:
                logger.info("[ATS] Loading sentence-transformers semantic model…")
                self._sem_model = SentenceTransformer(_SEMANTIC_MODEL_ID)
                logger.info("[ATS] Pre-computing resume embedding…")
                self._resume_embed = self._sem_model.encode(
                    self._resume_text, convert_to_tensor=True
                )
                self._ml_ready = True
                logger.info("[ATS] Semantic model ready")
            except Exception as e:
                logger.warning(
                    f"[ATS] sentence-transformers failed to load ({e}) — "
                    "falling back to static keyword matching only"
                )
                self._ml_ready = False

    # ── jobbert skill extraction from JD ──────────────────────────────────────

    def _extract_skills_jobbert(self, text: str) -> list[str]:
        try:
            entities = self._skill_pipe(text[:1400])
            skills = []
            for ent in entities:
                # Only SKILL entities (not EXPERIENCE) above confidence threshold
                if ent.get("entity_group") != "SKILL":
                    continue
                if ent.get("score", 0) < 0.80:
                    continue
                word = ent.get("word", "").strip()
                word = re.sub(r"\s*##\s*", "", word).strip(".,;:!?\"'()- ")
                word = word.lower()
                if word and len(word) > 2 and word not in _STOP_WORDS:
                    skills.append(word)
            return list(dict.fromkeys(skills))
        except Exception as e:
            logger.debug(f"[ATS] jobbert extraction error: {e}")
            return []

    # ── keyword matching (JD skills vs resume) ────────────────────────────────

    def _build_keyword_list(self, jd_text: str) -> list[str]:
        """
        When ML is available: jobbert SKILL entities (high-confidence) supplemented
        by _ALWAYS_EXTRACT terms present in the JD. The high-frequency word pass is
        skipped — it produces noise terms like 'data', 'intern', 'engineering'.
        When ML is unavailable: full static extraction as fallback.
        """
        if self._ml_ready:
            ml_skills     = self._extract_skills_jobbert(jd_text)
            static_skills = _extract_always_extract_terms(jd_text)
        else:
            ml_skills     = []
            static_skills = _extract_keywords_static(jd_text, top_n=25)

        merged: list[str] = list(ml_skills)
        for kw in static_skills:
            if kw not in merged:
                merged.append(kw)

        return merged[:30]

    # ── semantic similarity ───────────────────────────────────────────────────

    def _semantic_score(self, jd_text: str, resume_embed=None) -> int:
        try:
            embed = resume_embed if resume_embed is not None else self._resume_embed
            jd_embed = self._sem_model.encode(jd_text, convert_to_tensor=True)
            sim = float(st_util.cos_sim(embed, jd_embed))
            # Cosine similarity for related text is typically 0.2–0.9;
            # normalise to a 0–100 scale that feels intuitive.
            score = round(max(0.0, min(1.0, sim)) * 100)
            return score
        except Exception as e:
            logger.debug(f"[ATS] Semantic scoring error: {e}")
            return 0

    # ── public interface ──────────────────────────────────────────────────────

    def analyze(self, resume_text: str, job: Job) -> dict:
        """
        Returns:
            {
                "ats_score":        int  0-100  (combined: 60 % keyword + 40 % semantic)
                "keyword_score":    int  0-100
                "semantic_score":   int  0-100
                "matched_keywords": list[str]
                "missing_keywords": list[str]
                "critical_missing": list[str]
                "tip":              str
            }
        """
        if not job.description:
            return _empty("No job description available")

        try:
            resume_lower = resume_text.lower()
            jd_text      = job.description[:5000]

            # ── keyword layer ─────────────────────────────────────────────────
            keywords = self._build_keyword_list(jd_text)
            matched  = [kw for kw in keywords if _keyword_present(kw, resume_lower)]
            missing  = [kw for kw in keywords if not _keyword_present(kw, resume_lower)]
            critical = [kw for kw in missing if _is_critical(kw)]

            total         = len(keywords) if keywords else 1
            keyword_score = round((len(matched) / total) * 100)

            # ── semantic layer ────────────────────────────────────────────────
            # Compute resume embedding per-call so analyze() is thread-safe
            # (allows different resumes across concurrent API requests).
            # Falls back to the pre-computed embed when resume_text is unchanged.
            resume_embed = None
            if self._ml_ready:
                if resume_text and resume_text != self._resume_text:
                    try:
                        resume_embed = self._sem_model.encode(resume_text, convert_to_tensor=True)
                    except Exception:
                        resume_embed = self._resume_embed
                else:
                    resume_embed = self._resume_embed
            semantic_score = self._semantic_score(jd_text, resume_embed=resume_embed) if self._ml_ready else 0

            # ── combined score ────────────────────────────────────────────────
            if self._ml_ready:
                ats_score = round(0.6 * keyword_score + 0.4 * semantic_score)
            else:
                ats_score = keyword_score

            # ── tip ───────────────────────────────────────────────────────────
            if not keywords:
                tip = (
                    "Job description doesn't list specific technical requirements — "
                    "keyword match unavailable. Use the semantic score and Groq reasoning to judge fit."
                )
            elif critical:
                gaps = ", ".join(f'"{k}"' for k in critical[:3])
                tip = (
                    f"Add {gaps} to your resume — "
                    f"{'these are' if len(critical) > 1 else 'this is'} likely ATS "
                    "filter keyword(s) pulled directly from the job description."
                )
            elif missing:
                gaps = ", ".join(f'"{k}"' for k in missing[:3])
                tip = f"Consider adding {gaps} to improve keyword coverage."
            else:
                tip = "Strong keyword coverage — your resume matches this job's listed requirements well."

            logger.info(
                f"[ATS] {job.role} @ {job.company} → "
                f"combined {ats_score}%  "
                f"(keywords {keyword_score}% · semantic {semantic_score}%)"
            )
            return {
                "ats_score":        ats_score,
                "keyword_score":    keyword_score,
                "semantic_score":   semantic_score,
                "matched_keywords": matched,
                "missing_keywords": missing,
                "critical_missing": critical,
                "tip":              tip,
            }

        except Exception as e:
            logger.error(f"[ATS] Failed for '{job.role}' at '{job.company}': {e}")
            return _empty(str(e))
