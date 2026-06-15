import argparse
import json
import subprocess
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple
import re

import yaml
from src.logging import logger
from src.job import Job
from src.utils.constants import (
    PLAIN_TEXT_RESUME_YAML,
    SECRETS_YAML,
    WORK_PREFERENCES_YAML,
)
from src.job_searcher import JobSearchManager
from src.job_fit_scorer import JobFitScorer
from src.ats_scorer import ATSScorer

DEFAULT_FIXTURES = Path("data_folder") / "fixtures" / "sample_jobs.json"


class ConfigError(Exception):
    pass


class ConfigValidator:
    EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    REQUIRED_CONFIG_KEYS = {
        "remote": bool,
        "experience_level": dict,
        "job_types": dict,
        "date": dict,
        "positions": list,
        "locations": list,
        "location_blacklist": list,
        "distance": int,
        "company_blacklist": list,
        "title_blacklist": list,
    }
    EXPERIENCE_LEVELS = [
        "internship", "entry", "associate",
        "mid_senior_level", "director", "executive",
    ]
    JOB_TYPES = [
        "full_time", "contract", "part_time",
        "temporary", "internship", "other", "volunteer",
    ]
    DATE_FILTERS = ["all_time", "month", "week", "24_hours"]
    APPROVED_DISTANCES = [0, 5, 10, 25, 50, 100]

    @staticmethod
    def load_yaml(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @classmethod
    def validate_config(cls, config_yaml_path: Path) -> dict:
        parameters = cls.load_yaml(config_yaml_path)
        for key, expected_type in cls.REQUIRED_CONFIG_KEYS.items():
            if key not in parameters:
                if key in ["company_blacklist", "title_blacklist", "location_blacklist"]:
                    parameters[key] = []
                else:
                    raise ConfigError(f"Missing required key '{key}' in {config_yaml_path}")
            elif not isinstance(parameters[key], expected_type):
                if key in ["company_blacklist", "title_blacklist", "location_blacklist"] and parameters[key] is None:
                    parameters[key] = []
                else:
                    raise ConfigError(
                        f"Invalid type for key '{key}' in {config_yaml_path}. "
                        f"Expected {expected_type.__name__}."
                    )
        cls._validate_experience_levels(parameters["experience_level"], config_yaml_path)
        cls._validate_job_types(parameters["job_types"], config_yaml_path)
        cls._validate_date_filters(parameters["date"], config_yaml_path)
        cls._validate_list_of_strings(parameters, ["positions", "locations"], config_yaml_path)
        cls._validate_distance(parameters["distance"], config_yaml_path)
        cls._validate_blacklists(parameters, config_yaml_path)
        return parameters

    @classmethod
    def _validate_experience_levels(cls, experience_levels: dict, config_path: Path):
        for level in cls.EXPERIENCE_LEVELS:
            if not isinstance(experience_levels.get(level), bool):
                raise ConfigError(f"Experience level '{level}' must be a boolean in {config_path}")

    @classmethod
    def _validate_job_types(cls, job_types: dict, config_path: Path):
        for job_type in cls.JOB_TYPES:
            if not isinstance(job_types.get(job_type), bool):
                raise ConfigError(f"Job type '{job_type}' must be a boolean in {config_path}")

    @classmethod
    def _validate_date_filters(cls, date_filters: dict, config_path: Path):
        for date_filter in cls.DATE_FILTERS:
            if not isinstance(date_filters.get(date_filter), bool):
                raise ConfigError(f"Date filter '{date_filter}' must be a boolean in {config_path}")

    @classmethod
    def _validate_list_of_strings(cls, parameters: dict, keys: list, config_path: Path):
        for key in keys:
            if not all(isinstance(item, str) for item in parameters.get(key, [])):
                raise ConfigError(f"'{key}' must be a list of strings in {config_path}")

    @classmethod
    def _validate_distance(cls, distance: int, config_path: Path):
        if distance not in cls.APPROVED_DISTANCES:
            raise ConfigError(
                f"Invalid distance value '{distance}' in {config_path}. "
                f"Must be one of: {cls.APPROVED_DISTANCES}"
            )

    @classmethod
    def _validate_blacklists(cls, parameters: dict, config_path: Path):
        for blacklist in ["company_blacklist", "title_blacklist", "location_blacklist"]:
            if not isinstance(parameters.get(blacklist), list):
                raise ConfigError(f"'{blacklist}' must be a list in {config_path}")
            if parameters[blacklist] is None:
                parameters[blacklist] = []

    @staticmethod
    def validate_secrets(secrets_yaml_path: Path) -> dict:
        secrets = ConfigValidator.load_yaml(secrets_yaml_path)

        if not secrets.get("groq_api_key"):
            raise ConfigError(
                f"Missing or empty 'groq_api_key' in {secrets_yaml_path}. "
                "Get a free key at console.groq.com"
            )

        return {
            "groq_api_key":       secrets.get("groq_api_key", ""),
            "linkedin_email":     secrets.get("linkedin_email", ""),
            "linkedin_password":  secrets.get("linkedin_password", ""),
            "handshake_email":    secrets.get("handshake_email", ""),
            "handshake_password": secrets.get("handshake_password", ""),
            "resume_pdf_path":    secrets.get("resume_pdf_path", ""),
        }


class FileManager:
    REQUIRED_FILES = [SECRETS_YAML, WORK_PREFERENCES_YAML, PLAIN_TEXT_RESUME_YAML]

    @staticmethod
    def validate_data_folder(app_data_folder: Path) -> Tuple[Path, Path, Path, Path]:
        if not app_data_folder.is_dir():
            raise FileNotFoundError(f"Data folder not found: {app_data_folder}")

        missing_files = [
            f for f in FileManager.REQUIRED_FILES
            if not (app_data_folder / f).exists()
        ]
        if missing_files:
            raise FileNotFoundError(f"Missing files in data folder: {', '.join(missing_files)}")

        output_folder = app_data_folder / "output"
        output_folder.mkdir(exist_ok=True)
        reports_folder = output_folder / "reports"
        reports_folder.mkdir(exist_ok=True)

        return (
            app_data_folder / SECRETS_YAML,
            app_data_folder / WORK_PREFERENCES_YAML,
            app_data_folder / PLAIN_TEXT_RESUME_YAML,
            reports_folder,
        )

    @staticmethod
    def get_uploads(plain_text_resume_file: Path) -> Dict[str, Path]:
        if not plain_text_resume_file.exists():
            raise FileNotFoundError(f"Resume file not found: {plain_text_resume_file}")
        return {"plainTextResume": plain_text_resume_file}


def _compress_resume(yaml_text: str) -> str:
    """
    Convert the raw resume YAML into a compact plaintext summary for the scoring
    prompt. Drops structural YAML overhead (key names, indentation, list markers)
    and sections irrelevant to scoring (self_identification, legal_authorization,
    work_preferences, salary_expectations, availability). The result contains all
    signal the LLM needs while using ~3x fewer tokens than the raw YAML.
    """
    data = yaml.safe_load(yaml_text) or {}

    lines = []

    # ── Header ──────────────────────────────────────────────────────────────
    pi = data.get("personal_information") or {}
    name = f"{pi.get('name', '')} {pi.get('surname', '')}".strip()
    city = pi.get("city", "")
    linkedin = pi.get("linkedin", "") or ""
    header_parts = [p for p in [name, city, linkedin] if p]
    if header_parts:
        lines.append(" | ".join(header_parts))

    # ── Education ────────────────────────────────────────────────────────────
    for edu in data.get("education_details") or []:
        degree  = edu.get("education_level", "")
        field   = edu.get("field_of_study", "")
        inst    = edu.get("institution", "")
        start   = edu.get("start_date", "")
        end     = edu.get("year_of_completion", "")
        gpa     = edu.get("final_evaluation_grade", "")
        span    = f"{start}–{end}" if start and end else str(start or end)
        gpa_str = f" | GPA {gpa}" if gpa else ""
        lines.append(f"{degree} in {field} @ {inst} ({span}){gpa_str}")

    lines.append("")

    # ── Experience ───────────────────────────────────────────────────────────
    lines.append("EXPERIENCE")
    for exp in data.get("experience_details") or []:
        pos      = exp.get("position", "")
        company  = exp.get("company", "")
        period   = exp.get("employment_period", "")
        location = exp.get("location", "")
        industry = exp.get("industry", "")
        meta     = " | ".join(p for p in [period, location, industry] if p)
        lines.append(f"{pos} @ {company} ({meta})")

        for resp in exp.get("key_responsibilities") or []:
            if isinstance(resp, dict):
                for v in resp.values():
                    if v:
                        lines.append(f"  - {v}")
            elif isinstance(resp, str) and resp:
                lines.append(f"  - {resp}")

        skills = exp.get("skills_acquired") or []
        if skills:
            lines.append(f"  Skills: {', '.join(str(s) for s in skills if s)}")
        lines.append("")

    # ── Projects ─────────────────────────────────────────────────────────────
    projects = data.get("projects") or []
    if projects:
        lines.append("PROJECTS")
        for proj in projects:
            name_p = proj.get("name", "")
            desc   = proj.get("description", "")
            link   = proj.get("link") or ""
            entry  = f"{name_p}: {desc}"
            if link:
                entry += f" ({link})"
            lines.append(entry)
        lines.append("")

    # ── Achievements ─────────────────────────────────────────────────────────
    achievements = data.get("achievements") or []
    if achievements:
        lines.append("ACHIEVEMENTS")
        for ach in achievements:
            lines.append(f"{ach.get('name', '')}: {ach.get('description', '')}")
        lines.append("")

    # ── Certifications ───────────────────────────────────────────────────────
    certs = [c for c in (data.get("certifications") or []) if c]
    if certs:
        lines.append(f"Certifications: {', '.join(str(c) for c in certs)}")

    # ── Languages ────────────────────────────────────────────────────────────
    langs = data.get("languages") or []
    if langs:
        lang_str = ", ".join(
            f"{l.get('language', '')} ({l.get('proficiency', '')})" for l in langs if l
        )
        lines.append(f"Languages: {lang_str}")

    return "\n".join(lines)


_CACHE_TTL_DAYS = 14


def _load_score_cache(output_dir: Path) -> dict:
    cache_path = output_dir.parent / "score_cache.json"
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, encoding="utf-8") as f:
            raw: dict = json.load(f)
        cutoff = datetime.now() - timedelta(days=_CACHE_TTL_DAYS)
        return {
            link: entry for link, entry in raw.items()
            if datetime.fromisoformat(entry.get("cached_at", "2000-01-01")) > cutoff
        }
    except Exception as e:
        logger.warning(f"[Cache] Could not load score cache: {e}")
        return {}


def _save_score_cache(output_dir: Path, cache: dict) -> None:
    cache_path = output_dir.parent / "score_cache.json"
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"[Cache] Could not save score cache: {e}")


def _cache_get(cache: dict, link: str) -> dict | None:
    entry = cache.get(link)
    if not entry:
        return None
    try:
        cached_at = datetime.fromisoformat(entry["cached_at"])
        if datetime.now() - cached_at > timedelta(days=_CACHE_TTL_DAYS):
            return None
    except Exception:
        return None
    return entry["result"]


def _cache_set(cache: dict, link: str, result: dict) -> None:
    cache[link] = {"cached_at": datetime.now().isoformat(), "result": result}


_ROLE_PRIORITY_PATTERNS = [
    ("data engineer", 0),
    ("data engineering", 0),
    ("data scientist", 1),
    ("data science", 1),
    ("machine learning", 2),
    ("ml engineer", 2),
    ("software engineer", 3),
    ("software engineering", 3),
]


def _role_priority(role: str) -> int:
    r = role.lower()
    for pattern, pri in _ROLE_PRIORITY_PATTERNS:
        if pattern in r:
            return pri
    return 99


def _cleanup_old_outputs(output_dir: Path, keep: int = 10):
    for pattern in ("job_matches_*.md", "job_matches_*.json"):
        old_files = sorted(output_dir.glob(pattern))
        for f in old_files[:-keep]:
            try:
                f.unlink()
                logger.debug(f"[Cleanup] Removed old output: {f.name}")
            except OSError as e:
                logger.warning(f"[Cleanup] Could not remove {f.name}: {e}")


def load_jobs_from_fixtures(path: Path) -> list[Job]:
    """Load job records from a cached JSON fixture for dry-run scoring.

    The fixture is either a top-level list of records or an object with a
    "jobs" key. Each record maps to a Job; "source" is accepted as an alias
    for "apply_method". This lets the scoring pipeline be exercised end-to-end
    without launching Selenium or hitting any job site.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    records = raw.get("jobs", []) if isinstance(raw, dict) else raw
    jobs: list[Job] = []
    for rec in records:
        jobs.append(Job(
            role=rec.get("role", ""),
            company=rec.get("company", ""),
            location=rec.get("location", ""),
            link=rec.get("link", ""),
            apply_method=rec.get("apply_method") or rec.get("source", "fixture"),
            description=rec.get("description", ""),
        ))
    return jobs


def search_and_score_jobs(config: dict, secrets: dict, resume_yaml_text: str):
    """
    Search LinkedIn, Handshake, and Simplify for recently posted jobs, then
    score them. Scraping and scoring are kept separate so the scoring half can
    be exercised offline via load_jobs_from_fixtures (see --dry-run).
    """
    groq_key = secrets.get("groq_api_key", "")
    if not groq_key:
        logger.error(
            "No 'groq_api_key' found in secrets.yaml. "
            "Get a free key at console.groq.com (no credit card required)."
        )
        return

    logger.info("Launching browsers for job search (LinkedIn, Handshake, Simplify)…")

    # Browsers are closed inside JobSearchManager._run_searcher.
    manager  = JobSearchManager(config, secrets)
    all_jobs = manager.search_all()
    logger.info(f"Total jobs collected: {len(all_jobs)}")

    score_jobs(all_jobs, config, secrets, resume_yaml_text)


def score_jobs(all_jobs: list[Job], config: dict, secrets: dict, resume_yaml_text: str):
    """
    Score a list of already-collected jobs. Each job is scored for fit (1-10)
    by Groq/Llama. Jobs that pass the threshold also get an ATS analysis:
    keyword coverage %, matched/missing keywords, and one actionable tip.
    Results are saved as a Markdown report + JSON archive.
    """
    groq_key = secrets.get("groq_api_key", "")
    if not groq_key:
        logger.error(
            "No 'groq_api_key' found in secrets.yaml. "
            "Get a free key at console.groq.com (no credit card required)."
        )
        return

    output_dir: Path = config["outputFileDirectory"]
    datestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    all_jobs.sort(key=lambda j: _role_priority(j.role))

    max_to_score = config.get("max_jobs_to_score", 0)
    if max_to_score and len(all_jobs) > max_to_score:
        logger.info(
            f"[Cap] {len(all_jobs)} jobs found; scoring only the first "
            f"{max_to_score} (max_jobs_to_score in work_preferences.yaml)"
        )
        all_jobs = all_jobs[:max_to_score]

    if not all_jobs:
        logger.warning(
            "No jobs to score. For a live run, check your positions/locations in "
            "work_preferences.yaml; for a dry run, check the fixtures file."
        )
        return

    # ── 2. Fit score every job with Groq/Llama ────────────────────────
    fit_scorer = JobFitScorer(groq_api_key=groq_key)
    ats_scorer = ATSScorer(resume_text=resume_yaml_text)
    matched: list[dict] = []
    skipped: list[dict] = []

    score_cache = _load_score_cache(output_dir)
    cache_hits  = 0

    print(f"\nAnalyzing {len(all_jobs)} jobs…\n")
    for i, job in enumerate(all_jobs, 1):
        print(
            f"  [{i:>2}/{len(all_jobs)}] {job.role} @ {job.company} "
            f"({job.apply_method})",
            end="  ", flush=True,
        )

        if len(job.description) < 150:
            print("skipped [no description]")
            skipped.append({
                "role":      job.role,
                "company":   job.company,
                "location":  job.location,
                "source":    job.apply_method,
                "link":      job.link,
                "fit_score": 0,
                "reasoning": "No usable description scraped.",
            })
            continue

        cached_fit = _cache_get(score_cache, job.link) if job.link else None
        if cached_fit:
            fit       = cached_fit
            cache_tag = " [cached]"
            cache_hits += 1
        else:
            try:
                fit = fit_scorer.score(resume_yaml_text, job)
            except Exception as e:
                if "tokens per day" in str(e).lower() or "tpd" in str(e).lower():
                    print(f"\n[Scorer] Daily token quota exhausted after {i-1} jobs scored. "
                          f"Rerun tomorrow or reduce max_jobs_to_score.")
                    break
                raise
            if job.link:
                _cache_set(score_cache, job.link, fit)
            cache_tag = ""

        entry = {
            "role":      job.role,
            "company":   job.company,
            "location":  job.location,
            "source":    job.apply_method,
            "link":      job.link,
            "fit_score": fit["score"],
            "reasoning": fit["reasoning"],
        }

        if fit["is_match"]:
            # ── 3. ATS analysis only on jobs that pass fit threshold ──
            ats = ats_scorer.analyze(resume_yaml_text, job)
            entry.update({
                "ats_score":        ats["ats_score"],
                "keyword_score":    ats["keyword_score"],
                "semantic_score":   ats["semantic_score"],
                "matched_keywords": ats["matched_keywords"],
                "missing_keywords": ats["missing_keywords"],
                "critical_missing": ats["critical_missing"],
                "ats_tip":          ats["tip"],
            })
            matched.append(entry)
            print(
                f"Fit {fit['score']}/10 ✓{cache_tag}  |  "
                f"ATS {ats['ats_score']}%  "
                f"(kw {ats['keyword_score']}% · sem {ats['semantic_score']}%)"
            )
        else:
            skipped.append(entry)
            print(f"Fit {fit['score']}/10 ✗{cache_tag}")

    _save_score_cache(output_dir, score_cache)
    if cache_hits:
        print(
            f"  [Cache] {cache_hits}/{len(all_jobs)} jobs loaded from cache "
            f"(saved ~{cache_hits * 2500:,} Groq tokens)"
        )

    # ── 4. Write Markdown report ──────────────────────────────────────
    md_path = output_dir / f"job_matches_{datestamp}.md"
    resume_pdf = secrets.get("resume_pdf_path", "")
    _write_markdown_report(
        md_path, matched, skipped, len(all_jobs), fit_scorer.threshold,
        resume_pdf_path=resume_pdf,
    )

    # ── 5. Write JSON archive ─────────────────────────────────────────
    json_path = output_dir / f"job_matches_{datestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at":   datestamp,
                "total_searched": len(all_jobs),
                "total_matched":  len(matched),
                "matched":        matched,
                "skipped":        skipped,
            },
            f, ensure_ascii=False, indent=2,
        )

    # ── 6. Prune old output files (keep latest 10) ───────────────────
    _cleanup_old_outputs(output_dir, keep=10)

    # ── 7. Auto-open report on macOS ──────────────────────────────────
    try:
        subprocess.run(["open", str(md_path)], check=False)
    except Exception:
        pass

    print(f"\n{'─' * 62}")
    print(f"  Searched : {len(all_jobs)} jobs")
    print(f"  Matched  : {len(matched)} jobs  (fit ≥ {fit_scorer.threshold}/10)")
    print(f"  Report   : {md_path}")
    print(f"  Archive  : {json_path}")
    print(f"{'─' * 62}\n")


def _write_markdown_report(
    path: Path,
    matched: list,
    skipped: list,
    total: int,
    threshold: int,
    resume_pdf_path: str = "",
):
    lines = [
        f"# Job Matches — {datetime.now().strftime('%B %d, %Y')}",
        "",
        (
            f"Searched **{total}** recently posted jobs across "
            "LinkedIn, Handshake, and Simplify.  "
        ),
        (
            f"**{len(matched)} jobs matched** your resume "
            f"(fit score ≥ {threshold}/10). Click a link to apply manually."
        ),
        "",
        "---",
        "",
    ]

    if matched:
        for job in sorted(matched, key=lambda j: j["fit_score"], reverse=True):
            ats       = job.get("ats_score", 0)
            kw_score  = job.get("keyword_score", ats)
            sem_score = job.get("semantic_score", 0)

            kw_filled  = round(kw_score / 10)
            kw_bar     = "#" * kw_filled + "-" * (10 - kw_filled)
            sem_filled = round(sem_score / 10)
            sem_bar    = "#" * sem_filled + "-" * (10 - sem_filled)

            matched_kw  = ", ".join(job.get("matched_keywords", [])) or "—"
            critical_kw = ", ".join(job.get("critical_missing", [])) or "None"
            ats_tip     = job.get("ats_tip", "")

            resume_row = (
                f"| **Resume attached** | {Path(resume_pdf_path).name} |"
                if resume_pdf_path else ""
            )
            lines += [
                f"## Fit {job['fit_score']}/10 | ATS {ats}% — {job['role']} | {job['company']}",
                "",
                f"| | |",
                f"|---|---|",
                f"| **Location** | {job['location'] or 'Not specified'} |",
                f"| **Source** | {job['source'].capitalize()} |",
                f"| **Keyword match** | `{kw_bar}` {kw_score}% |",
                f"| **Semantic alignment** | `{sem_bar}` {sem_score}% |",
                f"| **Keywords you have** | {matched_kw} |",
                f"| **Critical gaps** | {critical_kw} |",
                *([resume_row] if resume_row else []),
                "",
                f"**Why you match:** {job['reasoning']}",
                "",
            ]
            if ats_tip:
                lines.append(f"**Quick ATS fix:** {ats_tip}")
                lines.append("")
            lines += [
                f"**→ Apply here:** {job['link']}",
                "",
                "---",
                "",
            ]
    else:
        lines += [
            "> No jobs met the fit threshold.  ",
            "> Lower `JOB_SUITABILITY_SCORE` in [config.py](config.py) or "
            "broaden your positions/locations in "
            "[work_preferences.yaml](data_folder/work_preferences.yaml).",
            "",
        ]

    if skipped:
        lines += [
            "<details>",
            f"<summary>Lower-scoring jobs ({len(skipped)} — click to expand)</summary>",
            "",
        ]
        for job in sorted(skipped, key=lambda j: j["fit_score"], reverse=True):
            lines.append(
                f"- **{job['fit_score']}/10** &nbsp; {job['role']} @ {job['company']} "
                f"({job['source']}) — {job['link']}"
            )
        lines += ["", "</details>", ""]

    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search, fit-score, and ATS-analyze internship/job listings. "
            "Use --dry-run to score cached fixtures without live scraping."
        )
    )
    parser.add_argument(
        "--dry-run",
        nargs="?",
        const=str(DEFAULT_FIXTURES),
        default=None,
        metavar="FIXTURES",
        help=(
            "Score jobs from a cached fixtures JSON file instead of live "
            f"scraping. Defaults to {DEFAULT_FIXTURES} when no path is given."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    try:
        data_folder = Path("data_folder")
        secrets_file, config_file, plain_text_resume_file, output_folder = \
            FileManager.validate_data_folder(data_folder)

        config  = ConfigValidator.validate_config(config_file)
        secrets = ConfigValidator.validate_secrets(secrets_file)

        config["uploads"] = FileManager.get_uploads(plain_text_resume_file)
        config["outputFileDirectory"] = output_folder
        config["dataFolder"] = data_folder

        with open(plain_text_resume_file, "r", encoding="utf-8") as f:
            resume_yaml_text = f.read()

        resume_for_scoring = _compress_resume(resume_yaml_text)

        if args.dry_run:
            fixtures_path = Path(args.dry_run)
            logger.info(f"[Dry-run] Loading cached jobs from {fixtures_path}")
            jobs = load_jobs_from_fixtures(fixtures_path)
            logger.info(f"[Dry-run] Loaded {len(jobs)} jobs — scoring without scraping")
            score_jobs(jobs, config, secrets, resume_for_scoring)
        else:
            search_and_score_jobs(config, secrets, resume_for_scoring)

    except ConfigError as ce:
        logger.error(f"Configuration error: {ce}")
    except FileNotFoundError as fnf:
        logger.error(f"File not found: {fnf}")
    except RuntimeError as rt:
        logger.error(f"Runtime error: {rt}")
        logger.debug(traceback.format_exc())
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
