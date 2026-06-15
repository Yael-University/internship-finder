"""
Job searcher module: searches LinkedIn, Handshake, and Simplify for recently posted
jobs matching the user's work_preferences.yaml settings, then returns Job objects
for fit scoring.

Scrapers run in parallel — one browser per site — and results are deduplicated
before being returned to the caller.
"""
from __future__ import annotations

import random
import re
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from src.job import Job
from src.logging import logger
from src.utils.chrome_utils import init_browser, init_browser_handshake, init_browser_stealth

# Shared lock so only one scraper thread can pause for CAPTCHA input at a time
_captcha_lock = threading.Lock()


class BaseJobSearcher(ABC):
    WAIT_TIMEOUT = 12

    def __init__(self, driver, config: dict):
        self.driver = driver
        self.config = config
        self.positions: list[str] = config.get("positions", [])
        self.locations: list[str] = config.get("locations", [])
        self.company_blacklist: list[str] = [
            c.lower() for c in config.get("company_blacklist", []) if c
        ]
        self.title_blacklist: list[str] = [
            t.lower() for t in config.get("title_blacklist", [])
            if t and t not in ("word1", "word2")  # ignore placeholder values
        ]
        self.location_blacklist: list[str] = [
            l.lower() for l in config.get("location_blacklist", []) if l
        ]
        self.experience_levels: dict = config.get("experience_level", {})
        self.job_types: dict = config.get("job_types", {})
        self.date_filter: dict = config.get("date", {})
        self.apply_once_at_company: bool = config.get("apply_once_at_company", False)

    @abstractmethod
    def search(self) -> list[Job]:
        pass

    # ── filtering ──────────────────────────────────────────────────────────

    def _is_blacklisted(self, job: Job) -> bool:
        if any(b in job.company.lower() for b in self.company_blacklist):
            logger.debug(f"Blacklisted company: {job.company}")
            return True
        if any(b in job.role.lower() for b in self.title_blacklist):
            logger.debug(f"Blacklisted title: {job.role}")
            return True
        if any(b in job.location.lower() for b in self.location_blacklist):
            logger.debug(f"Blacklisted location: {job.location}")
            return True
        return False

    # ── preference → URL param helpers ────────────────────────────────────

    def _active_experience_codes_linkedin(self) -> list[int]:
        mapping = {
            "internship": 1, "entry": 2, "associate": 3,
            "mid_senior_level": 4, "director": 5, "executive": 6,
        }
        return [mapping[k] for k, v in self.experience_levels.items() if v]

    def _active_job_type_codes_linkedin(self) -> list[str]:
        mapping = {
            "full_time": "F", "contract": "C", "part_time": "P",
            "temporary": "T", "internship": "I", "volunteer": "V", "other": "O",
        }
        return [mapping[k] for k, v in self.job_types.items() if v]

    def _date_code_linkedin(self) -> str:
        if self.date_filter.get("24_hours"): return "r86400"
        if self.date_filter.get("week"):     return "r604800"
        if self.date_filter.get("month"):    return "r2592000"
        return ""

    def _date_param(self) -> str:
        """Shared date-posted query param for Handshake and Simplify (same schema)."""
        if self.date_filter.get("24_hours"): return "&date_posted=today"
        if self.date_filter.get("week"):     return "&date_posted=this_week"
        if self.date_filter.get("month"):    return "&date_posted=this_month"
        return ""

    # ── selenium helpers ───────────────────────────────────────────────────

    def _wait_for(self, by, selector, timeout: int | None = None):
        t = timeout or self.WAIT_TIMEOUT
        return WebDriverWait(self.driver, t).until(
            EC.presence_of_element_located((by, selector))
        )

    def _safe_text(self, root, by, selector) -> str:
        try:
            return root.find_element(by, selector).text.strip()
        except NoSuchElementException:
            return ""

    def _safe_text_any(self, root, *selectors: str) -> str:
        """Try each CSS selector in turn; return the first that yields non-empty text.

        Unlike a comma-joined selector (which Selenium resolves in DOM order and may
        return an element whose .text is empty), this method stops at the first
        selector that actually produces visible text.
        """
        for sel in selectors:
            try:
                text = root.find_element(By.CSS_SELECTOR, sel).text.strip()
                if text:
                    return text
            except NoSuchElementException:
                pass
        return ""

    def _safe_attr(self, root, by, selector, attr) -> str:
        try:
            return root.find_element(by, selector).get_attribute(attr) or ""
        except NoSuchElementException:
            return ""

    @staticmethod
    def _delay(lo: float = 1.5, hi: float = 3.5):
        time.sleep(random.uniform(lo, hi))

    # ── CAPTCHA / bot-detection pause ─────────────────────────────────────

    _BLOCK_URL_SIGNALS = (
        "authwall", "checkpoint", "/uas/login", "cf-challenge",
        "security-check", "captcha", "recaptcha", "interstitial",
    )
    _BLOCK_TITLE_SIGNALS = (
        "just a moment", "access denied", "security check",
        "verify you are human", "are you a human",
        "robot", "unusual traffic", "blocked",
    )

    def _pause_if_blocked(self, site: str = "") -> None:
        """
        Detect CAPTCHA / bot-block pages and pause so the user can solve
        the challenge in the browser, then press Enter to continue.
        """
        try:
            url   = self.driver.current_url.lower()
            title = self.driver.title.lower()
        except Exception:
            return

        url_blocked   = any(s in url   for s in self._BLOCK_URL_SIGNALS)
        title_blocked = any(s in title for s in self._BLOCK_TITLE_SIGNALS)

        if url_blocked or title_blocked:
            tag = f"[{site}] " if site else ""
            with _captcha_lock:
                print(
                    f"\n{tag}Bot-detection / CAPTCHA screen detected.\n"
                    "  → Solve the challenge in the browser window, then press Enter here."
                )
                try:
                    input("Press Enter to continue > ")
                except EOFError:
                    pass  # non-interactive stdin (IDE/pipe) — continue automatically
            self._delay(2, 3)


# ══════════════════════════════════════════════════════════════════════════════
# LinkedIn
# ══════════════════════════════════════════════════════════════════════════════

class LinkedInJobSearcher(BaseJobSearcher):
    BASE_URL = "https://www.linkedin.com/jobs/search/"

    def __init__(self, driver, config: dict, email: str = "", password: str = ""):
        super().__init__(driver, config)
        self.email    = email
        self.password = password
        self._logged_in = False

    @staticmethod
    def _linkedin_logged_in(url: str) -> bool:
        """True when the current URL is a normal authenticated LinkedIn page."""
        _AUTH_PAGES = ("login", "checkpoint", "authwall", "challenge", "uas", "signup")
        return "linkedin.com" in url and not any(k in url for k in _AUTH_PAGES)

    def _login(self) -> bool:
        """
        Log in to LinkedIn.  On first run the profile directory is empty so a
        full password login is performed.  On subsequent runs Chrome loads the
        saved profile and the session is still active, so we land directly on
        the feed and skip typing credentials entirely.
        """
        try:
            self.driver.get("https://www.linkedin.com/feed/")
            self._delay(3, 5)
            cur = self.driver.current_url
            if self._linkedin_logged_in(cur):
                logger.info("[LinkedIn] Session already active (loaded from profile)")
                return True

            # Not logged in — go to the login form
            self.driver.get("https://www.linkedin.com/login")
            self._delay(2, 4)

            email_field = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "username"))
            )
            email_field.clear()
            for ch in self.email:
                email_field.send_keys(ch)
                time.sleep(random.uniform(0.05, 0.15))

            self._delay(0.5, 1.0)

            pwd_field = self.driver.find_element(By.ID, "password")
            for ch in self.password:
                pwd_field.send_keys(ch)
                time.sleep(random.uniform(0.05, 0.15))

            self._delay(0.8, 1.5)
            self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
            self._delay(4, 7)

            cur = self.driver.current_url
            if self._linkedin_logged_in(cur):
                logger.info("[LinkedIn] Logged in successfully")
                return True

            # Security checkpoint or email verification — poll until the user
            # approves on their phone; no Enter needed.
            if any(k in cur for k in ("checkpoint", "challenge", "verify")):
                print(
                    "\n[LinkedIn] Security verification required.\n"
                    "  → Approve it on your phone/other device.\n"
                    "  → Waiting automatically (up to 3 min)..."
                )
                for _ in range(180):
                    time.sleep(1)
                    try:
                        cur = self.driver.current_url
                        if self._linkedin_logged_in(cur):
                            logger.info("[LinkedIn] Logged in after verification")
                            return True
                    except Exception:
                        break
                logger.warning("[LinkedIn] Verification timed out after 3 min")

            logger.warning(f"[LinkedIn] Login unsuccessful — ended at: {self.driver.current_url}")
            return False

        except Exception as e:
            logger.error(f"[LinkedIn] Login error: {e}")
            return False

    def search(self) -> list[Job]:
        if self.email and self.password:
            self._logged_in = self._login()
            if not self._logged_in:
                logger.warning("[LinkedIn] Login failed — scraping in unauthenticated mode")

        jobs: list[Job] = []
        seen_companies: set[str] = set()

        exp   = ",".join(str(c) for c in self._active_experience_codes_linkedin())
        jtype = ",".join(self._active_job_type_codes_linkedin())
        date  = self._date_code_linkedin()

        for position in self.positions:
            for location in self.locations:
                logger.info(f"[LinkedIn] '{position}' in '{location}'")
                url   = self._build_search_url(position, location, exp, jtype, date)
                found = self._scrape_listing_page(url, seen_companies)
                jobs.extend(found)
                self._delay(3, 5)

        logger.info(f"[LinkedIn] Total matched: {len(jobs)}")
        return jobs

    def _build_search_url(self, position, location, exp, jtype, date) -> str:
        p = (
            f"keywords={quote_plus(position)}"
            f"&location={quote_plus(location)}"
            f"&sortBy=DD"
        )
        if exp:   p += f"&f_E={exp}"
        if jtype: p += f"&f_JT={jtype}"
        if date:  p += f"&f_TPR={date}"
        return f"{self.BASE_URL}?{p}"

    def _scrape_listing_page(self, url: str, seen_companies: set) -> list[Job]:
        jobs = []
        try:
            self.driver.get(url)
            self._delay(3, 5)
            self._pause_if_blocked("LinkedIn")
            self._scroll_page(times=5)   # more passes to trigger lazy-render
            self._delay(2, 3)            # extra wait for content to settle

            # Try job-card-specific selectors first (authenticated DOM), then fall
            # back to the public board list class.
            _card_sel = (
                ".job-card-container, "
                "[data-view-name='job-card'], "
                "ul.jobs-search__results-list li, "
                ".jobs-search-results-list__list-item"
            )
            try:
                self._wait_for(By.CSS_SELECTOR, _card_sel, timeout=15)
            except TimeoutException:
                logger.warning(
                    "[LinkedIn] No job cards appeared — empty results or bot block"
                )
                return jobs

            # Prefer the authenticated job-card selector; fall back progressively.
            cards = self.driver.find_elements(
                By.CSS_SELECTOR, ".job-card-container"
            )
            if not cards:
                cards = self.driver.find_elements(
                    By.CSS_SELECTOR, "[data-view-name='job-card']"
                )
            if not cards:
                cards = self.driver.find_elements(
                    By.CSS_SELECTOR, "ul.jobs-search__results-list li"
                )
            if not cards:
                cards = self.driver.find_elements(
                    By.CSS_SELECTOR, ".jobs-search-results-list__list-item"
                )

            # Drop structural list items that contain no text (headers, spacers)
            cards = [c for c in cards if c.text.strip()]
            logger.info(f"[LinkedIn] {len(cards)} cards found")

            # Pass 1: parse all card metadata while still on the search results page.
            # Must happen before any navigation or card elements go stale.
            stubs: list[Job] = []
            _debug_logged = False
            for card in cards[:25]:
                try:
                    job = self._parse_card(card)
                    if job and not self._is_blacklisted(job):
                        stubs.append(job)
                    elif job is None and not _debug_logged:
                        # Log the first failing card's raw text to help diagnose selector drift
                        try:
                            logger.debug(
                                f"[LinkedIn] First card parse failed — text preview: "
                                f"{card.text[:200]!r}"
                            )
                        except Exception:
                            pass
                        _debug_logged = True
                except Exception as e:
                    logger.debug(f"[LinkedIn] Card parse error: {e}")

            # Pass 2: navigate to each job page and fetch the full description.
            for job in stubs:
                company_key = job.company.lower().strip()
                if self.apply_once_at_company and company_key and company_key in seen_companies:
                    continue
                try:
                    desc = self._fetch_description(job.link)
                    if desc is None:
                        # Login wall — keep job stub; scorer returns 0 for empty desc
                        job.description = ""
                        if company_key:
                            seen_companies.add(company_key)
                        jobs.append(job)
                    elif desc:
                        job.description = desc
                        if company_key:
                            seen_companies.add(company_key)
                        jobs.append(job)
                    self._delay(1, 2)
                except Exception as e:
                    logger.debug(f"[LinkedIn] Description fetch error: {e}")
        except Exception as e:
            logger.error(f"[LinkedIn] Page scrape error: {e}")
        return jobs

    def _scroll_page(self, times: int = 3):
        for _ in range(times):
            self.driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
            time.sleep(1.5)

    @staticmethod
    def _clean_linkedin_title(raw: str) -> str:
        """Remove LinkedIn's appended verification / accessibility noise from titles."""
        if not raw:
            return raw
        # LinkedIn appends the title again with "with verification" for accessibility.
        # First line is always the real title.
        return raw.splitlines()[0].strip()

    def _parse_card(self, card) -> Optional[Job]:
        # ── Authenticated job-card-container DOM (2024-2026) ──────────────────
        # Title: prefer aria-hidden span which has clean display text only
        # (LinkedIn duplicates the title with "with verification" for accessibility).
        # Use _safe_text_any so we stop at the first selector that returns real text,
        # rather than letting Selenium pick the first DOM-order match (which may be empty).
        title = self._safe_text_any(card,
            ".job-card-list__title--link span[aria-hidden='true']",
            ".job-card-list__title span[aria-hidden='true']",
            ".job-card-list__title--link",
            ".job-card-list__title",
        )
        # Company: no aria-hidden span needed — the primary-description element
        # is plain text, not duplicated with a badge. Try multiple known selectors.
        company = self._safe_text_any(card,
            ".job-card-container__primary-description",
            ".artdeco-entity-lockup__subtitle",
            ".job-card-list__company-name",
            "a[href*='/company/']",
        )
        loc = self._safe_text_any(card,
            ".job-card-container__metadata-wrapper li",
            ".job-card-container__metadata-item",
            ".artdeco-entity-lockup__caption li",
        )
        link = self._safe_attr(card, By.CSS_SELECTOR,
            "a.job-card-container__link, "
            "a.job-card-list__title--link",
            "href")

        # ── Unauthenticated public board (fallback) ───────────────────────────
        if not title:
            title = self._safe_text(card, By.CSS_SELECTOR, "h3.base-search-card__title")
        if not company:
            company = self._safe_text(card, By.CSS_SELECTOR, "h4.base-search-card__subtitle")
        if not loc:
            loc = self._safe_text(card, By.CSS_SELECTOR, "span.job-search-card__location")
        if not link:
            link = self._safe_attr(card, By.CSS_SELECTOR, "a.base-card__full-link", "href")

        # ── Any anchor with a LinkedIn job view URL (last resort for link) ────
        if not link:
            try:
                for a in card.find_elements(By.TAG_NAME, "a"):
                    href = a.get_attribute("href") or ""
                    if re.search(r"/jobs/view/\d+", href):
                        link = href
                        break
            except Exception:
                pass

        title = self._clean_linkedin_title(title)

        if not title or not link:
            return None

        job             = Job()
        job.role        = title
        job.company     = company
        job.location    = loc
        job.link        = link.split("?")[0]
        job.apply_method = "linkedin"
        return job

    def _fetch_description(self, url: str) -> Optional[str]:
        """
        Returns description text on success, None if a LinkedIn login/auth wall
        redirected the request (job should still be kept), or "" on genuine failure.
        """
        try:
            self.driver.get(url)
            self._delay(3, 4)
            self._pause_if_blocked("LinkedIn")
            cur = self.driver.current_url.lower()
            logger.debug(f"[LinkedIn] Job page URL: {self.driver.current_url}")
            if (
                "linkedin.com/login" in cur
                or "authwall" in cur
                or "/checkpoint/" in cur
                or "linkedin.com/uas/login" in cur
            ):
                logger.warning(
                    f"[LinkedIn] Login wall detected — keeping job without description "
                    f"(redirected to: {self.driver.current_url})"
                )
                return None  # sentinel: login wall, not a real failure

            # Scroll to trigger lazy-render, then click "Show more" if present
            self.driver.execute_script("window.scrollTo(0, 600);")
            time.sleep(1.5)
            try:
                self.driver.find_element(
                    By.CSS_SELECTOR,
                    "button.show-more-less-html__button, "
                    "button[aria-label*='more'], "
                    "button.jobs-description__footer-button"
                ).click()
                time.sleep(1)
            except NoSuchElementException:
                pass

            time.sleep(3)  # extra wait for LinkedIn's lazy-rendered description component

            # JS: try named selectors first, then fall back to a keyword scan of
            # all large text blocks — avoids hard-coding class names that LinkedIn
            # rotates frequently.
            _JS = """
            const selectors = [
                '.jobs-description-content__text',
                '.jobs-description__content',
                '#job-details',
                'article.jobs-description',
                '.jobs-box__html-content',
                '.show-more-less-html__markup',
                'div.description__text',
                'section.description'
            ];
            for (const sel of selectors) {
                const el = document.querySelector(sel);
                const txt = el && el.innerText ? el.innerText.trim() : '';
                if (txt.length > 80) return txt.slice(0, 3000);
            }
            // Keyword fallback: find the largest div/section containing job keywords
            const keywords = ['responsibilities', 'requirements', 'qualifications',
                               'what you', 'about the role', 'about this job',
                               'we are looking', 'you will'];
            const blocks = document.querySelectorAll('div, article, section, li');
            let best = '';
            for (const el of blocks) {
                const txt = el.innerText ? el.innerText.trim() : '';
                if (txt.length > best.length && txt.length < 10000) {
                    const lower = txt.toLowerCase();
                    if (keywords.some(k => lower.includes(k))) {
                        best = txt;
                    }
                }
            }
            if (best.length > 80) return best.slice(0, 3000);
            // Broad fallback: grab all of <main> — noisy but better than nothing
            const main = document.querySelector('main');
            if (main) {
                const mainTxt = main.innerText ? main.innerText.trim() : '';
                if (mainTxt.length > 400) return mainTxt.slice(0, 3000);
            }
            return null;
            """
            for attempt in range(3):
                text = self.driver.execute_script(_JS)
                if text and len(text) > 80:
                    return text
                if attempt < 2:
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(3)

            logger.debug(
                f"[LinkedIn] JS extraction found nothing — page title: {self.driver.title!r}"
            )
        except Exception as e:
            logger.debug(f"[LinkedIn] Description fetch failed: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# Handshake  (email + password login — works with .edu or regular email)
# ══════════════════════════════════════════════════════════════════════════════

class HandshakeJobSearcher(BaseJobSearcher):
    LOGIN_URL = "https://app.joinhandshake.com/login"
    JOBS_URL  = "https://app.joinhandshake.com/job-search"

    def __init__(self, driver, config: dict, email: str, password: str):
        super().__init__(driver, config)
        self.email    = email
        self.password = password

    def search(self) -> list[Job]:
        if not self.email:
            logger.info("[Handshake] No credentials in secrets.yaml — skipping")
            return []

        if not self._login():
            logger.warning("[Handshake] Login failed — skipping")
            return []

        jobs: list[Job] = []
        seen_companies: set[str] = set()

        # Handshake search doesn't filter by location via URL — search per position only.
        for position in self.positions:
            logger.info(f"[Handshake] '{position}'")
            found = self._scrape_listing_page(position, "", seen_companies)
            jobs.extend(found)
            self._delay(2, 4)

        logger.info(f"[Handshake] Total matched: {len(jobs)}")
        return jobs

    def _login(self) -> bool:
        try:
            print("\n" + "=" * 60)
            print("[Handshake] Opening login page — please wait...")
            print("=" * 60)
            try:
                self.driver.get(self.LOGIN_URL)
            except Exception:
                pass
            self._delay(3, 5)

            url = self.driver.current_url
            if "handshake.com" in url and "login" not in url and "access" not in url:
                logger.info("[Handshake] Persistent profile session valid — already logged in")
                print("[Handshake] Already logged in via saved profile.\n")
                return True

            print("\n" + "=" * 60)
            print("[Handshake] ACTION REQUIRED: log in to the Handshake browser window.")
            print("  1. Complete any Cloudflare verification if prompted")
            print("  2. Enter your credentials and log in")
            print("  Watching for login automatically (up to 5 minutes)...")
            print("=" * 60 + "\n")

            deadline = time.time() + 300
            last_status = time.time()
            while time.time() < deadline:
                try:
                    url = self.driver.current_url
                except Exception:
                    break
                if "handshake.com" in url and "login" not in url and "access" not in url:
                    print("\n[Handshake] Login detected! Continuing with scrape...\n")
                    logger.info("[Handshake] Login detected automatically")
                    return True
                if time.time() - last_status >= 30:
                    remaining = int(deadline - time.time())
                    print(f"[Handshake] Still waiting for login... ({remaining}s left)")
                    last_status = time.time()
                time.sleep(2)

            logger.warning("[Handshake] Login timed out after 5 minutes")
            return False

        except Exception as e:
            logger.error(f"[Handshake] Login error: {e}")
            return False

    def _scrape_listing_page(
        self, position: str, location: str, seen_companies: set
    ) -> list[Job]:
        jobs = []
        try:
            # Navigate to the search page and type the query — Handshake's SPA
            # ignores URL query params on direct navigation, so we must interact
            # with the search input.
            self.driver.get(self.JOBS_URL)
            self._delay(4, 6)

            try:
                query_input = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "input[name='query']"))
                )
            except TimeoutException:
                logger.warning(f"[Handshake] Search input not found for '{position}'")
                return jobs

            query_input.clear()
            query_input.send_keys(position)
            self._delay(0.5, 1)
            # Re-find to avoid stale ref after React re-render
            query_input = self.driver.find_element(By.CSS_SELECTOR, "input[name='query']")
            query_input.send_keys(Keys.RETURN)
            self._delay(4, 7)

            try:
                self._wait_for(By.CSS_SELECTOR, "[data-hook^='job-hide-button-']", timeout=15)
            except TimeoutException:
                logger.warning(f"[Handshake] No job cards loaded for '{position}'")
                return jobs

            # Extract all IDs as strings NOW, before any navigation away from this page.
            hide_btns = self.driver.find_elements(
                By.CSS_SELECTOR, "[data-hook^='job-hide-button-']"
            )
            ids = [
                btn.get_attribute("data-hook").replace("job-hide-button-", "")
                for btn in hide_btns[:10]
            ]
            logger.info(f"[Handshake] {len(ids)} cards for '{position}'")

            # Parse all card metadata while still on the listing page.
            candidates: list[Job] = []
            for job_id in ids:
                try:
                    job = self._parse_card(job_id)
                    if not job or self._is_blacklisted(job):
                        continue
                    if self.apply_once_at_company and job.company.lower() in seen_companies:
                        continue
                    candidates.append(job)
                except Exception as e:
                    logger.debug(f"[Handshake] Card parse error: {e}")

            # Navigate to each job's detail page for the description.
            for job in candidates:
                seen_companies.add(job.company.lower())
                job.description = self._fetch_description(job.link)
                if job.description:
                    jobs.append(job)
                self._delay(1, 2)

        except Exception as e:
            logger.error(f"[Handshake] Scrape error: {e}")
        return jobs

    def _parse_card(self, job_id: str) -> Optional[Job]:
        try:
            card = self.driver.find_element(
                By.XPATH, f"//div[@data-hook='job-result-card | {job_id}']"
            )
        except NoSuchElementException:
            return None

        try:
            title_el = card.find_element(
                By.XPATH, ".//div[@aria-label[starts-with(., 'View ')]]"
            )
            title = title_el.get_attribute("aria-label").replace("View ", "", 1)
        except Exception:
            return None

        try:
            company = card.find_element(By.CSS_SELECTOR, "img[alt]").get_attribute("alt") or ""
        except Exception:
            company = ""

        if not company:
            try:
                for span in card.find_elements(By.CSS_SELECTOR, "span"):
                    text = span.text.strip()
                    if text:
                        company = text
                        break
            except Exception:
                pass

        try:
            footer = card.find_element(
                By.CSS_SELECTOR, "[data-hook='job-result-card-footer']"
            )
            loc = footer.find_element(By.CSS_SELECTOR, "span").text.strip()
        except Exception:
            loc = ""

        job = Job()
        job.role = title
        job.company = company
        job.location = loc
        job.link = f"https://app.joinhandshake.com/jobs/{job_id}"
        job.apply_method = "handshake"
        return job

    def _fetch_description(self, url: str) -> str:
        try:
            # Navigate to the full job page (/jobs/{id}) — direct navigation to
            # /job-search/{id} now redirects back to /job-search because Handshake's
            # SPA panel only opens via in-page card clicks, not direct URL entry.
            self.driver.get(url)
            self._delay(2, 3)

            cur = self.driver.current_url
            if "login" in cur or "access" in cur:
                logger.warning("[Handshake] Redirected to login — session may have expired")
                return ""

            # Expand truncated description if present
            try:
                self.driver.find_element(
                    By.CSS_SELECTOR, "button.view-more-button, [data-hook='view-more-button']"
                ).click()
                self._delay(0.5, 1)
            except Exception:
                pass

            _JS = """
            const selectors = [
                '[data-hook="job-description"]',
                '[data-hook="description"]',
                '[data-hook="right-content"]',
                '.job-description',
                'section.description',
                'article',
            ];
            for (const sel of selectors) {
                const el = document.querySelector(sel);
                const txt = el && el.innerText ? el.innerText.trim() : '';
                if (txt.length > 80) return txt.slice(0, 5000);
            }
            const keywords = ['responsibilities', 'requirements', 'qualifications',
                              'what you', 'about the role', 'job description',
                              'we are looking', 'you will'];
            const blocks = document.querySelectorAll('div, section');
            let best = '';
            for (const el of blocks) {
                const txt = el.innerText ? el.innerText.trim() : '';
                if (txt.length > best.length && txt.length < 15000) {
                    if (keywords.some(k => txt.toLowerCase().includes(k))) best = txt;
                }
            }
            if (best.length > 80) return best.slice(0, 5000);
            const main = document.querySelector('main');
            if (main) {
                const t = main.innerText ? main.innerText.trim() : '';
                if (t.length > 200) return t.slice(0, 5000);
            }
            return null;
            """

            text = self.driver.execute_script(_JS)
            if not text or len(text) < 50:
                return ""

            # Strip boilerplate sections that follow the description
            for stop in ["About the employer", "Similar jobs", "About the company"]:
                if stop in text:
                    text = text.split(stop, 1)[0].strip()

            # Strip the "Job description" header if the JS grabbed it
            if text.startswith("Job description"):
                text = text[len("Job description"):].strip()

            return text if len(text) > 50 else ""

        except Exception as e:
            logger.debug(f"[Handshake] Description fetch failed: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# Simplify
# ══════════════════════════════════════════════════════════════════════════════

class SimplifyJobSearcher(BaseJobSearcher):
    BASE_URL = "https://simplify.jobs/jobs"

    # Selectors used to find the search input on the Simplify SPA
    _SEARCH_INPUT_SEL = (
        "[data-testid='search-input'], "
        "input[placeholder*='earch'], "
        "input[type='search'], "
        "input[name='search'], "
        "input[name='query']"
    )

    def search(self) -> list[Job]:
        jobs: list[Job] = []
        seen_companies: set[str] = set()

        for position in self.positions:
            for location in self.locations:
                logger.info(f"[Simplify] '{position}' in '{location}'")
                found = self._scrape_listing_page(position, location, seen_companies)
                jobs.extend(found)
                self._delay(3, 5)

        logger.info(f"[Simplify] Total matched: {len(jobs)}")
        return jobs

    def _scrape_listing_page(self, position: str, location: str, seen_companies: set) -> list[Job]:
        jobs = []
        try:
            self.driver.get(self.BASE_URL)
            self._delay(4, 6)

            # Dismiss cookie banner if present
            try:
                self.driver.find_element(
                    By.CSS_SELECTOR, "[data-testid='cookie-banner-accept-button']"
                ).click()
                self._delay(0.5, 1)
            except NoSuchElementException:
                pass

            # Type the search term into the search input.
            # Simplify's URL query-params are not reliable — the site filters
            # client-side only after the user types in the box.
            try:
                search_input = WebDriverWait(self.driver, 12).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, self._SEARCH_INPUT_SEL)
                    )
                )
                search_input.clear()
                search_input.send_keys(position)
                search_input.send_keys(Keys.RETURN)
                logger.debug(f"[Simplify] Typed '{position}' into search box")
                self._delay(4, 6)  # let the React SPA re-render results
            except TimeoutException:
                logger.warning("[Simplify] Search input not found — trying URL params fallback")
                p = (
                    f"search={quote_plus(position)}"
                    f"&location={quote_plus(location)}"
                    f"&sort=date"
                    f"{self._date_param()}"
                )
                self.driver.get(f"{self.BASE_URL}?{p}")
                self._delay(6, 9)

            # Simplify is a React SPA — wait for card elements
            try:
                self._wait_for(By.CSS_SELECTOR, "[data-testid='job-card']", timeout=15)
            except TimeoutException:
                logger.warning("[Simplify] Job cards didn't appear in time")
                return jobs

            cards = self.driver.find_elements(By.CSS_SELECTOR, "[data-testid='job-card']")
            logger.info(f"[Simplify] {len(cards)} cards found")
            n = min(len(cards), 20)

            for idx in range(n):
                try:
                    # Re-fetch cards each iteration to avoid stale element refs
                    cards = self.driver.find_elements(
                        By.CSS_SELECTOR, "[data-testid='job-card']"
                    )
                    if idx >= len(cards):
                        break
                    job = self._parse_card(cards[idx])
                    if not job or self._is_blacklisted(job):
                        continue
                    if self.apply_once_at_company and job.company.lower() in seen_companies:
                        continue
                    seen_companies.add(job.company.lower())
                    jobs.append(job)
                    self._delay(1, 2)
                except Exception as e:
                    err = str(e).lower()
                    if "no such window" in err or "web view not found" in err or "target window already closed" in err:
                        logger.warning(f"[Simplify] Browser window closed at card {idx} — stopping early")
                        return jobs
                    logger.debug(f"[Simplify] Card {idx} parse error: {e}")
        except Exception as e:
            logger.error(f"[Simplify] Scrape error: {e}")
        return jobs

    def _parse_card(self, card) -> Optional[Job]:
        """Click a job card, read the sidebar details view, return a populated Job."""
        # Try data-testid attributes first (most reliable)
        role    = self._safe_text(card, By.CSS_SELECTOR, "[data-testid='job-title']")
        company = self._safe_text(card, By.CSS_SELECTOR, "[data-testid='company-name']")
        loc     = self._safe_text(card, By.CSS_SELECTOR, "[data-testid='job-location']")

        # Fallback: parse visible text lines
        if not role or not company:
            lines = [l.strip() for l in card.text.splitlines() if l.strip()]
            _WORK_MODES = {"in person", "remote", "hybrid", "in-person", "on-site", "onsite"}
            # Filter out work-mode tags, badges, and short noise words
            content = [
                l for l in lines
                if l.lower() not in _WORK_MODES and len(l) > 2
            ]
            # Simplify cards show company first, then role title
            if not company and len(content) >= 1:
                company = content[0]
            if not role and len(content) >= 2:
                role = content[1]

            if not loc:
                for i, ln in enumerate(lines):
                    if ln.lower() in _WORK_MODES and i >= 1:
                        candidate = lines[i - 1]
                        if candidate.lower() not in _WORK_MODES and len(candidate) > 2:
                            loc = candidate
                        break

        if not role or not company:
            return None

        original_window = self.driver.current_window_handle
        original_handles = set(self.driver.window_handles)
        try:
            card.click()
            self._delay(1, 2)
        except Exception:
            return None

        # If clicking opened a new tab, switch to it (some Simplify cards do this)
        new_handles = set(self.driver.window_handles) - original_handles
        switched_tab = bool(new_handles)
        if switched_tab:
            self.driver.switch_to.window(new_handles.pop())

        # Capture jobId from URL for a stable apply link
        m = re.search(r"jobId=([^&]+)", self.driver.current_url)
        job_id = m.group(1) if m else ""
        link = (
            f"https://simplify.jobs/jobs?jobId={job_id}"
            if job_id else "https://simplify.jobs/jobs"
        )

        # Read description from the sidebar — try multiple known selectors
        description = ""
        for sel in (
            "[data-testid='job-description']",
            "[data-testid='details-view']",
            "[data-testid='job-details']",
            ".job-description",
        ):
            try:
                details = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, sel))
                )
                text = details.text.strip()
                if len(text) >= 50:
                    description = text
                    break
            except Exception:
                continue

        # If we switched to a new tab, close it and return to the listing page
        if switched_tab:
            try:
                self.driver.close()
                self.driver.switch_to.window(original_window)
            except Exception:
                pass

        if not description:
            return None

        job              = Job()
        job.role         = role
        job.company      = company
        job.location     = loc
        job.link         = link
        job.apply_method = "simplify"
        job.description  = description
        return job


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

class JobSearchManager:
    """
    Runs LinkedIn, Handshake, and Simplify scrapers in parallel — each site gets
    its own browser instance — then deduplicates results before returning.

    LinkedIn uses undetected-chromedriver and, when credentials are provided,
    a persistent Chrome profile so the browser fingerprint stays consistent
    across runs (stored at data_folder/linkedin_profile/).
    """

    def __init__(self, config: dict, secrets: dict):
        self._config          = config
        self._apply_once      = config.get("apply_once_at_company", False)
        self._handshake_email = secrets.get("handshake_email", "")
        self._handshake_pass  = secrets.get("handshake_password", "")
        self._linkedin_email  = secrets.get("linkedin_email", "")
        self._linkedin_pass   = secrets.get("linkedin_password", "")

        output_dir = config.get("outputFileDirectory", Path("data_folder") / "output")
        data_folder = Path(config.get("dataFolder", Path(output_dir).parent))
        self._handshake_profile_dir = (
            str(data_folder / "handshake_profile")
            if self._handshake_email else None
        )
        self._linkedin_profile_dir = (
            str(data_folder / "linkedin_profile")
            if self._linkedin_email else None
        )

    def _run_searcher(
        self, label: str, factory, stealth: bool = False,
        profile_dir: str | None = None, browser_init=None
    ) -> list[Job]:
        """Create a fresh browser, run one searcher, quit the browser, return jobs."""
        if browser_init is not None:
            driver = browser_init()
        elif stealth:
            driver = init_browser_stealth(profile_dir)
        else:
            driver = init_browser()
        try:
            logger.info(f"── Starting {label} ──")
            return factory(driver).search()
        except Exception as e:
            logger.error(f"{label} search crashed: {e}")
            return []
        finally:
            try:
                driver.quit()
            except Exception:
                pass

    def search_all(self) -> list[Job]:
        cfg      = self._config
        hs_email = self._handshake_email
        hs_pass  = self._handshake_pass
        li_email = self._linkedin_email
        li_pass  = self._linkedin_pass
        li_prof  = self._linkedin_profile_dir

        hs_prof  = self._handshake_profile_dir

        # Handshake uses standard Chrome (NOT uc) via browser_init — avoids sharing
        # the patched uc ChromeDriver binary with LinkedIn and causing a binary conflict
        # when the Handshake browser quits while LinkedIn is still running.
        hs_browser_init = lambda: init_browser_handshake(hs_prof)

        # (label, factory, use_stealth_browser, profile_dir, browser_init)
        tasks = [
            ("LinkedIn",  lambda d: LinkedInJobSearcher(d, cfg, li_email, li_pass), True,  li_prof, None),
            ("Handshake", lambda d: HandshakeJobSearcher(d, cfg, hs_email, hs_pass), False, None,    hs_browser_init),
            ("Simplify",  lambda d: SimplifyJobSearcher(d, cfg),                     False, None,    None),
        ]

        raw: list[Job] = []
        per_source: dict[str, int] = {label: 0 for label, *_ in tasks}
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {
                pool.submit(self._run_searcher, label, factory, stealth, profile_dir, browser_init): label
                for label, factory, stealth, profile_dir, browser_init in tasks
            }
            for future in as_completed(futures):
                label = futures[future]
                try:
                    jobs = future.result()
                    per_source[label] = len(jobs)
                    logger.info(f"[{label}] returned {len(jobs)} jobs")
                    if not jobs:
                        logger.warning(
                            f"[{label}] returned 0 jobs — possible selector breakage, "
                            "auth wall, or simply no matches"
                        )
                    raw.extend(jobs)
                except Exception as e:
                    logger.error(f"{label} future raised: {e}")

        summary = ", ".join(f"{label}={per_source[label]}" for label, *_ in tasks)
        logger.info(f"[Search] Per-source job counts — {summary}")

        return self._deduplicate(raw)

    def _deduplicate(self, jobs: list[Job]) -> list[Job]:
        """Remove cross-site duplicates by URL and (company, role) pair.
        If apply_once_at_company is set, also keep only the first job per company."""
        seen_urls:      set[str]   = set()
        seen_pairs:     set[tuple] = set()
        seen_companies: set[str]   = set()
        unique: list[Job] = []

        for job in jobs:
            # LinkedIn links have ?-params stripped already in _parse_card;
            # Simplify links carry ?jobId=X which IS the unique identifier, so
            # don't strip the query string here.
            url_key     = job.link.rstrip("/").lower()
            pair_key    = (job.company.lower().strip(), job.role.lower().strip())
            company_key = job.company.lower().strip()

            if url_key in seen_urls:
                logger.debug(f"[Dedup] duplicate URL — {job.role} @ {job.company}")
                continue
            if pair_key in seen_pairs:
                logger.debug(f"[Dedup] duplicate (company, role) — {job.role} @ {job.company}")
                continue
            if self._apply_once and company_key in seen_companies:
                logger.debug(f"[Dedup] apply_once_at_company — skipping {job.company}")
                continue

            seen_urls.add(url_key)
            seen_pairs.add(pair_key)
            seen_companies.add(company_key)
            unique.append(job)

        removed = len(jobs) - len(unique)
        if removed:
            logger.info(f"[Dedup] Removed {removed} duplicates — {len(unique)} unique jobs remain")
        return unique
