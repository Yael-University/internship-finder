import os
import re
import ssl
import threading
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from src.logging import logger

try:
    from selenium_stealth import stealth as _selenium_stealth
    _STEALTH_AVAILABLE = True
except ImportError:
    _STEALTH_AVAILABLE = False

# undetected-chromedriver patches the ChromeDriver binary in a shared path.
# Parallel threads must not initialize uc simultaneously or they race on the file.
_UC_INIT_LOCK = threading.Lock()

def chrome_browser_options():
    logger.debug("Setting Chrome browser options")
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-gpu")
    options.add_argument("window-size=1200x800")
    options.add_argument("--disable-background-timer-throttling")
    options.add_argument("--disable-backgrounding-occluded-windows")
    options.add_argument("--disable-translate")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")
    options.add_argument("--disable-logging")
    options.add_argument("--disable-autofill")
    options.add_argument("--disable-plugins")
    options.add_argument("--disable-animations")
    options.add_argument("--disable-cache")
    options.add_argument("--incognito")
    logger.debug("Using Chrome in incognito mode")

    return options

def init_browser_handshake(profile_dir: str | None = None) -> webdriver.Chrome:
    """
    Standard (non-uc) Chrome for Handshake with only automation-removal flags.

    Uses standard webdriver_manager Chrome — NOT undetected-chromedriver — so it
    does not share a patched binary with the LinkedIn uc instance.  Cloudflare is
    bypassed via cf_clearance cookie injection before navigation, so uc patching
    is not needed here.
    """
    try:
        options = Options()
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=1200,800")
        # Remove navigator.webdriver at the Blink engine level.
        options.add_argument("--disable-blink-features=AutomationControlled")
        # Strip the ChromeDriver automation banner and automation extension.
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        if profile_dir:
            options.add_argument(f"--user-data-dir={profile_dir}")
        driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()), options=options
        )
        # Inject webdriver removal before ANY page JS runs.
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        })
        if _STEALTH_AVAILABLE:
            _selenium_stealth(
                driver,
                languages=["en-US", "en"],
                vendor="Google Inc.",
                platform="MacIntel",
                webgl_vendor="Apple Inc.",
                renderer="Apple M2",
                fix_hairline=True,
            )
            logger.debug("Handshake Chrome: selenium-stealth patches applied")
        logger.debug(f"Handshake Chrome initialized (profile_dir={profile_dir!r})")
        return driver
    except Exception as e:
        logger.error(f"Failed to initialize Handshake browser: {str(e)}")
        raise RuntimeError(f"Failed to initialize Handshake browser: {str(e)}")


def init_browser() -> webdriver.Chrome:
    try:
        options = chrome_browser_options()
        # Use webdriver_manager to handle ChromeDriver
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        logger.debug("Chrome browser initialized successfully.")
        return driver
    except Exception as e:
        logger.error(f"Failed to initialize browser: {str(e)}")
        raise RuntimeError(f"Failed to initialize browser: {str(e)}")


def _chrome_major_version() -> int | None:
    """Return the installed Chrome major version, or None if undetectable."""
    import subprocess
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "google-chrome",
        "chromium",
        "chromium-browser",
    ]
    for candidate in candidates:
        try:
            out = subprocess.check_output(
                [candidate, "--version"], stderr=subprocess.DEVNULL, text=True
            )
            m = re.search(r"(\d+)\.", out)
            if m:
                return int(m.group(1))
        except Exception:
            continue
    return None


def init_browser_stealth(profile_dir: str | None = None) -> webdriver.Chrome:
    """
    Undetected-chromedriver instance for LinkedIn.

    Patches the ChromeDriver binary to remove automation fingerprints that
    LinkedIn's bot detection looks for. When profile_dir is given, Chrome
    reuses the same profile across runs so the browser fingerprint stays
    consistent (harder to flag as a bot than a fresh fingerprint every time).
    Falls back to the standard browser if undetected-chromedriver is not installed.
    """
    try:
        import undetected_chromedriver as uc
        # macOS Python ships without system CA certs in its SSL bundle.
        # Temporarily use certifi so uc can verify the ChromeDriver download.
        try:
            import certifi
            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
            os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        except ImportError:
            # certifi not installed — disable SSL verification for the download only
            _orig_ctx = ssl._create_default_https_context
            ssl._create_default_https_context = ssl._create_unverified_context
        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1200,800")
        if profile_dir:
            options.add_argument(f"--user-data-dir={profile_dir}")
        version_main = _chrome_major_version()
        if version_main:
            logger.debug(f"Stealth Chrome: detected Chrome {version_main}")
        # uc patches the ChromeDriver binary in a shared path — serialise across threads.
        with _UC_INIT_LOCK:
            driver = uc.Chrome(options=options, version_main=version_main)
        logger.debug(f"Stealth Chrome initialized (profile_dir={profile_dir!r})")
        return driver
    except ImportError:
        logger.warning(
            "undetected-chromedriver not installed — falling back to standard Chrome. "
            "Run: pip install undetected-chromedriver"
        )
        return init_browser()
    except Exception as e:
        logger.error(f"Failed to initialize stealth browser: {e}")
        raise RuntimeError(f"Failed to initialize stealth browser: {e}")
