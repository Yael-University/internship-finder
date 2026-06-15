import os

from src.utils.constants import ERROR

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_SELENIUM_LEVEL = ERROR
LOG_TO_FILE = False
LOG_TO_CONSOLE = True

JOB_SUITABILITY_SCORE = 7
