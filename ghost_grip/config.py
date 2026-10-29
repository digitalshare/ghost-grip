"""Environment, model ids and a polite retry helper shared by every partner client."""
from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Callable, TypeVar

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
# CrewAI otherwise asks an interactive first-run question about traces, which blocks a server or piped run.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

YDC_API_KEY = os.getenv("YDC_API_KEY", "")
PARASAIL_API_KEY = os.getenv("PARASAIL_API_KEY", "")
PARASAIL_BASE_URL = "https://api.parasail.io/v1"
PARASAIL_MODEL = os.getenv("PARASAIL_MODEL", "parasail-deepseek-v41-flash")
PARASAIL_THINKING = os.getenv("PARASAIL_THINKING", "off").lower() in ("on", "1", "true")
PARASAIL_EMBED_MODEL = os.getenv("PARASAIL_EMBED_MODEL", "parasail-bge-m3")
ONE_SECRET = os.getenv("ONE_SECRET", "")
_BREW_NODE24 = "/opt/homebrew/opt/node@24/bin"
ONE_NODE_DIR = os.getenv("ONE_NODE_DIR") or (_BREW_NODE24 if os.path.isdir(_BREW_NODE24) else "")
CALL_SHEET_TO = os.getenv("CALL_SHEET_TO", "")
SLACK_CHANNEL = os.getenv("SLACK_CHANNEL", "")
DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID", "")
TEAM_HANDLE = os.getenv("TEAM_HANDLE", "team-ghost-grip")

CACHE_DIR = ROOT / ".cache"
RUNS_DIR = ROOT / "runs"
MAX_REVISIONS = 4

T = TypeVar("T")


class RetryableError(Exception):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


def with_backoff(fn: Callable[[], T], attempts: int = 4, base: float = 1.5) -> T:
    """Retry on 429/5xx-style failures, honouring Retry-After when the partner sends it."""
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as e:
            if attempt == attempts - 1:
                raise
            wait = e.retry_after if e.retry_after is not None else base * 2**attempt + random.random()
            time.sleep(min(wait, 30))
    raise RuntimeError("unreachable")
