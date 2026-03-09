"""
config.py – Centralised configuration for BalapBoY v2.0.

All secrets are loaded from environment variables (or a .env file via
python-dotenv).  Nothing is hard-coded here; update .env.example and your
real .env file to change values.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# ── Supabase ──────────────────────────────────────────────────────────────────
SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "")

# ── Pollinations ──────────────────────────────────────────────────────────────
POLLINATIONS_TEXT_URL: str = os.environ.get(
    "POLLINATIONS_TEXT_URL", "https://text.pollinations.ai/openai"
)
POLLINATIONS_IMAGE_URL: str = os.environ.get(
    "POLLINATIONS_IMAGE_URL", "https://image.pollinations.ai/prompt"
)

# ── Model defaults ────────────────────────────────────────────────────────────
DEFAULT_TEXT_MODEL: str = os.environ.get("DEFAULT_TEXT_MODEL", "openai")
DEFAULT_IMAGE_MODEL: str = os.environ.get("DEFAULT_IMAGE_MODEL", "flux")

# ── Pollen economy ────────────────────────────────────────────────────────────
# Daily free grant per user (number of text requests)
FREE_DAILY_GRANT: int = 10

# Cost per action (in "Pollen" credits)
POLLEN_COST_TEXT: float = 0.01       # per request
POLLEN_COST_IMAGE: float = 0.05      # per image
POLLEN_COST_VIDEO: float = 0.05      # per second of video

# Models that are ONLY available to premium / BYOP users
PAID_ONLY_MODELS: set = {"veo", "gpt-4o", "claude-large"}

# ── Context window ────────────────────────────────────────────────────────────
# Number of past messages kept in short-term memory per user
CONTEXT_WINDOW: int = 5

# ── Audio (TTS / STT) ─────────────────────────────────────────────────────────
POLLINATIONS_AUDIO_URL: str = os.environ.get(
    "POLLINATIONS_AUDIO_URL", "https://audio.pollinations.ai"
)

# ── Admin ─────────────────────────────────────────────────────────────────────
# Comma-separated list of Telegram user IDs that may run admin commands
ADMIN_USER_IDS: set = {
    int(x)
    for x in os.environ.get("ADMIN_USER_IDS", "").split(",")
    if x.strip().isdigit()
}
