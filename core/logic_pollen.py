"""
core/logic_pollen.py – Pollen economy decision layer for BalapBoY v2.0.

This module sits between the Agentic loop (pollinations.py) and the database
layer (database/crud.py).  It answers a single, critical question before any
AI action is executed:

    "Is this user allowed to run *this* action right now?"

Decision matrix
───────────────
Action type   | Free user w/ daily grant | Free user, grant exhausted | Premium / BYOP
──────────────┼──────────────────────────┼────────────────────────────┼───────────────
text (free)   |  ✅ allow, deduct grant   | ❌ reject, suggest top-up  | ✅ allow
image (free)  |  ✅ allow, deduct grant   | ❌ reject, suggest top-up  | ✅ allow
video (paid)  |  ❌ always reject         | ❌ always reject            | ✅ allow
any (BYOP)    |  ✅ allow (user's own key)|  ✅ allow (user's own key)  | ✅ allow
"""

from __future__ import annotations

import logging
from typing import Tuple

from config import (
    FREE_DAILY_GRANT,
    POLLEN_COST_TEXT,
    POLLEN_COST_IMAGE,
    POLLEN_COST_VIDEO,
    POLLEN_COST_AUDIO,
    PAID_ONLY_MODELS,
)
from database import crud

logger = logging.getLogger(__name__)

# ── Action type constants (mirror the [CALL_TOOL] tags) ──────────────────────
ACTION_TEXT = "text"
ACTION_IMAGE = "image_gen"
ACTION_VISION = "vision"
ACTION_VIDEO = "video_gen"
ACTION_TTS = "audio_tts"
ACTION_STT = "audio_stt"

# Cost per action
_COST_MAP: dict[str, float] = {
    ACTION_TEXT: POLLEN_COST_TEXT,
    ACTION_IMAGE: POLLEN_COST_IMAGE,
    ACTION_VISION: POLLEN_COST_TEXT,
    ACTION_VIDEO: POLLEN_COST_VIDEO,
    ACTION_TTS: POLLEN_COST_AUDIO,
    ACTION_STT: POLLEN_COST_AUDIO,
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def can_use_model(user_id: int, model: str) -> Tuple[bool, str]:
    """
    Check whether *user_id* may use *model*.

    Returns ``(True, "")`` if allowed, or ``(False, reason_message)`` if not.
    """
    if model not in PAID_ONLY_MODELS:
        return True, ""

    # Paid-only model: must be premium or have a BYOP key
    if crud.is_premium(user_id):
        return True, ""

    byop_key = crud.get_user_byop_key(user_id)
    if byop_key:
        return True, ""

    return (
        False,
        (
            f"⚠️ Model *{model}* hanya tersedia untuk pengguna Premium atau "
            "yang sudah daftar API Key sendiri.\n\n"
            "Gunakan */login* untuk menghubungkan API Key kamu, atau top-up "
            "Pollen dulu ya! 🌸"
        ),
    )


def check_and_charge(
    user_id: int,
    action: str,
    model: str = "",
    units: float = 1.0,
) -> Tuple[bool, str]:
    """
    Verify that *user_id* can run *action* and deduct the cost.

    Parameters
    ──────────
    user_id : Telegram user id
    action  : One of the ACTION_* constants defined in this module
    model   : Model name – checked against PAID_ONLY_MODELS
    units   : Multiplier (e.g. seconds of video, number of images)

    Returns ``(True, "")`` on success, or ``(False, rejection_message)``.
    """
    # 1. Model eligibility check
    if model:
        allowed, reason = can_use_model(user_id, model)
        if not allowed:
            return False, reason

    cost = _COST_MAP.get(action, POLLEN_COST_TEXT) * units

    # 2. BYOP key holders always bypass the daily / balance gate
    byop_key = crud.get_user_byop_key(user_id)
    if byop_key:
        return True, ""

    # 3. Premium users deduct from their Pollen balance
    if crud.is_premium(user_id):
        success = crud.deduct_pollen(user_id, cost)
        if not success:
            balance = crud.get_pollen_balance(user_id)
            return (
                False,
                (
                    f"💸 Pollen kamu tidak cukup untuk aksi ini "
                    f"(butuh {cost:.3f}, sisa {balance:.3f}).\n"
                    "Silakan top-up Pollen dulu ya! 🌸"
                ),
            )
        return True, ""

    # 4. Free users: check daily grant for non-video actions
    if action == ACTION_VIDEO:
        return (
            False,
            (
                "🎬 Pembuatan video (*veo*) adalah fitur Premium.\n"
                "Gunakan */login* untuk menghubungkan API Key kamu, atau "
                "top-up Pollen! 🌸"
            ),
        )

    if not crud.check_daily_grant(user_id):
        return (
            False,
            (
                f"📭 Kamu sudah menggunakan semua {FREE_DAILY_GRANT} request "
                "gratis hari ini.\nCoba lagi besok, atau top-up Pollen untuk "
                "akses tak terbatas! 🌸"
            ),
        )

    crud.increment_daily_usage(user_id)
    return True, ""
