"""
database/crud.py – Supabase CRUD helpers for BalapBoY v2.0.

All database interactions go through this module so that the rest of the
codebase never imports `supabase` directly.

Expected Supabase table schema
──────────────────────────────
Table: users
  id            bigint (primary key, Telegram user id)
  username      text
  is_premium    boolean  default false
  pollen_balance float    default 0
  daily_used    int      default 0
  daily_reset   date     (date of last reset, UTC)
  byop_key      text     nullable  (user-supplied Pollinations API key)
  created_at    timestamptz default now()
"""

from __future__ import annotations

import json
import logging
from datetime import date, timezone, datetime
from typing import Optional

from supabase import create_client, Client

from config import SUPABASE_URL, SUPABASE_KEY, FREE_DAILY_GRANT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supabase client (lazy singleton so tests can monkey-patch before import)
# ---------------------------------------------------------------------------

_client: Optional[Client] = None


def get_client() -> Client:
    """Return (and lazily create) the shared Supabase client."""
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_KEY must be set in the environment."
            )
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _today_utc() -> str:
    """Return today's date in ISO format (UTC)."""
    return date.today().isoformat()


def _ensure_user(user_id: int, username: str = "") -> dict:
    """Upsert a user row and return the current record."""
    client = get_client()
    today = _today_utc()

    resp = (
        client.table("users")
        .upsert(
            {
                "id": user_id,
                "username": username,
                "daily_reset": today,
            },
            on_conflict="id",
            ignore_duplicates=True,
        )
        .execute()
    )

    # Fetch the definitive row after upsert
    row = (
        client.table("users")
        .select("*")
        .eq("id", user_id)
        .single()
        .execute()
    )
    return row.data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_user(user_id: int, username: str = "") -> dict:
    """
    Fetch (or create) a user record.

    Returns a dict with at least these keys:
        id, username, is_premium, pollen_balance,
        daily_used, daily_reset, byop_key
    """
    return _ensure_user(user_id, username)


def get_user_byop_key(user_id: int) -> Optional[str]:
    """Return the user's BYOP (Bring Your Own Pollen) key, or None."""
    client = get_client()
    row = (
        client.table("users")
        .select("byop_key")
        .eq("id", user_id)
        .single()
        .execute()
    )
    return row.data.get("byop_key") if row.data else None


def is_premium(user_id: int) -> bool:
    """Return True if the user has a premium subscription."""
    client = get_client()
    row = (
        client.table("users")
        .select("is_premium")
        .eq("id", user_id)
        .single()
        .execute()
    )
    return bool(row.data.get("is_premium")) if row.data else False


def check_daily_grant(user_id: int) -> bool:
    """
    Return True if the user still has free daily grant remaining.

    Automatically resets `daily_used` to 0 if the stored `daily_reset`
    date is before today (UTC).
    """
    client = get_client()
    today = _today_utc()

    row = (
        client.table("users")
        .select("daily_used, daily_reset")
        .eq("id", user_id)
        .single()
        .execute()
    )
    if not row.data:
        return False

    # Reset counter if it's a new day
    if row.data.get("daily_reset") != today:
        client.table("users").update(
            {"daily_used": 0, "daily_reset": today}
        ).eq("id", user_id).execute()
        row.data["daily_used"] = 0

    return int(row.data.get("daily_used", 0)) < FREE_DAILY_GRANT


def increment_daily_usage(user_id: int) -> None:
    """Increment the daily_used counter for a user by 1."""
    client = get_client()
    today = _today_utc()

    row = (
        client.table("users")
        .select("daily_used, daily_reset")
        .eq("id", user_id)
        .single()
        .execute()
    )
    if not row.data:
        return

    current = int(row.data.get("daily_used", 0))
    reset_day = row.data.get("daily_reset")

    if reset_day != today:
        # New day – reset first
        client.table("users").update(
            {"daily_used": 1, "daily_reset": today}
        ).eq("id", user_id).execute()
    else:
        client.table("users").update(
            {"daily_used": current + 1}
        ).eq("id", user_id).execute()


def deduct_pollen(user_id: int, amount: float) -> bool:
    """
    Deduct *amount* Pollen from the user's balance.

    Returns True on success, False if the balance is insufficient.
    """
    client = get_client()

    row = (
        client.table("users")
        .select("pollen_balance")
        .eq("id", user_id)
        .single()
        .execute()
    )
    if not row.data:
        return False

    balance = float(row.data.get("pollen_balance", 0))
    if balance < amount:
        return False

    client.table("users").update(
        {"pollen_balance": round(balance - amount, 6)}
    ).eq("id", user_id).execute()
    return True


def add_pollen(user_id: int, amount: float) -> None:
    """Add *amount* Pollen credits to the user's balance (top-up)."""
    client = get_client()

    row = (
        client.table("users")
        .select("pollen_balance")
        .eq("id", user_id)
        .single()
        .execute()
    )
    if not row.data:
        return

    balance = float(row.data.get("pollen_balance", 0))
    client.table("users").update(
        {"pollen_balance": round(balance + amount, 6)}
    ).eq("id", user_id).execute()


def get_pollen_balance(user_id: int) -> float:
    """Return the current Pollen balance for *user_id*."""
    client = get_client()

    row = (
        client.table("users")
        .select("pollen_balance")
        .eq("id", user_id)
        .single()
        .execute()
    )
    return float(row.data.get("pollen_balance", 0)) if row.data else 0.0


def save_byop_key(user_id: int, api_key: str) -> None:
    """Persist a BYOP API key for *user_id*."""
    client = get_client()
    client.table("users").update({"byop_key": api_key}).eq(
        "id", user_id
    ).execute()


def save_context(user_id: int, messages: list) -> None:
    """
    Persist the short-term conversation context (list of message dicts)
    for *user_id*.

    The list is stored as JSON in the ``context`` column.
    """
    client = get_client()
    client.table("users").update(
        {"context": json.dumps(messages)}
    ).eq("id", user_id).execute()


def load_context(user_id: int) -> list:
    """
    Load the short-term conversation context for *user_id*.

    Returns an empty list if no context has been saved yet.
    """
    client = get_client()
    row = (
        client.table("users")
        .select("context")
        .eq("id", user_id)
        .single()
        .execute()
    )
    if not row.data or not row.data.get("context"):
        return []
    try:
        return json.loads(row.data["context"])
    except (json.JSONDecodeError, TypeError):
        return []
