"""
tests/test_logic_pollen.py – Unit tests for core/logic_pollen.py.

All Supabase calls are mocked so these tests run without a live database.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from core.logic_pollen import (
    ACTION_IMAGE,
    ACTION_TEXT,
    ACTION_VIDEO,
    can_use_model,
    check_and_charge,
)
from config import PAID_ONLY_MODELS, FREE_DAILY_GRANT


# ── can_use_model ──────────────────────────────────────────────────────────────

class TestCanUseModel:
    def test_free_model_always_allowed(self):
        # "flux" is not in PAID_ONLY_MODELS
        ok, reason = can_use_model(user_id=1, model="flux")
        assert ok is True
        assert reason == ""

    def test_paid_model_blocked_for_free_user(self):
        with (
            patch("core.logic_pollen.crud.is_premium", return_value=False),
            patch("core.logic_pollen.crud.get_user_byop_key", return_value=None),
        ):
            for model in PAID_ONLY_MODELS:
                ok, reason = can_use_model(user_id=1, model=model)
                assert ok is False
                assert "Premium" in reason or "API Key" in reason

    def test_paid_model_allowed_for_premium_user(self):
        with patch("core.logic_pollen.crud.is_premium", return_value=True):
            for model in PAID_ONLY_MODELS:
                ok, reason = can_use_model(user_id=1, model=model)
                assert ok is True

    def test_paid_model_allowed_with_byop_key(self):
        with (
            patch("core.logic_pollen.crud.is_premium", return_value=False),
            patch(
                "core.logic_pollen.crud.get_user_byop_key",
                return_value="sk-test-key",
            ),
        ):
            for model in PAID_ONLY_MODELS:
                ok, reason = can_use_model(user_id=1, model=model)
                assert ok is True


# ── check_and_charge ──────────────────────────────────────────────────────────

class TestCheckAndCharge:
    def _base_patches(
        self,
        *,
        is_premium=False,
        byop_key=None,
        daily_grant=True,
        deduct_ok=True,
    ):
        """Return a dict of patch context managers for common mocks."""
        return {
            "is_premium": patch(
                "core.logic_pollen.crud.is_premium", return_value=is_premium
            ),
            "byop_key": patch(
                "core.logic_pollen.crud.get_user_byop_key",
                return_value=byop_key,
            ),
            "daily_grant": patch(
                "core.logic_pollen.crud.check_daily_grant",
                return_value=daily_grant,
            ),
            "increment": patch(
                "core.logic_pollen.crud.increment_daily_usage"
            ),
            "deduct": patch(
                "core.logic_pollen.crud.deduct_pollen", return_value=deduct_ok
            ),
            "balance": patch(
                "core.logic_pollen.crud.get_pollen_balance", return_value=0.0
            ),
        }

    def test_free_user_text_with_daily_grant(self):
        p = self._base_patches()
        with p["is_premium"], p["byop_key"], p["daily_grant"], p["increment"]:
            ok, reason = check_and_charge(user_id=1, action=ACTION_TEXT)
        assert ok is True
        assert reason == ""

    def test_free_user_text_no_daily_grant(self):
        p = self._base_patches(daily_grant=False)
        with p["is_premium"], p["byop_key"], p["daily_grant"], p["increment"]:
            ok, reason = check_and_charge(user_id=1, action=ACTION_TEXT)
        assert ok is False
        assert "gratis" in reason or "habis" in reason

    def test_free_user_video_always_rejected(self):
        p = self._base_patches()
        with p["is_premium"], p["byop_key"], p["daily_grant"], p["increment"]:
            ok, reason = check_and_charge(user_id=1, action=ACTION_VIDEO)
        assert ok is False
        assert "Premium" in reason

    def test_premium_user_text_deducts_pollen(self):
        p = self._base_patches(is_premium=True, deduct_ok=True)
        with (
            p["is_premium"],
            p["byop_key"],
            p["daily_grant"],
            p["increment"],
            p["deduct"],
        ):
            ok, reason = check_and_charge(user_id=1, action=ACTION_TEXT)
        assert ok is True

    def test_premium_user_insufficient_balance(self):
        p = self._base_patches(is_premium=True, deduct_ok=False)
        with (
            p["is_premium"],
            p["byop_key"],
            p["daily_grant"],
            p["increment"],
            p["deduct"],
            p["balance"],
        ):
            ok, reason = check_and_charge(user_id=1, action=ACTION_TEXT)
        assert ok is False
        assert "Pollen" in reason

    def test_byop_user_bypasses_all_checks(self):
        """A BYOP user should always be allowed regardless of other state."""
        p = self._base_patches(byop_key="sk-mykey", is_premium=False, daily_grant=False)
        with p["is_premium"], p["byop_key"], p["daily_grant"], p["increment"]:
            ok, reason = check_and_charge(user_id=1, action=ACTION_TEXT)
        assert ok is True
        assert reason == ""

    def test_paid_model_blocked_for_free_user(self):
        """A paid-only model should be blocked even for an otherwise valid free request."""
        p = self._base_patches()
        with p["is_premium"], p["byop_key"]:
            ok, reason = check_and_charge(
                user_id=1, action=ACTION_IMAGE, model="veo"
            )
        assert ok is False

    def test_units_multiplier(self):
        """units > 1 should multiply the deducted cost."""
        p = self._base_patches(is_premium=True, deduct_ok=True)
        deduct_mock = MagicMock(return_value=True)
        with (
            p["is_premium"],
            p["byop_key"],
            p["daily_grant"],
            p["increment"],
            patch("core.logic_pollen.crud.deduct_pollen", deduct_mock),
        ):
            check_and_charge(user_id=1, action=ACTION_IMAGE, units=3.0)

        # deduct_pollen should have been called with 3× the IMAGE cost
        from config import POLLEN_COST_IMAGE
        deduct_mock.assert_called_once_with(1, pytest.approx(POLLEN_COST_IMAGE * 3))
