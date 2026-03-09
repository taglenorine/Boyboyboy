"""
tests/test_pollinations.py – Unit tests for core/pollinations.py.

These tests exercise the parts of the agentic core that do NOT require live
network connections: the regex interceptor, parameter parser, tag stripper,
and the high-level ``run_agent`` function (with all I/O mocked).
"""

from __future__ import annotations

import asyncio
import httpx
import re
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.pollinations import (
    AgentResponse,
    MediaResult,
    _parse_params,
    _strip_internal_tags,
    _TOOL_PATTERN,
    _fetch_tts,
    _transcribe_audio,
    run_agent,
)


# ── _parse_params ─────────────────────────────────────────────────────────────

class TestParseParams:
    def test_single_param(self):
        result = _parse_params("prompt: a red dragon")
        assert result == {"prompt": "a red dragon"}

    def test_multiple_params(self):
        result = _parse_params("prompt: a red dragon | model: flux")
        assert result == {"prompt": "a red dragon", "model": "flux"}

    def test_empty_string(self):
        assert _parse_params("") == {}

    def test_no_colon(self):
        # Parts without a colon are silently ignored
        result = _parse_params("no-colon-here")
        assert result == {}

    def test_extra_whitespace(self):
        result = _parse_params("  prompt :   big sky  |  model :  openai  ")
        assert result["prompt"] == "big sky"
        assert result["model"] == "openai"


# ── _TOOL_PATTERN regex ───────────────────────────────────────────────────────

class TestToolPattern:
    def test_matches_image_gen(self):
        text = "[CALL_TOOL: image_gen | prompt: naga merah | model: flux]"
        match = _TOOL_PATTERN.search(text)
        assert match is not None
        assert match.group("tool").lower() == "image_gen"

    def test_matches_text(self):
        text = "[CALL_TOOL: text | prompt: harga Bitcoin | model: gemini-search]"
        match = _TOOL_PATTERN.search(text)
        assert match is not None
        assert match.group("tool").lower() == "text"

    def test_case_insensitive(self):
        text = "[CALL_TOOL: IMAGE_GEN | prompt: test]"
        match = _TOOL_PATTERN.search(text)
        assert match is not None

    def test_no_params(self):
        text = "[CALL_TOOL: vision]"
        match = _TOOL_PATTERN.search(text)
        assert match is not None
        assert match.group("params") is None

    def test_no_match(self):
        text = "Ini adalah balasan biasa tanpa directive."
        assert _TOOL_PATTERN.search(text) is None

    def test_multiple_directives(self):
        text = (
            "Here is a plan:\n"
            "[CALL_TOOL: text | prompt: describe kafe]\n"
            "[CALL_TOOL: image_gen | prompt: kafe futuristik | model: flux]\n"
            "[CALL_TOOL: image_gen | prompt: kafe cyberpunk | model: flux]"
        )
        matches = list(_TOOL_PATTERN.finditer(text))
        assert len(matches) == 3


# ── _strip_internal_tags ──────────────────────────────────────────────────────

class TestStripInternalTags:
    def test_removes_call_tool(self):
        text = "Oke, gue akan bikin gambar.\n[CALL_TOOL: image_gen | prompt: cat]"
        result = _strip_internal_tags(text)
        assert "[CALL_TOOL" not in result
        assert "Oke, gue akan bikin gambar." in result

    def test_removes_thought_tags(self):
        text = (
            "[THOUGHT]: User minta gambar naga.\n"
            "[ACTION]: Panggil image_gen.\n"
            "Ini adalah jawaban akhir."
        )
        result = _strip_internal_tags(text)
        assert "[THOUGHT]" not in result
        assert "[ACTION]" not in result
        assert "Ini adalah jawaban akhir." in result

    def test_removes_state_check(self):
        text = "[STATE CHECK]: Free user, no premium.\nSilakan top-up."
        result = _strip_internal_tags(text)
        assert "[STATE CHECK]" not in result

    def test_plain_text_unchanged(self):
        text = "Halo! Gue siap bantu kamu."
        assert _strip_internal_tags(text) == text


# ── run_agent (integration-style with mocks) ──────────────────────────────────

class TestRunAgent:
    """
    Test ``run_agent`` by mocking:
    - ``crud.get_user``
    - ``crud.get_user_byop_key``
    - ``crud.load_context``
    - ``crud.save_context``
    - ``_call_text_api``
    - ``_fetch_image``  (where needed)
    """

    def _patch_crud(self):
        """Return a context-manager dict of crud patches."""
        return {
            "get_user": patch("core.pollinations.crud.get_user", return_value={}),
            "get_user_byop_key": patch(
                "core.pollinations.crud.get_user_byop_key", return_value=None
            ),
            "load_context": patch(
                "core.pollinations.crud.load_context", return_value=[]
            ),
            "save_context": patch("core.pollinations.crud.save_context"),
        }

    @pytest.mark.asyncio
    async def test_plain_text_reply(self):
        patches = self._patch_crud()
        with (
            patches["get_user"],
            patches["get_user_byop_key"],
            patches["load_context"],
            patches["save_context"],
            patch(
                "core.pollinations._call_text_api",
                new=AsyncMock(return_value="Halo, ini jawaban AI!"),
            ),
            patch(
                "core.pollinations.check_and_charge",
                return_value=(True, ""),
            ),
        ):
            response = await run_agent(
                user_id=12345,
                user_message="Apa kabar?",
                username="testuser",
            )

        assert isinstance(response, AgentResponse)
        assert response.error is False
        assert "Halo" in response.text
        assert response.media == []

    @pytest.mark.asyncio
    async def test_image_gen_directive_triggers_fetch(self):
        """When the LLM emits an image_gen directive, _fetch_image is called."""
        fake_image_bytes = b"\x89PNG\r\n\x1a\n"

        # First LLM call returns a directive; second returns final text
        llm_responses = [
            "[CALL_TOOL: image_gen | prompt: naga merah | model: flux]",
            "Nih gambar naga merahnya sudah jadi! 🐉",
        ]
        call_count = {"n": 0}

        async def mock_llm(*args, **kwargs):
            resp = llm_responses[call_count["n"]]
            call_count["n"] += 1
            return resp

        patches = self._patch_crud()
        with (
            patches["get_user"],
            patches["get_user_byop_key"],
            patches["load_context"],
            patches["save_context"],
            patch("core.pollinations._call_text_api", new=mock_llm),
            patch(
                "core.pollinations._fetch_image",
                new=AsyncMock(return_value=fake_image_bytes),
            ),
            patch(
                "core.pollinations.check_and_charge",
                return_value=(True, ""),
            ),
        ):
            response = await run_agent(
                user_id=12345,
                user_message="Buatin gambar naga merah",
                username="testuser",
            )

        assert response.error is False
        assert len(response.media) == 1
        assert response.media[0].kind == "image"
        assert response.media[0].data == fake_image_bytes

    @pytest.mark.asyncio
    async def test_pollen_rejection_propagated(self):
        """When check_and_charge returns False, the rejection message appears."""

        async def mock_llm(*args, **kwargs):
            return "[CALL_TOOL: video_gen | prompt: naga terbang | model: veo]"

        async def mock_llm_2(*args, **kwargs):
            # After the observation is fed back, the agent produces a final reply
            return "Maaf Bos, video tidak bisa dibuat."

        call_count = {"n": 0}

        async def mock_llm_seq(*args, **kwargs):
            responses = [
                "[CALL_TOOL: video_gen | prompt: naga terbang | model: veo]",
                "Maaf Bos, video tidak bisa dibuat.",
            ]
            resp = responses[call_count["n"]]
            call_count["n"] += 1
            return resp

        patches = self._patch_crud()
        with (
            patches["get_user"],
            patches["get_user_byop_key"],
            patches["load_context"],
            patches["save_context"],
            patch("core.pollinations._call_text_api", new=mock_llm_seq),
            patch(
                "core.pollinations.check_and_charge",
                return_value=(False, "⚠️ Fitur Premium saja"),
            ),
        ):
            response = await run_agent(
                user_id=12345,
                user_message="Buatin video naga terbang",
                username="testuser",
            )

        assert response.error is False
        # The observation (rejection message) was fed back; final answer should
        # be the second LLM response
        assert "Maaf" in response.text

    @pytest.mark.asyncio
    async def test_llm_error_triggers_fallback(self):
        """An HTTP error on the primary model triggers a fallback model call."""
        call_count = {"n": 0}

        async def mock_llm_failing(*args, **kwargs):
            if call_count["n"] == 0:
                call_count["n"] += 1
                # Simulate a 429 from the primary model
                raise httpx.HTTPStatusError(
                    "Too Many Requests",
                    request=MagicMock(),
                    response=MagicMock(status_code=429),
                )
            return "Fallback berhasil!"

        patches = self._patch_crud()
        with (
            patches["get_user"],
            patches["get_user_byop_key"],
            patches["load_context"],
            patches["save_context"],
            patch("core.pollinations._call_text_api", new=mock_llm_failing),
            patch(
                "core.pollinations.check_and_charge",
                return_value=(True, ""),
            ),
        ):
            response = await run_agent(
                user_id=12345,
                user_message="Tes fallback",
                username="testuser",
            )

        assert "Fallback berhasil!" in response.text
        assert response.error is False


# ── _fetch_tts ────────────────────────────────────────────────────────────────

class TestFetchTts:
    @pytest.mark.asyncio
    async def test_fetch_tts_returns_bytes(self):
        """_fetch_tts should return the audio bytes from the HTTP response."""
        fake_audio = b"\xff\xfb\x90\x00"  # minimal MP3 header bytes
        mock_resp = MagicMock()
        mock_resp.content = fake_audio
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("core.pollinations.httpx.AsyncClient", return_value=mock_client):
            result = await _fetch_tts("Halo dunia", voice="alloy")

        assert result == fake_audio

    @pytest.mark.asyncio
    async def test_fetch_tts_includes_voice_param(self):
        """The voice parameter should appear in the constructed URL."""
        mock_resp = MagicMock()
        mock_resp.content = b"audio"
        mock_resp.raise_for_status = MagicMock()

        captured_url: list[str] = []

        async def mock_get(url, **kwargs):
            captured_url.append(url)
            return mock_resp

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = mock_get

        with patch("core.pollinations.httpx.AsyncClient", return_value=mock_client):
            await _fetch_tts("test text", voice="nova")

        assert "voice=nova" in captured_url[0]

    @pytest.mark.asyncio
    async def test_fetch_tts_truncates_long_text(self):
        """Text longer than 500 characters should be truncated before URL encoding."""
        long_text = "A" * 600
        mock_resp = MagicMock()
        mock_resp.content = b"audio"
        mock_resp.raise_for_status = MagicMock()

        captured_url: list[str] = []

        async def mock_get(url, **kwargs):
            captured_url.append(url)
            return mock_resp

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = mock_get

        with patch("core.pollinations.httpx.AsyncClient", return_value=mock_client):
            await _fetch_tts(long_text)

        # The URL should contain at most 500 "A"s (not 600)
        assert "A" * 600 not in captured_url[0]
        assert "A" * 500 in captured_url[0]


# ── _transcribe_audio ─────────────────────────────────────────────────────────

class TestTranscribeAudio:
    @pytest.mark.asyncio
    async def test_transcribe_audio_calls_text_api(self):
        """_transcribe_audio should delegate to _call_text_api and return its result."""
        expected_text = "Halo, ini transkripsi."
        with patch(
            "core.pollinations._call_text_api",
            new=AsyncMock(return_value=expected_text),
        ) as mock_call:
            result = await _transcribe_audio(b"\x00\x01\x02", byop_key="")

        assert result == expected_text
        mock_call.assert_called_once()
        # Verify the model used for STT
        _, model_arg, _ = mock_call.call_args.args
        assert model_arg == "openai-audio"


# ── TTS tool execution ────────────────────────────────────────────────────────

class TestTtsTool:
    def _patch_crud(self):
        return {
            "get_user": patch("core.pollinations.crud.get_user", return_value={}),
            "get_user_byop_key": patch(
                "core.pollinations.crud.get_user_byop_key", return_value=None
            ),
            "load_context": patch(
                "core.pollinations.crud.load_context", return_value=[]
            ),
            "save_context": patch("core.pollinations.crud.save_context"),
        }

    @pytest.mark.asyncio
    async def test_tts_directive_produces_audio_media(self):
        """A TTS directive in the LLM response should result in an audio MediaResult."""
        fake_audio_bytes = b"\xff\xfb\x90\x00"
        llm_responses = [
            "[CALL_TOOL: audio_tts | text: Halo dunia | voice: alloy]",
            "Audio sudah siap! 🔊",
        ]
        call_count = {"n": 0}

        async def mock_llm(*args, **kwargs):
            resp = llm_responses[call_count["n"]]
            call_count["n"] += 1
            return resp

        patches = self._patch_crud()
        with (
            patches["get_user"],
            patches["get_user_byop_key"],
            patches["load_context"],
            patches["save_context"],
            patch("core.pollinations._call_text_api", new=mock_llm),
            patch(
                "core.pollinations._fetch_tts",
                new=AsyncMock(return_value=fake_audio_bytes),
            ),
            patch(
                "core.pollinations.check_and_charge",
                return_value=(True, ""),
            ),
        ):
            response = await run_agent(
                user_id=99,
                user_message="Bacakan teks ini",
                username="tester",
            )

        assert response.error is False
        assert len(response.media) == 1
        assert response.media[0].kind == "audio"
        assert response.media[0].data == fake_audio_bytes

    @pytest.mark.asyncio
    async def test_tts_directive_no_text_param(self):
        """A TTS directive with no 'text' param should return an error observation."""
        llm_responses = [
            "[CALL_TOOL: audio_tts]",
            "Maaf, teks tidak ditemukan.",
        ]
        call_count = {"n": 0}

        async def mock_llm(*args, **kwargs):
            resp = llm_responses[call_count["n"]]
            call_count["n"] += 1
            return resp

        patches = self._patch_crud()
        with (
            patches["get_user"],
            patches["get_user_byop_key"],
            patches["load_context"],
            patches["save_context"],
            patch("core.pollinations._call_text_api", new=mock_llm),
            patch(
                "core.pollinations.check_and_charge",
                return_value=(True, ""),
            ),
        ):
            response = await run_agent(
                user_id=99,
                user_message="Bacakan teks",
                username="tester",
            )

        # No media should be produced when text param is missing
        assert response.media == []


# ── Vision tool execution ─────────────────────────────────────────────────────

class TestVisionTool:
    def _patch_crud(self):
        return {
            "get_user": patch("core.pollinations.crud.get_user", return_value={}),
            "get_user_byop_key": patch(
                "core.pollinations.crud.get_user_byop_key", return_value=None
            ),
            "load_context": patch(
                "core.pollinations.crud.load_context", return_value=[]
            ),
            "save_context": patch("core.pollinations.crud.save_context"),
        }

    @pytest.mark.asyncio
    async def test_vision_directive_downloads_and_analyses(self):
        """A vision directive with a URL should download the image and call the text API."""
        fake_img = b"\x89PNG\r\n\x1a\n"
        analysis_text = "Gambar ini menampilkan kucing berwarna oranye."

        # Call sequence:
        # 0 → agentic loop first call → emits vision directive
        # 1 → _execute_tool/vision calls _call_text_api for the analysis
        # 2 → agentic loop second call (after observation) → final answer
        llm_responses = [
            "[CALL_TOOL: vision | url: http://example.com/cat.jpg | prompt: Apa ini?]",
            analysis_text,
            f"Analisis selesai: {analysis_text}",
        ]
        call_count = {"n": 0}

        async def mock_llm(*args, **kwargs):
            resp = llm_responses[call_count["n"]]
            call_count["n"] += 1
            return resp

        mock_img_resp = MagicMock()
        mock_img_resp.content = fake_img
        mock_img_resp.raise_for_status = MagicMock()

        async def mock_get(url, **kwargs):
            return mock_img_resp

        mock_http_client = AsyncMock()
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)
        mock_http_client.get = mock_get

        patches = self._patch_crud()
        with (
            patches["get_user"],
            patches["get_user_byop_key"],
            patches["load_context"],
            patches["save_context"],
            patch("core.pollinations._call_text_api", new=mock_llm),
            patch(
                "core.pollinations.check_and_charge",
                return_value=(True, ""),
            ),
            patch(
                "core.pollinations.httpx.AsyncClient",
                return_value=mock_http_client,
            ),
        ):
            response = await run_agent(
                user_id=99,
                user_message="Analisis gambar dari URL",
                username="tester",
            )

        assert response.error is False
        assert analysis_text in response.text

    @pytest.mark.asyncio
    async def test_vision_directive_no_url_returns_error_obs(self):
        """A vision directive without a URL should produce an error observation."""
        llm_responses = [
            "[CALL_TOOL: vision | prompt: Apa ini?]",
            "Tidak ada URL gambar yang diberikan.",
        ]
        call_count = {"n": 0}

        async def mock_llm(*args, **kwargs):
            resp = llm_responses[call_count["n"]]
            call_count["n"] += 1
            return resp

        patches = self._patch_crud()
        with (
            patches["get_user"],
            patches["get_user_byop_key"],
            patches["load_context"],
            patches["save_context"],
            patch("core.pollinations._call_text_api", new=mock_llm),
            patch(
                "core.pollinations.check_and_charge",
                return_value=(True, ""),
            ),
        ):
            response = await run_agent(
                user_id=99,
                user_message="Analisis gambar",
                username="tester",
            )

        assert response.error is False
        assert response.media == []


# ── run_agent with audio_bytes (STT path) ─────────────────────────────────────

class TestRunAgentStt:
    def _patch_crud(self):
        return {
            "get_user": patch("core.pollinations.crud.get_user", return_value={}),
            "get_user_byop_key": patch(
                "core.pollinations.crud.get_user_byop_key", return_value=None
            ),
            "load_context": patch(
                "core.pollinations.crud.load_context", return_value=[]
            ),
            "save_context": patch("core.pollinations.crud.save_context"),
        }

    @pytest.mark.asyncio
    async def test_audio_bytes_triggers_transcription(self):
        """When audio_bytes is provided, _transcribe_audio is called first."""
        patches = self._patch_crud()
        with (
            patches["get_user"],
            patches["get_user_byop_key"],
            patches["load_context"],
            patches["save_context"],
            patch(
                "core.pollinations._call_text_api",
                new=AsyncMock(return_value="Jawaban setelah transkripsi"),
            ),
            patch(
                "core.pollinations._transcribe_audio",
                new=AsyncMock(return_value="teks hasil transkripsi"),
            ) as mock_stt,
            patch(
                "core.pollinations.check_and_charge",
                return_value=(True, ""),
            ),
        ):
            response = await run_agent(
                user_id=99,
                user_message="",
                username="tester",
                audio_bytes=b"\x00\x01\x02",
            )

        mock_stt.assert_called_once()
        assert response.error is False

    @pytest.mark.asyncio
    async def test_stt_charge_rejected_returns_early(self):
        """If the STT Pollen check fails, run_agent returns immediately with the reason."""
        patches = self._patch_crud()
        with (
            patches["get_user"],
            patches["get_user_byop_key"],
            patches["load_context"],
            patches["save_context"],
            patch(
                "core.pollinations.check_and_charge",
                return_value=(False, "⚠️ Tidak cukup Pollen untuk STT"),
            ),
        ):
            response = await run_agent(
                user_id=99,
                user_message="",
                username="tester",
                audio_bytes=b"\x00\x01\x02",
            )

        assert response.error is False
        assert "STT" in response.text or "Pollen" in response.text

    @pytest.mark.asyncio
    async def test_stt_failure_falls_back_to_original_message(self):
        """If STT raises an exception, the original user_message is used."""
        patches = self._patch_crud()
        with (
            patches["get_user"],
            patches["get_user_byop_key"],
            patches["load_context"],
            patches["save_context"],
            patch(
                "core.pollinations._call_text_api",
                new=AsyncMock(return_value="Jawaban fallback"),
            ),
            patch(
                "core.pollinations._transcribe_audio",
                side_effect=Exception("STT network error"),
            ),
            patch(
                "core.pollinations.check_and_charge",
                return_value=(True, ""),
            ),
        ):
            response = await run_agent(
                user_id=99,
                user_message="pesan asli",
                username="tester",
                audio_bytes=b"\x00\x01\x02",
            )

        # Should not crash; falls back to the text LLM reply
        assert response.error is False
        assert "Jawaban fallback" in response.text
