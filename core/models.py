"""
core/models.py – Comprehensive Pollinations.ai model catalog for BalapBoY v2.0.

This module defines a structured catalog of all available Pollinations.ai
models, organised by category (image, video, audio, text).  Each entry
includes:
  - model_id   : The identifier used in API requests.
  - name       : Human-readable display name.
  - paid_only  : Whether the model requires purchased Pollen (True) or
                 is accessible via daily grants (False).
  - pricing    : Dict with cost details (unit, cost).
  - features   : Set of capability flags (vision, reasoning, search,
                 code_exec, audio_in, audio_out).
  - status     : Space-separated status tags, e.g. "NEW", "ALPHA",
                 "NEW ALPHA", or "DEPRECATED".

Pricing units follow the Pollinations.ai conventions:
  "/img"    – flat cost per image
  "/M"      – cost per million tokens (input/output separated)
  "/sec"    – cost per second (video / audio)
  "/1k chars" – cost per 1 000 characters (TTS)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelPricing:
    """Pricing details for a single model."""
    unit: str                        # e.g. "/img", "/M", "/sec", "/1k chars"
    cost: Optional[float] = None     # flat cost (image / video / audio)
    input_cost: Optional[float] = None   # per-million input tokens
    output_cost: Optional[float] = None  # per-million output tokens
    audio_in_cost: Optional[float] = None  # per-million audio input tokens
    audio_out_cost: Optional[float] = None  # per-million audio output tokens
    requests_per_pollen: Optional[int] = None  # approximate requests per 1 pollen


@dataclass
class ModelInfo:
    """Full metadata for a single Pollinations.ai model."""
    model_id: str
    name: str
    category: str           # "image" | "video" | "audio" | "text"
    paid_only: bool = False
    pricing: Optional[ModelPricing] = None
    features: set[str] = field(default_factory=set)
    status: str = ""        # "" | "NEW" | "ALPHA" | "DEPRECATED"


# ---------------------------------------------------------------------------
# Image models
# ---------------------------------------------------------------------------

IMAGE_MODELS: list[ModelInfo] = [
    ModelInfo(
        model_id="flux-schnell",
        name="Flux Schnell",
        category="image",
        paid_only=False,
        pricing=ModelPricing(unit="/img", cost=0.001, requests_per_pollen=3200),
    ),
    ModelInfo(
        model_id="flux",
        name="Flux",
        category="image",
        paid_only=False,
        pricing=ModelPricing(unit="/img", cost=0.002, requests_per_pollen=1600),
    ),
    ModelInfo(
        model_id="zimage",
        name="Z-Image Turbo",
        category="image",
        paid_only=False,
        pricing=ModelPricing(unit="/img", cost=0.002, requests_per_pollen=1600),
    ),
    ModelInfo(
        model_id="flux-2-dev",
        name="FLUX.2 Dev (api.airforce)",
        category="image",
        paid_only=False,
        pricing=ModelPricing(unit="/img", cost=0.001, requests_per_pollen=1000),
        features={"vision"},
        status="NEW ALPHA",
    ),
    ModelInfo(
        model_id="imagen-4",
        name="Imagen 4 (api.airforce)",
        category="image",
        paid_only=False,
        pricing=ModelPricing(unit="/img", cost=0.025, requests_per_pollen=400),
        status="NEW ALPHA",
    ),
    ModelInfo(
        model_id="grok-imagine",
        name="Grok Imagine (api.airforce)",
        category="image",
        paid_only=False,
        pricing=ModelPricing(unit="/img", cost=0.025, requests_per_pollen=400),
        status="NEW ALPHA",
    ),
    ModelInfo(
        model_id="klein",
        name="FLUX.2 Klein 4B",
        category="image",
        paid_only=False,
        pricing=ModelPricing(unit="/img", cost=0.01, requests_per_pollen=100),
    ),
    ModelInfo(
        model_id="gptimage",
        name="GPT Image 1 Mini",
        category="image",
        paid_only=False,
        pricing=ModelPricing(
            unit="/M",
            input_cost=2.0,
            output_cost=8.0,
            requests_per_pollen=80,
        ),
        features={"vision"},
    ),
    ModelInfo(
        model_id="klein-large",
        name="FLUX.2 Klein 9B",
        category="image",
        paid_only=False,
        pricing=ModelPricing(unit="/img", cost=0.015, requests_per_pollen=75),
        features={"vision"},
    ),
    ModelInfo(
        model_id="seeddream",
        name="Seeddream 4.0",
        category="image",
        paid_only=True,
        pricing=ModelPricing(unit="/img", cost=0.03, requests_per_pollen=35),
        features={"vision"},
    ),
    ModelInfo(
        model_id="konttext",
        name="FLUX1 Konttext",
        category="image",
        paid_only=True,
        pricing=ModelPricing(unit="/img", cost=0.04, requests_per_pollen=25),
        features={"vision"},
    ),
    ModelInfo(
        model_id="nanobanana",
        name="NanoBanana",
        category="image",
        paid_only=True,
        pricing=ModelPricing(
            unit="/M",
            input_cost=0.3,
            output_cost=30.0,
            requests_per_pollen=25,
        ),
        features={"vision"},
    ),
    ModelInfo(
        model_id="seeddream-pro",
        name="Seeddream 4.5 Pro",
        category="image",
        paid_only=True,
        pricing=ModelPricing(unit="/img", cost=0.04, requests_per_pollen=25),
    ),
    ModelInfo(
        model_id="nanobanana-2",
        name="NanoBanana 2",
        category="image",
        paid_only=True,
        pricing=ModelPricing(
            unit="/M",
            input_cost=0.6,
            output_cost=60.0,
            requests_per_pollen=15,
        ),
    ),
    ModelInfo(
        model_id="gptimage-large",
        name="GPT Image 1.5",
        category="image",
        paid_only=True,
        pricing=ModelPricing(
            unit="/M",
            input_cost=8.0,
            output_cost=32.0,
            requests_per_pollen=15,
        ),
        features={"vision"},
    ),
    ModelInfo(
        model_id="nanobanana-pro",
        name="NanoBanana Pro",
        category="image",
        paid_only=True,
        pricing=ModelPricing(
            unit="/M",
            input_cost=1.25,
            output_cost=120.0,
            requests_per_pollen=7,
        ),
        features={"vision"},
    ),
    ModelInfo(
        model_id="seeddream5",
        name="Seeddream 5.0 Lite",
        category="image",
        paid_only=True,
        pricing=ModelPricing(unit="/img", cost=0.035),
        features={"vision"},
    ),
]

# ---------------------------------------------------------------------------
# Video models
# ---------------------------------------------------------------------------

VIDEO_MODELS: list[ModelInfo] = [
    ModelInfo(
        model_id="grok-video",
        name="Grok Video (api.airforce)",
        category="video",
        paid_only=False,
        pricing=ModelPricing(unit="/sec", cost=0.003, requests_per_pollen=70),
        status="NEW ALPHA",
    ),
    ModelInfo(
        model_id="ltx-2",
        name="LTX-2",
        category="video",
        paid_only=True,
        pricing=ModelPricing(unit="/sec", cost=0.010, requests_per_pollen=20),
    ),
    ModelInfo(
        model_id="seedance",
        name="Seedance Lite",
        category="video",
        paid_only=True,
        pricing=ModelPricing(unit="/M", cost=1.8, requests_per_pollen=4),
        features={"vision"},
    ),
    ModelInfo(
        model_id="wan",
        name="Wan 2.6",
        category="video",
        paid_only=True,
        pricing=ModelPricing(unit="/sec", cost=0.050, requests_per_pollen=3),
        features={"vision"},
        status="NEW",
    ),
    ModelInfo(
        model_id="veo",
        name="Veo 3.1 Fast",
        category="video",
        paid_only=True,
        pricing=ModelPricing(unit="/sec", cost=0.150, requests_per_pollen=1),
        features={"vision"},
    ),
    ModelInfo(
        model_id="seedance-pro",
        name="Seedance Pro-Fast",
        category="video",
        paid_only=True,
        pricing=ModelPricing(unit="/M", cost=0.10),
    ),
]

# ---------------------------------------------------------------------------
# Audio models
# ---------------------------------------------------------------------------

AUDIO_MODELS: list[ModelInfo] = [
    ModelInfo(
        model_id="whisper",
        name="Whisper Large V3",
        category="audio",
        paid_only=False,
        pricing=ModelPricing(unit="/sec", cost=0.00004, requests_per_pollen=1300),
        features={"audio_in"},
        status="NEW ALPHA",
    ),
    ModelInfo(
        model_id="elevenlabs",
        name="ElevenLabs V TTS",
        category="audio",
        paid_only=False,
        pricing=ModelPricing(unit="/1k chars", cost=0.18, requests_per_pollen=45),
        features={"audio_out"},
        status="NEW",
    ),
    ModelInfo(
        model_id="elevenmusic",
        name="ElevenLabs Music",
        category="audio",
        paid_only=False,
        pricing=ModelPricing(unit="/sec", cost=0.0050, requests_per_pollen=3),
        features={"audio_out"},
        status="NEW",
    ),
    ModelInfo(
        model_id="suno",
        name="Suno V (api.airforce)",
        category="audio",
        paid_only=False,
        pricing=ModelPricing(unit="/sec", cost=0.001),
        features={"audio_out"},
        status="NEW ALPHA",
    ),
    ModelInfo(
        model_id="scribe",
        name="ElevenLabs Scribe V2",
        category="audio",
        paid_only=False,
        pricing=ModelPricing(unit="/sec", cost=0.0011),
        features={"audio_in"},
        status="NEW",
    ),
]

# ---------------------------------------------------------------------------
# Text models
# ---------------------------------------------------------------------------

TEXT_MODELS: list[ModelInfo] = [
    ModelInfo(
        model_id="qwen-safety",
        name="Qwen3Guard 8B",
        category="text",
        paid_only=False,
        pricing=ModelPricing(
            unit="/M",
            input_cost=0.001,
            output_cost=0.001,
            requests_per_pollen=200000,
        ),
        status="NEW",
    ),
    ModelInfo(
        model_id="step-3.5-flash",
        name="Step 3.5 Flash (api.airforce)",
        category="text",
        paid_only=False,
        pricing=ModelPricing(
            unit="/M",
            input_cost=0.001,
            output_cost=0.003,
            requests_per_pollen=18700,
        ),
        status="NEW ALPHA",
    ),
    ModelInfo(
        model_id="nova-fast",
        name="Amazon Nova Micro",
        category="text",
        paid_only=False,
        pricing=ModelPricing(
            unit="/M",
            input_cost=0.04,
            output_cost=0.15,
            requests_per_pollen=10300,
        ),
    ),
    ModelInfo(
        model_id="gemini-fast",
        name="Google Gemini 2.5 Flash Lite",
        category="text",
        paid_only=False,
        pricing=ModelPricing(
            unit="/M",
            input_cost=0.001,
            output_cost=0.004,
            requests_per_pollen=4900,
        ),
        features={"vision", "search", "code_exec"},
    ),
    ModelInfo(
        model_id="qwen-coder",
        name="Qwen3 Coder 30B",
        category="text",
        paid_only=False,
        pricing=ModelPricing(
            unit="/M",
            input_cost=0.006,
            output_cost=0.022,
            requests_per_pollen=3000,
        ),
        features={"code_exec"},
    ),
    ModelInfo(
        model_id="mistral",
        name="Mistral Small 3.2 24B",
        category="text",
        paid_only=False,
        pricing=ModelPricing(
            unit="/M",
            input_cost=0.01,
            output_cost=0.03,
            requests_per_pollen=2100,
        ),
    ),
    ModelInfo(
        model_id="gemini-search",
        name="Google Gemini 2.5 Flash Lite (Search)",
        category="text",
        paid_only=False,
        pricing=ModelPricing(
            unit="/M",
            input_cost=0.001,
            output_cost=0.004,
            requests_per_pollen=2500,
        ),
        features={"search"},
    ),
    ModelInfo(
        model_id="openai-fast",
        name="OpenAI GPT-5 Nano",
        category="text",
        paid_only=False,
        pricing=ModelPricing(
            unit="/M",
            input_cost=0.006,
            output_cost=0.044,
            requests_per_pollen=1000,
        ),
        features={"vision"},
    ),
    ModelInfo(
        model_id="openai",
        name="OpenAI GPT-5 Mini",
        category="text",
        paid_only=False,
        pricing=ModelPricing(
            unit="/M",
            input_cost=0.04,
            output_cost=0.08,
            requests_per_pollen=900,
        ),
    ),
    ModelInfo(
        model_id="perplexity-fast",
        name="Perplexity Sonar",
        category="text",
        paid_only=False,
        pricing=ModelPricing(
            unit="/M",
            input_cost=1.0,
            output_cost=1.0,
            requests_per_pollen=800,
        ),
        features={"search"},
    ),
    ModelInfo(
        model_id="minimax",
        name="MiniMax M2.5",
        category="text",
        paid_only=False,
        pricing=ModelPricing(
            unit="/M",
            input_cost=0.03,
            output_cost=0.12,
            requests_per_pollen=550,
        ),
        features={"reasoning"},
    ),
    ModelInfo(
        model_id="deepseek",
        name="DeepSeek V3.2",
        category="text",
        paid_only=False,
        pricing=ModelPricing(
            unit="/M",
            input_cost=0.27,
            output_cost=1.68,
            requests_per_pollen=250,
        ),
        features={"reasoning"},
    ),
    ModelInfo(
        model_id="grok",
        name="xAI Grok 4 Fast",
        category="text",
        paid_only=True,
        pricing=ModelPricing(
            unit="/M",
            input_cost=0.2,
            output_cost=0.5,
            requests_per_pollen=250,
        ),
    ),
    ModelInfo(
        model_id="openai-large",
        name="OpenAI GPT-5.2",
        category="text",
        paid_only=True,
        pricing=ModelPricing(
            unit="/M",
            input_cost=1.75,
            output_cost=14.0,
            requests_per_pollen=200,
        ),
        features={"vision", "reasoning"},
    ),
    ModelInfo(
        model_id="gemini",
        name="Google Gemini 3 Flash",
        category="text",
        paid_only=True,
        pricing=ModelPricing(
            unit="/M",
            input_cost=0.15,
            output_cost=3.0,
            requests_per_pollen=200,
        ),
        features={"vision", "search", "code_exec"},
    ),
    ModelInfo(
        model_id="claude-fast",
        name="Anthropic Claude Haiku 4.5",
        category="text",
        paid_only=False,
        pricing=ModelPricing(
            unit="/M",
            input_cost=0.1,
            output_cost=0.5,
            requests_per_pollen=200,
        ),
        features={"vision"},
    ),
    ModelInfo(
        model_id="openai-audio",
        name="OpenAI GPT-4o Mini Audio",
        category="text",
        paid_only=False,
        pricing=ModelPricing(
            unit="/M",
            input_cost=0.27,
            output_cost=0.66,
            audio_in_cost=11.0,
            audio_out_cost=22.0,
            requests_per_pollen=150,
        ),
        features={"vision", "audio_in", "audio_out"},
    ),
    ModelInfo(
        model_id="perplexity-reasoning",
        name="Perplexity Sonar Reasoning",
        category="text",
        paid_only=False,
        pricing=ModelPricing(
            unit="/M",
            input_cost=2.0,
            output_cost=8.0,
            requests_per_pollen=100,
        ),
        features={"reasoning", "search"},
    ),
    ModelInfo(
        model_id="kimi",
        name="Moonshot Kimi K2.5",
        category="text",
        paid_only=False,
        pricing=ModelPricing(
            unit="/M",
            input_cost=0.13,
            output_cost=3.0,
            requests_per_pollen=100,
        ),
        features={"vision", "reasoning"},
    ),
    ModelInfo(
        model_id="glm",
        name="ZAI GLM-5",
        category="text",
        paid_only=False,
        pricing=ModelPricing(
            unit="/M",
            # Unusually, GLM-5 output cost (0.21) is lower than input cost
            # (0.8) per Pollinations.ai pricing data.
            input_cost=0.8,
            output_cost=0.21,
            requests_per_pollen=90,
        ),
        features={"reasoning"},
        status="NEW",
    ),
    ModelInfo(
        model_id="midjourney",
        name="MidJourney",
        category="text",
        paid_only=False,
        pricing=ModelPricing(
            unit="/M",
            input_cost=2.21,
            output_cost=8.81,
            requests_per_pollen=65,
        ),
    ),
    ModelInfo(
        model_id="claude",
        name="Anthropic Claude Sonnet 4.6",
        category="text",
        paid_only=True,
        pricing=ModelPricing(
            unit="/M",
            input_cost=3.0,
            output_cost=15.0,
            requests_per_pollen=25,
        ),
        features={"vision"},
    ),
    ModelInfo(
        model_id="gemini-large",
        name="Google Gemini 3.1 Pro",
        category="text",
        paid_only=True,
        pricing=ModelPricing(
            unit="/M",
            input_cost=0.2,
            output_cost=12.0,
            requests_per_pollen=20,
        ),
        features={"vision", "reasoning", "search"},
    ),
    ModelInfo(
        model_id="claude-large",
        name="Anthropic Claude Opus 4.6",
        category="text",
        paid_only=True,
        pricing=ModelPricing(
            unit="/M",
            input_cost=5.0,
            output_cost=25.0,
            requests_per_pollen=10,
        ),
        features={"vision"},
    ),
    ModelInfo(
        model_id="claude-legacy",
        name="Anthropic Claude Opus 4.5",
        category="text",
        paid_only=True,
        pricing=ModelPricing(
            unit="/M",
            input_cost=5.0,
            output_cost=25.0,
        ),
        features={"vision"},
    ),
    ModelInfo(
        model_id="gemini-pro-preview",
        name="Google Gemini 3 Pro (deprecated)",
        category="text",
        paid_only=True,
        pricing=ModelPricing(
            unit="/M",
            input_cost=0.2,
            output_cost=12.0,
            requests_per_pollen=20,
        ),
        status="DEPRECATED",
    ),
    ModelInfo(
        model_id="gemini-pro",
        name="Google Gemini 2.5 Pro",
        category="text",
        paid_only=True,
        pricing=ModelPricing(
            unit="/M",
            input_cost=1.25,
            output_cost=10.0,
        ),
        features={"reasoning"},
    ),
    ModelInfo(
        model_id="nomnom",
        name="NomNom by @itachi-1824",
        category="text",
        paid_only=False,
        status="ALPHA",
    ),
    ModelInfo(
        model_id="polly",
        name="Polly by @itachi-1824",
        category="text",
        paid_only=False,
        status="NEW ALPHA",
    ),
    ModelInfo(
        model_id="qwen-character",
        name="Qwen Character (api.airforce)",
        category="text",
        paid_only=False,
        pricing=ModelPricing(
            unit="/M",
            input_cost=0.001,
            output_cost=0.01,
        ),
        status="NEW ALPHA",
    ),
]

# ---------------------------------------------------------------------------
# Combined catalog and lookup helpers
# ---------------------------------------------------------------------------

ALL_MODELS: list[ModelInfo] = IMAGE_MODELS + VIDEO_MODELS + AUDIO_MODELS + TEXT_MODELS

# Fast lookup by model_id
_MODEL_BY_ID: dict[str, ModelInfo] = {m.model_id: m for m in ALL_MODELS}

# Set of all paid-only model IDs (used by config.py / logic_pollen.py)
PAID_ONLY_MODEL_IDS: set[str] = {
    m.model_id for m in ALL_MODELS if m.paid_only
}


def get_model(model_id: str) -> Optional[ModelInfo]:
    """Return the :class:`ModelInfo` for *model_id*, or ``None`` if not found."""
    return _MODEL_BY_ID.get(model_id)


def is_paid_only(model_id: str) -> bool:
    """Return ``True`` if *model_id* is a paid-only model."""
    model = _MODEL_BY_ID.get(model_id)
    return model.paid_only if model is not None else False


def list_models_by_category(category: str) -> list[ModelInfo]:
    """Return all models for a given *category* (image/video/audio/text)."""
    return [m for m in ALL_MODELS if m.category == category]


def list_free_models() -> list[ModelInfo]:
    """Return all models accessible via daily grants (not paid-only)."""
    return [m for m in ALL_MODELS if not m.paid_only]


def list_paid_models() -> list[ModelInfo]:
    """Return all models that require purchased Pollen."""
    return [m for m in ALL_MODELS if m.paid_only]
