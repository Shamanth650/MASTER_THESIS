"""
RAG2/generators/orchestrator.py

Purpose:
- Single high-level API to generate scenario artifacts (Python + XOSC).
- UI/CLI should call ONLY this module.

AEB demo policy:
- No silent fallback XOSC for AEB (always disabled).
- Provider routing is controlled by UI selection ("openai" or "claude").
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from ..scenario_utils import _family_of
from .python_generator import generate_python_rag
from .xosc_generator import generate_xosc_rag


LLMProvider = Literal["openai", "claude"]


def generate_scenario_artifacts(
    scenario: Dict[str, Any],
    *,
    k: int | None = None,
    provider: LLMProvider = "openai",
    enable_xosc_fallback: bool = True,
) -> Dict[str, str]:
    """
    Generate all artifacts (Python + XOSC) for a single scenario.

    Parameters
    ----------
    scenario : dict
        One scenario entry from uniform_scenarios.json / structured_scenarios.json
    k : int | None
        Optional override for Chroma TOP_K retrieval
    provider : "openai" | "claude"
        LLM provider selected by UI
    enable_xosc_fallback : bool
        Whether to allow deterministic xosc_builder fallback.
        NOTE: For AEB, fallback is always forced OFF.

    Returns
    -------
    dict:
        { "carla_py": "...", "xosc": "..." }
    """
    family = _family_of(scenario)

    # AEB demo safety: never allow deterministic fallback because it can look like "success"
    if (family or "").strip().upper() == "AEB":
        enable_xosc_fallback = False

    carla_py = generate_python_rag(
        scenario,
        k=k,
        provider=provider,
    )

    xosc = generate_xosc_rag(
        scenario,
        k=k,
        provider=provider,
        enable_fallback_builder=enable_xosc_fallback,
    )

    return {
        "carla_py": carla_py,
        "xosc": xosc,
    }
