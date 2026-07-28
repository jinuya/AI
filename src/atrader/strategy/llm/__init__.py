"""LLM agent strategy and its safety layer (spec §7.7).

The model produces a schema-constrained intent and nothing else. Symbol
whitelisting, numeric sanity checks, confidence scaling and prompt-injection
isolation all sit here — but the real defence is the deterministic risk engine
downstream, not any of this.
"""

from __future__ import annotations

from atrader.strategy.llm.agent import LLMAgentStrategy
from atrader.strategy.llm.client import (
    AnthropicLLMClient,
    FakeLLMClient,
    LLMClient,
    LLMRequest,
    LLMResponse,
    LLMTradingDecision,
    LLMTradingResponse,
    LLMUsage,
    RecordedLLMClient,
)
from atrader.strategy.llm.guards import GuardRejection, GuardResult, apply_guards
from atrader.strategy.llm.prompt import build_system_prompt, build_user_content, render_untrusted

__all__ = [
    "AnthropicLLMClient",
    "FakeLLMClient",
    "GuardRejection",
    "GuardResult",
    "LLMAgentStrategy",
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "LLMTradingDecision",
    "LLMTradingResponse",
    "LLMUsage",
    "RecordedLLMClient",
    "apply_guards",
    "build_system_prompt",
    "build_user_content",
    "render_untrusted",
]
