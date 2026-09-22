"""Stable content-free routing contracts for rule-based and future policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class RoutingFeatures:
    """Structural request features only; never stores prompt or context text."""

    estimated_input_tokens: int
    block_count: int
    protected_block_count: int
    protected_ratio: float
    selectable_block_count: int
    candidate_tokens: int
    local_provider: str
    local_model: str
    cloud_provider: str
    cloud_model: str
    router_enabled: bool
    schema_version: str = "routing_features_v1"

    def as_metrics(self) -> dict[str, Any]:
        return {
            "routing_features_version": self.schema_version,
            "estimated_input_tokens": self.estimated_input_tokens,
            "block_count": self.block_count,
            "protected_block_count": self.protected_block_count,
            "protected_ratio": self.protected_ratio,
            "selectable_block_count": self.selectable_block_count,
            "candidate_tokens": self.candidate_tokens,
            "local_provider": self.local_provider,
            "local_model": self.local_model,
            "cloud_provider": self.cloud_provider,
            "cloud_model": self.cloud_model,
        }


@dataclass(frozen=True)
class RouteDecision:
    route: str
    reason: str
    local_decision: Any
    source: str
    input_tokens: int | None = None
    threshold_tokens: int | None = None
    features: RoutingFeatures | None = None


@runtime_checkable
class RoutingPolicy(Protocol):
    """A policy may choose a route but cannot bypass the safety assessment."""

    def decide(self, payload: dict, settings: Any, raw: Any, local_decision: Any,
               safety: Any) -> RouteDecision: ...
