"""Universal safety assessment that remains authoritative over routing policy."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .evaluation import estimate_input_tokens
from .routing import RoutingFeatures


@dataclass(frozen=True)
class SafetyAssessment:
    features: RoutingFeatures
    tool_chain_protected: bool
    opaque_state_protected: bool
    has_optional_context: bool
    has_selectable_blocks: bool
    has_rule_reduction: bool


class UniversalSafetyLayer:
    """Extract safe structural features and non-bypassable protection signals."""

    _KNOWN_FIELDS = {
        "model", "messages", "stream", "stream_options", "temperature", "top_p", "seed",
        "max_tokens", "max_completion_tokens", "response_format", "stop", "n",
    }

    def assess(self, payload, settings, raw) -> SafetyAssessment:
        # Local import avoids coupling the context module's rendering code back
        # into the policy module during import initialization.
        from .context import filter_blocks, protected

        rendered = raw.render_all(settings)
        estimated = estimate_input_tokens(rendered)
        kept, discarded = filter_blocks(raw, settings)
        candidates = [entry for entry in kept if entry["reason"] == "selection_candidate"]
        selectable_ids = {entry["id"] for entry in candidates}
        block_count = len(raw.blocks)
        protected_count = sum(protected(block) for block in raw.blocks)
        candidate_tokens = sum(
            math.ceil(len(entry["content"].encode("utf-8")) / 4) for entry in candidates
        )
        messages = payload.get("messages", [])
        tool_chain = bool(payload.get("tools")) or any(
            message.get("role") == "tool" or message.get("tool_calls") or message.get("function_call")
            for message in messages if isinstance(message, dict)
        )
        opaque = bool(payload.keys() - self._KNOWN_FIELDS) or any(
            not isinstance(message.get("content"), str)
            for message in messages if isinstance(message, dict)
        )
        features = RoutingFeatures(
            estimated_input_tokens=estimated,
            block_count=block_count,
            protected_block_count=protected_count,
            protected_ratio=protected_count / block_count if block_count else 0.0,
            selectable_block_count=len(selectable_ids),
            candidate_tokens=candidate_tokens,
            local_provider=settings.local.provider,
            local_model=settings.local.model,
            cloud_provider=settings.cloud.provider,
            cloud_model=settings.cloud.model,
            router_enabled=settings.router.enabled,
        )
        return SafetyAssessment(
            features=features,
            tool_chain_protected=tool_chain,
            opaque_state_protected=opaque,
            has_optional_context=any(block.optional for block in raw.blocks),
            has_selectable_blocks=bool(candidates),
            has_rule_reduction=bool(discarded),
        )
