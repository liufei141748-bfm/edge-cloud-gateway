"""Usage normalization and deliberately conservative price-table estimates.

An omitted cache counter is unknown, not zero. Adapters that know a provider
does not use cache billing should explicitly supply zero counters. Reasoning
tokens are a subset of output tokens and are never charged a second time.
"""

from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation
from typing import Any


USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
)


def _count(value: Any) -> int | None:
    # bool is an int subclass but is not a token count.
    return value if type(value) is int and value >= 0 else None


def _first(mapping: dict, *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def usage_from_response(body: dict) -> dict | None:
    """Read Chat Completions or canonical usage without storing raw responses.

    Missing/invalid individual fields remain None. Missing usage returns None,
    including a stream that ended before its usage event. Cache counters are
    not inferred from total input. The caller separately tracks whether the
    response/stream completed successfully.
    """
    if not isinstance(body, dict) or not isinstance(body.get("usage"), dict):
        return None
    raw = body["usage"]
    input_details = raw.get("prompt_tokens_details", raw.get("input_tokens_details"))
    output_details = raw.get("completion_tokens_details", raw.get("output_tokens_details"))
    input_details = input_details if isinstance(input_details, dict) else {}
    output_details = output_details if isinstance(output_details, dict) else {}
    cached = _first(raw, "cached_input_tokens", "cache_read_input_tokens")
    if cached is None:
        cached = input_details.get("cached_tokens")
    written = _first(raw, "cache_write_tokens", "cache_creation_input_tokens")
    if written is None:
        written = input_details.get("cache_write_tokens")
    reasoning = raw.get("reasoning_tokens", output_details.get("reasoning_tokens"))
    return {
        "input_tokens": _count(_first(raw, "input_tokens", "prompt_tokens")),
        "output_tokens": _count(_first(raw, "output_tokens", "completion_tokens")),
        "cached_input_tokens": _count(cached),
        "cache_write_tokens": _count(written),
        "reasoning_tokens": _count(reasoning),
    }


def _rate(prices: dict, key: str) -> Decimal | None:
    value = prices.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() and result >= 0 else None


def calculate_cost(usage: dict | None, prices: dict | None) -> dict | None:
    """Estimate a charge from explicit counts and configured per-million rates.

    All four input/output/cache counts must be known, even when zero. Unknown
    usage, invalid/inconsistent counts, or a missing rate for a nonzero billing
    category returns None. `complete=False`/`usage_complete=False` also refuses
    an estimate; normally the caller checks completion before calling.

    `included`: cache writes are part of reported input, so uncached input is
    input - cache reads - cache writes. `additional`: writes are a surcharge,
    so uncached input is input - cache reads. The selected provider's adapter
    must normalize counters to this convention; do not guess its billing rules.
    """
    if not isinstance(usage, dict) or not isinstance(prices, dict):
        return None
    if usage.get("complete") is False or usage.get("usage_complete") is False:
        return None
    counts = {field: _count(usage.get(field)) for field in USAGE_FIELDS[:4]}
    if any(value is None for value in counts.values()):
        return None
    incoming, outgoing, cached, written = (counts[field] for field in USAGE_FIELDS[:4])
    reasoning = usage.get("reasoning_tokens")
    if reasoning is not None and (_count(reasoning) is None or reasoning > outgoing):
        return None
    mode = prices.get("cache_write_mode", "included")
    if mode not in {"included", "additional"}:
        return None
    uncached = incoming - cached - (written if mode == "included" else 0)
    if uncached < 0:
        return None
    if "total_tokens" in usage and usage["total_tokens"] is not None:
        if _count(usage["total_tokens"]) != incoming + outgoing:
            return None
    currency = prices.get("currency", "CNY")
    if not isinstance(currency, str) or len(currency) != 3 or not currency.isascii() or not currency.isalpha():
        return None
    amount = Decimal(0)
    categories = (
        (uncached, "input_per_million"),
        (outgoing, "output_per_million"),
        (cached, "cached_input_per_million"),
        (written, "cache_write_per_million"),
    )
    # Invalid supplied rates are rejected even for a zero-count category.
    for count, key in categories:
        rate = _rate(prices, key)
        if key in prices and rate is None:
            return None
        if count and rate is None:
            return None
        if rate is not None:
            amount += Decimal(count) * rate / Decimal(1_000_000)
    try:
        result = float(amount)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return {"amount": result, "currency": currency.upper(), "estimated": True}
