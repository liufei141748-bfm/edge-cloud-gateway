"""Reproducible estimates and comparison helpers; not a model tokenizer."""
import json
import math

ESTIMATE_METHOD = "utf8_bytes_div4_v1"


def canonical_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def estimate_input_tokens(payload: dict) -> int:
    # Count prompt-bearing fields, including tool schemas. This is a consistent
    # comparison heuristic, not an exact billing/tokenizer calculation.
    prompt = {key: value for key, value in payload.items() if key in {
        "messages", "tools", "functions", "response_format", "tool_choice",
    }}
    return math.ceil(len(canonical_bytes(prompt)) / 4)


def input_comparison(raw: dict, working: dict) -> dict:
    before, after = estimate_input_tokens(raw), estimate_input_tokens(working)
    return {
        "raw_input_tokens": before, "working_input_tokens": after,
        "raw_context_tokens": before, "working_context_tokens": after,
        "context_compression_ratio": after / before if before else None,
        "input_usage_source": "estimated", "input_estimation_method": ESTIMATE_METHOD,
        "raw_payload_bytes": len(canonical_bytes(raw)),
        "working_payload_bytes": len(canonical_bytes(working)),
    }


def compare_runs(baseline: dict, optimized: dict, *, quality_passed: bool | None = None) -> dict:
    """Only actual, complete, nonsimulated usage supports a measured reduction."""
    actual = all(
        item.get("usage_source") == "actual" and not item.get("simulated")
        and all(type(item.get(key)) is int and item[key] >= 0
                for key in ("cloud_input_tokens", "cloud_output_tokens"))
        for item in (baseline, optimized)
    )
    before = baseline.get("cloud_input_tokens", 0) + baseline.get("cloud_output_tokens", 0) if actual else 0
    after = optimized.get("cloud_input_tokens", 0) + optimized.get("cloud_output_tokens", 0) if actual else 0
    return {
        "usage_comparable": bool(actual and before),
        "cloud_token_reduction": 1 - after / before if actual and before else None,
        "quality_passed": quality_passed,
        "quality_preserving_savings_verified": bool(actual and before and after < before and quality_passed is True),
    }
