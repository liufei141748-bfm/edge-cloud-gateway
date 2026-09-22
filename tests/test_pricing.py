import math

import pytest

from edge_cloud_gateway.pricing import calculate_cost, usage_from_response


def usage(**changes):
    result = {
        "input_tokens": 1_000_000,
        "output_tokens": 500_000,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": None,
    }
    return result | changes


def prices(**changes):
    return {"input_per_million": 10, "output_per_million": 20, "currency": "CNY"} | changes


def test_basic_cost_and_reasoning_not_double_counted():
    result = calculate_cost(usage(reasoning_tokens=200_000), prices())
    assert result == {"amount": 20.0, "currency": "CNY", "estimated": True}


def test_cache_read_write_included_in_input():
    result = calculate_cost(
        usage(cached_input_tokens=200_000, cache_write_tokens=100_000),
        prices(cached_input_per_million=2, cache_write_per_million=12),
    )
    assert result["amount"] == pytest.approx(18.6)


def test_cache_write_additional_charge():
    result = calculate_cost(
        usage(cached_input_tokens=200_000, cache_write_tokens=100_000),
        prices(cached_input_per_million=2, cache_write_per_million=12, cache_write_mode="additional"),
    )
    assert result["amount"] == pytest.approx(19.6)


def test_decimal_prices_and_zero_counts():
    assert calculate_cost(usage(input_tokens=1, output_tokens=0), prices(input_per_million="0.1"))["amount"] == 0.0000001
    assert calculate_cost(usage(input_tokens=0, output_tokens=0), {"currency": "usd"}) == {
        "amount": 0.0, "currency": "USD", "estimated": True,
    }


@pytest.mark.parametrize("changes", [
    {"input_tokens": None}, {"output_tokens": None}, {"cached_input_tokens": None},
    {"cache_write_tokens": None}, {"input_tokens": -1}, {"output_tokens": 1.0},
    {"input_tokens": True}, {"output_tokens": float("nan")},
    {"cached_input_tokens": 1_000_001},
    {"cached_input_tokens": 900_000, "cache_write_tokens": 200_000},
    {"reasoning_tokens": 500_001}, {"reasoning_tokens": -1},
    {"usage_complete": False}, {"complete": False}, {"total_tokens": 2},
])
def test_invalid_unknown_or_partial_counts_are_not_priced(changes):
    assert calculate_cost(usage(**changes), prices()) is None


@pytest.mark.parametrize("changes", [
    {"input_per_million": None}, {"input_per_million": -1},
    {"input_per_million": float("nan")}, {"input_per_million": float("inf")},
    {"input_per_million": True}, {"input_per_million": "invalid"},
    {"cache_write_mode": "guess"}, {"currency": ""}, {"currency": "人民币"},
])
def test_unknown_or_invalid_prices_are_not_priced(changes):
    assert calculate_cost(usage(), prices(**changes)) is None


def test_nonzero_cache_category_requires_its_price():
    assert calculate_cost(usage(cached_input_tokens=10), prices()) is None
    assert calculate_cost(usage(cache_write_tokens=10), prices()) is None
    assert calculate_cost(None, prices()) is None
    assert calculate_cost(usage(), None) is None
    assert calculate_cost(usage(), {}) is None


def test_normalize_chat_completion_details_and_discard_raw_fields():
    normalized = usage_from_response({"usage": {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 30, "cache_write_tokens": 10},
        "completion_tokens_details": {"reasoning_tokens": 5},
        "untrusted_text": "never stored",
    }})
    assert normalized == {
        "input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 30,
        "cache_write_tokens": 10, "reasoning_tokens": 5,
    }


def test_missing_details_remain_unknown_and_are_not_priced():
    normalized = usage_from_response({"usage": {"prompt_tokens": 100, "completion_tokens": 20}})
    assert normalized["input_tokens"] == 100
    assert normalized["cached_input_tokens"] is None
    assert normalized["cache_write_tokens"] is None
    assert calculate_cost(normalized, prices()) is None


def test_canonical_usage_and_missing_or_invalid_stream_usage():
    assert usage_from_response({"usage": usage()}) == usage()
    assert usage_from_response({"choices": []}) is None
    assert usage_from_response({"usage": None}) is None
    normalized = usage_from_response({"usage": {"input_tokens": -1, "output_tokens": True}})
    assert normalized["input_tokens"] is None
    assert normalized["output_tokens"] is None
    assert calculate_cost(normalized, prices()) is None


def test_extreme_rates_do_not_create_infinite_costs():
    assert calculate_cost(usage(), prices(input_per_million="1e999")) is None
