import json

import pytest

from edge_cloud_gateway.storage import Store


@pytest.fixture
def store(tmp_path):
    instance = Store(tmp_path / "metrics.sqlite3")
    yield instance
    instance.close()


def request(request_id="r1", **changes):
    return {
        "request_id": request_id, "task_id": "task-1", "status": "success",
        "route": "cloud", "reason": "explicit_cloud_model", "model": "mock-cloud",
        "latency_ms": 100.0, "first_byte_ms": 20.0, "cache_hit": False,
        "simulated": True,
    } | changes


def attempt(attempt_id="a1", **changes):
    return {
        "request_id": "r1", "task_id": "task-1", "attempt_id": attempt_id,
        "provider": "cloud", "model": "mock-cloud", "status": "success",
        "latency_ms": 80.0,
        "usage": {"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 0,
                  "cache_write_tokens": 0, "reasoning_tokens": None},
        "cost": {"amount": 0.1, "currency": "CNY", "estimated": True},
        "simulated": True, "error_type": None, "usage_complete": True,
    } | changes


def test_cache_hit_miss_expiration_isolation_and_replacement(store):
    response = {"choices": [{"message": {"content": "中文回复"}}]}
    assert store.get_cache("missing", now=0) is None
    store.put_cache("tenant-a:key", response, ttl_seconds=10, now=100)
    assert store.get_cache("tenant-a:key", now=109) == response
    assert store.get_cache("tenant-b:key", now=109) is None
    assert store.get_cache("tenant-a:changed-input", now=109) is None
    assert store.get_cache("tenant-a:changed-config", now=109) is None
    assert store.get_cache("tenant-a:key", now=110) is None
    store.put_cache("tenant-a:key", {"new": True}, ttl_seconds=20, now=110)
    assert store.get_cache("tenant-a:key", now=111) == {"new": True}
    assert store.get_cache("tenant-a:key", now=130) is None


@pytest.mark.parametrize("ttl", [0, -1, True, float("nan"), float("inf")])
def test_cache_rejects_invalid_ttl(store, ttl):
    with pytest.raises(ValueError):
        store.put_cache("key", {}, ttl_seconds=ttl, now=0)


def test_cache_rejects_invalid_clock_and_nonfinite_response(store):
    with pytest.raises(ValueError):
        store.get_cache("key", now=float("nan"))
    with pytest.raises(ValueError):
        store.put_cache("key", {"value": float("nan")}, ttl_seconds=10, now=0)


def test_metadata_whitelist_does_not_persist_prompt_headers_or_raw_usage(store):
    store.record_request(request(prompt="TOP_SECRET", headers={"Authorization": "TOP_SECRET"}, api_key="TOP_SECRET"))
    raw_usage = attempt()["usage"] | {"prompt": "TOP_SECRET"}
    store.record_attempt(attempt(
        usage=raw_usage, headers={"Authorization": "TOP_SECRET"},
        error_type="Exception TOP_SECRET: contains arbitrary message",
        cost=attempt()["cost"] | {"api_key": "TOP_SECRET"},
    ))
    persisted = json.dumps(store.recent_requests() + store.recent_attempts())
    assert "TOP_SECRET" not in persisted
    assert "headers" not in persisted
    assert "api_key" not in persisted
    assert store.recent_attempts()[0]["error_type"] is None


def test_summary_separates_simulated_and_real_and_keeps_unknown_unknown(store):
    store.record_request(request())
    store.record_attempt(attempt())
    store.record_request(request("r2", simulated=False, latency_ms=300))
    store.record_attempt(attempt("a2", request_id="r2", simulated=False, usage=None, cost=None, usage_complete=False))
    store.record_request(request("r3", route="cache", cache_hit=True, latency_ms=5))
    summary = store.summary()
    assert summary["requests_total"] == 3
    assert summary["attempts_total"] == 2
    assert summary["requests_by_route"] == {"cloud": 2, "cache": 1}
    assert summary["attempts_by_provider"] == {"cloud": 2}
    simulated = summary["groups"]["simulated"]
    real = summary["groups"]["real"]
    assert simulated["requests_total"] == 2
    assert simulated["attempts_total"] == 1
    assert simulated["usage"]["input_tokens"] == {"known_total": 100, "known_attempts": 1, "unknown_attempts": 0}
    assert simulated["estimated_cost_by_currency"] == {"CNY": 0.1}
    assert real["usage"]["input_tokens"]["known_total"] is None
    assert real["unknown_usage_attempts"] == 1
    assert real["estimated_cost_by_currency"] == {}
    assert real["unpriced_attempts"] == 1
    assert real["latency_ms"]["requests"] == {"samples": 1, "p50": 300.0, "p95": 300.0, "p99": 300.0}


def test_incomplete_attempt_does_not_contribute_partial_tokens_or_cost(store):
    store.record_attempt(attempt(usage_complete=False, status="cancelled", error_type="CancelledError"))
    persisted = store.recent_attempts()[0]
    assert persisted["cost"] is None
    assert persisted["error_type"] == "CancelledError"
    group = store.summary()["groups"]["simulated"]
    assert group["usage"]["output_tokens"]["known_total"] is None
    assert group["unknown_usage_attempts"] == 1
    assert group["unpriced_attempts"] == 1


def test_missing_cache_counter_does_not_hide_known_input_output_totals(store):
    store.record_attempt(attempt(usage={"input_tokens": 5, "output_tokens": 0}, cost=None))
    group = store.summary()["groups"]["simulated"]
    assert group["usage"]["input_tokens"]["known_total"] == 5
    assert group["usage"]["output_tokens"]["known_total"] == 0
    assert group["usage"]["cached_input_tokens"]["known_total"] is None
    assert group["unknown_usage_attempts"] == 0


def test_cost_currencies_are_not_added_together(store):
    store.record_attempt(attempt("a1"))
    store.record_attempt(attempt("a2", cost={"amount": 0.2, "currency": "USD", "estimated": True}))
    group = store.summary()["groups"]["simulated"]
    assert group["estimated_cost_by_currency"] == {"CNY": 0.1, "USD": 0.2}
    assert group["priced_attempts"] == 2


def test_percentiles_have_sample_counts_and_empty_values_are_null(store):
    empty = store.summary()["groups"]["real"]
    assert empty["latency_ms"]["requests"] == {"samples": 0, "p50": None, "p95": None, "p99": None}
    for index in range(1, 101):
        store.record_request(request(f"r{index}", latency_ms=index, first_byte_ms=None))
    group = store.summary()["groups"]["simulated"]
    assert group["latency_ms"]["requests"] == {"samples": 100, "p50": 50.5, "p95": 95.05, "p99": 99.01}
    assert group["latency_ms"]["first_byte"]["samples"] == 0


def test_request_record_update_is_idempotent_and_recent_limit_works(store):
    store.record_request(request("r1", status="pending"))
    store.record_request(request("r1", status="success"))
    store.record_request(request("r2"))
    assert store.summary()["requests_total"] == 2
    assert store.recent_requests(1)[0]["request_id"] == "r2"
    assert store.recent_requests(2)[1]["status"] == "success"
    assert store.recent_requests(0) == []


def test_store_reopens_without_losing_data(tmp_path):
    path = tmp_path / "metrics.sqlite3"
    first = Store(path)
    first.record_request(request())
    first.put_cache("key", {"content": "cached"}, ttl_seconds=10, now=0)
    first.close()
    second = Store(path)
    try:
        assert second.summary()["requests_total"] == 1
        assert second.get_cache("key", now=1) == {"content": "cached"}
    finally:
        second.close()


def test_invalid_numeric_metadata_is_marked_unknown(store):
    store.record_request(request(latency_ms=float("nan")))
    store.record_attempt(attempt(usage={"input_tokens": -1, "output_tokens": True}, cost={"amount": float("inf")}))
    assert store.recent_requests()[0]["latency_ms"] is None
    assert store.recent_attempts()[0]["usage"]["input_tokens"] is None
    assert store.recent_attempts()[0]["cost"] is None
