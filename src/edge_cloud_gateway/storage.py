"""Small local SQLite store: safe metadata, explicit response cache, no prompts.

Request/attempt methods whitelist fields and deliberately ignore extra fields.
Only put_cache stores full response data; callers must enforce opt-in policy.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .pricing import USAGE_FIELDS


REQUEST_FIELDS = {
    "request_id", "task_id", "status", "route", "reason", "model",
    "latency_ms", "first_byte_ms", "cache_hit", "simulated",
    "route_reason", "raw_input_tokens", "working_input_tokens", "raw_context_tokens", "working_context_tokens",
    "cloud_input_tokens", "cloud_output_tokens", "local_model_used", "cloud_model_used",
    "latency_local", "latency_cloud", "latency_total", "fallback_used", "context_compression_ratio",
    "usage_source", "input_usage_source", "input_estimation_method", "raw_payload_bytes", "working_payload_bytes",
    "context_snapshot_saved",
    "route_source", "route_decision_reason", "context_reason", "router_enabled",
    "router_input_tokens", "router_threshold_tokens",
    "initial_route_decision_reason", "routing_features_version", "estimated_input_tokens",
    "block_count", "protected_block_count", "protected_ratio", "selectable_block_count",
    "candidate_tokens", "local_provider", "local_model", "cloud_provider", "cloud_model",
    "local_input_tokens", "local_output_tokens",
}
ATTEMPT_FIELDS = {
    "request_id", "task_id", "attempt_id", "provider", "model", "status",
    "latency_ms", "usage", "cost", "simulated", "error_type", "usage_complete",
    "usage_source",
}


def _nonnegative_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _sanitize(record: dict, fields: set[str]) -> dict:
    result = {}
    for key in fields:
        value = record.get(key)
        if key in {"latency_ms", "first_byte_ms", "latency_local", "latency_cloud", "latency_total",
                   "context_compression_ratio", "protected_ratio"}:
            result[key] = value if _nonnegative_number(value) else None
        elif key in {"raw_input_tokens", "working_input_tokens", "raw_context_tokens", "working_context_tokens",
                     "cloud_input_tokens", "cloud_output_tokens", "raw_payload_bytes", "working_payload_bytes",
                     "router_input_tokens", "router_threshold_tokens", "estimated_input_tokens",
                     "block_count", "protected_block_count", "selectable_block_count", "candidate_tokens",
                     "local_input_tokens", "local_output_tokens"}:
            result[key] = value if type(value) is int and value >= 0 else None
        elif key in {"simulated", "cache_hit", "usage_complete", "local_model_used", "cloud_model_used", "fallback_used",
                     "context_snapshot_saved", "router_enabled"}:
            result[key] = value is True
        elif key == "usage":
            result[key] = (
                {name: value.get(name) if type(value.get(name)) is int and value[name] >= 0 else None
                 for name in USAGE_FIELDS}
                if isinstance(value, dict) else None
            )
        elif key == "cost":
            if (isinstance(value, dict) and _nonnegative_number(value.get("amount"))
                    and isinstance(value.get("currency"), str)
                    and len(value["currency"]) == 3 and value["currency"].isascii()
                    and value["currency"].isalpha() and value.get("estimated") is True):
                result[key] = {"amount": value["amount"], "currency": value["currency"].upper(), "estimated": True}
            else:
                result[key] = None
        elif key == "error_type":
            # Accept class/category labels, never arbitrary exception messages.
            result[key] = value if isinstance(value, str) and len(value) <= 80 and value.replace("_", "").isalnum() else None
        else:
            result[key] = value[:512] if isinstance(value, str) else None
    if "usage_complete" in fields and not result["usage_complete"]:
        result["cost"] = None
    return result


def _percentiles(values: list[float]) -> dict:
    values = sorted(values)
    if not values:
        return {"samples": 0, "p50": None, "p95": None, "p99": None}

    def quantile(p: float) -> float:
        position = (len(values) - 1) * p
        lower = math.floor(position)
        upper = math.ceil(position)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return {"samples": len(values), "p50": quantile(0.50), "p95": quantile(0.95), "p99": quantile(0.99)}


def _group_summary(requests: list[dict], attempts: list[dict]) -> dict:
    usage = {}
    for field in USAGE_FIELDS:
        known = [item["usage"][field] for item in attempts
                 if item.get("usage_complete") and isinstance(item.get("usage"), dict)
                 and item["usage"].get(field) is not None]
        usage[field] = {
            "known_total": sum(known) if known else None,
            "known_attempts": len(known),
            "unknown_attempts": len(attempts) - len(known),
        }
    costs: dict[str, float] = {}
    priced = 0
    for item in attempts:
        cost = item.get("cost")
        if cost is not None:
            priced += 1
            currency = cost["currency"]
            costs[currency] = costs.get(currency, 0) + cost["amount"]
    return {
        "requests_total": len(requests),
        "attempts_total": len(attempts),
        "requests_by_route": dict(Counter(item.get("route") or "unknown" for item in requests)),
        "attempts_by_provider": dict(Counter(item.get("provider") or "unknown" for item in attempts)),
        "usage": usage,
        "unknown_usage_attempts": sum(
            not item.get("usage_complete") or not isinstance(item.get("usage"), dict)
            or item["usage"].get("input_tokens") is None or item["usage"].get("output_tokens") is None
            for item in attempts
        ),
        "estimated_cost_by_currency": costs,
        "priced_attempts": priced,
        "unpriced_attempts": len(attempts) - priced,
        "latency_ms": {
            "requests": _percentiles([item["latency_ms"] for item in requests if item.get("latency_ms") is not None]),
            "attempts": _percentiles([item["latency_ms"] for item in attempts if item.get("latency_ms") is not None]),
            "first_byte": _percentiles([item["first_byte_ms"] for item in requests if item.get("first_byte_ms") is not None]),
        },
    }


class Store:
    def __init__(self, db_path: str | Path):
        path = str(db_path)
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY, recorded_at REAL NOT NULL, metadata TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY, recorded_at REAL NOT NULL, metadata TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS response_cache (
                cache_key TEXT PRIMARY KEY, expires_at REAL NOT NULL, response TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS context_snapshots (
                request_id TEXT PRIMARY KEY, recorded_at REAL NOT NULL,
                raw_context TEXT NOT NULL, working_context TEXT NOT NULL, package TEXT NOT NULL
            );
            """
        )
        self._connection.commit()

    def record_request(self, record: dict) -> None:
        self._record("requests", "request_id", _sanitize(record, REQUEST_FIELDS))

    def save_context(self, request_id: str, raw: dict, working: dict, package: dict) -> None:
        """Opt-in debug/evaluation snapshots retain raw material locally.

        Unlike metric records these snapshots contain user data. The gateway
        never stores HTTP headers/credentials here or uploads this database.
        Callers must enforce observability.save_context_snapshots.
        """
        values = [json.dumps(value, ensure_ascii=False, allow_nan=False) for value in (raw, working, package)]
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO context_snapshots VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(request_id) DO UPDATE SET working_context=excluded.working_context, package=excluded.package",
                (request_id, time.time(), *values),
            )

    def get_context(self, request_id: str) -> dict | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT raw_context, working_context, package FROM context_snapshots WHERE request_id=?", (request_id,),
            ).fetchone()
        return dict(zip(("raw_context", "working_context", "package"), map(json.loads, row))) if row else None

    def record_attempt(self, record: dict) -> None:
        self._record("attempts", "attempt_id", _sanitize(record, ATTEMPT_FIELDS))

    def _record(self, table: str, identifier: str, record: dict) -> None:
        # Table/column names are private fixed constants, never external input.
        key = record.get(identifier)
        if not key:
            raise ValueError(f"{identifier} is required")
        payload = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with self._lock, self._connection:
            self._connection.execute(
                f"INSERT INTO {table} ({identifier}, recorded_at, metadata) VALUES (?, ?, ?) "
                f"ON CONFLICT({identifier}) DO UPDATE SET metadata=excluded.metadata",
                (key, time.time(), payload),
            )

    def get_cache(self, key: str, now: float | None = None) -> dict | None:
        instant = time.time() if now is None else now
        if not _nonnegative_number(instant):
            raise ValueError("now must be a finite nonnegative timestamp")
        with self._lock:
            row = self._connection.execute(
                "SELECT response FROM response_cache WHERE cache_key=? AND expires_at>?",
                (key, instant),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put_cache(self, key: str, value: dict, ttl_seconds: float, now: float | None = None) -> None:
        instant = time.time() if now is None else now
        if not isinstance(key, str) or not key:
            raise ValueError("cache key must be nonempty")
        if not isinstance(value, dict):
            raise ValueError("cache value must be a response object")
        if not _nonnegative_number(ttl_seconds) or ttl_seconds == 0:
            raise ValueError("ttl_seconds must be finite and positive")
        if not _nonnegative_number(instant) or not _nonnegative_number(instant + ttl_seconds):
            raise ValueError("now and expiry must be finite nonnegative timestamps")
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False)
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO response_cache (cache_key, expires_at, response) VALUES (?, ?, ?) "
                "ON CONFLICT(cache_key) DO UPDATE SET expires_at=excluded.expires_at, response=excluded.response",
                (key, instant + ttl_seconds, payload),
            )

    def _recent(self, table: str, limit: int) -> list[dict]:
        if type(limit) is not int or limit < 0:
            raise ValueError("limit must be a nonnegative integer")
        with self._lock:
            rows = self._connection.execute(
                f"SELECT metadata FROM {table} ORDER BY recorded_at DESC, rowid DESC LIMIT ?", (limit,)
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def recent_requests(self, limit: int = 50) -> list[dict]:
        return self._recent("requests", limit)

    def recent_attempts(self, limit: int = 50) -> list[dict]:
        return self._recent("attempts", limit)

    def summary(self) -> dict:
        """Aggregate real and simulated work separately; unknown sums stay None.

        Token totals include only complete attempts and identify their sample
        counts. Costs are price-table estimates, grouped by currency, never an
        invoice. Cached requests count as requests but add no provider attempt.
        """
        with self._lock:
            requests = [json.loads(row[0]) for row in self._connection.execute("SELECT metadata FROM requests")]
            attempts = [json.loads(row[0]) for row in self._connection.execute("SELECT metadata FROM attempts")]
        result = {
            "requests_total": len(requests),
            "attempts_total": len(attempts),
            "requests_by_route": dict(Counter(item.get("route") or "unknown" for item in requests)),
            "attempts_by_provider": dict(Counter(item.get("provider") or "unknown" for item in attempts)),
            "groups": {
                label: _group_summary(
                    [item for item in requests if item.get("simulated") is flag],
                    [item for item in attempts if item.get("simulated") is flag],
                )
                for label, flag in (("real", False), ("simulated", True))
            },
        }
        for label, flag in (("real", False), ("simulated", True)):
            result["groups"][label]["usage_by_provider"] = {
                provider: _group_summary([], [item for item in attempts
                                             if item.get("simulated") is flag and item.get("provider") == provider])["usage"]
                for provider in ("local", "cloud")
            }
        return result

    def close(self) -> None:
        with self._lock:
            self._connection.close()
