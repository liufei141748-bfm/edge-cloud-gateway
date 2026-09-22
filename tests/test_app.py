import asyncio
from copy import deepcopy
from dataclasses import replace
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from edge_cloud_gateway.adapters import HTTPReply, StreamReply, MockAdapter
from edge_cloud_gateway.app import create_app
from edge_cloud_gateway.config import Settings, GatewaySettings, CacheSettings, RouterSettings, ObservabilitySettings
from edge_cloud_gateway.policy import cache_key, decide
from edge_cloud_gateway.storage import Store
from test_context import context_request


def local_request():
    return {"model": "local-json", "messages": [{"role": "user", "content": '{"name":"小明","amount":120}'}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "person", "schema": {
                "type": "object", "properties": {"name": {"type": "string"}, "amount": {"type": "number"}},
                "required": ["name", "amount"], "additionalProperties": False}}}}


def make_app(**kwargs):
    store = kwargs.pop("store", None) or Store(":memory:")
    settings = kwargs.pop("settings", Settings())
    return create_app(settings, store=store, **kwargs), store


class FailingStore(Store):
    def __init__(self, failure):
        super().__init__(":memory:")
        self.failure = failure

    def _maybe_fail(self, operation):
        if self.failure == operation:
            raise RuntimeError("storage fixture failure")

    def record_attempt(self, record):
        self._maybe_fail("record_attempt")
        return super().record_attempt(record)

    def record_request(self, record):
        self._maybe_fail("record_request")
        return super().record_request(record)

    def save_context(self, request_id, raw, working, package):
        self._maybe_fail("save_context")
        return super().save_context(request_id, raw, working, package)

    def get_cache(self, key, now=None):
        self._maybe_fail("get_cache")
        return super().get_cache(key, now)

    def put_cache(self, key, value, ttl_seconds, now=None):
        self._maybe_fail("put_cache")
        return super().put_cache(key, value, ttl_seconds, now)


def test_health_models_three_paths_and_snapshot_preserve_raw():
    cloud, local = MockAdapter("cloud"), MockAdapter("local")
    settings = replace(Settings(), observability=ObservabilitySettings(save_context_snapshots=True))
    app, store = make_app(cloud=cloud, local=local, settings=settings)
    with TestClient(app) as client:
        assert client.get("/health").json()["mode"] == "mock"
        assert {x["id"] for x in client.get("/v1/models").json()["data"]} == {"adaptive", "mock-cloud", "local-json"}
        direct = {"model": "chosen-cloud", "messages": [{"role": "user", "content": "你好"}], "vendor_extension": {"x": 3}}
        r = client.post("/v1/chat/completions", json=direct)
        assert r.status_code == 200
        assert r.headers["x-gateway-route"] == "direct_cloud"
        assert cloud.calls[-1] == direct
        r = client.post("/v1/chat/completions", json=local_request())
        assert r.headers["x-gateway-route"] == "direct_local"
        assert cloud.complete_calls == 1
        assert json.loads(r.json()["choices"][0]["message"]["content"])["amount"] == 120
        original = context_request()
        r = client.post("/v1/chat/completions", json=original, headers={"X-Gateway-Task-ID": "ctx-demo"})
        assert r.headers["x-gateway-route"] == "context_then_cloud"
        snapshot = client.get("/contexts/" + r.headers["x-gateway-request-id"]).json()
        assert snapshot["raw_context"] == original
        assert snapshot["working_context"] == cloud.calls[-1]
        assert "gateway_context" not in cloud.calls[-1]
        stats = client.get("/stats").json()
        row = stats["recent_requests"][0]
        assert row["raw_input_tokens"] > row["working_input_tokens"]
        assert row["local_model_used"] and row["cloud_model_used"]
        assert row["input_usage_source"] == "estimated"
        assert row["usage_source"] == "estimated" and row["simulated"]
        assert row["task_id"] == "ctx-demo"
        assert stats["requests_total"] == 3


def test_baseline_context_optout_sends_every_block():
    settings = replace(Settings(), observability=ObservabilitySettings(save_context_snapshots=True))
    app, store = make_app(settings=settings)
    data = context_request()
    data["gateway_context"]["optimize"] = False
    with TestClient(app) as client:
        r = client.post("/v1/chat/completions", json=data)
        assert r.headers["x-gateway-route"] == "direct_cloud"
        row = store.recent_requests()[0]
        assert not row["local_model_used"]
        assert row["context_compression_ratio"] == 1
        assert "gardening" in json.dumps(store.get_context(row["request_id"])["working_context"])


@pytest.mark.parametrize("failure", ["timeout", "bad_json", "bad_schema", "http"])
def test_local_failure_calls_cloud_exactly_once(failure):
    class FailingLocal(MockAdapter):
        async def complete(self, payload):
            self.complete_calls += 1
            if failure == "timeout":
                raise httpx.ReadTimeout("test")
            if failure == "http":
                return HTTPReply(500, b"{}", {})
            content = "invalid" if failure == "bad_json" else '{"name":true}'
            return HTTPReply(200, json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": content}}]}).encode(), {})
    cloud, local = MockAdapter("cloud"), FailingLocal("local")
    app, store = make_app(cloud=cloud, local=local)
    with TestClient(app) as client:
        r = client.post("/v1/chat/completions", json=local_request())
        assert r.status_code == 200
        assert local.complete_calls == cloud.complete_calls == 1
        row = store.recent_requests()[0]
        assert row["fallback_used"]
        assert row["initial_route_decision_reason"] == "explicit_standalone_json"
        assert row["route_decision_reason"] == "local_failed_cloud_fallback"
        assert len(store.recent_attempts()) == 2


def test_local_success_survives_attempt_metrics_failure_without_cloud_fallback():
    cloud, local = MockAdapter("cloud"), MockAdapter("local")
    app, _ = make_app(cloud=cloud, local=local, store=FailingStore("record_attempt"))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=local_request())
        assert response.status_code == 200
        assert response.headers["x-gateway-route"] == "direct_local"
        assert local.complete_calls == 1 and cloud.complete_calls == 0


def test_successful_request_survives_request_metrics_failure():
    cloud = MockAdapter("cloud")
    app, _ = make_app(cloud=cloud, store=FailingStore("record_request"))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "cloud", "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 200
        assert cloud.complete_calls == 1


def test_context_snapshot_failure_does_not_block_selector_or_cloud():
    cloud, local = MockAdapter("cloud"), MockAdapter("local")
    store = FailingStore("save_context")
    settings = replace(Settings(), observability=ObservabilitySettings(save_context_snapshots=True))
    app, _ = make_app(cloud=cloud, local=local, store=store, settings=settings)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=context_request())
        assert response.status_code == 200
        assert local.complete_calls == cloud.complete_calls == 1
        assert store.recent_requests()[0]["context_snapshot_saved"] is False


def test_local_stream_is_validated_before_synthetic_output():
    cloud, local = MockAdapter("cloud"), MockAdapter("local")
    app, store = make_app(cloud=cloud, local=local)
    with TestClient(app) as client:
        r = client.post("/v1/chat/completions", json=local_request() | {"stream": True, "stream_options": {"include_usage": True}})
        assert r.status_code == 200
        assert r.headers["x-gateway-local-stream"] == "buffered-after-validation"
        assert "[DONE]" in r.text
        assert local.complete_calls == 1 and cloud.complete_calls == 0


def test_cache_optin_bypass_refresh_config_fingerprint():
    settings = replace(Settings(), cache=CacheSettings(enabled=True))
    local = MockAdapter("local")
    app, store = make_app(local=local, settings=settings)
    payload = local_request()
    with TestClient(app) as client:
        first = client.post("/v1/chat/completions", json=payload, headers={"X-Gateway-Cache": "allow"})
        second = client.post("/v1/chat/completions", json=payload, headers={"X-Gateway-Cache": "allow"})
        assert first.headers["x-gateway-cache"] == "miss"
        assert second.headers["x-gateway-cache"] == "hit"
        assert second.json()["usage"]["total_tokens"] == 0
        assert local.complete_calls == 1
        assert len(store.recent_attempts()) == 1
        client.post("/v1/chat/completions", json=payload, headers={"X-Gateway-Cache": "bypass"})
        client.post("/v1/chat/completions", json=payload, headers={"X-Gateway-Cache": "refresh"})
        client.post("/v1/chat/completions", json=payload)
        assert local.complete_calls == 4
        changed = replace(settings, local=replace(settings.local, model_revision="changed"))
        assert cache_key(payload, settings) != cache_key(payload, changed)


@pytest.mark.parametrize("failure", ["get_cache", "put_cache"])
def test_optional_cache_failure_is_a_miss_or_skipped_write(failure):
    settings = replace(Settings(), cache=CacheSettings(enabled=True))
    cloud, local = MockAdapter("cloud"), MockAdapter("local")
    app, _ = make_app(settings=settings, cloud=cloud, local=local, store=FailingStore(failure))
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions", json=local_request(), headers={"X-Gateway-Cache": "allow"},
        )
        assert response.status_code == 200
        assert response.headers["x-gateway-route"] == "direct_local"
        assert response.headers["x-gateway-cache"] == "miss"
        assert local.complete_calls == 1 and cloud.complete_calls == 0


def test_default_cache_disabled_and_tool_context_never_local():
    settings = replace(Settings(), router=RouterSettings(enabled=True))
    app, store = make_app(settings=settings)
    with TestClient(app) as client:
        for _ in range(2):
            assert client.post("/v1/chat/completions", json=local_request(), headers={"X-Gateway-Cache": "allow"}).headers["x-gateway-cache"] == "miss"
        payload = context_request()
        payload["tools"] = [{"type": "function", "function": {"name": "edit"}}]
        r = client.post("/v1/chat/completions", json=payload)
        assert r.headers["x-gateway-reason"] == "tool_chain_protected"
        row = store.recent_requests()[0]
        assert row["local_model_used"] is False
        assert row["route_source"] == "auto_router"


def test_auto_router_records_decision_metadata_and_preserves_streaming():
    settings = replace(Settings(), router=RouterSettings(enabled=True),
                       context=replace(Settings().context, min_input_tokens=1))
    app, store = make_app(settings=settings)
    payload = context_request() | {"stream": True, "stream_options": {"include_usage": True}}
    with TestClient(app) as client:
        local_response = client.post("/v1/chat/completions", json=local_request())
        assert local_response.headers["x-gateway-route"] == "direct_local"
        r = client.post("/v1/chat/completions", json=payload)
        assert r.status_code == 200 and "[DONE]" in r.text
        assert r.headers["x-gateway-route"] == "context_then_cloud"
        assert r.headers["x-gateway-reason"] == "auto_at_or_above_context_threshold"
        row = store.recent_requests()[0]
        assert row["route_source"] == "auto_router"
        assert row["route_decision_reason"] == "auto_at_or_above_context_threshold"
        assert row["context_reason"] == "selected_original_context"
        assert row["router_enabled"] is True
        assert row["router_input_tokens"] >= row["router_threshold_tokens"] == 1
        assert row["local_model_used"] and row["cloud_model_used"]


def test_standard_messages_auto_route_and_record_reduction_metrics():
    settings = replace(Settings(), context=replace(Settings().context, min_input_tokens=1))
    app, store = make_app(settings=settings)
    payload = {"model": "adaptive", "messages": [
        {"role": "system", "content": "Answer only from relevant material."},
        {"role": "user", "content": (
            "gardening plants and soil are unrelated background prose. " * 50
            + "\n\nauthentication design checks token validity and session expiry."
            + "\n\nExplain authentication design."
        )},
    ]}
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert response.headers["x-gateway-route"] == "context_then_cloud"
        row = store.recent_requests()[0]
        assert row["route_source"] == "auto_router"
        assert row["route_decision_reason"] == "auto_at_or_above_context_threshold"
        assert row["router_enabled"] is True and row["router_input_tokens"] > 1
        assert row["latency_local"] > 0 and row["local_model_used"] is True
        assert row["working_input_tokens"] < row["raw_input_tokens"]
        assert row["working_context_tokens"] < row["raw_context_tokens"]


def test_stream_exact_bytes_unknown_usage_and_http_error():
    raw = 'data: {"choices":[{"delta":{"content":"中文"}}]}\n\ndata: [DONE]\n\n'.encode()
    class Cloud(MockAdapter):
        async def stream(self, payload):
            self.stream_calls += 1
            async def chunks():
                for b in raw:
                    yield bytes([b])
            async def close():
                self.closed = True
            return StreamReply(200, {"content-type": "text/event-stream"}, chunks(), close)
    cloud = Cloud()
    app, store = make_app(cloud=cloud)
    with TestClient(app) as client:
        r = client.post("/v1/chat/completions", json={"model": "x", "messages": [{"role": "user", "content": "hi"}], "stream": True})
        assert r.content == raw
        row = store.recent_requests()[0]
        assert row["cloud_input_tokens"] is None and row["usage_source"] == "unknown"
        assert store.recent_attempts()[0]["cost"] is None
        assert cloud.stream_calls == 1 and cloud.closed


def test_stream_failure_does_not_restart_or_fake_done():
    class Cloud(MockAdapter):
        async def stream(self, payload):
            self.stream_calls += 1
            async def chunks():
                yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
                raise httpx.ReadError("private fixture details")
            async def close(): self.closed = True
            return StreamReply(200, {"content-type": "text/event-stream"}, chunks(), close)
    cloud = Cloud()
    app, store = make_app(cloud=cloud)
    with TestClient(app) as client:
        r = client.post("/v1/chat/completions", json={"model": "x", "messages": [{"role": "user", "content": "hi"}], "stream": True})
        assert "partial" in r.text and "[DONE]" not in r.text
        assert "private fixture details" not in r.text
        assert cloud.stream_calls == 1 and cloud.closed
        assert store.recent_requests()[0]["status"] == "error"


@pytest.mark.parametrize("body", [b"", b"not-json", b"{}", b'{"choices":[]}'])
def test_cloud_rejects_obviously_invalid_nonstream_2xx(body):
    class InvalidCloud(MockAdapter):
        async def complete(self, payload):
            self.complete_calls += 1
            return HTTPReply(200, body, {"content-type": "application/json"})
    cloud = InvalidCloud("cloud")
    app, _ = make_app(cloud=cloud)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "cloud", "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 502
        assert response.json()["error"]["message"] == "Upstream request failed"
        assert cloud.complete_calls == 1


def test_cloud_accepts_tool_call_completion_without_usage_and_with_extensions():
    body = {"id": "chatcmpl-tool", "choices": [{"index": 0, "message": {
        "role": "assistant", "content": None, "tool_calls": [{
            "id": "call-1", "type": "function",
            "function": {"name": "lookup", "arguments": '{"id":1}'},
        }],
    }, "finish_reason": "tool_calls"}], "provider_extension": {"ok": True}}
    class ToolCloud(MockAdapter):
        async def complete(self, payload):
            self.complete_calls += 1
            return HTTPReply(200, json.dumps(body).encode(), {"content-type": "application/json"})
    cloud = ToolCloud("cloud")
    app, _ = make_app(cloud=cloud)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "cloud", "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
        })
        assert response.status_code == 200 and response.json() == body
        assert cloud.complete_calls == 1


def test_cloud_rejects_non_sse_streaming_2xx_before_sending_body():
    consumed = False
    closed = False
    class NonSSECloud(MockAdapter):
        async def stream(self, payload):
            self.stream_calls += 1
            async def chunks():
                nonlocal consumed
                consumed = True
                yield b'{"choices":[]}'
            async def close():
                nonlocal closed
                closed = True
            return StreamReply(200, {"content-type": "application/json"}, chunks(), close)
    cloud = NonSSECloud("cloud")
    app, _ = make_app(cloud=cloud)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "cloud", "messages": [{"role": "user", "content": "hello"}], "stream": True,
        })
        assert response.status_code == 502
        assert cloud.stream_calls == 1 and not consumed and closed


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before_headers", "during_stream"])
async def test_disconnect_cancels_pending_upstream(phase):
    waiting, cancelled, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    class Cloud(MockAdapter):
        async def stream(self, payload):
            self.stream_calls += 1
            if phase == "before_headers":
                waiting.set()
                try: await asyncio.Event().wait()
                finally: cancelled.set()
            async def chunks():
                yield b"data: partial\n\n"
                waiting.set()
                try: await asyncio.Event().wait()
                finally: cancelled.set()
            async def close(): closed.set()
            return StreamReply(200, {"content-type": "text/event-stream"}, chunks(), close)
    cloud = Cloud()
    app, store = make_app(cloud=cloud)
    delivered = False
    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": json.dumps({"model": "x", "messages": [{"role": "user", "content": "hi"}], "stream": True}).encode(), "more_body": False}
        await waiting.wait()
        return {"type": "http.disconnect"}
    async def send(message): pass
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "http_version": "1.1", "method": "POST",
             "scheme": "http", "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions", "query_string": b"",
             "headers": [(b"host", b"testserver"), (b"content-type", b"application/json")], "server": ("testserver", 80), "client": ("127.0.0.1", 1)}
    await asyncio.wait_for(app(scope, receive, send), 2)
    assert cancelled.is_set()
    assert cloud.stream_calls == 1
    if phase == "during_stream": assert closed.is_set()
    assert store.recent_requests()[0]["status"] == "cancelled"
    await app.state.runtime.close()


def test_live_without_cloud_configuration_does_not_mock_or_call_cloud():
    settings = replace(Settings(), gateway=GatewaySettings(mode="live", database=":memory:"))
    app, store = make_app(settings=settings, local=MockAdapter("local"))
    with TestClient(app) as client:
        r = client.post("/v1/chat/completions", json={"model": "real", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 503
        assert not store.recent_requests()[0]["cloud_model_used"]


def test_disabled_cloud_context_route_fails_before_selector():
    settings = replace(Settings(), gateway=GatewaySettings(mode="live", database=":memory:"),
                       context=replace(Settings().context, min_input_tokens=1))
    local = MockAdapter("local")
    app, _ = make_app(settings=settings, local=local)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=context_request())
        assert response.status_code == 503
        assert local.complete_calls == 0


def test_invalid_json_schema_external_refs_and_invalid_request():
    data = local_request()
    data["response_format"]["json_schema"]["schema"]["$ref"] = "https://must-not-fetch.test/schema"
    assert decide(data, Settings()).local is False
    app, _ = make_app()
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", content="hi").status_code == 415
        assert client.post("/v1/chat/completions", json={"model": "x", "messages": []}).status_code == 400
        assert client.post("/v1/chat/completions", content='{"a": NaN}', headers={"content-type": "application/json"}).status_code == 400


def test_context_worker_invalid_selection_falls_back_to_full_cloud_payload():
    class BadSelector(MockAdapter):
        async def complete(self, payload):
            self.complete_calls += 1
            return HTTPReply(200, b'{"choices":[{"finish_reason":"stop","message":{"content":"oops"}}]}', {})
    cloud, local = MockAdapter(), BadSelector("local")
    settings = replace(Settings(), router=RouterSettings(enabled=True),
                       context=replace(Settings().context, min_input_tokens=1))
    app, store = make_app(cloud=cloud, local=local, settings=settings)
    with TestClient(app) as client:
        r = client.post("/v1/chat/completions", json=context_request())
        assert r.headers["x-gateway-route"] == "direct_cloud"
        assert r.headers["x-gateway-reason"] == "context_worker_failed_raw_fallback"
        row = store.recent_requests()[0]
        assert row["fallback_used"] and row["context_compression_ratio"] == 1
        assert row["route_source"] == "auto_router"
        assert row["route_decision_reason"] == "context_worker_failed_raw_fallback"
        assert row["context_reason"] == "context_worker_failed_raw_fallback"
        assert "gardening" in json.dumps(cloud.calls[0])
        assert cloud.complete_calls == local.complete_calls == 1


@pytest.mark.parametrize("stream", [False, True])
def test_gateway_keeps_upstream_429_and_body(stream):
    class Limited(MockAdapter):
        async def complete(self, payload):
            self.complete_calls += 1
            return HTTPReply(429, b'{"error":"limited"}', {"retry-after": "9", "content-type": "application/json"})
        async def stream(self, payload):
            self.stream_calls += 1
            async def chunks(): yield b'{"error":"limited"}'
            async def close(): pass
            return StreamReply(429, {"retry-after": "9", "content-type": "application/json"}, chunks(), close)
    cloud = Limited()
    app, store = make_app(cloud=cloud)
    with TestClient(app) as client:
        r = client.post("/v1/chat/completions", json={"model": "x", "messages": [{"role": "user", "content": "hi"}], "stream": stream})
        assert r.status_code == 429 and r.json() == {"error": "limited"}
        assert r.headers["retry-after"] == "9"
        assert cloud.stream_calls + cloud.complete_calls == 1


def test_prices_only_apply_to_the_configured_model():
    settings = Settings()
    settings = replace(settings, cloud=replace(settings.cloud, prices={"input_per_million": 1, "output_per_million": 2}))
    app, store = make_app(settings=settings)
    with TestClient(app) as client:
        for model in (settings.cloud.model, "other-model"):
            client.post("/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": "hi"}]})
        rows = store.recent_attempts()
        assert rows[0]["cost"] is None
        assert rows[1]["cost"]["amount"] > 0


def test_context_added_to_local_alias_is_not_silently_ignored():
    data = local_request()
    data["gateway_context"] = {"constraints": ["Do not change numbers."], "blocks": []}
    app, store = make_app()
    with TestClient(app) as client:
        r = client.post("/v1/chat/completions", json=data)
        assert r.headers["x-gateway-route"] == "direct_cloud"
        assert not store.recent_requests()[0]["local_model_used"]
