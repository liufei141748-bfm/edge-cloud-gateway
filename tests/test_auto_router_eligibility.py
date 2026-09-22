"""Offline regressions for the threshold-to-selector path and final reasons."""

from dataclasses import replace
import json

from fastapi.testclient import TestClient
import pytest

from edge_cloud_gateway.adapters import MockAdapter
from edge_cloud_gateway.config import ObservabilitySettings, RouterSettings, Settings
from edge_cloud_gateway.context import RawContext, filter_blocks
from edge_cloud_gateway.evaluation import estimate_input_tokens
from edge_cloud_gateway.policy import decide_route
from test_app import local_request, make_app
from test_context import context_request, standard_message_request


class SelectAllAdapter(MockAdapter):
    def _body(self, payload):
        body = super()._body(payload)
        candidates = json.loads(payload["messages"][1]["content"])["blocks"]
        body["choices"][0]["message"]["content"] = json.dumps({
            "selected_ids": [block["id"] for block in candidates],
        })
        return body


class FailingSelector(MockAdapter):
    async def complete(self, payload):
        self.complete_calls += 1
        raise TimeoutError("offline selector failure fixture")


@pytest.mark.parametrize("at_boundary", [False, True])
def test_standard_messages_at_or_below_threshold_skip_selector(at_boundary):
    payload = {"model": "adaptive", "messages": [{"role": "user", "content": "hello"}]}
    if at_boundary:
        # The estimate counts UTF-8 bytes: construct the exact public boundary.
        payload["messages"][0]["content"] = "x" * (
            len("hello") + 4 * (512 - estimate_input_tokens(payload))
        )
        assert estimate_input_tokens(payload) == 512
    cloud, local = MockAdapter("cloud"), MockAdapter("local")
    app, store = make_app(cloud=cloud, local=local)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert response.headers["x-gateway-route"] == "direct_cloud"
        row = store.recent_requests()[0]
        assert row["route_source"] == "auto_router"
        assert row["router_enabled"] is True
        assert row["router_input_tokens"] <= row["router_threshold_tokens"] == 512
        assert row["route_decision_reason"] == "auto_below_context_threshold"
        assert row["latency_local"] == 0
        assert local.complete_calls == 0
        assert cloud.calls == [payload | {"model": Settings().cloud.model}]


@pytest.mark.parametrize("stream", [False, True])
def test_long_selectable_standard_messages_call_selector_and_preserve_sources(stream):
    payload = standard_message_request() | {"stream": stream}
    if stream:
        payload["stream_options"] = {"include_usage": True}
    cloud, local = MockAdapter("cloud"), MockAdapter("local")
    settings = replace(Settings(), observability=ObservabilitySettings(save_context_snapshots=True))
    app, store = make_app(cloud=cloud, local=local, settings=settings)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert response.headers["x-gateway-route"] == "context_then_cloud"
        if stream:
            assert "[DONE]" in response.text
            assert cloud.stream_calls == 1 and cloud.complete_calls == 0
        else:
            assert cloud.complete_calls == 1 and cloud.stream_calls == 0
        assert local.complete_calls == 1
        assert local.calls[0]["response_format"]["json_schema"]["name"] == "context_selection"
        row = store.recent_requests()[0]
        assert row["router_enabled"] is True and row["route_source"] == "auto_router"
        assert row["router_input_tokens"] > row["router_threshold_tokens"] == 512
        assert row["route_decision_reason"] == "auto_at_or_above_context_threshold"
        assert row["context_reason"] == "selected_original_context"
        assert row["local_model_used"] is True and row["latency_local"] > 0
        assert row["working_input_tokens"] < row["raw_input_tokens"]
        assert row["working_context_tokens"] < row["raw_context_tokens"]
        snapshot = store.get_context(row["request_id"])
        assert snapshot["raw_context"] == payload
        assert snapshot["working_context"] == cloud.calls[-1]
        raw = RawContext.from_request(payload, Settings())
        originals = {block.id: block for block in raw.blocks}
        for entry in snapshot["package"]["relevant_context"]:
            assert entry["content"] == originals[entry["id"]].content[entry["start"]:entry["end"]]
        assert cloud.calls[-1]["messages"][0] == payload["messages"][0]
        assert "authentication design checks" in cloud.calls[-1]["messages"][-1]["content"]
        assert cloud.calls[-1]["messages"][-1]["content"].endswith("Explain authentication design.")
        assert "gateway_context" not in cloud.calls[-1]


def test_long_protected_standard_messages_choose_direct_with_specific_reason():
    payload = {"model": "adaptive", "messages": [
        {"role": "system", "content": "Preserve the supplied policy."},
        {"role": "user", "content": (
            "The timeout must remain 30 seconds. " * 100
            + "Explain the timeout policy."
        )},
    ]}
    raw = RawContext.from_request(payload, Settings())
    kept, discarded = filter_blocks(raw, Settings())
    assert estimate_input_tokens(payload) > 512
    assert any(block.optional for block in raw.blocks)
    assert not discarded and all(entry["reason"] == "protected_original" for entry in kept)
    routing = decide_route(raw.payload, Settings(), raw)
    assert (routing.route, routing.reason) == ("direct_cloud", "context_no_selectable_blocks")
    cloud, local = MockAdapter("cloud"), MockAdapter("local")
    app, store = make_app(cloud=cloud, local=local)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert response.headers["x-gateway-route"] == "direct_cloud"
        row = store.recent_requests()[0]
        assert row["router_enabled"] is True and row["route_source"] == "auto_router"
        assert row["router_input_tokens"] > 512
        assert row["route_decision_reason"] == "context_no_selectable_blocks"
        assert row["route_reason"] == "context_no_selectable_blocks"
        assert row["working_context_tokens"] == row["raw_context_tokens"]
        assert row["latency_local"] == 0 and local.complete_calls == 0
        assert cloud.calls == [payload | {"model": Settings().cloud.model}]


def test_rule_only_net_reduction_remains_eligible_without_selector_candidates():
    payload = context_request()
    payload["gateway_context"]["blocks"] = [
        block for block in payload["gateway_context"]["blocks"] if block["id"] in {"code", "big"}
    ]
    raw = RawContext.from_request(payload, Settings())
    kept, discarded = filter_blocks(raw, Settings())
    assert discarded and all(entry["reason"] != "selection_candidate" for entry in kept)
    assert decide_route(raw.payload, Settings(), raw).route == "context_then_cloud"
    cloud, local = MockAdapter("cloud"), MockAdapter("local")
    app, store = make_app(cloud=cloud, local=local)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert response.headers["x-gateway-route"] == "context_then_cloud"
        row = store.recent_requests()[0]
        assert row["route_source"] == "auto_router"
        assert row["context_reason"] == "selected_original_context"
        assert row["working_context_tokens"] < row["raw_context_tokens"]
        assert row["latency_local"] == 0 and local.complete_calls == 0
        cloud_material = json.dumps(cloud.calls[-1], ensure_ascii=False)
        assert "ValueError: invalid token" in cloud_material
        assert "def validate_token(token)" in cloud_material


@pytest.mark.parametrize("cause,expected_reason,expected_calls", [
    ("budget", "context_worker_budget_exceeded", 0),
    ("select_all", "context_no_net_reduction", 1),
    ("failure", "context_worker_failed_raw_fallback", 1),
])
def test_auto_router_final_reason_explains_unoptimized_full_request(cause, expected_reason, expected_calls):
    payload = standard_message_request()
    settings = Settings()
    if cause == "budget":
        settings = replace(settings, context=replace(settings.context, worker_max_bytes=128))
    local = (FailingSelector("local") if cause == "failure" else
             SelectAllAdapter("local") if cause == "select_all" else MockAdapter("local"))
    cloud = MockAdapter("cloud")
    app, store = make_app(settings=settings, cloud=cloud, local=local)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert response.headers["x-gateway-route"] == "direct_cloud"
        assert response.headers["x-gateway-reason"] == expected_reason
        row = store.recent_requests()[0]
        assert row["route_source"] == "auto_router" and row["router_enabled"] is True
        assert row["router_input_tokens"] > 512
        assert row["route_decision_reason"] == expected_reason
        assert row["context_reason"] == expected_reason
        assert row["route_reason"] == expected_reason
        assert row["fallback_used"] is (cause == "failure")
        assert row["working_input_tokens"] == row["raw_input_tokens"]
        assert row["working_context_tokens"] == row["raw_context_tokens"]
        assert local.complete_calls == expected_calls
        assert row["local_model_used"] is bool(expected_calls)
        assert cloud.calls == [payload | {"model": settings.cloud.model}]


@pytest.mark.parametrize("optimize", [False, True])
def test_explicit_optimize_keeps_priority_with_auto_router_disabled(optimize):
    payload = context_request()
    payload["gateway_context"]["optimize"] = optimize
    settings = replace(Settings(), router=RouterSettings(enabled=False))
    cloud, local = MockAdapter("cloud"), MockAdapter("local")
    app, store = make_app(settings=settings, cloud=cloud, local=local)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert response.headers["x-gateway-route"] == ("context_then_cloud" if optimize else "direct_cloud")
        row = store.recent_requests()[0]
        assert row["route_source"] == "explicit_optimize"
        assert row["route_decision_reason"] == f"explicit_optimize_{str(optimize).lower()}"
        assert local.complete_calls == int(optimize)


def test_explicit_optimize_retains_decision_metadata_after_selector_failure():
    payload = context_request()
    payload["gateway_context"]["optimize"] = True
    cloud, local = MockAdapter("cloud"), FailingSelector("local")
    app, store = make_app(cloud=cloud, local=local)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert response.headers["x-gateway-route"] == "direct_cloud"
        row = store.recent_requests()[0]
        assert row["route_source"] == "explicit_optimize"
        assert row["initial_route_decision_reason"] == "explicit_optimize_true"
        assert row["route_decision_reason"] == "context_worker_failed_raw_fallback"
        assert row["route_reason"] == "context_worker_failed_raw_fallback"
        assert row["fallback_used"] is True and local.complete_calls == 1
        assert cloud.calls == [RawContext.from_request(payload, Settings()).render_all(Settings())]


def test_explicit_local_json_keeps_local_route():
    cloud, local = MockAdapter("cloud"), MockAdapter("local")
    app, store = make_app(cloud=cloud, local=local)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=local_request())
        assert response.status_code == 200
        assert response.headers["x-gateway-route"] == "direct_local"
        assert local.complete_calls == 1 and cloud.complete_calls == 0
        row = store.recent_requests()[0]
        assert row["route_source"] == "explicit_model"
        assert row["route_decision_reason"] == "explicit_standalone_json"
