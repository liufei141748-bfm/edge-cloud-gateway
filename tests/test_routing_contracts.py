import json

from fastapi.testclient import TestClient

from edge_cloud_gateway.adapters import HTTPReply, MockAdapter
from edge_cloud_gateway.app import create_app
from edge_cloud_gateway.config import Settings
from edge_cloud_gateway.routing import RoutingFeatures, RoutingPolicy
from edge_cloud_gateway.policy import RuleBasedPolicy
from edge_cloud_gateway.storage import Store
from test_context import standard_message_request


def test_rule_based_policy_exposes_stable_policy_contract():
    assert isinstance(RuleBasedPolicy(), RoutingPolicy)
    fields = set(RoutingFeatures.__dataclass_fields__)
    assert fields == {
        "schema_version", "estimated_input_tokens", "block_count", "protected_block_count",
        "protected_ratio", "selectable_block_count", "candidate_tokens", "local_provider",
        "local_model", "cloud_provider", "cloud_model", "router_enabled",
    }


def test_routing_features_and_outcomes_are_recorded_without_prompt_text():
    marker = "PRIVATE_PROMPT_MUST_NOT_PERSIST_7f35"
    payload = standard_message_request()
    payload["messages"][-1]["content"] += marker
    store = Store(":memory:")
    app = create_app(Settings(), cloud=MockAdapter("cloud"), local=MockAdapter("local"), store=store)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        row = store.recent_requests()[0]
        assert row["routing_features_version"] == "routing_features_v1"
        assert row["estimated_input_tokens"] == row["router_input_tokens"]
        assert row["block_count"] > 0
        assert 0 <= row["protected_block_count"] <= row["block_count"]
        assert 0 <= row["protected_ratio"] <= 1
        assert row["selectable_block_count"] > 0 and row["candidate_tokens"] > 0
        assert row["local_provider"] == "ollama" and row["local_model"] == "local-model"
        assert row["cloud_provider"] == "openai_compatible" and row["cloud_model"] == "mock-cloud"
        assert row["local_input_tokens"] is not None and row["local_output_tokens"] is not None
        assert row["cloud_input_tokens"] is not None and row["cloud_output_tokens"] is not None
        assert row["working_context_tokens"] < row["raw_context_tokens"]
        assert row["context_compression_ratio"] < 1
        assert row["context_snapshot_saved"] is False
        assert store.get_context(row["request_id"]) is None
        persisted_metrics = json.dumps(store.recent_requests() + store.recent_attempts())
        assert marker not in persisted_metrics


def test_direct_cloud_records_no_fake_reduction_or_local_usage():
    store = Store(":memory:")
    app = create_app(Settings(), cloud=MockAdapter("cloud"), local=MockAdapter("local"), store=store)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "adaptive", "messages": [{"role": "user", "content": "short"}],
        })
        assert response.status_code == 200
        row = store.recent_requests()[0]
        assert row["route"] == "direct_cloud"
        assert row["raw_context_tokens"] == row["working_context_tokens"]
        assert row["context_compression_ratio"] == 1
        assert row["local_input_tokens"] == row["local_output_tokens"] == 0
        assert row["cloud_input_tokens"] is not None


def test_valid_openai_compatible_refusal_is_passed_through():
    body = {"id": "chatcmpl-refusal", "choices": [{
        "index": 0,
        "message": {"role": "assistant", "refusal": "Cannot help with that request."},
        "finish_reason": "stop",
    }]}

    class RefusalProvider(MockAdapter):
        async def complete(self, payload):
            return HTTPReply(200, json.dumps(body).encode(), {"content-type": "application/json"})

    app = create_app(Settings(), cloud=RefusalProvider("cloud"), store=Store(":memory:"))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "adaptive", "messages": [{"role": "user", "content": "short"}],
        })
        assert response.status_code == 200
        assert response.json() == body
