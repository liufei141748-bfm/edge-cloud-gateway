from copy import deepcopy
from dataclasses import replace
import json

import pytest

from edge_cloud_gateway.adapters import MockAdapter, HTTPReply
from edge_cloud_gateway.config import Settings, ContextSettings, RouterSettings
from edge_cloud_gateway.context import RawContext, NeedContext, build_working_context, filter_blocks
from edge_cloud_gateway.evaluation import compare_runs, estimate_input_tokens, input_comparison
from edge_cloud_gateway.policy import decide_route, route_request


def context_request():
    return {
        "model": "adaptive", "messages": [
            {"role": "system", "content": "请严格保留约束，不能泄露内部信息。"},
            {"role": "user", "content": "Explain authentication design."},
        ],
        "gateway_context": {"constraints": ["Do not change validate_token; timeout must remain 30 seconds."], "blocks": [
            {"id": "code", "source": "src/auth.py", "kind": "code", "optional": True,
             "content": "def validate_token(token):\n    return token is not None\n"},
            {"id": "auth", "source": "auth-notes", "kind": "text", "optional": True,
             "content": "authentication design checks token validity and session expiry"},
            {"id": "garden", "source": "unrelated-notes", "kind": "text", "optional": True,
             "content": "gardening discusses plants and soil " * 18},
            {"id": "big", "source": "long-unrelated-log", "kind": "log", "optional": True,
             "content": "background heartbeat normal\n" * 90 + "Traceback ERROR /src/auth.py:42\nValueError: invalid token\n" + "background heartbeat normal\n" * 90},
            {"id": "empty", "source": "empty", "kind": "text", "optional": True, "content": "  \n"},
        ]},
    }


@pytest.mark.asyncio
async def test_context_worker_selects_ids_and_cloud_uses_verbatim_sources():
    settings = Settings()
    original = context_request()
    unchanged = deepcopy(original)
    raw = RawContext.from_request(original, settings)
    worker = MockAdapter("local")
    working = await build_working_context(raw, settings, worker.complete)
    assert original == unchanged
    assert working.optimized
    assert worker.complete_calls == 1
    kept = working.package["relevant_context"]
    assert {x["id"] for x in kept} >= {"code", "auth", "big"}
    assert "garden" not in {x["id"] for x in kept}
    sources = {block.id: block for block in raw.blocks}
    for entry in kept:
        assert entry["content"] == sources[entry["id"]].content[entry["start"]:entry["end"]]
    sent = json.dumps(working.payload, ensure_ascii=False)
    assert "Traceback ERROR /src/auth.py:42" in sent
    assert "ValueError: invalid token" in sent
    assert "timeout must remain 30 seconds" in sent
    assert working.payload["messages"][0] == original["messages"][0]
    assert working.payload["messages"][-1] == original["messages"][-1]
    assert working.metrics["context_compression_ratio"] < 1
    assert working.metrics["input_usage_source"] == "estimated"
    assert working.package["compressed_context"] == []
    assert raw.resolve(NeedContext("symbol", "validate_token"))[0]["content"] == sources["code"].content


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["invalid_json", "unknown_id", "timeout", "rewritten_content"])
async def test_worker_failure_restores_all_raw_context(failure):
    raw = RawContext.from_request(context_request(), Settings())
    async def worker(payload):
        if failure == "timeout":
            raise TimeoutError()
        value = {"selected_ids": ["not-in-request"]} if failure == "unknown_id" else {"selected_ids": [], "content": "rewritten"}
        content = "oops" if failure == "invalid_json" else json.dumps(value)
        return HTTPReply(200, json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": content}}]}).encode(), {})
    working = await build_working_context(raw, Settings(), worker)
    assert not working.optimized
    assert working.reason == "context_worker_failed_raw_fallback"
    assert working.payload == raw.render_all(Settings())
    assert working.metrics["context_compression_ratio"] == 1


@pytest.mark.parametrize("content", ["amount=1000", "不要删除", "不允许删除备份。", "未经批准", "无权限", "勿泄露", "never delete", "no deletion", "without consent", "unless authorized", "don't delete", "shouldn’t delete", "src/auth.py", "def validate_token(x):", "class User:", "```python\na = 1\n```", '{"role":"admin"}', "@@ -1 +1 @@"])
def test_critical_content_not_offered_for_model_discard(content):
    data = context_request()
    data["gateway_context"]["blocks"] = [{"id": "critical", "source": "x", "kind": "text", "content": content, "optional": True}]
    kept, dropped = filter_blocks(RawContext.from_request(data, Settings()), Settings())
    assert dropped == []
    assert kept[0]["content"] == content
    assert kept[0]["reason"] == "protected_original"


def test_dedup_requires_same_source_and_preserves_required_blocks():
    data = context_request()
    base = {"id": "a", "source": "same", "kind": "text", "content": "identical prose", "optional": True}
    data["gateway_context"]["blocks"] = [base, base | {"id": "b"}, base | {"id": "c", "source": "different"}, base | {"id": "d", "optional": False}]
    kept, dropped = filter_blocks(RawContext.from_request(data, Settings()), Settings())
    assert [x["id"] for x in kept] == ["a", "c", "d"]
    assert dropped[0]["id"] == "b"


def test_router_handles_three_paths_and_protects_tool_chain():
    data = context_request()
    raw = RawContext.from_request(data, Settings())
    assert route_request(raw.payload, Settings(), raw)[0] == "context_then_cloud"
    data["tools"] = [{"type": "function", "function": {"name": "run"}}]
    raw = RawContext.from_request(data, Settings())
    assert route_request(raw.payload, Settings(), raw)[:2] == ("direct_cloud", "tool_chain_protected")
    assert raw.render_all(Settings())["tools"] == data["tools"]
    small = {"model": "any-model", "messages": [{"role": "user", "content": "hello"}]}
    raw = RawContext.from_request(small, Settings())
    assert route_request(small, Settings(), raw)[0] == "direct_cloud"


def test_auto_router_is_default_threshold_based_and_explicit_optimize_wins():
    data = context_request()
    default_raw = RawContext.from_request(data, Settings())
    observed = estimate_input_tokens(default_raw.render_all(Settings()))

    default = decide_route(default_raw.payload, Settings(), default_raw)
    assert default.route == "context_then_cloud"
    assert default.source == "auto_router"
    assert default.reason == "auto_at_or_above_context_threshold"

    auto = Settings()
    at_threshold = replace(auto, context=replace(auto.context, min_input_tokens=observed - 1))
    raw = RawContext.from_request(data, at_threshold)
    routing = decide_route(raw.payload, at_threshold, raw)
    assert (routing.route, routing.reason, routing.source) == (
        "context_then_cloud", "auto_at_or_above_context_threshold", "auto_router",
    )
    assert routing.input_tokens == observed
    assert routing.threshold_tokens == observed - 1

    below_threshold = replace(auto, context=replace(auto.context, min_input_tokens=observed))
    raw = RawContext.from_request(data, below_threshold)
    routing = decide_route(raw.payload, below_threshold, raw)
    assert (routing.route, routing.reason) == ("direct_cloud", "auto_below_context_threshold")

    disabled = replace(auto, router=RouterSettings(enabled=False))
    raw = RawContext.from_request(data, disabled)
    routing = decide_route(raw.payload, disabled, raw)
    assert (routing.route, routing.reason, routing.source) == (
        "direct_cloud", "auto_router_disabled", "auto_router",
    )
    assert routing.input_tokens == observed

    explicit_false = deepcopy(data)
    explicit_false["gateway_context"]["optimize"] = False
    explicit_settings = replace(at_threshold, router=RouterSettings(enabled=False))
    raw = RawContext.from_request(explicit_false, explicit_settings)
    routing = decide_route(raw.payload, explicit_settings, raw)
    assert (routing.route, routing.reason, routing.source) == (
        "direct_cloud", "explicit_optimize_false", "explicit_optimize",
    )

    explicit_true = deepcopy(data)
    explicit_true["gateway_context"]["optimize"] = True
    raw = RawContext.from_request(explicit_true, explicit_settings)
    routing = decide_route(raw.payload, explicit_settings, raw)
    assert (routing.route, routing.reason, routing.source) == (
        "context_then_cloud", "explicit_optimize_true", "explicit_optimize",
    )


@pytest.mark.asyncio
async def test_worker_budget_preserves_unseen_full_blocks():
    data = context_request()
    data["gateway_context"]["blocks"] = [{"id": "long", "source": "x", "kind": "text", "content": "long prose " * 1000, "optional": True}]
    raw = RawContext.from_request(data, Settings())
    worker = MockAdapter("local")
    working = await build_working_context(raw, Settings(), worker.complete)
    assert worker.complete_calls == 0
    assert working.payload == raw.render_all(Settings())


@pytest.mark.asyncio
async def test_selector_sees_complete_conversation_and_output_requirements():
    original = context_request()
    original["messages"] = [
        {"role": "system", "content": "When asked about it, discuss bluegreen release."},
        {"role": "developer", "content": "Use the supplied release documentation."},
        {"role": "user", "content": "We are comparing release strategies."},
        {"role": "assistant", "content": "The bluegreen approach is our focus."},
        {"role": "user", "content": "Explain it."},
    ]
    original["response_format"] = {"type": "json_object"}
    original["gateway_context"]["blocks"][1]["content"] = "bluegreen release preserves service continuity"
    mock = MockAdapter("local")

    async def worker(payload):
        data = json.loads(payload["messages"][1]["content"])
        assert data["conversation"] == original["messages"]
        assert data["response_format"] == original["response_format"]
        return await mock.complete(payload)

    working = await build_working_context(RawContext.from_request(original, Settings()), Settings(), worker)
    assert mock.complete_calls == 1
    assert "auth" in {entry["id"] for entry in working.package["relevant_context"]}


@pytest.mark.asyncio
async def test_oversized_conversation_is_never_truncated_for_worker():
    original = context_request()
    original["messages"][0]["content"] = "Complete conversation context must remain visible. " * 100
    mock = MockAdapter("local")
    settings = replace(Settings(), context=replace(Settings().context, worker_max_bytes=3072))
    working = await build_working_context(RawContext.from_request(original, settings), settings, mock.complete)
    assert mock.complete_calls == 0
    assert {"auth", "garden"} <= {entry["id"] for entry in working.package["relevant_context"]}
    assert working.payload["messages"][0] == original["messages"][0]


def standard_message_request():
    return {"model": "adaptive", "messages": [
        {"role": "system", "content": "Answer only from relevant material."},
        {"role": "user", "content": (
            "gardening plants and soil are unrelated background prose. " * 50
            + "\n\nauthentication design checks token validity and session expiry."
            + "\n\nExplain authentication design."
        )},
    ]}


@pytest.mark.asyncio
async def test_standard_messages_are_literal_selectable_blocks_without_custom_envelope():
    settings = replace(Settings(), context=replace(Settings().context, min_input_tokens=1))
    original = standard_message_request()
    raw = RawContext.from_request(original, settings)
    routing = decide_route(raw.payload, settings, raw)
    assert raw.messages_derived and raw.has_context and not raw.optimize_explicit
    assert routing.route == "context_then_cloud"
    assert routing.source == "auto_router" and routing.input_tokens > routing.threshold_tokens

    worker = MockAdapter("local")
    working = await build_working_context(raw, settings, worker.complete)
    assert working.optimized and worker.complete_calls == 1
    assert working.metrics["working_input_tokens"] < working.metrics["raw_input_tokens"]
    assert working.payload["messages"][0] == original["messages"][0]
    assert working.payload["messages"][-1]["content"].endswith("Explain authentication design.")
    assert "authentication design checks" in working.payload["messages"][-1]["content"]
    assert "gateway_context" not in working.payload


def test_standard_adaptive_short_tool_and_nonadaptive_routes_are_conservative():
    short = {"model": "adaptive", "messages": [{"role": "user", "content": "hello"}]}
    raw = RawContext.from_request(short, Settings())
    routing = decide_route(raw.payload, Settings(), raw)
    assert (routing.route, routing.reason, routing.source) == (
        "direct_cloud", "auto_below_context_threshold", "auto_router",
    )
    assert routing.input_tokens is not None

    with_tools = standard_message_request() | {"tools": [{"type": "function", "function": {"name": "run"}}]}
    raw = RawContext.from_request(with_tools, Settings())
    routing = decide_route(raw.payload, Settings(), raw)
    assert (routing.route, routing.reason) == ("direct_cloud", "tool_chain_protected")
    assert routing.input_tokens is not None

    cloud = {"model": "other-cloud", "messages": short["messages"]}
    raw = RawContext.from_request(cloud, Settings())
    routing = decide_route(raw.payload, Settings(), raw)
    assert routing.route == "direct_cloud" and routing.source == "explicit_model"
    assert routing.input_tokens is not None
    assert routing.features is not None


def test_invalid_context_and_internal_retrieval_contract():
    data = context_request()
    data["gateway_context"]["blocks"].append(data["gateway_context"]["blocks"][0])
    with pytest.raises(ValueError):
        RawContext.from_request(data, Settings())
    with pytest.raises(ValueError):
        RawContext.from_request({"gateway_context": {"blocks": "bad"}}, Settings())
    raw = RawContext.from_request(context_request(), Settings())
    with pytest.raises(ValueError):
        raw.resolve(NeedContext("execute", "shell"))


def test_comparison_never_calls_mock_a_real_saving():
    baseline = {"cloud_input_tokens": 100, "cloud_output_tokens": 10, "usage_source": "actual", "simulated": False}
    improved = baseline | {"cloud_input_tokens": 50}
    assert compare_runs(baseline, improved, quality_passed=True)["quality_preserving_savings_verified"]
    assert compare_runs(baseline, improved)["quality_preserving_savings_verified"] is False
    assert compare_runs(baseline, improved | {"simulated": True})["cloud_token_reduction"] is None


def test_complete_traceback_and_structured_log_are_preserved():
    data = context_request()
    trace = "Traceback (most recent call last):\n" + "    continuation without markers\n" * 20 + "ValueError: invalid token\n"
    text = "noise\n" * 50 + trace + "noise\n" * 50
    data["gateway_context"]["blocks"] = [{"id": "trace", "source": "log", "kind": "log", "content": text, "optional": True}]
    raw = RawContext.from_request(data, Settings())
    kept, _ = filter_blocks(raw, Settings())
    assert trace in "".join(x["content"] for x in kept)
    structured = "noise\n" * 60 + '{"message":"keep unchanged"}'
    data["gateway_context"]["blocks"][0]["content"] = structured
    kept, discarded = filter_blocks(RawContext.from_request(data, Settings()), Settings())
    assert kept[0]["content"] == structured and not discarded
