from copy import deepcopy
from dataclasses import replace
import json

import httpx
import pytest

from edge_cloud_gateway.adapters import HTTPReply
from edge_cloud_gateway.app import create_app
from edge_cloud_gateway.config import CloudSettings, GatewaySettings, Settings
from edge_cloud_gateway.context import RawContext
from edge_cloud_gateway.danger_eval import (
    COST_WARNING, CATEGORIES, LiveCancelled, ScriptedProvider, build_request, evaluation_settings,
    load_cases, load_fixtures, main, redact, run_arm, run_suite,
)
from edge_cloud_gateway.danger_quality import assess_case, check_must_keep
from edge_cloud_gateway.policy import route_request
from edge_cloud_gateway.storage import Store


def live_settings():
    return Settings(gateway=GatewaySettings(mode="live"),
                    cloud=CloudSettings(enabled=True, base_url="https://cloud.example/v1",
                                        model="cloud-model", api_key_env="CLOUD_API_KEY"))


def test_all_36_cases_have_complete_fields_and_three_per_category():
    from collections import Counter
    cases = load_cases()
    assert len(cases) == len({case["id"] for case in cases}) == 36
    assert Counter(case["category"] for case in cases) == {name: 3 for name in CATEGORIES}
    fixtures = load_fixtures()
    assert set(fixtures) == {case["id"] for case in cases}
    for case in cases:
        raw = RawContext.from_request(build_request(case, False, Settings()), Settings())
        checked = check_must_keep(case, raw.package_all())
        assert checked["retained"] == checked["total"] > 0
        assert isinstance(fixtures[case["id"]]["answer_a"], dict)
        assert isinstance(fixtures[case["id"]]["answer_b"], dict)
        assert set(fixtures[case["id"]]["selected_ids"]) <= {b["id"] for b in case["context"]}


@pytest.mark.parametrize("index", range(36))
def test_a_b_only_switch_optimize_and_force_planned_routes(index):
    case = load_cases()[index]
    original = deepcopy(case)
    settings = evaluation_settings()
    a, b = build_request(case, False, settings), build_request(case, True, settings)
    assert a["gateway_context"]["optimize"] is False
    assert b["gateway_context"]["optimize"] is True
    a["gateway_context"]["optimize"] = True
    assert a == b
    assert "must_keep" not in b and "expected_answer" not in b
    assert b["gateway_context"]["constraints"] == []
    assert b["messages"][-1]["content"] == case["question"]
    for optimize, route in ((False, "direct_cloud"), (True, "context_then_cloud")):
        request = build_request(case, optimize, settings)
        raw = RawContext.from_request(request, settings)
        actual = route_request(raw.payload, settings, raw)
        assert actual[0] == route
        assert actual[1] == ("explicit_optimize_true" if optimize else "explicit_optimize_false")
    assert case == original


def test_dry_run_ignores_live_settings_and_preserves_original_config():
    original = live_settings()
    evaluated = evaluation_settings(original)
    assert evaluated.gateway.mode == "mock"
    assert evaluated.cloud.enabled is False
    assert evaluated.gateway.database == ":memory:"
    assert original.gateway.mode == "live" and original.gateway.database == "data/gateway.sqlite3"
    assert original.context.min_input_tokens == 512


@pytest.mark.asyncio
async def test_suite_is_network_free_and_has_72_cloud_and_36_local_attempts(monkeypatch):
    import edge_cloud_gateway.app as module
    def forbidden(*args, **kwargs):
        raise AssertionError("Real adapter must not be constructed in dry-run")
    monkeypatch.setattr(module, "HttpCloudAdapter", forbidden)
    monkeypatch.setattr(module, "OllamaAdapter", forbidden)
    report = {"schema_version": 1, "run_id": "offline-test", "cases": []}
    await run_suite(load_cases(), live_settings(), report)
    assert report["simulated"] is True and report["mode"] == "dry-run"
    assert len(report["cases"]) == 36
    for item in report["cases"]:
        a, b = item["A"], item["B"]
        assert a["optimize"] is False and b["optimize"] is True
        assert a["route"] == "direct_cloud" and not a["local_model_used"]
        assert a["local_attempt_count"] == 0 and b["local_attempt_count"] == 1
        assert a["cloud_attempt_count"] == b["cloud_attempt_count"] == 1
        assert b["local_model"] == "local-model"
        assert a["cloud_model"] == b["cloud_model"] == "fixture-cloud"
        assert a["raw_input_tokens"] == b["raw_input_tokens"]
        assert a["working_input_tokens"] == a["raw_input_tokens"]
        # Fixed wrong answers remain wrong; source-retention regressions must
        # fail independently without modifying fixtures or the quality judge.
        assert item["quality"]["b_must_keep"]["missing"] == [], item["id"]
        assert item["quality"]["b_must_keep"]["retained"] == len(item["must_keep"])
        assert b["working_payload_bytes"] < b["raw_payload_bytes"]
        for run in (a, b):
            for field in ("answer", "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens",
                          "raw_payload_bytes", "working_payload_bytes", "context_compression_ratio",
                          "latency_local", "latency_cloud", "latency_total", "fallback_used", "package"):
                assert field in run
            assert run["request_status"] == "success"
            assert run["usage_source"] == "estimated"
            assert run["total_tokens"] == run["prompt_tokens"] + run["completion_tokens"]
            assert run["end_to_end_total_tokens"] == run["total_tokens"] + run["local_total_tokens"]
    assert {item["quality"]["status"] for item in report["cases"]} == {"PASS", "FAIL", "MANUAL_REVIEW"}


def test_default_cli_ignores_even_missing_live_config_and_generates_reports(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "does-not-exist.toml"), "--output-dir", str(tmp_path)]) == 0
    directories = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(directories) == 1
    directory = directories[0]
    assert {p.name for p in directory.iterdir()} == {"details.json", "details.csv", "summary.md"}
    saved = json.loads((directory / "details.json").read_text())
    assert saved["simulated"] is True and len(saved["cases"]) == 36
    assert COST_WARNING not in capsys.readouterr().out


@pytest.mark.parametrize("choice,tty", [("NO", True), ("", False)])
def test_live_warns_and_cancelled_or_noninteractive_never_calls_api(monkeypatch, capsys, tmp_path, choice, tty):
    import edge_cloud_gateway.danger_eval as module
    monkeypatch.setenv("CLOUD_API_KEY", "private-unit-test-value")
    monkeypatch.setattr(module, "load_settings", lambda path: live_settings())
    monkeypatch.setattr(module.sys.stdin, "isatty", lambda: tty)
    monkeypatch.setattr("builtins.input", lambda prompt: choice)
    def forbidden(*args, **kwargs):
        raise AssertionError("Cancelled live invocation must not start a suite")
    monkeypatch.setattr(module, "create_app", forbidden)
    assert main(["--live", "--output-dir", str(tmp_path)]) == 2
    output = capsys.readouterr()
    assert COST_WARNING in output.out and "72" in output.out
    assert "private-unit-test-value" not in output.out + output.err
    assert not any(path.is_file() for path in tmp_path.rglob("*"))


def test_live_requires_enabled_cloud_and_local_providers():
    with pytest.raises(ValueError):
        evaluation_settings(Settings(), live=True)
    with pytest.raises(ValueError):
        evaluation_settings(replace(live_settings(), local=replace(Settings().local, enabled=False)), live=True)
    configured = evaluation_settings(
        replace(live_settings(), cloud=replace(live_settings().cloud, base_url="https://other.example/v1"),
                local=replace(Settings().local, model="another-local-model")),
        live=True,
    )
    assert configured.cloud.base_url == "https://other.example/v1"
    assert configured.local.model == "another-local-model"


@pytest.mark.asyncio
async def test_worker_failure_keeps_raw_and_is_not_a_successful_b_trial():
    case = load_cases()[0]
    settings = evaluation_settings()
    cloud = ScriptedProvider("cloud")
    cloud.fixture = load_fixtures()[case["id"]]
    class FailingLocal:
        simulated = True
        async def complete(self, payload):
            raise TimeoutError()
        async def close(self):
            pass
    app = create_app(settings, cloud=cloud, local=FailingLocal(), store=Store(":memory:"))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            a = await run_arm(client, app.state.runtime, case, "A", settings)
            b = await run_arm(client, app.state.runtime, case, "B", settings)
        assert b["fallback_used"] is True and b["raw_returned"] is True
        assert b["route"] == "direct_cloud" and b["local_attempt_count"] == 1
        assert b["working_payload"] == a["working_payload"]
        assert b["local_total_tokens"] is None
        assert assess_case(case, a, b)["status"] == "MANUAL_REVIEW"
    finally:
        await app.state.runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("usage,expected_total,reasoning", [
    ({"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130,
      "completion_tokens_details": {"reasoning_tokens": 20}}, 130, 20),
    ({"prompt_tokens": 100, "completion_tokens": 30}, None, None),
    ({"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 999}, None, None),
    ({"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130,
      "completion_tokens_details": {"reasoning_tokens": 40}}, 130, None),
    ({}, None, None),
])
async def test_usage_is_actual_nullable_and_reasoning_never_double_counted(usage, expected_total, reasoning):
    case, settings = load_cases()[0], evaluation_settings()
    class RecordedCloud:
        simulated = False
        async def complete(self, payload):
            return HTTPReply(200, json.dumps({"model": "deepseek-chat", "usage": usage,
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(case["expected_answer"])}}]}).encode(), {})
        async def close(self):
            pass
    app = create_app(settings, cloud=RecordedCloud(), local=ScriptedProvider("local"), store=Store(":memory:"))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            a = await run_arm(client, app.state.runtime, case, "A", settings)
        assert a["total_tokens"] == expected_total
        assert a["reasoning_tokens"] == reasoning
        assert a["end_to_end_total_tokens"] == expected_total
        if usage:
            assert a["usage_source"] == "actual" and not a["simulated"]
    finally:
        await app.state.runtime.close()


def test_redaction_covers_nested_values_and_keys():
    assert redact({"secret-value": ["echo secret-value"]}, ("secret-value",)) == {
        "[REDACTED_SECRET]": ["echo [REDACTED_SECRET]"]}


@pytest.mark.asyncio
async def test_library_live_cannot_bypass_confirmation(monkeypatch, capsys):
    import edge_cloud_gateway.danger_eval as module
    monkeypatch.setenv("CLOUD_API_KEY", "private-unit-test-value")
    monkeypatch.setattr(module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "cancel")
    with pytest.raises(LiveCancelled):
        await run_suite(load_cases()[:1], live_settings(), {}, live=True)
    out = capsys.readouterr().out
    assert COST_WARNING in out and "private-unit-test-value" not in out


@pytest.mark.asyncio
async def test_interrupted_b_preserves_completed_a_and_unknown_b(monkeypatch):
    import edge_cloud_gateway.danger_eval as module
    real_run_arm = module.run_arm
    async def interrupted(*args, **kwargs):
        if args[3] == "B":
            raise InterruptedError()
        return await real_run_arm(*args, **kwargs)
    monkeypatch.setattr(module, "run_arm", interrupted)
    report = {"cases": []}
    with pytest.raises(InterruptedError):
        await run_suite(load_cases()[:1], Settings(), report)
    case = report["cases"][0]
    assert case["A"]["request_status"] == "success" and case["A"]["total_tokens"] > 0
    assert case["B"]["request_status"] == "not_run" and case["B"]["total_tokens"] is None
    assert case["completed"] is False and case["quality"]["status"] == "MANUAL_REVIEW"
    assert case["comparison"]["cloud_input_token_reduction"] is None


@pytest.mark.asyncio
async def test_conversation_preserved_in_both_rendered_cloud_payloads():
    case = next(c for c in load_cases() if c["id"] == "reference_02")
    report = {"cases": []}
    await run_suite([case], Settings(), report)
    for arm in ("A", "B"):
        messages = report["cases"][0][arm]["working_payload"]["messages"]
        assert messages[1:1 + len(case["conversation"])] == case["conversation"]
        assert "相应字段填 null" in messages[0]["content"]


@pytest.mark.asyncio
async def test_confirmed_live_library_auto_redacts_key_with_injected_offline_gateway(monkeypatch, capsys):
    import edge_cloud_gateway.danger_eval as module
    monkeypatch.setenv("CLOUD_API_KEY", "unit-secret-value")
    monkeypatch.setattr(module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "LIVE")
    case = load_cases()[0]
    def offline_app(settings, **kwargs):
        assert COST_WARNING in capsys.readouterr().out
        local, cloud = ScriptedProvider("local"), ScriptedProvider("cloud")
        local.fixture = cloud.fixture = {"selected_ids": ["focus"],
                                        "answer_a": {"echo": "unit-secret-value"},
                                        "answer_b": {"echo": "unit-secret-value"}}
        return create_app(evaluation_settings(), local=local, cloud=cloud, store=kwargs["store"])
    monkeypatch.setattr(module, "create_app", offline_app)
    report = {"cases": []}
    await run_suite([case], live_settings(), report, live=True)
    encoded = json.dumps(report)
    assert "unit-secret-value" not in encoded and "[REDACTED_SECRET]" in encoded


def test_unusable_report_destination_fails_before_starting_suite(monkeypatch, tmp_path):
    import edge_cloud_gateway.danger_eval as module
    destination = tmp_path / "a-file"
    destination.write_text("existing file")
    def forbidden(*args, **kwargs):
        raise AssertionError("Must check output destination before any API work")
    monkeypatch.setattr(module, "run_suite", forbidden)
    assert main(["--output-dir", str(destination)]) == 2
    assert destination.read_text() == "existing file"
