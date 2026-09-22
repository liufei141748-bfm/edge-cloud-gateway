"""Explicit, isolated A/B evaluation through the existing Gateway ASGI endpoint.

Default execution uses only scripted providers and an in-memory Store. No
production routing/selection/adapter code or user's local settings are changed.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from importlib.resources import files
import json
import hashlib
import math
import os
from pathlib import Path
import sys
import uuid

import httpx

from .adapters import HTTPReply
from .app import create_app
from .config import CloudSettings, Settings, load_settings
from .context import RawContext
from .danger_quality import assess_case
from .danger_report import compare_ab, write_reports
from .evaluation import canonical_bytes, estimate_input_tokens
from .policy import route_request
from .pricing import usage_from_response
from .storage import Store

CATEGORIES = (
    "数字阈值混淆", "否定词/禁止条件", "单位混淆", "新旧版本冲突",
    "跨文件依赖", "前后文引用/代词指代", "长日志中的关键错误行", "多个条件必须同时成立",
    "条件优先级", "代码与注释冲突", "相似实体", "关键条件藏在长文本中",
)
ANSWER_INSTRUCTION = (
    "请依据参考资料回答最后的问题。只输出一个完整 JSON 对象，遵守问题指定的字段和类型，"
    "不要增加字段、说明或 Markdown。资料不足时相应字段填 null。"
)
COST_WARNING = "即将执行真实云 API 调用，可能产生费用。"


class LiveCancelled(Exception):
    """Live work was not explicitly confirmed at the provider boundary."""


def _authorize_live(settings: Settings, case_count: int) -> str:
    key = os.environ.get(settings.cloud.api_key_env)
    if not key:
        raise ValueError("Missing configured cloud key environment variable")
    print(COST_WARNING, flush=True)
    print(
        f"本次 {case_count} 个案例，每例 A/B 各调用一次配置的云模型 "
        f"{settings.cloud.model}，共 {case_count * 2} 次；本地模型 {settings.local.model}。",
        flush=True,
    )
    if not sys.stdin.isatty():
        print("未检测到交互终端，已取消，未执行真实 API 调用。", file=sys.stderr)
        raise LiveCancelled()
    if input("确认付费测试请输入 LIVE（其他输入取消）：").strip() != "LIVE":
        raise LiveCancelled()
    return key


def load_cases(path: str | Path | None = None) -> list[dict]:
    source = Path(path) if path is not None else files("edge_cloud_gateway").joinpath("data/danger_cases.json")
    document = json.loads(source.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("cases"), list):
        raise ValueError("Invalid danger case document")
    cases = document["cases"]
    from collections import Counter
    if len(cases) != 36 or Counter(case.get("category") for case in cases) != Counter({c: 3 for c in CATEGORIES}):
        raise ValueError("Exactly 36 cases and three cases per category are required")
    ids = set()
    for case in cases:
        if (not isinstance(case.get("id"), str) or not case["id"] or case["id"] in ids
                or not isinstance(case.get("question"), str) or not case["question"].strip()
                or not isinstance(case.get("expected_answer"), dict) or not case["expected_answer"]
                or type(case.get("manual_review_needed")) is not bool
                or not isinstance(case.get("context"), list) or not case["context"]
                or not isinstance(case.get("must_keep"), list) or not case["must_keep"]):
            raise ValueError("Incomplete or duplicate danger case")
        ids.add(case["id"])
        blocks = {block["id"]: block for block in case["context"]}
        for condition in case["must_keep"]:
            block = blocks.get(condition.get("block_id"))
            text = condition.get("text")
            if block is None or not isinstance(text, str) or not text or text not in block["content"]:
                raise ValueError("must_keep must reference literal source text")
        conversation = case.get("conversation", [])
        if not isinstance(conversation, list) or any(
            not isinstance(m, dict) or set(m) != {"role", "content"}
            or m["role"] not in {"system", "developer", "user", "assistant"}
            or not isinstance(m["content"], str) for m in conversation
        ):
            raise ValueError("Invalid case conversation")
        RawContext.from_request(build_request(case, False, Settings()), Settings())
    return cases


def load_fixtures() -> dict:
    document = json.loads(files("edge_cloud_gateway").joinpath("data/danger_fixtures.json").read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("cases"), dict):
        raise ValueError("Invalid danger fixture document")
    return document["cases"]


def build_request(case: dict, optimize: bool, settings: Settings) -> dict:
    if type(optimize) is not bool:
        raise ValueError("optimize must be an explicit boolean")
    # Ground truth is never available to either model. A/B differ only here.
    return {
        "model": settings.context.alias,
        "messages": [{"role": "system", "content": ANSWER_INSTRUCTION}]
            + deepcopy(case.get("conversation", []))
            + [{"role": "user", "content": case["question"]}],
        "gateway_context": {"optimize": optimize, "blocks": deepcopy(case["context"]), "constraints": []},
        "response_format": {"type": "json_object"}, "temperature": 0,
        "max_tokens": 512, "stream": False,
    }


def evaluation_settings(base: Settings | None = None, *, live: bool = False) -> Settings:
    if live:
        if base is None or base.gateway.mode != "live" or not base.cloud.enabled:
            raise ValueError("Live evaluation requires an enabled live config")
        if not base.local.enabled:
            raise ValueError("Live evaluation requires an enabled local provider")
        settings = base
    else:
        # Intentionally ignore even a supplied live configuration in dry-run.
        settings = Settings(cloud=CloudSettings(model="fixture-cloud"))
    settings = replace(
        settings,
        gateway=replace(settings.gateway, database=":memory:"),
        cache=replace(settings.cache, enabled=False),
        observability=replace(settings.observability, save_context_snapshots=True),
        context=replace(settings.context, enabled=True, min_input_tokens=1, worker_max_bytes=12000),
        local=replace(settings.local, num_ctx=max(8192, settings.local.num_ctx)),
    )
    settings.validate()
    return settings


class ScriptedProvider:
    """Fixed offline scenarios, deliberately unrelated to model competence.

    The fixture provides answers and selected IDs; it never reads the expected
    answer. Input usage comes from the actual Gateway-rendered payload heuristic.
    """
    simulated = True

    def __init__(self, kind: str):
        self.kind = kind
        self.fixture: dict = {}
        self.arm = "A"
        self.calls: list[dict] = []

    async def complete(self, payload: dict) -> HTTPReply:
        self.calls.append(deepcopy(payload))
        if self.kind == "local":
            data = json.loads(payload["messages"][-1]["content"])
            offered = {block["id"] for block in data["blocks"]}
            content = json.dumps({"selected_ids": [identifier for identifier in self.fixture["selected_ids"]
                                                    if identifier in offered]}, ensure_ascii=False)
        else:
            content = json.dumps(self.fixture["answer_" + self.arm.lower()], ensure_ascii=False)
        incoming = estimate_input_tokens(payload)
        outgoing = math.ceil(len(content.encode("utf-8")) / 4)
        body = {
            "model": payload["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": incoming, "completion_tokens": outgoing,
                      "total_tokens": incoming + outgoing,
                      "completion_tokens_details": {"reasoning_tokens": 0}},
        }
        return HTTPReply(200, canonical_bytes(body), {"content-type": "application/json"})

    async def close(self):
        pass


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _sum_known(*values):
    return sum(values) if all(value is not None for value in values) else None


def redact(value, secrets: tuple[str, ...]):
    """Remove configured key values from response-derived artifacts as well."""
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED_SECRET]")
        return value
    if isinstance(value, dict):
        return {redact(k, secrets): redact(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, secrets) for v in value]
    return value


async def run_arm(client, runtime, case, arm, settings, *, secrets=()) -> dict:
    optimize = arm == "B"
    payload = build_request(case, optimize, settings)
    raw = RawContext.from_request(payload, settings)
    planned_route = "context_then_cloud" if optimize else "direct_cloud"
    if route_request(raw.payload, settings, raw)[0] != planned_route:
        raise ValueError("Evaluation preflight route mismatch")
    response = await client.post("/v1/chat/completions", json=payload,
                                 headers={"x-gateway-task-id": case["id"] + ":" + arm,
                                          "x-gateway-cache": "bypass"})
    request_id = response.headers.get("x-gateway-request-id")
    record = next((r for r in runtime.store.recent_requests(2) if r["request_id"] == request_id), {})
    attempts = [a for a in runtime.store.recent_attempts(4) if a["request_id"] == request_id]
    snapshot = runtime.store.get_context(request_id) if request_id else None
    try:
        body = response.json()
        body = body if isinstance(body, dict) else {}
    except ValueError:
        body = {}
    choices = body.get("choices")
    choice = choices[0] if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    answer = message.get("content") if isinstance(message.get("content"), str) else ""
    finish = choice.get("finish_reason")
    if message.get("tool_calls") or message.get("refusal"):
        finish = "invalid_answer"
    successful = response.is_success and record.get("status") == "success"
    usage = (usage_from_response(body) or {}) if successful else {}
    raw_usage = body.get("usage") if isinstance(body.get("usage"), dict) and successful else {}
    prompt, completion = usage.get("input_tokens"), usage.get("output_tokens")
    total = _count(raw_usage.get("total_tokens"))
    # An inconsistent total is unknown, never silently used to advertise savings.
    if total is not None and prompt is not None and completion is not None and total != prompt + completion:
        total = None
    reasoning = usage.get("reasoning_tokens")
    if reasoning is not None and completion is not None and reasoning > completion:
        reasoning = None
    local_attempts = [a for a in attempts if a["provider"] == "local"]
    local_in = _sum_known(*[(a.get("usage") or {}).get("input_tokens") if a["usage_complete"] else None for a in local_attempts])
    local_out = _sum_known(*[(a.get("usage") or {}).get("output_tokens") if a["usage_complete"] else None for a in local_attempts])
    result = {key: record.get(key) for key in (
        "raw_input_tokens", "working_input_tokens", "raw_payload_bytes", "working_payload_bytes",
        "context_compression_ratio", "input_usage_source", "input_estimation_method",
        "local_model_used", "cloud_model_used", "route", "route_reason", "latency_local",
        "latency_cloud", "latency_total", "fallback_used", "usage_source", "simulated",
    )}
    # Preserve gateway semantics; also identify a no-reduction raw return for
    # evaluation purposes without changing production fallback telemetry.
    result.update(
        arm=arm, optimize=optimize, planned_route=planned_route, request_id=request_id,
        request_status=record.get("status", "missing"), http_status=response.status_code,
        answer=answer, finish_reason=finish,
        prompt_tokens=prompt, completion_tokens=completion,
        cloud_input_tokens=prompt, cloud_output_tokens=completion,
        reasoning_tokens=reasoning, total_tokens=total,
        local_model=settings.local.model if record.get("local_model_used") else None,
        cloud_model=settings.cloud.model if record.get("cloud_model_used") else None,
        response_model=body.get("model") if successful else None,
        local_input_tokens=local_in, local_output_tokens=local_out,
        local_total_tokens=_sum_known(local_in, local_out),
        end_to_end_total_tokens=_sum_known(total, local_in, local_out),
        local_attempt_count=len(local_attempts),
        cloud_attempt_count=sum(a["provider"] == "cloud" for a in attempts),
        raw_returned=optimize and record.get("route") == "direct_cloud",
        package=snapshot["package"] if snapshot else None,
        working_payload=snapshot["working_context"] if snapshot else None,
        error_type=next((a["error_type"] for a in attempts if a.get("error_type")), None),
        latency_source="fixture_execution" if record.get("simulated") else "measured",
    )
    return redact(result, secrets)


def _pending_run(arm: str) -> dict:
    # Preserve a paid completed arm even when the other arm is interrupted.
    return {
        "arm": arm, "optimize": arm == "B", "answer": "", "finish_reason": None,
        "request_status": "not_run", "package": None, "working_payload": None,
        "route": None, "planned_route": "direct_cloud" if arm == "A" else "context_then_cloud",
        "local_model_used": False, "cloud_model_used": False, "fallback_used": False,
        "usage_source": "unknown", "simulated": None,
        **{field: None for field in (
            "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens",
            "raw_input_tokens", "working_input_tokens", "raw_payload_bytes", "working_payload_bytes",
            "context_compression_ratio", "local_model", "cloud_model", "latency_local", "latency_cloud",
            "latency_total", "local_input_tokens", "local_output_tokens", "local_total_tokens",
            "end_to_end_total_tokens",
        )},
    }


async def run_suite(cases: list[dict], settings: Settings, report: dict, *, live: bool = False,
                    fixtures: dict | None = None, secrets: tuple[str, ...] = ()) -> dict:
    # The API also defaults offline even if its caller provides live Settings.
    settings = evaluation_settings(settings if live else None, live=live)
    local = None if live else ScriptedProvider("local")
    cloud = None if live else ScriptedProvider("cloud")
    if not live:
        fixtures = load_fixtures() if fixtures is None else fixtures
        if any(case["id"] not in fixtures for case in cases):
            raise ValueError("Missing offline fixture")
    for case in cases:
        for optimize, expected in ((False, "direct_cloud"), (True, "context_then_cloud")):
            request = build_request(case, optimize, settings)
            raw = RawContext.from_request(request, settings)
            if route_request(raw.payload, settings, raw)[0] != expected:
                raise ValueError("Evaluation case cannot use its required route")
    if live:
        # Library callers face the same confirmation and redaction as the CLI.
        secrets = (*secrets, _authorize_live(settings, len(cases)))
    app = create_app(settings, cloud=cloud, local=local, store=Store(":memory:"))
    runtime = app.state.runtime
    report.update(mode="live" if live else "dry-run", simulated=not live)
    report["metadata"] = {
        "cases_planned": len(cases), "cloud_calls_planned": 2 * len(cases),
        "local_model": settings.local.model, "cloud_model": settings.cloud.model,
        "context_min_input_tokens": settings.context.min_input_tokens,
        "context_worker_max_bytes": settings.context.worker_max_bytes,
        "context_min_savings_ratio": settings.context.min_savings_ratio,
        "local_num_ctx": settings.local.num_ctx, "cache_enabled": False,
        "temperature": 0, "max_tokens": 512, "order": "AB / BA alternating",
        "usage_note": "estimated fixture usage; not real savings" if not live else "provider usage; missing fields remain null",
        "dataset_sha256": hashlib.sha256(canonical_bytes({"cases": cases})).hexdigest(),
    }
    report.setdefault("cases", [])
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver", trust_env=False) as client:
            for index, case in enumerate(cases):
                item = {key: deepcopy(case[key]) for key in (
                    "id", "category", "manual_review_needed", "question", "context", "expected_answer", "must_keep",
                )}
                item.update(A=_pending_run("A"), B=_pending_run("B"), completed=False)
                item["quality"] = assess_case(case, item["A"], item["B"])
                item["comparison"] = compare_ab(item["A"], item["B"])
                report["cases"].append(item)
                # Alternate pair order to reduce systematic order bias.
                for arm in (("A", "B") if index % 2 == 0 else ("B", "A")):
                    if not live:
                        local.fixture = cloud.fixture = fixtures[case["id"]]
                        local.arm = cloud.arm = arm
                    item[arm] = await run_arm(client, runtime, case, arm, settings, secrets=secrets)
                    item["quality"] = assess_case(case, item["A"], item["B"])
                    item["comparison"] = compare_ab(item["A"], item["B"])
                item["completed"] = True
    finally:
        report["cases"] = redact(report["cases"], secrets)
        await runtime.close()
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="危险案例 A/B 评测；默认离线，不产生云 API 费用。")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="离线剧本，首次只用此模式（默认）")
    mode.add_argument("--live", action="store_true", help="使用已配置的本地与云端 provider；提示费用后需输入 LIVE 确认")
    parser.add_argument("--config", default="config.local.toml", help="仅 live 读取；密钥只从指定环境变量获取")
    parser.add_argument("--output-dir", default="evaluation-private/danger-ab", help="报告父目录，每次创建独立子目录")
    args = parser.parse_args(argv)
    report = {"schema_version": 1, "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8],
              "started_at": datetime.now(timezone.utc).isoformat(), "cases": []}
    secrets = ()
    try:
        cases = load_cases()
        base = load_settings(args.config) if args.live else None
        settings = evaluation_settings(base, live=args.live)
        if args.live:
            secrets = (os.environ.get(settings.cloud.api_key_env, ""),)
        else:
            print("dry-run：离线 fixture，无网络、无需密钥；回答和 Token 是模拟数据。", flush=True)
        output = Path(args.output_dir).resolve() / report["run_id"]
        # Fail on an unusable destination before any paid work can begin.
        output.mkdir(parents=True, exist_ok=False)
        try:
            asyncio.run(run_suite(cases, settings, report, live=args.live, secrets=secrets))
        finally:
            if "mode" in report:
                report["completed"] = (len(report["cases"]) == len(cases)
                                       and all(item["completed"] for item in report["cases"]))
                report["finished_at"] = datetime.now(timezone.utc).isoformat()
                paths = write_reports(redact(report, secrets), output)
                for label, path in paths.items():
                    print(f"{label}: {redact(str(path), secrets)}")
    except LiveCancelled:
        print("已取消，未执行真实 API 调用。")
        return 2
    except KeyboardInterrupt:
        print("评测中断；已完成案例会保存在本次报告中。", file=sys.stderr)
        return 130
    except Exception as exc:
        # Never echo exception bodies/config contents, upstream errors or keys.
        print(f"评测未完成（{type(exc).__name__}）。请检查案例、配置和环境变量；未输出敏感错误详情。", file=sys.stderr)
        return 2
    print(f"完成 {len(report['cases'])} 个案例。dry-run 结果仅验证流程。" if not args.live else f"完成 {len(report['cases'])} 个真实 A/B 案例。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
