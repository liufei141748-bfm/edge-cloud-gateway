"""Deterministic comparisons and Chinese reports for the danger A/B suite.

This module never calls a model or a network service. Unknown measurements stay
unknown, and simulated measurements cannot support a production recommendation.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def _number(value: Any) -> float | int | None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        return None
    return value


def _delta(before: Any, after: Any) -> tuple[float | int | None, float | None]:
    before, after = _number(before), _number(after)
    if before is None or after is None:
        return None, None
    return after - before, (after - before) / before if before else None


def compare_ab(a: dict, b: dict) -> dict:
    """Compare like measurement sources; reasoning tokens are already in totals."""
    actual = all(run.get("usage_source") == "actual" and run.get("simulated") is False
                 for run in (a, b))
    estimated = all(run.get("usage_source") == "estimated" and run.get("simulated") is True
                    for run in (a, b))
    comparable = actual or estimated
    latency_comparable = (type(a.get("simulated")) is bool
                          and a.get("simulated") is b.get("simulated"))
    prompt_a, prompt_b = _number(a.get("prompt_tokens")), _number(b.get("prompt_tokens"))
    total_delta, total_ratio = _delta(a.get("total_tokens"), b.get("total_tokens"))
    latency_delta, latency_ratio = _delta(a.get("latency_total"), b.get("latency_total"))
    end_delta, end_ratio = _delta(a.get("end_to_end_total_tokens"), b.get("end_to_end_total_tokens"))
    return {
        "usage_comparable": comparable,
        "comparison_basis": "actual" if actual else "simulated_estimate" if estimated else "unavailable",
        "cloud_input_token_reduction": 1 - prompt_b / prompt_a
        if comparable and prompt_a and prompt_b is not None else None,
        "total_tokens_delta": total_delta if comparable else None,
        "total_tokens_change_ratio": total_ratio if comparable else None,
        "end_to_end_total_tokens_delta": end_delta if comparable else None,
        "end_to_end_total_tokens_change_ratio": end_ratio if comparable else None,
        "latency_comparable": latency_comparable,
        "latency_delta_ms": latency_delta if latency_comparable else None,
        "latency_change_ratio": latency_ratio if latency_comparable else None,
        "context_compression_ratio": _number(b.get("context_compression_ratio")),
    }


def _stats(values: list[float | int | None]) -> dict:
    values = sorted(value for value in values
                    if type(value) in (int, float) and math.isfinite(value))

    def percentile(probability: float) -> float | None:
        if not values:
            return None
        position = (len(values) - 1) * probability
        lower, upper = math.floor(position), math.ceil(position)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return {"count": len(values), "mean": mean(values) if values else None,
            "p50": percentile(0.50), "p95": percentile(0.95)}


def _missing(case: dict) -> list:
    missing = case.get("quality", {}).get("b_must_keep", {}).get("missing")
    return missing if isinstance(missing, list) else []


def _constraint_totals(cases: list[dict]) -> dict:
    retained = known_total = unknown_total = unknown_cases = 0
    for case in cases:
        result = case.get("quality", {}).get("b_must_keep", {})
        count, total = result.get("retained"), result.get("total")
        if type(count) is int and type(total) is int and 0 <= count <= total:
            retained += count
            known_total += total
        else:
            unknown_cases += 1
            unknown_total += total if type(total) is int and total >= 0 else len(case.get("must_keep", []))
    return {"retained": retained, "known_total": known_total,
            "unknown_total": unknown_total, "unknown_cases": unknown_cases,
            "rate": retained / known_total if known_total else None}


def _b_task_status(case: dict) -> str:
    answer = case.get("quality", {}).get("b_answer", {})
    if answer.get("status") == "FAIL" or answer.get("correct") is False:
        return "FAIL"
    if case.get("manual_review_needed") or answer.get("status") == "MANUAL_REVIEW" or answer.get("correct") is None:
        return "MANUAL_REVIEW"
    if answer.get("correct") is True and answer.get("status") == "PASS":
        return "PASS"
    return "FAIL"


def _recommendation(simulated: bool, counts: Counter, unknown: int, total: int) -> str:
    if simulated:
        return "仅完成离线演练，不能据此建议生产启用；需经费用确认后进行真实 A/B 测试。"
    if not total:
        return "没有案例，无法判断是否适合启用。"
    if counts["FAIL"]:
        return "暂不建议启用；先修复失败案例及错误删除的关键条件，再重新验证。"
    if counts["MANUAL_REVIEW"] or unknown:
        return "暂不建议启用；先完成人工复核并补齐未知的关键约束检查。"
    return "可扩大样本继续验证；本套件每类仅 3 例，不足以建议生产启用。"


def _summarize_group(cases: list[dict], simulated: bool) -> dict:
    total = len(cases)
    statuses = Counter(case.get("quality", {}).get("status", "MANUAL_REVIEW") for case in cases)
    # An unrecognized status must not silently become a pass or disappear.
    statuses["MANUAL_REVIEW"] += sum(value for key, value in statuses.items()
                                    if key not in {"PASS", "FAIL", "MANUAL_REVIEW"})
    b_statuses = Counter(_b_task_status(case) for case in cases)
    constraints = _constraint_totals(cases)
    comparisons = [compare_ab(case.get("A", {}), case.get("B", {})) for case in cases]
    return {
        "total_cases": total,
        "status_counts": {key: statuses[key] for key in ("PASS", "FAIL", "MANUAL_REVIEW")},
        "b_task_pass_count": b_statuses["PASS"],
        "b_task_fail_count": b_statuses["FAIL"],
        "b_task_manual_review_count": b_statuses["MANUAL_REVIEW"],
        "b_task_pass_rate": b_statuses["PASS"] / total if total else None,
        "danger_case_pass_rate": statuses["PASS"] / total if total else None,
        "danger_case_failure_rate": statuses["FAIL"] / total if total else None,
        "constraint_retention": constraints,
        "cloud_input_token_reduction": _stats([item["cloud_input_token_reduction"] for item in comparisons]),
        "latency_delta_ms": _stats([item["latency_delta_ms"] for item in comparisons]),
        "enablement_recommendation": _recommendation(simulated, statuses, constraints["unknown_cases"], total),
    }


def summarize(cases: list[dict], *, simulated: bool) -> dict:
    summary = _summarize_group(cases, simulated)
    summary["simulated"] = simulated
    summary["categories"] = {
        category: _summarize_group([case for case in cases if (case.get("category") or "未分类") == category], simulated)
        for category in sorted({case.get("category") or "未分类" for case in cases})
    }
    summary["failures"] = [
        {"id": case.get("id"), "category": case.get("category"),
         "reason": case.get("quality", {}).get("reason"),
         "missing_constraints": _missing(case),
         "selection_omission_evidence": case.get("quality", {}).get("selection_omission_evidence")}
        for case in cases if case.get("quality", {}).get("status") == "FAIL"
    ]
    summary["manual_reviews"] = [
        {"id": case.get("id"), "category": case.get("category"),
         "reason": case.get("quality", {}).get("reason"), "missing_constraints": _missing(case)}
        for case in cases if case.get("quality", {}).get("status") not in {"PASS", "FAIL"}
    ]
    summary["deleted_constraints"] = [
        {"id": case.get("id"), "category": case.get("category"), "missing_constraints": _missing(case),
         "discard_reasons": sorted({entry.get("reason", "unknown")
             for entry in (case.get("B", {}).get("package") or {}).get("discarded_context", [])
             if entry.get("id") in {rule.get("block_id") for rule in _missing(case)}})}
        for case in cases if _missing(case)
    ]
    return summary


_OMIT_KEYS = {"headers", "requestheaders", "responseheaders", "config", "settings", "credentials"}
_SECRET_KEYS = {"apikey", "authorization", "proxyauthorization", "password", "secret", "accesstoken", "refreshtoken"}


def _safe_report(value: Any) -> Any:
    """Defensive filtering, in addition to the runner's value redaction."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("_", "").replace("-", "")
            if normalized in _OMIT_KEYS or normalized in _SECRET_KEYS or normalized.endswith("apikey"):
                continue
            result[key] = _safe_report(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_safe_report(item) for item in value]
    if type(value) is float and not math.isfinite(value):
        return None
    return value


def _cell(value: Any) -> str | int | float | bool:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False, allow_nan=False)
    # Keep report text safe when a beginner opens the CSV in a spreadsheet.
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


_RUN_COLUMNS = (
    "answer", "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens",
    "raw_input_tokens", "working_input_tokens", "raw_payload_bytes", "working_payload_bytes",
    "context_compression_ratio", "local_model_used", "cloud_model_used", "local_model", "cloud_model",
    "route", "planned_route", "latency_local", "latency_cloud", "latency_total", "fallback_used",
    "usage_source", "simulated", "finish_reason", "request_status", "error_type",
    "local_input_tokens", "local_output_tokens", "local_total_tokens", "end_to_end_total_tokens",
)


def _csv_rows(cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        quality = case.get("quality", {})
        comparison = compare_ab(case.get("A", {}), case.get("B", {}))
        for group in ("A", "B"):
            run = case.get(group, {})
            answer_quality = quality.get(group.lower() + "_answer", {})
            constraints = quality.get(group.lower() + "_must_keep", {})
            row = {"id": case.get("id"), "category": case.get("category"), "group": group,
                   "manual_review_needed": case.get("manual_review_needed"),
                   "expected_answer": case.get("expected_answer"), "must_keep": case.get("must_keep"),
                   **{column: run.get(column) for column in _RUN_COLUMNS},
                   "case_status": quality.get("status"), "case_reason": quality.get("reason"),
                   "answer_status": answer_quality.get("status"), "answer_correct": answer_quality.get("correct"),
                   "answer_reason": answer_quality.get("reason"), "must_keep_status": constraints.get("status"),
                   "must_keep_retained": constraints.get("retained"), "must_keep_total": constraints.get("total"),
                   "must_keep_retention_rate": constraints.get("retention_rate"), "missing_constraints": constraints.get("missing"),
                   "selection_omission_evidence": quality.get("selection_omission_evidence"),
                   **{"comparison_" + key: value for key, value in comparison.items()}}
            # Keep additional future scalar measurements visible without dumping payloads.
            for key, value in run.items():
                if key not in row and value is not None and isinstance(value, (str, int, float, bool)):
                    row[key] = value
            rows.append(row)
    return rows


def _percent(value: Any) -> str:
    return "未知" if value is None else f"{value * 100:.2f}%"


def _ms(value: Any) -> str:
    return "未知" if value is None else f"{value:+.2f} ms"


def _escape(value: Any) -> str:
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False)
    return str(value if value is not None else "未知").replace("|", "\\|").replace("\n", "<br>").replace("\r", "")


def _markdown(report: dict) -> str:
    summary = report["summary"]
    counts, constraints = summary["status_counts"], summary["constraint_retention"]
    tokens, latency = summary["cloud_input_token_reduction"], summary["latency_delta_ms"]
    lines = [
        "# 危险案例自动 A/B 评测报告", "",
        f"- 运行编号：{_escape(report.get('run_id'))}",
        f"- 开始时间：{_escape(report.get('started_at'))}",
        f"- 模式：{'dry-run（离线模拟）' if summary['simulated'] else 'live（真实调用）'}", "",
        "A = 完整上下文经 Gateway 直达云端，强制 optimize=false。",
        "B = 配置的本地模型筛选后经 Gateway 上云，强制 optimize=true。", "",
    ]
    if summary["simulated"]:
        lines += ["**本报告是 fixture / mock 离线演练。回答来自固定剧本，Token 为估算数据；延迟是本机 fixture 执行耗时，不代表任何真实 provider 的效果、账单或性能，也不能作为生产启用依据。**", ""]
    else:
        lines += ["本报告仅反映本次真实调用返回的测量值。缺失 usage 或筛选证据的项目保持未知，不据此推断节省或保留成功。", ""]
    if report.get("completed") is False:
        planned = report.get("metadata", {}).get("cases_planned", "未知")
        finished = sum(case.get("completed") is True for case in report.get("cases", []))
        lines += [f"**本次评测未完成：计划 {planned} 例，已保存 {summary['total_cases']} 例，其中完整 A/B {finished} 例。未完成分组保持未知；以下比例只针对已保存案例，不能代表完整套件。**", ""]
    lines += [
        "## 总体结果", "",
        "| 指标 | 结果 |", "| --- | --- |",
        f"| 总案例数 | {summary['total_cases']} |",
        f"| PASS / FAIL / MANUAL_REVIEW | {counts['PASS']} / {counts['FAIL']} / {counts['MANUAL_REVIEW']} |",
        f"| B 任务通过率 | {_percent(summary['b_task_pass_rate'])}（{summary['b_task_pass_count']} / {summary['total_cases']}） |",
        f"| 危险案例通过率 | {_percent(summary['danger_case_pass_rate'])} |",
        f"| 危险案例失败率 | {_percent(summary['danger_case_failure_rate'])} |",
        f"| B 关键约束保留率 | {_percent(constraints['rate'])}（{constraints['retained']} / {constraints['known_total']} 条已知约束） |",
        f"| 未知约束 / 涉及案例 | {constraints['unknown_total']} / {constraints['unknown_cases']} |",
        f"| 平均云端输入 Token 降幅 | {_percent(tokens['mean'])} |",
        f"| P50 / P95 云端输入 Token 降幅 | {_percent(tokens['p50'])} / {_percent(tokens['p95'])} |",
        f"| 平均延迟变化（B − A） | {_ms(latency['mean'])} |",
        f"| P50 / P95 延迟变化（B − A） | {_ms(latency['p50'])} / {_ms(latency['p95'])} |", "",
        "### 统计口径", "",
        "- B 任务通过：B 回答的确定性判定为 PASS，且该案例不要求人工复核；分母为全部案例。",
        "- 危险案例通过 / 失败率：整体 PASS / FAIL 数分别除以全部案例；MANUAL_REVIEW 不计为通过，也不当作已确定的失败。",
        "- 关键约束保留率按约束条数加权，只使用有实际检查结果的约束；未知约束单列，不能按已保留或已删除处理。",
        "- 云端输入 Token 降幅 = 1 − B.prompt_tokens / A.prompt_tokens；正数表示减少。总 Token 变化 = B.total_tokens − A.total_tokens，reasoning_tokens 不重复相加。",
        "- context_compression_ratio = Working / Raw 输入 Token 的估算比值；越小表示压缩越多，1 表示未压缩。A / B 的原始值保存在 JSON / CSV。",
        "- 延迟变化 = B 总延迟 − A 总延迟；正数表示更慢，负数表示更快。",
        "- JSON null / CSV 空白 / 本文“未知”表示缺失、不可比或分母为 0。实际与估算 usage 混用时不计算 Token 变化；延迟独立比较同类执行，不因缺 usage 而丢弃已知耗时。",
        f"- 输入 Token 降幅有效样本 {tokens['count']} 个；延迟变化有效样本 {latency['count']} 个。均值和分位数仅计算有效样本，P50 / P95 使用排序后的线性插值。",
        "- 正确性来自案例预设规则及 must_keep 检查；需要语义判断的项目交给人工复核，不依赖另一个云模型自评。", "",
        "## 每种危险类型的表现", "",
        "| 危险类型 | 案例 | PASS / FAIL / 复核 | B 任务通过率 | 危险通过率 | 关键约束保留率 | 未知约束 | 平均输入 Token 降幅 | 平均延迟变化 | 建议 |",
        "| --- | ---: | --- | --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for category, group in summary["categories"].items():
        group_counts, group_constraints = group["status_counts"], group["constraint_retention"]
        lines.append(f"| {_escape(category)} | {group['total_cases']} | {group_counts['PASS']} / {group_counts['FAIL']} / {group_counts['MANUAL_REVIEW']} | {_percent(group['b_task_pass_rate'])} | {_percent(group['danger_case_pass_rate'])} | {_percent(group_constraints['rate'])} | {group_constraints['unknown_total']} | {_percent(group['cloud_input_token_reduction']['mean'])} | {_ms(group['latency_delta_ms']['mean'])} | {_escape(group['enablement_recommendation'])} |")
    lines += ["", "## 失败案例清单", ""]
    if summary["failures"]:
        lines += ["| 案例 | 危险类型 | 原因 | B 遗失的关键条件 | 上下文筛选遗漏证据 |", "| --- | --- | --- | --- | --- |"]
        for failure in summary["failures"]:
            lines.append("| " + " | ".join(_escape(failure[key]) for key in ("id", "category", "reason", "missing_constraints", "selection_omission_evidence")) + " |")
    else:
        lines.append("没有已判定为 FAIL 的案例；请同时查看待人工复核项。")
    lines += ["", "## 哪些信息被错误删除", ""]
    if summary["deleted_constraints"]:
        for item in summary["deleted_constraints"]:
            lines.append(f"- {_escape(item['id'])}（{_escape(item['category'])}）：{_escape(item['missing_constraints'])}；删除原因：{_escape(item['discard_reasons'])}")
        lines += ["", "`local_worker_not_selected` 表示本地选择器舍弃；`optional_log_outside_protected_window` 表示日志窗口规则舍弃。两者不能都归因于本地模型。"]
    else:
        lines.append("已知检查结果中未发现关键条件被删除；未知项不代表保留成功。")
    lines += ["", "## 待人工复核", ""]
    if summary["manual_reviews"]:
        for item in summary["manual_reviews"]:
            lines.append(f"- {_escape(item['id'])}（{_escape(item['category'])}）：{_escape(item['reason'])}")
    else:
        lines.append("无。")
    lines += ["", "## 是否建议启用", "", summary["enablement_recommendation"], "",
              "## 明细文件", "", "- [JSON 明细](details.json)：完整 A/B 回答、指标、质量检查、对比与汇总。",
              "- [CSV 明细](details.csv)：每案例两行（A / B），便于表格软件检查。", ""]
    return "\n".join(lines)


def write_reports(report: dict, output_dir: Path) -> dict[str, Path]:
    """Write one run into an existing unique directory supplied by the caller."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise ValueError("报告目录必须由调用方提前创建")
    safe = _safe_report(report)
    for case in safe.get("cases", []):
        case["comparison"] = compare_ab(case.get("A", {}), case.get("B", {}))
    simulated = safe.get("mode") == "dry-run" or bool(safe.get("simulated", True))
    safe["simulated"] = simulated
    safe["summary"] = summarize(safe.get("cases", []), simulated=simulated)
    if safe.get("completed") is False:
        advice = "评测未完成，暂不建议启用；请复核已保存结果并完成剩余评测。"
        safe["summary"]["enablement_recommendation"] = advice
        for group in safe["summary"]["categories"].values():
            group["enablement_recommendation"] = advice
    paths = {"json": output_dir / "details.json", "csv": output_dir / "details.csv",
             "markdown": output_dir / "summary.md"}
    # Refuse accidental overwrite of a previous run's evidence.
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("报告文件已存在，请为本次运行创建新的目录")
    paths["json"].write_text(json.dumps(safe, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows = _csv_rows(safe.get("cases", []))
    columns = list(dict.fromkeys(key for row in rows for key in row)) or ["id", "category", "group"]
    with paths["csv"].open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: _cell(value) for key, value in row.items()} for row in rows)
    paths["markdown"].write_text(_markdown(safe), encoding="utf-8")
    return paths
