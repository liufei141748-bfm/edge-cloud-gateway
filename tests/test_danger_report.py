"""Report checks cover missing data, measurement provenance and arithmetic."""
import copy
import csv
import json

import pytest

from edge_cloud_gateway.danger_report import compare_ab, summarize, write_reports


def run(*, prompt=100, total=150, latency=20, simulated=True, source="estimated"):
    return {"answer": '{"answer": "正确"}', "prompt_tokens": prompt,
            "completion_tokens": total - prompt, "reasoning_tokens": 10, "total_tokens": total,
            "raw_input_tokens": 100, "working_input_tokens": prompt,
            "raw_payload_bytes": 400, "working_payload_bytes": prompt * 4,
            "context_compression_ratio": prompt / 100, "local_model_used": True,
            "cloud_model_used": True, "local_model": "qwen3.5:4b", "cloud_model": "deepseek-chat",
            "route": "context_then_cloud", "planned_route": "context_then_cloud",
            "latency_local": 1, "latency_cloud": latency - 1, "latency_total": latency,
            "fallback_used": False, "usage_source": source, "simulated": simulated,
            "finish_reason": "stop", "request_status": "success", "error_type": None,
            "local_input_tokens": 10, "local_output_tokens": 2, "local_total_tokens": 12,
            "end_to_end_total_tokens": total + 12}


def case(identifier="threshold-1", *, status="PASS", correct=True, manual=False,
         retained=2, total=2, prompt=50, latency=25, category="数字阈值混淆"):
    b_answer = {"status": "MANUAL_REVIEW" if correct is None or manual else "PASS" if correct else "FAIL",
                "correct": correct, "reason": "答案判定"}
    keep = {"status": "MANUAL_REVIEW" if retained is None else "PASS" if retained == total else "FAIL",
            "retained": retained, "total": total,
            "retention_rate": retained / total if retained is not None and total else None,
            "missing": [] if retained is None or retained == total else [{"block_id": "rule", "text": "禁止删除"}]}
    return {"id": identifier, "category": category, "manual_review_needed": manual,
            "expected_answer": {"answer": "正确"},
            "must_keep": [{"block_id": "rule", "text": "禁止删除"}] * total,
            "A": run(), "B": run(prompt=prompt, total=prompt + 50, latency=latency),
            "quality": {"status": status, "a_answer": {"status": "PASS", "correct": True, "reason": "正确"},
                        "b_answer": b_answer, "a_must_keep": {"status": "PASS", "retained": total,
                        "total": total, "retention_rate": 1, "missing": []}, "b_must_keep": keep,
                        "selection_omission_evidence": correct is False and bool(keep["missing"]),
                        "reason": "测试原因"}, "comparison": {"cloud_input_token_reduction": 999}}


def test_comparison_uses_cloud_prompt_and_existing_total_without_double_counting_reasoning():
    result = compare_ab(run(), run(prompt=60, total=120, latency=25))
    assert result["cloud_input_token_reduction"] == pytest.approx(0.4)
    assert result["total_tokens_delta"] == -30
    assert result["total_tokens_change_ratio"] == pytest.approx(-0.2)
    assert result["end_to_end_total_tokens_delta"] == -30
    assert result["latency_delta_ms"] == 5
    assert result["latency_change_ratio"] == pytest.approx(0.25)
    assert result["context_compression_ratio"] == pytest.approx(0.6)
    assert result["comparison_basis"] == "simulated_estimate"


@pytest.mark.parametrize("a_source,b_source,a_simulated,b_simulated", [
    ("actual", "estimated", False, True), ("actual", "actual", False, True),
    ("estimated", "estimated", False, False), ("unknown", "unknown", True, True),
])
def test_mixed_or_unsupported_sources_are_not_comparable(a_source, b_source, a_simulated, b_simulated):
    result = compare_ab(run(source=a_source, simulated=a_simulated),
                        run(source=b_source, simulated=b_simulated))
    assert result["usage_comparable"] is False
    for key in ("cloud_input_token_reduction", "total_tokens_delta", "total_tokens_change_ratio"):
        assert result[key] is None
    assert result["latency_delta_ms"] == (0 if a_simulated is b_simulated else None)


def test_actual_sources_are_identified_as_measured():
    result = compare_ab(run(source="actual", simulated=False),
                        run(source="actual", simulated=False, prompt=50))
    assert result["usage_comparable"] is True
    assert result["comparison_basis"] == "actual"


@pytest.mark.parametrize("bad_value", [None, False, -1, float("nan"), float("inf"), "100"])
def test_invalid_measurement_is_unknown(bad_value):
    a, b = run(), run()
    b.update(prompt_tokens=bad_value, total_tokens=bad_value, latency_total=bad_value)
    result = compare_ab(a, b)
    assert result["cloud_input_token_reduction"] is None
    assert result["total_tokens_delta"] is None
    assert result["latency_delta_ms"] is None


def test_zero_denominators_are_unknown_and_absolute_deltas_still_available():
    a, b = run(), run()
    a.update(prompt_tokens=0, total_tokens=0, latency_total=0)
    result = compare_ab(a, b)
    assert result["cloud_input_token_reduction"] is None
    assert result["total_tokens_change_ratio"] is None
    assert result["latency_change_ratio"] is None
    assert result["total_tokens_delta"] == 150
    assert result["latency_delta_ms"] == 20


def test_summary_distinguishes_task_success_full_quality_unknown_and_weighted_retention():
    cases = [case("pass", retained=2, total=2),
             case("fail", status="FAIL", correct=False, retained=1, total=4),
             case("unknown", status="MANUAL_REVIEW", correct=None, retained=None, total=3),
             case("manual", status="MANUAL_REVIEW", correct=True, manual=True, retained=1, total=1)]
    result = summarize(cases, simulated=True)
    assert result["total_cases"] == 4
    assert result["status_counts"] == {"PASS": 1, "FAIL": 1, "MANUAL_REVIEW": 2}
    assert result["b_task_pass_rate"] == 0.25
    assert result["danger_case_pass_rate"] == 0.25
    assert result["danger_case_failure_rate"] == 0.25
    assert result["constraint_retention"] == {
        "retained": 4, "known_total": 7, "unknown_total": 3, "unknown_cases": 1, "rate": 4 / 7}
    assert result["failures"][0]["id"] == "fail"
    assert result["failures"][0]["selection_omission_evidence"] is True
    assert result["deleted_constraints"][0]["missing_constraints"][0]["text"] == "禁止删除"


def test_b_answer_success_does_not_hide_constraint_failure():
    result = summarize([case(status="FAIL", correct=True, retained=1)], simulated=True)
    assert result["b_task_pass_rate"] == 1
    assert result["danger_case_pass_rate"] == 0
    assert result["danger_case_failure_rate"] == 1


def test_manual_flag_does_not_hide_a_known_wrong_b_answer():
    result = summarize([case(status="FAIL", correct=False, manual=True)], simulated=True)
    assert result["b_task_fail_count"] == 1
    assert result["b_task_manual_review_count"] == 0


def test_percentiles_use_linear_interpolation_and_exclude_unknown_pairs():
    cases = [case("one", prompt=100, latency=10), case("two", prompt=50, latency=20),
             case("three", prompt=0, latency=50), case("unknown")]
    cases[-1]["B"]["usage_source"] = "actual"
    cases[-1]["B"]["simulated"] = False
    result = summarize(cases, simulated=True)
    assert result["cloud_input_token_reduction"] == {"count": 3, "mean": 0.5, "p50": 0.5, "p95": 0.95}
    assert result["latency_delta_ms"]["count"] == 3
    assert result["latency_delta_ms"]["mean"] == pytest.approx(20 / 3)
    assert result["latency_delta_ms"]["p50"] == 0
    assert result["latency_delta_ms"]["p95"] == pytest.approx(27)


def test_empty_and_single_case_statistics():
    empty = summarize([], simulated=False)
    assert empty["danger_case_pass_rate"] is None
    assert empty["constraint_retention"]["rate"] is None
    assert empty["cloud_input_token_reduction"] == {"count": 0, "mean": None, "p50": None, "p95": None}
    single = summarize([case()], simulated=True)
    assert single["cloud_input_token_reduction"]["p95"] == 0.5


def test_categories_keep_separate_recommendations_and_dry_run_cannot_recommend_production():
    cases = [case(category="阈值"), case("fail", category="单位", status="FAIL", correct=False, retained=0)]
    offline = summarize(cases, simulated=True)
    assert set(offline["categories"]) == {"阈值", "单位"}
    assert all("不能据此建议生产启用" in item["enablement_recommendation"] for item in offline["categories"].values())
    live = summarize(cases, simulated=False)
    assert "扩大样本" in live["categories"]["阈值"]["enablement_recommendation"]
    assert "暂不建议启用" in live["categories"]["单位"]["enablement_recommendation"]


def test_generate_json_csv_and_chinese_markdown_without_mutating_input(tmp_path):
    cases = [case(str(index), category=f"类型 {index // 3 + 1}") for index in range(36)]
    report = {"schema_version": 1, "run_id": "test-run", "mode": "dry-run", "simulated": True,
              "started_at": "2026-09-15T00:00:00Z", "metadata": {}, "cases": cases}
    before = copy.deepcopy(report)
    paths = write_reports(report, tmp_path)
    assert report == before
    assert set(paths) == {"json", "csv", "markdown"}
    details = json.loads(paths["json"].read_text())
    assert details["summary"]["total_cases"] == 36
    assert details["cases"][0]["comparison"]["cloud_input_token_reduction"] == 0.5
    with paths["csv"].open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 72
    assert rows[0]["group"] == "A" and rows[1]["group"] == "B"
    assert rows[0]["answer"] == '{"answer": "正确"}'
    assert rows[0]["reasoning_tokens"] == "10"
    assert rows[0]["comparison_cloud_input_token_reduction"] == "0.5"
    text = paths["markdown"].read_text()
    for required in ("危险案例自动 A/B", "离线演练", "总体结果", "每种危险类型", "失败案例清单",
                     "哪些信息被错误删除", "待人工复核", "是否建议启用", "MANUAL_REVIEW", "P50 / P95", "分母为 0"):
        assert required in text


def test_reports_omit_secret_configuration_and_headers(tmp_path):
    item = case()
    item["A"]["headers"] = {"Authorization": "Bearer secret-header"}
    item["B"]["working_payload"] = {"api_key": "secret-nested", "messages": []}
    report = {"simulated": True, "metadata": {"api_key": "secret-key", "cloud_api_key": "secret-cloud",
              "config": {"endpoint": "secret-endpoint"}}, "cases": [item]}
    paths = write_reports(report, tmp_path)
    for path in paths.values():
        assert "secret-" not in path.read_text(encoding="utf-8-sig")


def test_csv_protects_formula_strings_and_preserves_numeric_values(tmp_path):
    item = case()
    item["A"]["answer"] = "=HYPERLINK(\"https://example.invalid\")"
    paths = write_reports({"simulated": True, "cases": [item]}, tmp_path)
    with paths["csv"].open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["answer"].startswith("'=")
    assert row["comparison_total_tokens_delta"] == "-50"


def test_dry_run_mode_always_labels_output_as_simulated(tmp_path):
    paths = write_reports({"mode": "dry-run", "simulated": False, "cases": [case()]}, tmp_path)
    details = json.loads(paths["json"].read_text())
    assert details["simulated"] is True
    assert "不能据此建议生产启用" in details["summary"]["enablement_recommendation"]


def test_write_reports_requires_unique_existing_directory(tmp_path):
    with pytest.raises(ValueError):
        write_reports({"simulated": True, "cases": []}, tmp_path / "missing")
    write_reports({"simulated": True, "cases": []}, tmp_path)
    with pytest.raises(FileExistsError):
        write_reports({"simulated": True, "cases": []}, tmp_path)


def test_missing_usage_does_not_discard_measured_latency():
    a, b = run(simulated=False, source="unknown"), run(simulated=False, source="unknown", latency=30)
    a["prompt_tokens"] = b["prompt_tokens"] = None
    result = compare_ab(a, b)
    assert result["cloud_input_token_reduction"] is None
    assert result["latency_delta_ms"] == 10


def test_partial_report_preserves_unknown_and_cannot_recommend_enabling(tmp_path):
    from edge_cloud_gateway.danger_eval import _pending_run
    item = case(status="MANUAL_REVIEW", correct=None, retained=None)
    item["completed"] = False
    item["B"] = _pending_run("B")
    report = {"mode": "dry-run", "simulated": True, "completed": False,
              "metadata": {"cases_planned": 36}, "cases": [item]}
    paths = write_reports(report, tmp_path)
    result = json.loads(paths["json"].read_text())
    assert result["summary"]["cloud_input_token_reduction"]["count"] == 0
    assert result["summary"]["constraint_retention"]["unknown_total"] == 2
    assert "评测未完成" in paths["markdown"].read_text()
    assert "暂不建议启用" in result["summary"]["enablement_recommendation"]
