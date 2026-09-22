from copy import deepcopy
import json

import pytest

from edge_cloud_gateway.danger_quality import assess_case, check_answer, check_must_keep


def case_data():
    return {"id": "test", "expected_answer": {"allowed": False, "limit": 50}, "manual_review_needed": False,
            "context": [{"id": "rule", "source": "current-policy", "kind": "text", "content": "prefix 不得超过 50 毫秒 suffix"}],
            "must_keep": [{"block_id": "rule", "text": "不得超过 50 毫秒"}]}


def package(case, slices=None):
    block = case["context"][0]
    intervals = slices if slices is not None else [(0, len(block["content"]))]
    return {"relevant_context": [block | {"start": start, "end": end, "content": block["content"][start:end]}
                                  for start, end in intervals]}


def runs(case):
    a = {"answer": json.dumps(case["expected_answer"]), "finish_reason": "stop", "request_status": "success",
         "package": package(case), "planned_route": "direct_cloud", "route": "direct_cloud",
         "local_model_used": False, "fallback_used": False}
    b = deepcopy(a) | {"planned_route": "context_then_cloud", "route": "context_then_cloud", "local_model_used": True}
    return a, b


@pytest.mark.parametrize("answer", ['{"nested":{"value":[false,1,"毫秒"]}}',
                                    '```json\n{"nested":{"value":[false,1,"毫秒"]}}\n```',
                                    '```\n{"nested":{"value":[false,1,"毫秒"]}}\n```'])
def test_exact_json_answer_accepts_optional_complete_fence(answer):
    assert check_answer(answer, {"nested": {"value": [False, 1, "毫秒"]}})["status"] == "PASS"


@pytest.mark.parametrize("answer", ['{"value":true}', '{"value":1.0}', '{"value":"1"}',
                                    '{"value":1,"other":false}', '{"value":1,"value":1}',
                                    '{"value":1} 然而答案是 2', '[{"value":1}]',
                                    '答案如下：{"value":1}', '{"value":NaN}',
                                    '```python\n{"value":1}\n```', '{"value":1} {"value":2}'])
def test_answer_rejects_type_confusion_extra_fields_duplicates_and_prose(answer):
    result = check_answer(answer, {"value": 1})
    assert result["status"] == "FAIL" and result["correct"] is False


def test_nested_boolean_is_not_an_integer_and_list_order_matters():
    assert check_answer('{"values":[{"n":true}]}', {"values": [{"n": 1}]})["status"] == "FAIL"
    assert check_answer('{"values":[2,1]}', {"values": [1, 2]})["status"] == "FAIL"


def test_manual_flag_does_not_hide_a_definite_failure():
    assert check_answer('{"value":2}', {"value": 1}, True)["status"] == "FAIL"
    result = check_answer('{"value":1}', {"value": 1}, True)
    assert result["status"] == "MANUAL_REVIEW" and result["correct"] is True


def test_incomplete_answer_and_missing_expected_value_remain_unknown():
    assert check_answer('{"value":2}', {"value": 1}, complete=False)["correct"] is None
    assert check_answer('{}', None)["status"] == "MANUAL_REVIEW"


def test_must_keep_accepts_source_slice_and_rejects_missing_condition():
    case = case_data()
    result = check_must_keep(case, package(case))
    assert result["status"] == "PASS" and result["retained"] == result["total"] == 1
    assert result["retention_rate"] == 1
    result = check_must_keep(case, package(case, [(0, 6)]))
    assert result["status"] == "FAIL" and result["retention_rate"] == 0
    assert result["missing"] == case["must_keep"]


@pytest.mark.parametrize("change", [{"id": "similar-rule"}, {"source": "old-policy"}, {"kind": "code"},
                                    {"start": -1}, {"start": True}, {"end": 999}, {"content": "伪造的原文"}])
def test_must_keep_rejects_another_source_or_invalid_provenance(change):
    case = case_data()
    snapshot = package(case)
    snapshot["relevant_context"][0].update(change)
    assert check_must_keep(case, snapshot)["status"] == "FAIL"


def test_identical_sentence_from_a_different_block_does_not_satisfy_requirement():
    case = case_data()
    other = case["context"][0] | {"id": "other", "source": "old-policy"}
    case["context"].append(other)
    snapshot = {"relevant_context": [other | {"start": 0, "end": len(other["content"])}]}
    assert check_must_keep(case, snapshot)["status"] == "FAIL"


@pytest.mark.parametrize("slices", [[(0, 12), (12, 24)], [(11, 24), (0, 12)], [(0, 15), (12, 24)]])
def test_adjacent_overlapping_and_out_of_order_source_slices_preserve_condition(slices):
    case = case_data()
    assert check_must_keep(case, package(case, slices))["status"] == "PASS"


def test_disjoint_slices_cannot_join_to_invent_retained_condition():
    case = case_data()
    case["context"][0]["content"] = "ABCDE ABxxxCDE"
    case["must_keep"][0]["text"] = "ABCDE"
    # Concatenating these separated excerpts spells ABCDE, but the original
    # occurrence is absent, and the hidden xxx gap cannot be silently erased.
    assert check_must_keep(case, package(case, [(6, 8), (11, 14)]))["status"] == "FAIL"


def test_explicit_offsets_distinguish_repeated_occurrences():
    case = case_data()
    case["context"][0]["content"] = "repeat / repeat"
    case["must_keep"][0].update(text="repeat", start=0, end=6)
    assert check_must_keep(case, package(case, [(9, 15)]))["status"] == "FAIL"
    case["must_keep"][0].pop("start")
    case["must_keep"][0].pop("end")
    assert check_must_keep(case, package(case, [(9, 15)]))["status"] == "PASS"


@pytest.mark.parametrize("snapshot", [None, {}, {"relevant_context": None}])
def test_unavailable_snapshot_is_unknown_not_full_retention(snapshot):
    result = check_must_keep(case_data(), snapshot)
    assert result["status"] == "MANUAL_REVIEW"
    assert result["retained"] is None and result["retention_rate"] is None
    assert result["total"] == 1 and result["missing"] == []


def test_invalid_reference_requirement_requires_review():
    case = case_data()
    case["must_keep"][0]["text"] = "从未出现的句子"
    assert check_must_keep(case, package(case))["status"] == "MANUAL_REVIEW"


def test_retention_rate_counts_individual_rules_and_empty_rules_are_not_perfect_evidence():
    case = case_data()
    case["must_keep"].append({"block_id": "rule", "text": "prefix"})
    result = check_must_keep(case, package(case, [(0, 6)]))
    assert result["retained"] == 1 and result["total"] == 2 and result["retention_rate"] == 0.5
    case["must_keep"] = []
    assert check_must_keep(case, package(case))["retention_rate"] is None


def test_assessment_requires_correct_answers_constraints_and_actual_routes():
    case = case_data()
    a, b = runs(case)
    assert assess_case(case, a, b)["status"] == "PASS"
    assert assess_case(case, a, b)["selection_omission_evidence"] is False
    b["answer"] = '{"allowed":true,"limit":50}'
    result = assess_case(case, a, b)
    assert result["status"] == "FAIL" and result["selection_omission_evidence"] is False


def test_a_wrong_b_correct_requires_review_and_b_omission_fails_even_if_answer_correct():
    case = case_data()
    a, b = runs(case)
    a["answer"] = '{}'
    assert assess_case(case, a, b)["status"] == "MANUAL_REVIEW"
    b["package"] = {"relevant_context": []}
    result = assess_case(case, a, b)
    assert result["status"] == "FAIL" and result["selection_omission_evidence"] is False


def test_omission_evidence_requires_a_correct_b_wrong_and_verified_loss():
    case = case_data()
    a, b = runs(case)
    b.update(answer='{}', package={"relevant_context": []})
    result = assess_case(case, a, b)
    assert result["selection_omission_evidence"] is True
    assert "不能证明因果" in result["reason"]
    a["answer"] = '{}'
    assert assess_case(case, a, b)["selection_omission_evidence"] is False


@pytest.mark.parametrize("side,change", [
    ("b", {"fallback_used": True}), ("b", {"local_model_used": False}),
    ("b", {"route": "direct_cloud"}), ("b", {"planned_route": "direct_cloud"}),
    ("a", {"route": "context_then_cloud"}), ("a", {"local_model_used": True}),
    ("a", {"fallback_used": True}), ("a", {"planned_route": "direct_local"}),
    ("b", {"finish_reason": "length"}), ("b", {"request_status": "error"}),
    ("b", {"package": None}), ("a", {"package": None}),
])
def test_fallback_unexecuted_selector_invalid_routes_or_incomplete_run_require_review(side, change):
    case = case_data()
    a, b = runs(case)
    (a if side == "a" else b).update(change)
    assert assess_case(case, a, b)["status"] == "MANUAL_REVIEW"


def test_declared_manual_review_and_fallback_do_not_override_definite_failure():
    case = case_data()
    case["manual_review_needed"] = True
    a, b = runs(case)
    assert assess_case(case, a, b)["status"] == "MANUAL_REVIEW"
    b.update(answer='{}', fallback_used=True)
    assert assess_case(case, a, b)["status"] == "FAIL"
