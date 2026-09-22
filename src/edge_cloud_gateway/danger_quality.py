"""Deterministic answer and source-retention checks for danger-case evaluation.

Expected answers and retention rules are evaluation metadata, never model input.
These checks do not call a model or infer causality from a single A/B pair.
"""

from __future__ import annotations

import json
import re


def _strict_equal(actual: object, expected: object) -> bool:
    # Python considers True == 1; that is not an acceptable JSON answer match.
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _strict_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite JSON number")


def check_answer(answer: str, expected: dict, manual_review_needed: bool = False,
                 *, complete: bool = True) -> dict:
    """Require one complete JSON object, with an optional plain/json code fence."""
    if not complete:
        return {"status": "MANUAL_REVIEW", "correct": None, "reason": "回答未完整结束，无法判定正确性。"}
    if not isinstance(expected, dict):
        return {"status": "MANUAL_REVIEW", "correct": None, "reason": "缺少有效的预期 JSON 对象。"}
    if not isinstance(answer, str):
        return {"status": "FAIL", "correct": False, "reason": "最终回答不是文本 JSON 对象。"}
    source = answer.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", source, re.DOTALL | re.IGNORECASE)
    if fence:
        source = fence.group(1).strip()
    try:
        actual = json.loads(source, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (ValueError, TypeError, RecursionError):
        return {"status": "FAIL", "correct": False, "reason": "回答不是唯一、完整且有效的 JSON 对象。"}
    if not _strict_equal(actual, expected):
        return {"status": "FAIL", "correct": False, "reason": "回答与预期 JSON 对象不完全一致。"}
    if manual_review_needed:
        return {"status": "MANUAL_REVIEW", "correct": True,
                "reason": "自动答案检查通过，但案例明确要求人工复核。"}
    return {"status": "PASS", "correct": True, "reason": "回答与预期 JSON 对象完全一致。"}


def _unknown_retention(requirements: list, reason: str) -> dict:
    return {"status": "MANUAL_REVIEW", "retained": None, "total": len(requirements),
            "retention_rate": None, "missing": [], "checks": [
                {"block_id": item.get("block_id"), "text": item.get("text"),
                 "retained": None, "reason": reason}
                for item in requirements if isinstance(item, dict)
            ]}


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def check_must_keep(case: dict, package: dict | None) -> dict:
    """Check literal source ranges; only adjacent/overlapping slices may join.

    Each rule has ``block_id`` and ``text``. Optional ``start``/``end`` pin a
    particular occurrence; without them any occurrence in that block may match.
    A different block or source can never satisfy the rule.
    """
    requirements = case.get("must_keep", [])
    if not isinstance(requirements, list):
        return _unknown_retention([], "must_keep 不是有效的规则列表。")
    if not isinstance(package, dict) or not isinstance(package.get("relevant_context"), list):
        return _unknown_retention(requirements, "上下文快照不可用，无法核验原文。")
    blocks = case.get("context")
    if not isinstance(blocks, list) or any(
        not isinstance(block, dict) or any(not isinstance(block.get(key), str)
                                           for key in ("id", "source", "kind", "content"))
        for block in blocks
    ):
        return _unknown_retention(requirements, "案例缺少有效的原始上下文。")
    originals = {block["id"]: block for block in blocks}
    if len(originals) != len(blocks):
        return _unknown_retention(requirements, "案例原始上下文存在重复 block id。")
    ranges = {block_id: [] for block_id in originals}
    for entry in package["relevant_context"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            continue
        block = originals.get(entry["id"])
        if block is None:
            continue
        start, end = entry.get("start"), entry.get("end")
        if (type(start) is not int or type(end) is not int
                or not 0 <= start <= end <= len(block["content"])
                or entry.get("source") != block["source"]
                or entry.get("kind") != block["kind"]
                or entry.get("content") != block["content"][start:end]):
            continue
        ranges[block["id"]].append((start, end))
    ranges = {block_id: _merge_intervals(intervals) for block_id, intervals in ranges.items()}
    checks = []
    for rule in requirements:
        if (not isinstance(rule, dict) or not isinstance(rule.get("block_id"), str)
                or rule["block_id"] not in originals or not isinstance(rule.get("text"), str)
                or not rule["text"]):
            return _unknown_retention(requirements, "案例含无效的 must_keep 规则。")
        block_id, text = rule["block_id"], rule["text"]
        original = originals[block_id]["content"]
        occurrences = []
        if "start" in rule or "end" in rule:
            start, end = rule.get("start"), rule.get("end")
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(original) or original[start:end] != text):
                return _unknown_retention(requirements, "must_keep 指定的位置与原文不一致。")
            occurrences.append((start, end))
        else:
            position = original.find(text)
            while position != -1:
                occurrences.append((position, position + len(text)))
                position = original.find(text, position + 1)
        if not occurrences:
            return _unknown_retention(requirements, "must_keep 条件在指定的原文 block 中不存在。")
        retained = any(left <= start and end <= right
                       for start, end in occurrences for left, right in ranges[block_id])
        checks.append({"block_id": block_id, "text": text, "retained": retained,
                       "reason": "关键条件原文已完整保留。" if retained else "指定来源的连续原文切片未完整保留该条件。"})
    missing = [{"block_id": check["block_id"], "text": check["text"]}
               for check in checks if not check["retained"]]
    retained_count = sum(check["retained"] for check in checks)
    return {"status": "FAIL" if missing else "PASS", "retained": retained_count,
            "total": len(requirements), "retention_rate": retained_count / len(requirements) if requirements else None,
            "missing": missing, "checks": checks}


def assess_case(case: dict, a: dict, b: dict) -> dict:
    """Evaluate both tasks plus provenance; operational uncertainty stays manual."""
    expected = case.get("expected_answer")
    answers = [check_answer(run.get("answer"), expected, case.get("manual_review_needed", False),
                            complete=run.get("finish_reason") == "stop" and run.get("request_status") == "success")
               for run in (a, b)]
    a_answer, b_answer = answers
    a_keep, b_keep = (check_must_keep(case, run.get("package")) for run in (a, b))
    omission_evidence = (a_answer["correct"] is True and b_answer["correct"] is False
                         and bool(b_keep["missing"]))
    if b_answer["status"] == "FAIL" or b_keep["status"] == "FAIL":
        status = "FAIL"
        reason = "B 回答错误或关键条件遗漏，不能启用此类筛选。"
        if omission_evidence:
            reason += "A 正确且 B 错误并存在遗漏，提示筛选遗漏风险，但单次对照不能证明因果。"
    elif any(check["status"] != "PASS" for check in (a_answer, b_answer, a_keep, b_keep)):
        status = "MANUAL_REVIEW"
        reason = "A 基线、答案检查或原文保留检查尚未全部通过，需人工复核。"
    elif (a.get("planned_route") != "direct_cloud" or a.get("route") != "direct_cloud"
          or a.get("local_model_used") is not False or a.get("fallback_used") is not False
          or b.get("planned_route") != "context_then_cloud" or b.get("route") != "context_then_cloud"
          or b.get("local_model_used") is not True or b.get("fallback_used") is not False):
        status = "MANUAL_REVIEW"
        reason = "A/B 未按预定路径完成实际本地筛选，或发生全量回退，不能视为筛选通过。"
    else:
        status = "PASS"
        reason = "A/B 回答均正确，关键条件均保留，且实际执行了预定筛选路径。"
    return {"status": status, "a_answer": a_answer, "b_answer": b_answer,
            "a_must_keep": a_keep, "b_must_keep": b_keep,
            "selection_omission_evidence": omission_evidence, "reason": reason}
