"""Synthetic requirement-retention checks with a deliberately omitting selector."""

from dataclasses import replace
import json

import pytest

from edge_cloud_gateway.adapters import HTTPReply
from edge_cloud_gateway.config import Settings
from edge_cloud_gateway.context import RawContext, build_working_context, protected


PREFIX = "院子里的桂树长出嫩叶。\n" * 24
SUFFIX = "湖边的芦苇随着微风摆动。\n" * 24


def make_raw(content, *, kind="document"):
    return RawContext.from_request({
        "model": "adaptive",
        "messages": [{"role": "user", "content": "请依据资料给出交接说明，并遵循材料中的交付约定。"}],
        "gateway_context": {"blocks": [
            {"id": "handoff_notes", "source": "handoff-notes", "kind": kind,
             "optional": True, "content": content},
            {"id": "garden", "source": "garden-background", "kind": "text",
             "optional": True, "content": PREFIX + SUFFIX},
        ]},
    }, Settings())


async def omit_or_select(raw, selected_ids=()):
    settings = replace(Settings(), context=replace(
        Settings().context, worker_max_bytes=64000, min_savings_ratio=0.01,
    ))
    calls = []

    async def selector(payload):
        offered = json.loads(payload["messages"][-1]["content"])["blocks"]
        calls.append(offered)
        assert {item["id"] for item in offered} == {"handoff_notes", "garden"}
        return HTTPReply(200, json.dumps({"choices": [{
            "finish_reason": "stop", "message": {"role": "assistant", "content":
                json.dumps({"selected_ids": list(selected_ids)})},
        }]}).encode(), {})

    working = await build_working_context(raw, settings, selector)
    assert len(calls) == 1, "The omission must come from an actual offline selector callback"
    assert working.optimized
    assert working.reason == "selected_original_context"
    assert not any(entry["reason"] == "worker_budget_preserve_original"
                   for entry in working.package["relevant_context"])
    assert "garden" not in {entry["id"] for entry in working.package["relevant_context"]}
    return working


def retained_notes(working):
    return [entry for entry in working.package["relevant_context"]
            if entry["id"] == "handoff_notes"]


def assert_literal_partition(raw, working):
    """Kept excerpts and omitted ranges account for the same original source."""
    source = raw.blocks[0]
    kept = retained_notes(working)
    omitted = [entry for entry in working.package["discarded_context"]
               if entry["id"] == source.id]
    assert kept and omitted
    assert sum(len(entry["content"]) for entry in kept) < len(source.content) / 2
    assert [(entry["start"], entry["end"]) for entry in kept] == sorted(
        (entry["start"], entry["end"]) for entry in kept)
    ranges = []
    for entry in kept + omitted:
        assert entry["source"] == source.source
        assert type(entry.get("start")) is int and type(entry.get("end")) is int
        assert 0 <= entry["start"] < entry["end"] <= len(source.content)
        ranges.append((entry["start"], entry["end"]))
        if "content" in entry:
            assert entry["content"] == source.content[entry["start"]:entry["end"]]
    ranges.sort()
    assert ranges[0][0] == 0
    assert ranges[-1][1] == len(source.content)
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))


@pytest.mark.asyncio
@pytest.mark.parametrize("requirement", [
    "输出应当列出审阅人。",
    "提交材料要求附上签收记录，按职责顺序排列。",
    "交接说明需要注明经办人员。",
    "移交时需附上验收记录。",
    "最终输出采用表格，按姓名排列。",
    "交付文件应当包含签收摘要。",
    "答复要求按步骤列出办理流程。",
    "交付格式要求使用表格。",
    "输出要求先列出负责人，再列出送达地点。",
    "验收要求提交审核意见。",
    "The delivery note should include the reviewer's name.",
    "The output should list the responsible person before the delivery location.",
])
async def test_actionable_requirement_survives_selector_omission(requirement):
    raw = make_raw(PREFIX + requirement + "\n" + SUFFIX)
    assert not protected(raw.blocks[0]), "Whole-block protection must not mask this regression"
    working = await omit_or_select(raw)
    assert any(requirement in entry["content"] for entry in retained_notes(working))
    assert_literal_partition(raw, working)


@pytest.mark.asyncio
@pytest.mark.parametrize("description", [
    "课堂讲解‘要求’一词的含义。",
    "会议讨论客户需求和格式样式。",
    "花园需要雨水，溪流随着山势蜿蜒。",
    "这篇文章谈到了交付、输出、格式和顺序。",
    "树叶排列成扇形，枝条交错成荫。",
    "The garden needs rain and the trees provide shade.",
    "课堂讨论要求的含义，并提供阅读材料。",
    "作者研究输出，附录包含词汇解释。",
    "课堂讨论要求的含义；学生提供阅读材料。",
    "The dictionary explains should, and readers provide examples.",
])
async def test_descriptive_words_do_not_become_unconditional_protection(description):
    raw = make_raw(PREFIX + description + "\n" + SUFFIX)
    assert not protected(raw.blocks[0])
    working = await omit_or_select(raw)
    assert retained_notes(working) == []


@pytest.mark.asyncio
async def test_complete_conditional_sentence_keeps_comma_and_semicolon_branches():
    requirement = (
        "若材料已签收，提交说明应当附上签收凭据；"
        "签收尚在办理时，交付备注需要列明办理进度。"
    )
    raw = make_raw(PREFIX + requirement + "\n" + SUFFIX)
    assert not protected(raw.blocks[0])
    working = await omit_or_select(raw)
    assert any(requirement in entry["content"] for entry in retained_notes(working))
    assert_literal_partition(raw, working)


@pytest.mark.asyncio
async def test_neighboring_sentences_keep_local_antecedent_and_explanation():
    passage = (
        "材料分为原稿和副本。\n"
        "交付说明要求列明两者的存放位置。\n"
        "它们分别对应归档件与传阅件。"
    )
    raw = make_raw(PREFIX + passage + "\n" + SUFFIX)
    assert not protected(raw.blocks[0])
    working = await omit_or_select(raw)
    assert any(passage in entry["content"] for entry in retained_notes(working))
    assert_literal_partition(raw, working)


@pytest.mark.asyncio
async def test_colon_heading_and_multiline_materials_keep_literal_continuation():
    passage = "交付材料要求附上以下资料：\n- 签收凭据\n- 验收意见。"
    raw = make_raw(PREFIX + passage + "\n" + SUFFIX)
    assert not protected(raw.blocks[0])
    working = await omit_or_select(raw)
    assert any(passage in entry["content"] for entry in retained_notes(working))
    assert_literal_partition(raw, working)


@pytest.mark.asyncio
async def test_overlapping_requirement_neighborhoods_merge_without_duplication():
    passage = "提交材料要求附上签收凭据。\n交付说明应当注明经办人员。"
    raw = make_raw(PREFIX + passage + "\n" + SUFFIX)
    assert not protected(raw.blocks[0])
    working = await omit_or_select(raw)
    assert sum(passage in entry["content"] for entry in retained_notes(working)) == 1
    assert_literal_partition(raw, working)


@pytest.mark.asyncio
async def test_separated_requirements_keep_source_order_and_discard_middle_noise():
    first = "提交说明要求附上签收凭据。"
    second = "交付材料需要注明经办人员。"
    raw = make_raw(PREFIX + first + "\n" + PREFIX + second + "\n" + SUFFIX)
    assert not protected(raw.blocks[0])
    working = await omit_or_select(raw)
    entries = retained_notes(working)
    assert len(entries) == 2
    assert first in entries[0]["content"]
    assert second in entries[1]["content"]
    assert_literal_partition(raw, working)


@pytest.mark.asyncio
async def test_selected_requirement_block_stays_complete_and_unmodified():
    content = PREFIX + "提交说明要求附上签收凭据。\n" + SUFFIX
    raw = make_raw(content)
    working = await omit_or_select(raw, ["handoff_notes"])
    entries = retained_notes(working)
    assert len(entries) == 1
    assert entries[0]["content"] == content
    assert (entries[0]["start"], entries[0]["end"]) == (0, len(content))
    assert entries[0]["reason"] == "local_worker_selected_original"
    assert all(entry["id"] != "handoff_notes" for entry in working.package["discarded_context"])


@pytest.mark.asyncio
async def test_requirement_guard_does_not_change_optional_log_selection():
    content = "提交说明要求附上签收凭据。"
    raw = make_raw(content, kind="log")
    assert not protected(raw.blocks[0])
    working = await omit_or_select(raw)
    assert retained_notes(working) == []
