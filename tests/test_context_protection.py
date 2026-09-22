"""Generic source-retention regressions; no live provider or danger fixtures."""

from dataclasses import replace
import json

import pytest

from edge_cloud_gateway.adapters import HTTPReply
from edge_cloud_gateway.config import Settings
from edge_cloud_gateway.context import RawContext, build_working_context, filter_blocks


def block(identifier, content, *, source=None, kind="text"):
    return {"id": identifier, "source": source or identifier, "kind": kind,
            "content": content, "optional": True}


def raw_context(blocks):
    return RawContext.from_request({
        "model": "adaptive",
        "messages": [{"role": "user", "content": "请根据资料说明交接安排。"}],
        "gateway_context": {"blocks": blocks, "constraints": []},
    }, Settings())


def unrelated():
    return block("orchard_background", "榆树旁的花圃开满了小花。午后白云慢慢飘过。\n" * 60)


async def select_only(raw, identifiers):
    # Keep the full examples inside the worker budget, so an accidental budget
    # fallback cannot make a source-retention test pass.
    settings = replace(Settings(), context=replace(
        Settings().context, worker_max_bytes=24000, min_savings_ratio=0.01,
    ))

    async def selector(payload):
        offered = json.loads(payload["messages"][-1]["content"])["blocks"]
        offered_ids = {entry["id"] for entry in offered}
        content = json.dumps({"selected_ids": [identifier for identifier in identifiers
                                                if identifier in offered_ids]})
        return HTTPReply(200, json.dumps({"choices": [{
            "finish_reason": "stop", "message": {"role": "assistant", "content": content},
        }]}).encode(), {})

    return await build_working_context(raw, settings, selector)


def assert_source_slices(raw, entries):
    originals = {item.id: item for item in raw.blocks}
    original_order = {item.id: index for index, item in enumerate(raw.blocks)}
    positions = []
    for entry in entries:
        source = originals[entry["id"]]
        assert entry["source"] == source.source
        assert entry["kind"] == source.kind
        assert 0 <= entry["start"] < entry["end"] <= len(source.content)
        assert entry["content"] == source.content[entry["start"]:entry["end"]]
        positions.append((original_order[source.id], entry["start"]))
    assert positions == sorted(positions)


@pytest.mark.parametrize("content", [
    "探头 AURORA-X 的状态为维护中，交由紫竹组保管。",
    "Current status: awaiting calibration.",
    "state = paused",
    "tenant is Juniper",
    "当前版本为晨雾版，替代旧版本。",
    "发布仅在签名核验完成后执行。",
    "只有审核人可以解除封存。",
    "人工审批优先于自动处理。",
    "最新约定覆盖先前的交接规则。",
    "Only reviewers may unlock the archive.",
])
@pytest.mark.asyncio
async def test_facts_and_restrictions_survive_selector_omission(content):
    raw = raw_context([block("operational_fact", content), unrelated()])
    working = await select_only(raw, [])
    retained = working.package["relevant_context"]
    assert any(entry["id"] == "operational_fact" and entry["content"] == content
               for entry in retained)
    assert "orchard_background" not in {entry["id"] for entry in retained}
    assert working.optimized
    assert_source_slices(raw, retained)


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", [
    "后者签署交接意见。",
    "前者提交交接材料。",
    "The latter signed the handoff memo.",
    "The former prepared the handoff memo.",
])
async def test_former_and_latter_restore_both_preceding_people(reference):
    raw = raw_context([
        block("person_cedar", "岚歌负责采样。"),
        block("person_willow", "泽舟负责复核。"),
        block("handoff_action", reference),
        unrelated(),
    ])
    working = await select_only(raw, ["handoff_action"])
    retained = working.package["relevant_context"]
    assert [entry["id"] for entry in retained] == [
        "person_cedar", "person_willow", "handoff_action",
    ]
    assert working.optimized
    assert_source_slices(raw, retained)


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", ["previous", "next"])
async def test_explicit_adjacent_reference_restores_original_neighbor(direction):
    target = block("contact_card", "启岚负责联络，汀荷负责记录。")
    reference = block("handoff_pointer", "按上文的人员安排交接。" if direction == "previous"
                      else "按下文的人员安排交接。")
    ordered = [target, reference] if direction == "previous" else [reference, target]
    raw = raw_context(ordered + [unrelated()])
    working = await select_only(raw, ["handoff_pointer"])
    retained = working.package["relevant_context"]
    assert [entry["id"] for entry in retained] == [entry["id"] for entry in ordered]
    assert_source_slices(raw, retained)


@pytest.mark.asyncio
async def test_restored_reference_recursively_retains_its_antecedent():
    raw = raw_context([
        block("original_roster", "凌溪负责采样，闻舟负责复核。"),
        block("roster_pointer", "沿用上文的人员配置。"),
        block("final_handoff", "执行上文规定的交接安排。"),
        unrelated(),
    ])
    working = await select_only(raw, ["final_handoff"])
    retained = working.package["relevant_context"]
    assert [entry["id"] for entry in retained] == [
        "original_roster", "roster_pointer", "final_handoff",
    ]
    assert working.optimized
    assert_source_slices(raw, retained)


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", [
    "交接职责参见 staff-roster。",
    "See staff-roster for the handoff.",
    "See staff-roster.",
    "执行依赖 duty_anchor 的职责说明。",
])
async def test_named_reference_restores_a_nonadjacent_source_or_id(reference):
    raw = raw_context([
        block("duty_anchor", "映澜负责联络，星禾负责记录。", source="staff-roster"),
        block("distant_decoy", "桌边摆着一束鲜花。"),
        block("nearby_decoy", "窗外能看到整片树林。"),
        block("named_pointer", reference),
        unrelated(),
    ])
    working = await select_only(raw, ["named_pointer"])
    retained = working.package["relevant_context"]
    assert {entry["id"] for entry in retained} == {"duty_anchor", "named_pointer"}
    assert_source_slices(raw, retained)


@pytest.mark.asyncio
@pytest.mark.parametrize("short_name,long_name", [("ops-RAVEN", "ops-RAVENLY"), ("ops.ini", "ops.ini.bak")])
async def test_named_reference_does_not_confuse_prefix_sources(short_name, long_name):
    raw = raw_context([
        block("short_label", "溪禾负责送样。", source=short_name),
        block("long_label", "澜秋负责接样。", source=long_name),
        block("separator", "屋外种着一棵榆树。"),
        block("exact_pointer", f"See {long_name} for the handoff."),
        unrelated(),
    ])
    working = await select_only(raw, ["exact_pointer"])
    retained = working.package["relevant_context"]
    assert {entry["id"] for entry in retained} == {"long_label", "exact_pointer"}
    assert_source_slices(raw, retained)


@pytest.mark.asyncio
async def test_named_reference_does_not_restore_discarded_log_noise():
    gap = "ambient pulse\n" * 50
    content = gap + "WARNING transfer stalled\n" + gap
    raw = raw_context([
        block("channel_events", content, source="channel-note", kind="log"),
        block("log_pointer", "See channel-note for the handoff."),
        unrelated(),
    ])
    before, _ = filter_blocks(raw, Settings())
    expected = [(e["start"], e["end"]) for e in before if e["id"] == "channel_events"]
    working = await select_only(raw, ["log_pointer"])
    events = [e for e in working.package["relevant_context"] if e["id"] == "channel_events"]
    assert [(e["start"], e["end"]) for e in events] == expected
    assert sum(len(e["content"]) for e in events) < len(content) / 2
    assert any(e["id"] == "channel_events" and e["reason"] == "optional_log_outside_protected_window"
               for e in working.package["discarded_context"])
    assert_source_slices(raw, working.package["relevant_context"])


def event_log(events):
    # Digit-free noise avoids being accidentally protected as a numeric fact.
    gap = "ambient background pulse\n" * 12
    return gap + gap.join(event + "\n" for event in events) + gap


@pytest.mark.parametrize("events", [
    ["tenant: Juniper", "ERROR channel disconnected", "INFO channel recovered"],
    ["会话租户：青禾", "警告：连接中断", "恢复：传输正常"],
    ["state: waiting", "ERROR handoff interrupted", "status: ready"],
])
def test_distant_identity_error_and_recovery_survive_small_log_windows(events):
    content = event_log(events)
    raw = raw_context([block("handoff_log", content, source="service-juniper", kind="log")])
    settings = Settings()
    assert settings.context.log_window_lines == 2
    kept, discarded = filter_blocks(raw, settings)
    for event in events:
        assert any(event in entry["content"] for entry in kept)
    assert discarded, "Unrelated event-gap noise must still be removed"
    assert sum(len(entry["content"]) for entry in kept) < len(content)
    assert_source_slices(raw, kept)


def test_log_sources_keep_separate_subjects_and_original_chronology():
    raw = raw_context([
        block("cedar_log", event_log([
            "tenant: Cedar", "WARNING transfer stalled", "INFO transfer recovered",
        ]), source="channel-cedar", kind="log"),
        block("willow_log", event_log([
            "tenant: Willow", "ERROR review stalled", "state: waiting",
        ]), source="channel-willow", kind="log"),
    ])
    kept, discarded = filter_blocks(raw, Settings())
    assert_source_slices(raw, kept)
    assert discarded
    assert any("tenant: Cedar" in entry["content"] for entry in kept if entry["id"] == "cedar_log")
    assert any("tenant: Willow" in entry["content"] for entry in kept if entry["id"] == "willow_log")
    for entry in kept:
        assert "Willow" not in entry["content"] if entry["id"] == "cedar_log" else "Cedar" not in entry["content"]


@pytest.mark.parametrize("non_event", [
    "The prewarning lamp is decorative.",
    "The sign says warningly in its caption.",
])
def test_warning_anchor_requires_a_word_boundary(non_event):
    positive = "WARNING transfer stalled"
    content = event_log([non_event, positive, "ERROR transport closed"])
    raw = raw_context([block("boundary_log", content, kind="log")])
    kept, discarded = filter_blocks(raw, Settings())
    assert any(positive in entry["content"] for entry in kept)
    assert all(non_event not in entry["content"] for entry in kept)
    assert discarded
    assert_source_slices(raw, kept)


@pytest.mark.parametrize("event", [
    "WARNING queue pressure", "write failed", "access denied", "lease expired",
    "account locked", "route disabled", "channel recovered", "rollback completed",
    "session tenant is Pine", "status: ready", "state: pending",
    "reconnect succeeded; channel resumed", "告警：队列拥塞", "上传失败",
    "访问被拒绝", "租约过期", "账户锁定", "通道禁用", "传输恢复", "配置回滚",
])
def test_each_state_or_anomaly_is_an_independent_log_anchor(event):
    # No ERROR elsewhere to accidentally rescue the event through its window.
    gap = "ambient background pulse\n" * 40
    raw = raw_context([block("event_source", gap + event + "\n" + gap, kind="log")])
    kept, discarded = filter_blocks(raw, Settings())
    assert any(event in entry["content"] for entry in kept)
    assert discarded
    assert sum(len(entry["content"]) for entry in kept) < len(raw.blocks[0].content) / 2
    assert_source_slices(raw, kept)
