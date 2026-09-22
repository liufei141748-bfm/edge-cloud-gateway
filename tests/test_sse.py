import json

import pytest

from edge_cloud_gateway.sse import SSEObserver, completion_to_sse


def event(value, ending=b"\n\n"):
    return b"data: " + json.dumps(value, ensure_ascii=False).encode() + ending


def decode_events(chunks):
    return [json.loads(part[6:]) for part in b"".join(chunks).split(b"\n\n")
            if part.startswith(b"data: ") and part != b"data: [DONE]"]


@pytest.mark.parametrize("ending", [b"\n\n", b"\r\n\r\n", b"\r\r"])
def test_observer_handles_utf8_and_line_endings_across_every_byte(ending):
    usage = {"prompt_tokens": 17, "completion_tokens": 5, "total_tokens": 22}
    stream = event({"model": "云模型", "choices": [{"delta": {"content": "中文你好"}}]}, ending)
    stream += event({"model": "云模型", "choices": [], "usage": usage}, ending)
    stream += b"data: [DONE]" + ending
    observer = SSEObserver()
    for value in stream:
        observer.feed(bytes([value]))
    observer.finish()
    assert observer.first_content is True
    assert observer.model == "云模型"
    assert observer.usage == usage
    assert observer.saw_done is True


def test_tool_arguments_are_observed_without_reassembly_or_modification():
    stream = event({"choices": [{"delta": {"role": "assistant"}}]})
    observer = SSEObserver()
    observer.feed(stream)
    assert observer.first_content is False
    fragments = [
        {"index": 0, "id": "call_1", "type": "function", "function": {"name": "天气", "arguments": '{"城市":'}},
        {"index": 1, "id": "call_2", "type": "function", "function": {"name": "记录", "arguments": '{"内容":'}},
        {"index": 0, "function": {"arguments": '"北京"}'}},
        {"index": 1, "function": {"arguments": '"测试"}'}},
    ]
    data = b"".join(event({"choices": [{"delta": {"tool_calls": [part]}}]}) for part in fragments)
    original = data[:]
    observer.feed(data)
    assert observer.first_content is True
    assert data == original
    assert observer.usage is None


def test_malformed_metadata_and_json_do_not_prevent_later_usage():
    observer = SSEObserver()
    observer.feed(b": ping\n\nevent: completion\ndata: not json\n\n")
    for value in ([1, 2], None, {"choices": "bad"}, {"choices": [1, None, {"delta": []}]}):
        observer.feed(event(value))
    observer.feed(b'data: {"choices": [],\ndata: "usage": {"prompt_tokens": 4}}\n\n')
    observer.feed(event({"usage": None}))
    assert observer.usage == {"prompt_tokens": 4}
    assert observer.first_content is False


def test_oversized_event_is_bounded_and_next_event_still_parses():
    observer = SSEObserver(max_buffer=128)
    observer.feed(b"data: " + b"x" * 1_000_000)
    assert len(observer._line) == 0
    assert observer._data == []
    observer.feed(b"\r\n\r\n")
    observer.feed(event({"usage": {"prompt_tokens": 1}}))
    observer.feed(b"data: [DONE]\n\n")
    assert observer.dropped_events == 1
    assert observer.usage == {"prompt_tokens": 1}
    assert observer.saw_done is True


def test_multiline_event_cannot_bypass_memory_limit():
    observer = SSEObserver(max_buffer=20)
    observer.feed(b"data: abc\ndata: abc\ndata: abc\n")
    assert observer.dropped_events == 1
    assert observer._data == []
    observer.feed(b"\ndata: [DONE]\n\n")
    assert observer.saw_done is True


def test_unterminated_event_is_not_dispatched():
    observer = SSEObserver()
    observer.feed(b'data: {"usage":{"prompt_tokens":1}}\n')
    observer.finish()
    assert observer.usage is None
    observer.feed(b"data: [DONE]")
    observer.finish()
    assert observer.saw_done is False


def completion(message, finish_reason="stop"):
    return {"id": "chatcmpl-fixture", "created": 123, "model": "mock", "object": "chat.completion",
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}


@pytest.mark.parametrize("include_usage", [False, True])
def test_synthetic_text_stream_reconstructs_complete_answer(include_usage):
    text = "这是经过完整校验后才合成流式输出的中文。" * 4
    body = completion({"role": "assistant", "content": text})
    chunks = completion_to_sse(body, include_usage)
    events = decode_events(chunks)
    reconstructed = "".join(choice["delta"].get("content", "") for row in events for choice in row["choices"])
    assert reconstructed == text
    assert chunks[-1] == b"data: [DONE]\n\n"
    assert events[-2 if include_usage else -1]["choices"][0]["finish_reason"] == "stop"
    if include_usage:
        assert events[-1]["usage"] == body["usage"]
        assert events[-1]["choices"] == []
    else:
        assert all("usage" not in row for row in events)


def test_synthetic_multi_tool_stream_preserves_ids_names_and_json_arguments():
    calls = [{"id": f"call_{index}", "type": "function", "function": {
        "name": f"tool_{index}", "arguments": json.dumps({"城市": city, "count": index}, ensure_ascii=False)}}
        for index, city in enumerate(["上海", "北京"])]
    chunks = completion_to_sse(completion({"role": "assistant", "content": None, "tool_calls": calls}, "tool_calls"), True)
    assembled = {}
    for row in decode_events(chunks):
        for choice in row["choices"]:
            for delta in choice["delta"].get("tool_calls", []):
                current = assembled.setdefault(delta["index"], {"arguments": ""})
                current.update({key: delta[key] for key in ("id", "type") if key in delta})
                if "name" in delta["function"]:
                    current["name"] = delta["function"]["name"]
                current["arguments"] += delta["function"].get("arguments", "")
    for index, call in enumerate(calls):
        assert assembled[index] == {"id": call["id"], "type": call["type"], **call["function"]}


def test_nonpositive_buffer_is_rejected():
    with pytest.raises(ValueError):
        SSEObserver(max_buffer=0)
