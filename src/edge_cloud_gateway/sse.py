"""Observe SSE for metrics without changing or buffering the forwarded stream."""

from __future__ import annotations

import json
from typing import Any


class SSEObserver:
    """Best-effort, bounded side-channel parser; malformed events are ignored.

    UTF-8 is decoded only after an entire event has arrived. A too-large event
    is discarded until its terminating blank line; later events still work.
    Never use this parser to reconstruct the bytes sent to a client.
    """

    def __init__(self, max_buffer: int = 1024 * 1024) -> None:
        if max_buffer < 1:
            raise ValueError("max_buffer must be positive")
        self.max_buffer = max_buffer
        self.usage: dict[str, Any] | None = None
        self.model: str | None = None
        self.saw_done = False
        self.first_content = False
        self.dropped_events = 0
        self._line = bytearray()
        self._data: list[bytes] = []
        self._event_size = 0
        self._line_has_bytes = False
        self._dropping = False
        self._after_cr = False

    def feed(self, chunk: bytes) -> None:
        for value in chunk:
            if self._after_cr:
                self._after_cr = False
                if value == 10:  # CRLF is one line ending, even across chunks.
                    continue
            if value in (10, 13):
                self._end_line()
                self._after_cr = value == 13
                continue
            self._line_has_bytes = True
            if self._dropping:
                continue
            self._line.append(value)
            if self._event_size + len(self._line) > self.max_buffer:
                self._dropping = True
                self.dropped_events += 1
                self._line.clear()
                self._data.clear()

    def _end_line(self) -> None:
        if not self._line_has_bytes:
            if not self._dropping:
                self._dispatch()
            self._data.clear()
            self._event_size = 0
            self._dropping = False
        elif not self._dropping:
            line = bytes(self._line)
            self._event_size += len(line)
            if line.startswith(b"data:"):
                value = line[5:]
                self._data.append(value[1:] if value.startswith(b" ") else value)
            elif line == b"data":
                self._data.append(b"")
        self._line.clear()
        self._line_has_bytes = False

    def _dispatch(self) -> None:
        if not self._data:
            return
        raw = b"\n".join(self._data)
        if raw.strip() == b"[DONE]":
            self.saw_done = True
            return
        try:
            event = json.loads(raw)
        except (ValueError, UnicodeError, RecursionError):
            return
        if not isinstance(event, dict):
            return
        if isinstance(event.get("model"), str):
            self.model = event["model"]
        if isinstance(event.get("usage"), dict):
            self.usage = event["usage"]
        choices = event.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict) and (
                delta.get("content") or delta.get("tool_calls") or delta.get("refusal")
            ):
                self.first_content = True

    def finish(self) -> None:
        """Discard an unterminated event, which SSE clients do not dispatch."""
        self._line.clear()
        self._data.clear()
        self._event_size = 0
        self._line_has_bytes = False
        self._dropping = False
        self._after_cr = False


def _event(value: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n\n"


def completion_to_sse(body: dict[str, Any], include_usage: bool) -> list[bytes]:
    """Convert an already-complete, validated local/mock answer to SSE.

    This is synthetic streaming: no local token reaches the caller until the
    gateway has accepted the complete answer. Cloud SSE must bypass this helper.
    """
    common = {
        "id": body["id"],
        "object": "chat.completion.chunk",
        "created": body["created"],
        "model": body["model"],
    }
    for key in ("system_fingerprint", "service_tier"):
        if key in body:
            common[key] = body[key]
    chunks: list[bytes] = []

    def add(index: int, delta: dict[str, Any], finish_reason: str | None = None) -> None:
        event = {**common, "choices": [{"index": index, "delta": delta, "finish_reason": finish_reason}]}
        if include_usage:
            event["usage"] = None
        chunks.append(_event(event))

    for position, choice in enumerate(body["choices"]):
        index = choice.get("index", position)
        message = choice["message"]
        add(index, {"role": message.get("role", "assistant")})
        for field in ("content", "refusal"):
            value = message.get(field)
            if isinstance(value, str):
                for start in range(0, len(value), 16):
                    add(index, {field: value[start:start + 16]})
        for tool_index, call in enumerate(message.get("tool_calls") or []):
            function = call["function"]
            add(index, {"tool_calls": [{"index": tool_index, "id": call["id"], "type": "function", "function": {"name": function["name"], "arguments": ""}}]})
            arguments = function.get("arguments", "")
            width = max(1, len(arguments) // 2)
            for start in range(0, len(arguments), width):
                add(index, {"tool_calls": [{"index": tool_index, "function": {"arguments": arguments[start:start + width]}}]})
        add(index, {}, choice.get("finish_reason", "stop"))
    if include_usage:
        chunks.append(_event({**common, "choices": [], "usage": body.get("usage")}))
    chunks.append(b"data: [DONE]\n\n")
    return chunks
