"""HTTP providers and explicitly simulated, network-free development fixtures."""

from __future__ import annotations

import asyncio
import copy
import json
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .sse import completion_to_sse


@dataclass(slots=True)
class HTTPReply:
    status_code: int
    body: bytes
    headers: dict[str, str]


@dataclass(slots=True)
class StreamReply:
    status_code: int
    headers: dict[str, str]
    chunks: AsyncIterator[bytes]
    aclose: Callable[[], Awaitable[None]]


class Provider(Protocol):
    simulated: bool

    async def complete(self, payload: dict[str, Any]) -> HTTPReply: ...

    async def stream(self, payload: dict[str, Any]) -> StreamReply: ...

    async def close(self) -> None: ...


def _headers(response: httpx.Response) -> dict[str, str]:
    # aiter_bytes/content are decompressed by HTTPX. ASGI owns HTTP framing.
    excluded = {
        "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
        "te", "trailer", "transfer-encoding", "upgrade", "content-length", "content-encoding",
    }
    excluded.update(part.strip().lower() for part in response.headers.get("connection", "").split(","))
    return {key: value for key, value in response.headers.items() if key.lower() not in excluded}


def _json_reply(body: dict[str, Any], status: int = 200) -> HTTPReply:
    return HTTPReply(status, json.dumps(body, ensure_ascii=False).encode("utf-8"), {"content-type": "application/json"})


def _completion(model: str, message: dict[str, Any], usage: dict[str, Any], finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }


class OpenAICompatibleAdapter:
    simulated = False

    def __init__(self, base_url: str, api_key: str, timeout: float = 60,
                 client: httpx.AsyncClient | None = None) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.timeout = timeout
        self._request_headers = {"authorization": "Bearer " + api_key} if api_key else {}
        self._owns_client = client is None
        self._client = client if client is not None else httpx.AsyncClient(
            timeout=timeout, trust_env=False, follow_redirects=False,
            transport=httpx.AsyncHTTPTransport(retries=0, trust_env=False),
        )

    async def complete(self, payload: dict[str, Any]) -> HTTPReply:
        response = await self._client.post(self.url, json=payload, headers=self._request_headers,
                                           timeout=self.timeout, follow_redirects=False)
        return HTTPReply(response.status_code, response.content, _headers(response))

    async def stream(self, payload: dict[str, Any]) -> StreamReply:
        request = self._client.build_request("POST", self.url, json=payload,
                                             headers=self._request_headers, timeout=self.timeout)
        response = await self._client.send(request, stream=True, follow_redirects=False)

        async def chunks() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await response.aclose()

        return StreamReply(response.status_code, _headers(response), chunks(), response.aclose)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


# Backward-compatible internal name retained for downstream imports.
HttpCloudAdapter = OpenAICompatibleAdapter


class OllamaAdapter:
    simulated = False

    def __init__(self, base_url: str, model: str, timeout: float = 30,
                 client: httpx.AsyncClient | None = None, num_ctx: int = 4096,
                 keep_alive: str = "5m") -> None:
        self.url = base_url.rstrip("/") + "/api/chat"
        self.model = model
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self._owns_client = client is None
        self._client = client if client is not None else httpx.AsyncClient(
            timeout=timeout, trust_env=False, follow_redirects=False,
            transport=httpx.AsyncHTTPTransport(retries=0, trust_env=False),
        )

    async def complete(self, payload: dict[str, Any]) -> HTTPReply:
        options: dict[str, Any] = {"num_ctx": self.num_ctx}
        for key in ("temperature", "top_p", "seed", "stop", "frequency_penalty", "presence_penalty"):
            if key in payload:
                options[key] = payload[key]
        if "max_completion_tokens" in payload:
            options["num_predict"] = payload["max_completion_tokens"]
        elif "max_tokens" in payload:
            options["num_predict"] = payload["max_tokens"]
        request_body: dict[str, Any] = {
            "model": self.model, "messages": payload["messages"], "stream": False,
            "think": False, "keep_alive": self.keep_alive, "options": options,
        }
        response_format = payload.get("response_format") or {}
        if response_format.get("type") == "json_schema":
            request_body["format"] = response_format["json_schema"]["schema"]
        elif response_format.get("type") == "json_object":
            request_body["format"] = "json"
        response = await self._client.post(self.url, json=request_body, timeout=self.timeout,
                                           follow_redirects=False)
        if not response.is_success:
            return HTTPReply(response.status_code, response.content, _headers(response))
        try:
            data = response.json()
            message = data["message"]
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise ValueError("invalid message")
            if not data.get("done") or message.get("tool_calls"):
                raise ValueError("local response is incomplete or contains unexpected tools")
        except (ValueError, KeyError, TypeError):
            return _json_reply({"error": {"message": "Invalid Ollama completion", "type": "upstream_error"}}, 502)
        input_count = data.get("prompt_eval_count")
        output_count = data.get("eval_count")
        valid_counts = all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                           for value in (input_count, output_count))
        usage = {
            "prompt_tokens": input_count if valid_counts else None,
            "completion_tokens": output_count if valid_counts else None,
            "total_tokens": input_count + output_count if valid_counts else None,
            "prompt_tokens_details": {"cached_tokens": data.get("prompt_eval_cached_count", 0), "cache_write_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
        result = _completion(data.get("model") or self.model,
                             {"role": "assistant", "content": message["content"]}, usage,
                             "length" if data.get("done_reason") == "length" else "stop")
        return _json_reply(result)

    async def stream(self, payload: dict[str, Any]) -> StreamReply:
        # Included for the Provider interface. The gateway should call complete
        # and validate first, then use completion_to_sse itself.
        reply = await self.complete(payload)
        if 200 <= reply.status_code < 300:
            blocks = completion_to_sse(json.loads(reply.body), bool((payload.get("stream_options") or {}).get("include_usage")))
            headers = {"content-type": "text/event-stream"}
        else:
            blocks, headers = [reply.body], reply.headers

        async def chunks() -> AsyncIterator[bytes]:
            for block in blocks:
                yield block

        async def aclose() -> None:
            return None

        return StreamReply(reply.status_code, headers, chunks(), aclose)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class MockAdapter:
    """Protocol demonstration only; synthetic usage is never a benchmark."""

    simulated = True

    def __init__(self, kind: str = "cloud") -> None:
        if kind not in {"cloud", "local"}:
            raise ValueError("kind must be cloud or local")
        self.kind = kind
        self.complete_calls = 0
        self.stream_calls = 0
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def _body(self, payload: dict[str, Any]) -> dict[str, Any]:
        user = next((message.get("content", "") for message in reversed(payload.get("messages", []))
                     if message.get("role") == "user"), "")
        user_text = user if isinstance(user, str) else "[multimodal input]"
        content = "【模拟云服务；未调用真实模型】" + user_text[:160]
        if self.kind == "local":
            try:
                parsed = json.loads(user_text)
                if not isinstance(parsed, dict):
                    raise ValueError("expected object")
                content = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
            except (ValueError, TypeError):
                content = "【模拟本地服务】请提供完整 JSON 对象以演示格式校验；本服务不进行真实提取。"
            fmt = payload.get("response_format", {}).get("json_schema", {})
            if fmt.get("name") == "context_selection":
                # Deterministic fixture, not a learned relevance model. English
                # word overlap makes selection observable in the offline demo.
                try:
                    data = json.loads(user_text)
                    task_text = data["task"] + " " + " ".join(
                        str(message.get("content", "")) for message in data.get("conversation", [])
                    ) + " " + " ".join(data.get("constraints", []))
                    words = set(re.findall(r"[a-zA-Z_]{4,}", task_text.lower()))
                    selected = [block["id"] for block in data["blocks"]
                                if not words or words & set(re.findall(r"[a-zA-Z_]{4,}", block["content"].lower()))]
                    content = json.dumps({"selected_ids": selected})
                except (ValueError, KeyError, TypeError):
                    content = "invalid mock selection fixture"
        message: dict[str, Any] = {"role": "assistant", "content": content}
        finish_reason = "stop"
        tools = payload.get("tools") or []
        if self.kind == "cloud" and tools and payload.get("tool_choice") != "none":
            # No tools are executed. This fixture only emits protocol examples.
            functions = [tool["function"] for tool in tools if tool.get("type") == "function"]
            if functions:
                message = {"role": "assistant", "content": None, "tool_calls": [
                    {"id": f"call_mock_{index}", "type": "function", "function": {
                        "name": function["name"], "arguments": json.dumps({"mock": True, "示例": index}, ensure_ascii=False),
                    }} for index, function in enumerate(functions[:2])
                ]}
                finish_reason = "tool_calls"
        usage = {
            "prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
        body = _completion(payload.get("model", "mock-" + self.kind), message, usage, finish_reason)
        body["system_fingerprint"] = "mock-fixture-not-a-measurement"
        return body

    async def complete(self, payload: dict[str, Any]) -> HTTPReply:
        self.complete_calls += 1
        self.calls.append(copy.deepcopy(payload))
        reply = _json_reply(self._body(payload))
        reply.headers["x-gateway-simulated"] = "true"
        return reply

    async def stream(self, payload: dict[str, Any]) -> StreamReply:
        self.stream_calls += 1
        self.calls.append(copy.deepcopy(payload))
        body = self._body(payload)
        blocks = completion_to_sse(body, bool((payload.get("stream_options") or {}).get("include_usage")))
        cancelled = False

        async def chunks() -> AsyncIterator[bytes]:
            for block in blocks:
                # Deliberately split UTF-8 and JSON/tool arguments across chunks.
                for start in range(0, len(block), 7):
                    if cancelled:
                        return
                    await asyncio.sleep(0)
                    yield block[start:start + 7]

        async def aclose() -> None:
            nonlocal cancelled
            cancelled = True

        return StreamReply(200, {"content-type": "text/event-stream", "x-gateway-simulated": "true"}, chunks(), aclose)

    async def close(self) -> None:
        self.closed = True
