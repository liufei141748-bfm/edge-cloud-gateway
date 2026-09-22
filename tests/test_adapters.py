import asyncio
import gzip
import json

import httpx
import pytest

from edge_cloud_gateway.adapters import HttpCloudAdapter, MockAdapter, OllamaAdapter
from edge_cloud_gateway.sse import SSEObserver


class ByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_cloud_retains_prefix_unknown_fields_tools_and_business_headers():
    payload = {"model": "chosen-cloud", "messages": [{"role": "user", "content": "中文"}],
               "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
               "response_format": {"type": "json_object"}, "vendor_extension": {"one": [1, None]}}
    response_body = b'{"choices":[],"vendor_response":{"ok":true}}'
    calls = []

    def handler(request):
        calls.append(request)
        assert str(request.url) == "https://example.test/vendor/v1/chat/completions"
        assert json.loads(request.content) == payload
        assert request.headers["authorization"] == "Bearer test-secret"
        assert request.extensions["timeout"]["read"] == 17
        return httpx.Response(200, content=gzip.compress(response_body), headers={
            "content-type": "application/json", "content-encoding": "gzip", "x-request-id": "req-fixture",
            "connection": "keep-alive, x-hop-header", "x-hop-header": "remove-me",
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = HttpCloudAdapter("https://example.test/vendor/v1/", "test-secret", timeout=17, client=client)
        reply = await adapter.complete(payload)
        assert reply.status_code == 200
        assert reply.body == response_body
        assert reply.headers["x-request-id"] == "req-fixture"
        assert not {"content-encoding", "content-length", "connection", "x-hop-header"} & reply.headers.keys()
        await adapter.close()
        assert client.is_closed is False
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 429, 500, 503])
async def test_cloud_http_error_is_returned_once_without_retry(status):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(status, json={"error": {"message": "upstream fixture"}}, headers={"retry-after": "60"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        reply = await HttpCloudAdapter("https://example.test/v1", "", client=client).complete({"model": "x"})
    assert reply.status_code == status
    assert json.loads(reply.body)["error"]["message"] == "upstream fixture"
    assert reply.headers["retry-after"] == "60"
    assert calls == 1


@pytest.mark.asyncio
async def test_cloud_stream_retains_exact_event_bytes_and_closes_response():
    raw = 'data: {"choices":[{"delta":{"content":"中文"}}]}\r\n\r\ndata: [DONE]\r\n\r\n'.encode()
    source = ByteStream([raw[index:index + 1] for index in range(len(raw))])
    payload = {"model": "test", "stream": True, "stream_options": {"include_usage": True}, "unknown": [1, 2]}

    def handler(request):
        assert json.loads(request.content) == payload
        return httpx.Response(200, stream=source, headers={"content-type": "text/event-stream", "x-upstream": "yes"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        reply = await HttpCloudAdapter("https://example.test/v1", "", client=client).stream(payload)
        assert reply.headers["x-upstream"] == "yes"
        assert b"".join([chunk async for chunk in reply.chunks]) == raw
        await reply.aclose()
    assert source.closed is True


@pytest.mark.asyncio
async def test_cloud_stream_error_keeps_http_status_and_json_body():
    raw = b'{"error":{"message":"rate limit"}}'
    source = ByteStream([raw])

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(
        429, stream=source, headers={"content-type": "application/json", "retry-after": "8"}))) as client:
        reply = await HttpCloudAdapter("https://example.test/v1", "", client=client).stream({"stream": True})
        assert reply.status_code == 429
        assert reply.headers["retry-after"] == "8"
        assert b"".join([chunk async for chunk in reply.chunks]) == raw
    assert source.closed


@pytest.mark.asyncio
async def test_cloud_cancellation_closes_upstream_without_repeating_request():
    entered = asyncio.Event()
    closed = asyncio.Event()
    calls = 0

    class BlockingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: first\n\n"
            entered.set()
            await asyncio.Event().wait()

        async def aclose(self):
            closed.set()

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, stream=BlockingStream())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        reply = await HttpCloudAdapter("https://example.test/v1", "", client=client).stream({"stream": True})

        async def consume():
            async for _ in reply.chunks:
                pass

        task = asyncio.create_task(consume())
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()
        assert calls == 1


@pytest.mark.asyncio
async def test_unconsumed_stream_can_be_explicitly_closed():
    source = ByteStream([b"anything"])
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=source))) as client:
        reply = await HttpCloudAdapter("https://example.test/v1", "", client=client).stream({"stream": True})
        await reply.aclose()
        assert source.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["complete", "stream"])
async def test_cloud_timeout_is_propagated_without_retry(method):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("fixture timeout", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = HttpCloudAdapter("https://example.test/v1", "", client=client)
        with pytest.raises(httpx.ReadTimeout):
            await getattr(adapter, method)({"model": "test"})
        assert calls == 1


@pytest.mark.asyncio
async def test_owned_clients_close_and_do_not_use_environment_proxies():
    for adapter in [HttpCloudAdapter("https://example.test/v1", ""), OllamaAdapter("http://127.0.0.1:11434", "local")]:
        assert adapter._client._trust_env is False
        await adapter.close()
        assert adapter._client.is_closed is True


@pytest.mark.asyncio
async def test_ollama_maps_only_local_model_supported_options_and_schema():
    schema = {"type": "object", "properties": {"姓名": {"type": "string"}}, "required": ["姓名"]}
    payload = {"model": "must-not-load-this", "messages": [{"role": "user", "content": "张三"}],
               "stream": True, "temperature": 0, "top_p": 0.8, "seed": 4, "max_tokens": 2,
               "max_completion_tokens": 90, "stop": ["END"], "unknown": "not-sent",
               "response_format": {"type": "json_schema", "json_schema": {"name": "person", "schema": schema}}}
    calls = []

    def handler(request):
        calls.append(request)
        assert str(request.url) == "http://ollama.test/api/chat"
        assert json.loads(request.content) == {
            "model": "chosen-local", "messages": payload["messages"], "stream": False, "think": False,
            "keep_alive": "5m", "format": schema,
            "options": {"num_ctx": 4096, "temperature": 0, "top_p": 0.8, "seed": 4, "num_predict": 90, "stop": ["END"]},
        }
        return httpx.Response(200, json={"model": "chosen-local", "message": {"role": "assistant", "content": '{"姓名":"张三"}'},
            "done": True, "prompt_eval_count": 20, "eval_count": 8, "prompt_eval_cached_count": 5})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = OllamaAdapter("http://ollama.test", "chosen-local", client=client)
        reply = await adapter.complete(payload)
        await adapter.close()
        assert not client.is_closed
    body = json.loads(reply.body)
    assert body["model"] == "chosen-local"
    assert json.loads(body["choices"][0]["message"]["content"]) == {"姓名": "张三"}
    assert body["usage"] == {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28,
                             "prompt_tokens_details": {"cached_tokens": 5, "cache_write_tokens": 0},
                             "completion_tokens_details": {"reasoning_tokens": 0}}
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_ollama_missing_usage_stays_unknown_and_json_object_format_maps():
    def handler(request):
        assert json.loads(request.content)["format"] == "json"
        return httpx.Response(200, json={"message": {"content": "{}"}, "done": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        reply = await OllamaAdapter("http://ollama.test", "local", client=client).complete({
            "messages": [], "response_format": {"type": "json_object"}})
    assert json.loads(reply.body)["usage"]["prompt_tokens"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("data", [{}, {"message": {"content": "{}"}, "done": False},
                                 {"message": {"content": "{}", "tool_calls": [{"function": {}}]}, "done": True}])
async def test_ollama_incomplete_or_unexpected_response_fails(data):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=data))) as client:
        reply = await OllamaAdapter("http://ollama.test", "local", client=client).complete({"messages": []})
    assert reply.status_code == 502


@pytest.mark.asyncio
async def test_ollama_http_errors_are_preserved():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(404, json={"error": "missing model"}))) as client:
        reply = await OllamaAdapter("http://ollama.test", "local", client=client).complete({"messages": []})
    assert reply.status_code == 404
    assert json.loads(reply.body) == {"error": "missing model"}


@pytest.mark.asyncio
async def test_mock_local_echoes_json_but_never_fabricates_extraction():
    adapter = MockAdapter("local")
    reply = await adapter.complete({"model": "local", "messages": [{"role": "user", "content": '{"姓名": "张三", "数量": 5}'}]})
    assert json.loads(json.loads(reply.body)["choices"][0]["message"]["content"]) == {"姓名": "张三", "数量": 5}
    assert reply.headers["x-gateway-simulated"] == "true"
    reply = await adapter.complete({"messages": [{"role": "user", "content": "Extract the name from prose"}],
                                    "response_format": {"type": "json_schema", "json_schema": {"schema": {"const": {"name": "fabricated"}}}}})
    assert "模拟本地服务" in json.loads(reply.body)["choices"][0]["message"]["content"]
    assert adapter.complete_calls == 2
    assert adapter.stream_calls == 0
    assert adapter.simulated


@pytest.mark.asyncio
async def test_mock_stream_multi_tools_final_usage_and_close():
    adapter = MockAdapter("cloud")
    reply = await adapter.stream({"model": "mock-cloud", "messages": [{"role": "user", "content": "测试"}],
                                  "tools": [{"type": "function", "function": {"name": name}} for name in ("one", "two")],
                                  "stream_options": {"include_usage": True}})
    observer = SSEObserver()
    pieces = []
    async for piece in reply.chunks:
        pieces.append(piece)
        observer.feed(piece)
    observer.finish()
    assert observer.first_content and observer.saw_done
    assert observer.usage["total_tokens"] == 20
    raw = b"".join(pieces).decode()
    assert "call_mock_0" in raw and "call_mock_1" in raw
    assert '"finish_reason":"tool_calls"' in raw
    assert adapter.stream_calls == 1 and adapter.complete_calls == 0
    await reply.aclose()
    await adapter.close()
    assert adapter.closed
