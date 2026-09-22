"""API orchestration over the existing adapters, storage and selection policy."""
from contextlib import asynccontextmanager
from copy import deepcopy
import json
import os
import time
import uuid

import anyio
import httpx
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .adapters import OpenAICompatibleAdapter, MockAdapter, OllamaAdapter, HTTPReply

# Compatibility hook retained for existing tests and downstream monkeypatches.
HttpCloudAdapter = OpenAICompatibleAdapter
from .config import Settings
from .context import RawContext, build_working_context
from .evaluation import input_comparison
from .policy import cache_key, decide_route, json_bytes, prepare_local, validate_local
from .pricing import calculate_cost, usage_from_response
from .sse import SSEObserver, completion_to_sse
from .storage import Store


class CloudDisabledError(Exception):
    pass


class MissingEnvironmentVariableError(ValueError):
    """A configured secret name is safe to report; its value is never exposed."""


class LocalDisabledError(Exception):
    pass


class InvalidCloudResponseError(Exception):
    pass


class DisabledCloud:
    simulated = False

    async def complete(self, payload):
        raise CloudDisabledError()

    async def stream(self, payload):
        raise CloudDisabledError()

    async def close(self):
        pass


class DisabledLocal:
    simulated = False

    async def complete(self, payload):
        raise LocalDisabledError()

    async def stream(self, payload):
        raise LocalDisabledError()

    async def close(self):
        pass


def _configured_key(env_name: str, *, required: bool) -> str:
    if not env_name:
        if required:
            raise MissingEnvironmentVariableError("cloud.api_key_env must name an environment variable")
        return ""
    key = os.environ.get(env_name)
    if not key:
        raise MissingEnvironmentVariableError(f"Missing required environment variable: {env_name}")
    return key


def decode_body(body: bytes) -> dict:
    try:
        value = json.loads(body)
        return value if isinstance(value, dict) else {}
    except (ValueError, UnicodeError, RecursionError):
        return {}


def complete_usage(usage) -> bool:
    return isinstance(usage, dict) and all(type(usage.get(k)) is int for k in ("input_tokens", "output_tokens"))


def valid_cloud_completion(body: dict) -> bool:
    """Accept provider extensions while rejecting an obviously invalid 2xx body."""
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices:
        return False
    for choice in choices:
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            return False
        message = choice["message"]
        if not any(field in message for field in ("content", "tool_calls", "refusal")):
            return False
    return True


class Runtime:
    def __init__(self, settings: Settings, cloud=None, local=None, store=None):
        settings.validate()
        self.settings = settings
        if settings.gateway.mode == "mock":
            self.cloud = cloud or MockAdapter("cloud")
            self.local = local or MockAdapter("local")
        else:
            cloud_key = (
                _configured_key(settings.cloud.api_key_env, required=True)
                if cloud is None and settings.cloud.enabled else ""
            )
            local_key = (
                _configured_key(settings.local.api_key_env, required=False)
                if local is None and settings.local.enabled
                and settings.local.provider == "openai_compatible" and settings.local.api_key_env else ""
            )
            if cloud is not None:
                self.cloud = cloud
            elif settings.cloud.enabled:
                self.cloud = HttpCloudAdapter(
                    settings.cloud.base_url, cloud_key, settings.cloud.timeout_seconds,
                )
            else:
                self.cloud = DisabledCloud()
            if local is not None:
                self.local = local
            elif not settings.local.enabled:
                self.local = DisabledLocal()
            elif settings.local.provider == "openai_compatible":
                self.local = OpenAICompatibleAdapter(
                    settings.local.base_url, local_key, settings.local.timeout_seconds,
                )
            else:
                self.local = OllamaAdapter(
                    settings.local.base_url, settings.local.model, settings.local.timeout_seconds,
                    num_ctx=settings.local.num_ctx, keep_alive=settings.local.keep_alive,
                )
        self.store = store or Store(settings.gateway.database)
        self.local_lock = anyio.Semaphore(1)

    async def close(self):
        await self.cloud.close()
        await self.local.close()
        self.store.close()

    async def store_best_effort(self, operation, *args):
        """Keep synchronous SQLite work and failures outside the request event loop."""
        try:
            with anyio.CancelScope(shield=True):
                return True, await anyio.to_thread.run_sync(operation, *args)
        except Exception:
            return False, None

    async def record_attempt(self, ctx, provider, model, start, status, body=None, error_type=None, complete=True):
        elapsed = (time.perf_counter() - start) * 1000
        adapter = self.cloud if provider == "cloud" else self.local
        simulated = adapter.simulated
        usage = usage_from_response(body or {})
        valid = complete and complete_usage(usage)
        ctx["latency_cloud" if provider == "cloud" else "latency_local"] += elapsed
        source = "estimated" if simulated and valid else "actual" if valid else "unknown"
        if provider == "cloud":
            ctx["usage_source"] = source
            ctx["cloud_input_tokens"] = usage["input_tokens"] if valid else None
            ctx["cloud_output_tokens"] = usage["output_tokens"] if valid else None
        else:
            if valid and ctx["local_input_tokens"] is not None and ctx["local_output_tokens"] is not None:
                ctx["local_input_tokens"] += usage["input_tokens"]
                ctx["local_output_tokens"] += usage["output_tokens"]
            else:
                ctx["local_input_tokens"] = ctx["local_output_tokens"] = None
        await self.store_best_effort(self.store.record_attempt, {
            "request_id": ctx["request_id"], "task_id": ctx["task_id"], "attempt_id": uuid.uuid4().hex,
            "provider": provider, "model": model, "status": status, "latency_ms": elapsed,
            "usage": usage, "usage_complete": valid, "simulated": simulated,
            "usage_source": source, "error_type": error_type,
            "cost": calculate_cost(usage, self.settings.cloud.prices)
                if valid and provider == "cloud" and model == self.settings.cloud.model else None,
        })

    async def local_call(self, payload, ctx):
        start, reply, status, error = time.perf_counter(), None, "error", None
        try:
            if isinstance(self.local, DisabledLocal):
                raise LocalDisabledError()
            # One local model at a time; timeout includes queueing and reading.
            with anyio.fail_after(self.settings.local.timeout_seconds):
                async with self.local_lock:
                    ctx["local_model_used"] = True
                    ctx["simulated"] = ctx["simulated"] or self.local.simulated
                    reply = await self.local.complete(payload)
            status = "success" if 200 <= reply.status_code < 300 else "upstream_error"
            return reply
        except BaseException as exc:
            error = type(exc).__name__
            status = "cancelled" if isinstance(exc, anyio.get_cancelled_exc_class()) else "error"
            raise
        finally:
            await self.record_attempt(ctx, "local", self.settings.local.model, start, status,
                                      decode_body(reply.body) if reply else None, error, status == "success")

    def response_headers(self, ctx, headers=None):
        result = dict(headers or {})
        result.update({"x-gateway-request-id": ctx["request_id"], "x-gateway-route": ctx["route"],
                       "x-gateway-reason": ctx["route_reason"],
                       "x-gateway-simulated": str(ctx["simulated"]).lower(),
                       "x-gateway-cache": "hit" if ctx["cache_hit"] else "miss",
                       "cache-control": "no-store"})
        return result

    async def send_complete(self, ctx, reply, scope, receive, send):
        await Response(reply.body, reply.status_code, headers=self.response_headers(ctx, reply.headers))(scope, receive, send)

    async def send_local(self, ctx, body, stream, include_usage, scope, receive, send):
        if not stream:
            await self.send_complete(ctx, HTTPReply(200, json_bytes(body), {"content-type": "application/json"}), scope, receive, send)
            return
        headers = self.response_headers(ctx, {"content-type": "text/event-stream", "x-gateway-local-stream": "buffered-after-validation"})
        await send({"type": "http.response.start", "status": 200, "headers": [(k.encode(), v.encode()) for k, v in headers.items()]})
        for block in completion_to_sse(body, include_usage):
            await send({"type": "http.response.body", "body": block, "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def cloud_call(self, payload, ctx, scope, receive, send):
        start, reply, body, error, status = time.perf_counter(), None, None, None, "error"
        is_stream = payload.get("stream", False)
        observer = SSEObserver()
        stream_complete = False
        try:
            if isinstance(self.cloud, DisabledCloud):
                raise CloudDisabledError()
            ctx["cloud_model_used"] = True
            ctx["simulated"] = ctx["simulated"] or self.cloud.simulated
            if not is_stream:
                reply = await self.cloud.complete(payload)
                body = decode_body(reply.body)
                if 200 <= reply.status_code < 300 and not valid_cloud_completion(body):
                    status = "invalid_response"
                    ctx["status"] = status
                    raise InvalidCloudResponseError()
                status = "success" if 200 <= reply.status_code < 300 else "upstream_error"
                ctx["status"] = status
                await self.send_complete(ctx, reply, scope, receive, send)
                return
            reply = await self.cloud.stream(payload)
            headers = self.response_headers(ctx, reply.headers)
            is_sse = "text/event-stream" in headers.get("content-type", "").lower()
            if 200 <= reply.status_code < 300 and not is_sse:
                status = "invalid_response"
                ctx["status"] = status
                raise InvalidCloudResponseError()
            await send({"type": "http.response.start", "status": reply.status_code,
                        "headers": [(k.encode("latin-1"), v.encode("latin-1")) for k, v in headers.items()]})
            non_sse_body = bytearray()
            async for chunk in reply.chunks:
                if is_sse:
                    observer.feed(chunk)
                elif len(non_sse_body) + len(chunk) <= self.settings.gateway.max_request_bytes:
                    non_sse_body.extend(chunk)
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            observer.finish()
            stream_complete = observer.saw_done if is_sse else True
            body = {"usage": observer.usage} if is_sse else decode_body(bytes(non_sse_body))
            status = "upstream_error" if reply.status_code >= 400 else "success" if stream_complete else "incomplete"
            ctx["status"] = status
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        except BaseException as exc:
            error = type(exc).__name__
            status = "cancelled" if isinstance(exc, anyio.get_cancelled_exc_class()) else "error"
            ctx["status"] = status
            raise
        finally:
            if is_stream and reply is not None:
                with anyio.CancelScope(shield=True):
                    await reply.aclose()
            if body is None and observer.usage is not None:
                body = {"usage": observer.usage}
            if not isinstance(self.cloud, DisabledCloud):
                await self.record_attempt(ctx, "cloud", payload["model"], start, status, body, error,
                                          status == "success" and (not is_stream or stream_complete))

    async def dispatch(self, payload, raw, ctx, cache_mode, scope, receive, send):
        routing = decide_route(payload, self.settings, raw)
        route, reason, decision = routing.route, routing.reason, routing.local_decision
        ctx.update(route=route, reason=reason, route_reason=reason,
                   route_source=routing.source, route_decision_reason=reason,
                   initial_route_decision_reason=reason,
                   router_enabled=self.settings.router.enabled,
                   router_input_tokens=routing.input_tokens,
                   router_threshold_tokens=routing.threshold_tokens)
        if routing.features is not None:
            ctx.update(routing.features.as_metrics())
        cloud_payload = raw.render_all(self.settings)
        ctx.update(input_comparison(cloud_payload, cloud_payload))
        if route != "direct_local" and isinstance(self.cloud, DisabledCloud):
            raise CloudDisabledError()
        if (self.settings.observability.save_context_snapshots
                and raw.has_context and not raw.messages_derived):
            saved, _ = await self.store_best_effort(
                self.store.save_context, ctx["request_id"], json.loads(raw.original_json),
                cloud_payload, raw.package_all(),
            )
            ctx["context_snapshot_saved"] = saved
        if route == "direct_local":
            enabled = self.settings.cache.enabled and cache_mode in {"allow", "refresh"} and (
                self.settings.gateway.mode == "mock" or bool(self.settings.local.model_revision)
            )
            key = cache_key(payload, self.settings) if enabled else None
            if key and cache_mode == "allow":
                _, body = await self.store_best_effort(self.store.get_cache, key)
            else:
                body = None
            if body is not None:
                ctx.update(cache_hit=True, status="success", simulated=self.local.simulated,
                           route_reason="exact_local_cache_hit", reason="exact_local_cache_hit",
                           route_decision_reason="exact_local_cache_hit")
                # A cache hit consumes no new model tokens. Do not replay the
                # original completion's usage and accidentally double count it.
                body = deepcopy(body)
                body["id"] = "chatcmpl-cache-" + ctx["request_id"]
                body["created"] = int(time.time())
                body["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                                 "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                                 "completion_tokens_details": {"reasoning_tokens": 0}}
            else:
                try:
                    reply = await self.local_call(prepare_local(payload, self.settings), ctx)
                    body = decode_body(reply.body)
                    if not 200 <= reply.status_code < 300 or not validate_local(body, decision.schema):
                        body = None
                except Exception:
                    body = None
                if body is not None:
                    if key:
                        await self.store_best_effort(
                            self.store.put_cache, key, body, self.settings.cache.ttl_seconds,
                        )
                    ctx["status"] = "success"
            if body is not None:
                ctx.update(cloud_input_tokens=0, cloud_output_tokens=0,
                           usage_source="estimated" if ctx["simulated"] else "actual")
                await self.send_local(ctx, body, payload.get("stream", False),
                                      bool((payload.get("stream_options") or {}).get("include_usage")), scope, receive, send)
                return
            ctx.update(route="direct_cloud", fallback_used=True,
                       route_reason="local_failed_cloud_fallback", reason="local_failed_cloud_fallback",
                       route_decision_reason="local_failed_cloud_fallback")
        elif route == "context_then_cloud":
            working = await build_working_context(raw, self.settings, lambda p: self.local_call(p, ctx))
            cloud_payload = working.payload
            ctx.update(working.metrics, context_reason=working.reason)
            if not working.optimized:
                ctx.update(route="direct_cloud", route_reason=working.reason, reason=working.reason,
                           route_decision_reason=working.reason)
                ctx["fallback_used"] = working.reason == "context_worker_failed_raw_fallback"
            if self.settings.observability.save_context_snapshots:
                saved, _ = await self.store_best_effort(
                    self.store.save_context, ctx["request_id"], json.loads(raw.original_json),
                    cloud_payload, working.package,
                )
                ctx["context_snapshot_saved"] = saved
        await self.cloud_call(cloud_payload, ctx, scope, receive, send)


class ManagedGatewayResponse(Response):
    """Race the entire upstream operation with disconnect, including first-byte waits.

    Body input has already been consumed, so only this watcher reads receive.
    This also avoids relying on ASGI server versions to cancel a blocked stream.
    """
    def __init__(self, runtime, payload, raw, task_id, cache_mode):
        super().__init__(b"")
        self.runtime, self.payload, self.raw = runtime, payload, raw
        self.task_id, self.cache_mode = task_id, cache_mode

    async def __call__(self, scope, receive, send):
        start = time.perf_counter()
        ctx = {"request_id": uuid.uuid4().hex, "task_id": self.task_id or uuid.uuid4().hex,
               "status": "pending", "route": "direct_cloud", "route_reason": "pending", "reason": "pending",
               "model": self.payload.get("model"), "cache_hit": False, "simulated": False,
               "local_model_used": False, "cloud_model_used": False, "fallback_used": False,
               "latency_local": 0, "latency_cloud": 0, "first_byte_ms": None,
               "cloud_input_tokens": None, "cloud_output_tokens": None, "usage_source": "unknown",
               "local_input_tokens": 0, "local_output_tokens": 0,
               "route_source": None, "route_decision_reason": None,
               "initial_route_decision_reason": None, "context_reason": None,
               "router_enabled": self.runtime.settings.router.enabled,
               "router_input_tokens": None, "router_threshold_tokens": None,
               "routing_features_version": None, "estimated_input_tokens": None,
               "block_count": None, "protected_block_count": None, "protected_ratio": None,
               "selectable_block_count": None, "candidate_tokens": None,
               "local_provider": self.runtime.settings.local.provider,
               "local_model": self.runtime.settings.local.model,
               "cloud_provider": self.runtime.settings.cloud.provider,
               "cloud_model": self.runtime.settings.cloud.model}
        started = False

        async def counted_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            if message["type"] == "http.response.body" and message.get("body") and ctx["first_byte_ms"] is None:
                ctx["first_byte_ms"] = (time.perf_counter() - start) * 1000
            await send(message)

        async with anyio.create_task_group() as group:
            async def run():
                try:
                    await self.runtime.dispatch(self.payload, self.raw, ctx, self.cache_mode, scope, receive, counted_send)
                except anyio.get_cancelled_exc_class():
                    ctx["status"] = "cancelled"
                    raise
                except Exception as exc:
                    ctx["status"] = "error"
                    if not started:
                        status = 503 if isinstance(exc, CloudDisabledError) else 504 if isinstance(exc, (TimeoutError, httpx.TimeoutException)) else 502
                        message = "Cloud calls are disabled or not configured" if status == 503 else "Upstream request failed"
                        await JSONResponse({"error": {"type": type(exc).__name__, "message": message}}, status,
                                           headers=self.runtime.response_headers(ctx))(scope, receive, counted_send)
                    else:
                        # No retry or model switch once a response starts.
                        try:
                            await counted_send({"type": "http.response.body", "body": b"", "more_body": False})
                        except (OSError, RuntimeError):
                            pass
                finally:
                    ctx["latency_ms"] = ctx["latency_total"] = (time.perf_counter() - start) * 1000
                    await self.runtime.store_best_effort(self.runtime.store.record_request, ctx)
                    group.cancel_scope.cancel()

            group.start_soon(run)
            while True:
                event = await receive()
                if event["type"] == "http.disconnect":
                    group.cancel_scope.cancel()
                    break


def create_app(settings: Settings | None = None, *, cloud=None, local=None, store=None) -> FastAPI:
    runtime = Runtime(settings or Settings(), cloud, local, store)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await runtime.close()

    app = FastAPI(title="Adaptive Edge-Cloud LLM Gateway", version="0.1.0", lifespan=lifespan)
    app.state.runtime = runtime
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])

    @app.get("/health")
    async def health():
        return {"status": "ok", "mode": runtime.settings.gateway.mode, "cloud_enabled": runtime.settings.cloud.enabled,
                "local_enabled": runtime.settings.local.enabled,
                "cloud_provider": runtime.settings.cloud.provider, "local_provider": runtime.settings.local.provider,
                "context_snapshots_enabled": runtime.settings.observability.save_context_snapshots,
                "note": "mock is a protocol fixture" if runtime.settings.gateway.mode == "mock" else "no paid readiness probe"}

    @app.get("/v1/models")
    async def models():
        models = [runtime.settings.cloud.model, runtime.settings.context.alias]
        if runtime.settings.local.enabled:
            models.append(runtime.settings.local.alias)
        return {"object": "list", "data": [{"id": model, "object": "model", "created": 0,
                                               "owned_by": "configured-gateway"}
                                              for model in dict.fromkeys(models)]}

    @app.get("/stats")
    async def stats():
        return runtime.store.summary() | {"recent_requests": runtime.store.recent_requests(), "recent_attempts": runtime.store.recent_attempts()}

    @app.get("/contexts/{request_id}")
    async def context_snapshot(request_id: str):
        snapshot = runtime.store.get_context(request_id)
        return snapshot if snapshot else JSONResponse({"error": "context snapshot not found"}, 404)

    @app.post("/v1/chat/completions")
    async def completion(request: Request):
        if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
            return JSONResponse({"error": "application/json is required"}, 415)
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > runtime.settings.gateway.max_request_bytes:
                return JSONResponse({"error": "request too large"}, 413)
        try:
            original = json.loads(data, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            if (not isinstance(original, dict) or not isinstance(original.get("model"), str)
                    or not original["model"] or not isinstance(original.get("messages"), list)
                    or not original["messages"] or not all(isinstance(m, dict) and isinstance(m.get("role"), str) for m in original["messages"])
                    or type(original.get("stream", False)) is not bool
                    or (original.get("stream_options") is not None and not isinstance(original["stream_options"], dict))):
                raise ValueError("invalid request shape")
            raw = RawContext.from_request(original, runtime.settings)
        except (ValueError, TypeError, RecursionError):
            return JSONResponse({"error": "Invalid request or gateway_context; check the documented schema"}, 400)
        task_id = request.headers.get("x-gateway-task-id", "")[:128]
        cache_mode = request.headers.get("x-gateway-cache", "bypass")
        return ManagedGatewayResponse(runtime, raw.payload, raw, task_id, cache_mode)

    return app
