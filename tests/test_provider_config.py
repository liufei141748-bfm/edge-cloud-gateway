from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from edge_cloud_gateway.adapters import MockAdapter, OpenAICompatibleAdapter
from edge_cloud_gateway.app import MissingEnvironmentVariableError, Runtime, create_app
from edge_cloud_gateway.config import CloudSettings, GatewaySettings, LocalSettings, Settings, load_settings
from edge_cloud_gateway.policy import prepare_local
from edge_cloud_gateway.storage import Store


def live_settings(**changes):
    settings = Settings(
        gateway=GatewaySettings(mode="live", database=":memory:"),
        cloud=CloudSettings(
            enabled=True,
            provider="openai_compatible",
            base_url="https://cloud.example/custom/v1",
            model="custom-cloud-model",
            api_key_env="CLOUD_API_KEY",
        ),
        local=LocalSettings(
            enabled=True,
            provider="openai_compatible",
            base_url="http://local.example:8000/v1",
            model="custom-local-model",
            api_key_env="LOCAL_API_KEY",
        ),
    )
    return replace(settings, **changes)


@pytest.mark.asyncio
async def test_custom_openai_compatible_urls_and_api_key_env(monkeypatch):
    monkeypatch.setenv("LOCAL_API_KEY", "local-test-key")
    monkeypatch.setenv("CLOUD_API_KEY", "cloud-test-key")
    runtime = Runtime(live_settings(), store=Store(":memory:"))
    try:
        assert isinstance(runtime.local, OpenAICompatibleAdapter)
        assert isinstance(runtime.cloud, OpenAICompatibleAdapter)
        assert runtime.local.url == "http://local.example:8000/v1/chat/completions"
        assert runtime.cloud.url == "https://cloud.example/custom/v1/chat/completions"
        assert runtime.local._request_headers == {"authorization": "Bearer local-test-key"}
        assert runtime.cloud._request_headers == {"authorization": "Bearer cloud-test-key"}
    finally:
        await runtime.close()


@pytest.mark.parametrize("missing", ["LOCAL_API_KEY", "CLOUD_API_KEY"])
def test_missing_configured_api_key_names_the_environment_variable(monkeypatch, missing):
    monkeypatch.setenv("LOCAL_API_KEY", "local-test-key")
    monkeypatch.setenv("CLOUD_API_KEY", "cloud-test-key")
    monkeypatch.delenv(missing)
    with pytest.raises(MissingEnvironmentVariableError, match=missing):
        Runtime(live_settings(), store=Store(":memory:"))


def test_local_openai_compatible_key_is_optional(monkeypatch):
    monkeypatch.setenv("CLOUD_API_KEY", "cloud-test-key")
    settings = live_settings(local=replace(live_settings().local, api_key_env=""))
    runtime = Runtime(settings, store=Store(":memory:"))
    assert runtime.local._request_headers == {}


def test_custom_models_are_used_for_local_and_adaptive_cloud_requests():
    settings = replace(
        Settings(),
        cloud=replace(Settings().cloud, model="custom-cloud-model"),
        local=replace(Settings().local, model="custom-local-model"),
    )
    assert prepare_local({
        "model": settings.local.alias,
        "messages": [{"role": "user", "content": "{}"}],
        "response_format": {"type": "json_schema", "json_schema": {
            "schema": {"type": "object"},
        }},
    }, settings)["model"] == "custom-local-model"

    cloud = MockAdapter("cloud")
    app = create_app(settings, cloud=cloud, store=Store(":memory:"))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "adaptive", "messages": [{"role": "user", "content": "short request"}],
        })
        assert response.status_code == 200
        assert cloud.calls[-1]["model"] == "custom-cloud-model"


def test_local_disabled_skips_local_and_uses_cloud_quality_fallback():
    settings = replace(Settings(), local=replace(Settings().local, enabled=False))
    cloud, local = MockAdapter("cloud"), MockAdapter("local")
    app = create_app(settings, cloud=cloud, local=local, store=Store(":memory:"))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={
            "model": settings.local.alias,
            "messages": [{"role": "user", "content": "{}"}],
            "response_format": {"type": "json_schema", "json_schema": {
                "schema": {"type": "object"},
            }},
        })
        assert response.status_code == 200
        assert response.headers["x-gateway-route"] == "direct_cloud"
        assert response.headers["x-gateway-reason"] == "local_provider_disabled"
        assert local.complete_calls == 0 and cloud.complete_calls == 1
        assert settings.local.alias not in {m["id"] for m in client.get("/v1/models").json()["data"]}


def test_cloud_disabled_returns_clear_503_without_network():
    settings = Settings(gateway=GatewaySettings(mode="live", database=":memory:"),
                        local=replace(Settings().local, enabled=False))
    app = create_app(settings, store=Store(":memory:"))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "adaptive", "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 503
        assert response.json()["error"]["message"] == "Cloud calls are disabled or not configured"


def test_public_provider_examples_load_and_validate():
    for path in (
        "examples/configs/ollama-deepseek.toml",
        "examples/configs/ollama-openai.toml",
        "examples/configs/generic-openai-compatible.toml",
    ):
        settings = load_settings(path)
        assert settings.gateway.mode == "live"
        assert settings.local.enabled and settings.cloud.enabled
