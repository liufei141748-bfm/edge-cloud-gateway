from dataclasses import replace

import pytest

from edge_cloud_gateway.config import RouterSettings, Settings, load_settings


def test_example_config_loads_and_resolves_database():
    settings = load_settings("config.example.toml")
    assert settings.gateway.mode == "mock"
    assert settings.gateway.database.endswith("data/gateway.sqlite3")
    assert settings.context.enabled and not settings.cloud.enabled and not settings.cache.enabled
    assert settings.router.enabled
    assert settings.observability.save_context_snapshots is False


@pytest.mark.parametrize("field,value", [("min_savings_ratio", -1), ("min_savings_ratio", float("nan")),
                                         ("worker_max_bytes", 0), ("max_blocks", True), ("log_window_lines", -1),
                                         ("enabled", "true"), ("alias", "local-json")])
def test_invalid_context_configuration_is_rejected(field, value):
    settings = Settings()
    with pytest.raises(ValueError):
        replace(settings, context=replace(settings.context, **{field: value})).validate()


def test_nonloopback_bind_and_plaintext_live_cloud_are_rejected():
    settings = Settings()
    with pytest.raises(ValueError):
        replace(settings, gateway=replace(settings.gateway, host="0.0.0.0")).validate()
    with pytest.raises(ValueError):
        replace(settings, gateway=replace(settings.gateway, mode="live"),
                cloud=replace(settings.cloud, enabled=True, base_url="http://cloud.example/v1")).validate()


def test_router_enabled_requires_a_boolean():
    with pytest.raises(ValueError):
        replace(Settings(), router=RouterSettings(enabled="true")).validate()
