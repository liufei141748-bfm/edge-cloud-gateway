"""Configuration has no network or model-loading side effects."""

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
import ipaddress
import math
import tomllib


@dataclass(frozen=True)
class GatewaySettings:
    mode: str = "mock"
    host: str = "127.0.0.1"
    port: int = 8787
    database: str = "data/gateway.sqlite3"
    max_request_bytes: int = 4 * 1024 * 1024


@dataclass(frozen=True)
class CloudSettings:
    enabled: bool = False
    provider: str = "openai_compatible"
    base_url: str = ""
    model: str = "mock-cloud"
    api_key_env: str = "CLOUD_API_KEY"
    timeout_seconds: float = 60
    prices: dict = field(default_factory=dict)


@dataclass(frozen=True)
class LocalSettings:
    enabled: bool = True
    provider: str = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    model: str = "local-model"
    api_key_env: str = ""
    model_revision: str = ""
    alias: str = "local-json"
    num_ctx: int = 4096
    keep_alive: str = "5m"
    timeout_seconds: float = 30
    max_payload_bytes: int = 3072
    max_output_tokens: int = 512


@dataclass(frozen=True)
class CacheSettings:
    enabled: bool = False
    ttl_seconds: float = 86400


@dataclass(frozen=True)
class RouterSettings:
    enabled: bool = True


@dataclass(frozen=True)
class ObservabilitySettings:
    save_context_snapshots: bool = False


@dataclass(frozen=True)
class ContextSettings:
    enabled: bool = True
    alias: str = "adaptive"
    min_input_tokens: int = 512
    min_savings_ratio: float = 0.05
    worker_max_bytes: int = 12288
    max_blocks: int = 128
    log_window_lines: int = 2
    log_min_lines: int = 40


def is_loopback(host: str | None) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class Settings:
    gateway: GatewaySettings = field(default_factory=GatewaySettings)
    cloud: CloudSettings = field(default_factory=CloudSettings)
    local: LocalSettings = field(default_factory=LocalSettings)
    cache: CacheSettings = field(default_factory=CacheSettings)
    router: RouterSettings = field(default_factory=RouterSettings)
    observability: ObservabilitySettings = field(default_factory=ObservabilitySettings)
    context: ContextSettings = field(default_factory=ContextSettings)

    def validate(self) -> None:
        if self.gateway.mode not in {"mock", "live"}:
            raise ValueError("gateway.mode must be mock or live")
        if not is_loopback(self.gateway.host):
            raise ValueError("This single-user gateway must listen on loopback")
        for value, name in [
            (self.gateway.port, "gateway.port"),
            (self.gateway.max_request_bytes, "gateway.max_request_bytes"),
            (self.local.num_ctx, "local.num_ctx"),
            (self.local.max_payload_bytes, "local.max_payload_bytes"),
            (self.local.max_output_tokens, "local.max_output_tokens"),
            (self.context.min_input_tokens, "context.min_input_tokens"),
            (self.context.worker_max_bytes, "context.worker_max_bytes"),
            (self.context.max_blocks, "context.max_blocks"),
            (self.context.log_min_lines, "context.log_min_lines"),
        ]:
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.gateway.port > 65535:
            raise ValueError("gateway.port must be at most 65535")
        for value, name in [
            (self.cloud.timeout_seconds, "cloud.timeout_seconds"),
            (self.local.timeout_seconds, "local.timeout_seconds"),
            (self.cache.ttl_seconds, "cache.ttl_seconds"),
        ]:
            if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.local.alias or self.local.alias == self.cloud.model:
            raise ValueError("local.alias must be nonempty and distinct from cloud.model")
        if (type(self.cloud.enabled) is not bool or type(self.local.enabled) is not bool
                or type(self.cache.enabled) is not bool
                or type(self.router.enabled) is not bool):
            raise ValueError("enabled settings must be booleans")
        if type(self.context.enabled) is not bool:
            raise ValueError("context.enabled must be boolean")
        if type(self.observability.save_context_snapshots) is not bool:
            raise ValueError("observability.save_context_snapshots must be boolean")
        if (not self.context.alias or self.context.alias in {self.local.alias, self.cloud.model}):
            raise ValueError("context.alias must be distinct from local alias and cloud model")
        if (type(self.context.min_savings_ratio) not in {float, int}
                or not 0 <= self.context.min_savings_ratio < 1):
            raise ValueError("context.min_savings_ratio must be between zero and one")
        if type(self.context.log_window_lines) is not int or self.context.log_window_lines < 0:
            raise ValueError("context.log_window_lines must be a nonnegative integer")
        if self.cloud.provider != "openai_compatible":
            raise ValueError("cloud.provider must be openai_compatible")
        if self.local.provider not in {"ollama", "openai_compatible"}:
            raise ValueError("local.provider must be ollama or openai_compatible")
        if self.local.enabled:
            local_url = urlsplit(self.local.base_url)
            if (local_url.scheme not in {"http", "https"} or not local_url.hostname
                    or local_url.username or local_url.password or local_url.query or local_url.fragment):
                raise ValueError("local.base_url must be an HTTP(S) service root without embedded credentials")
            if not self.local.model:
                raise ValueError("enabled local provider requires a model")
            if self.local.api_key_env and not self.local.api_key_env.strip():
                raise ValueError("local.api_key_env must be empty or name an environment variable")
        if self.gateway.mode == "live" and self.cloud.enabled:
            cloud_url = urlsplit(self.cloud.base_url)
            if (cloud_url.scheme != "https" or not cloud_url.hostname or cloud_url.username
                    or cloud_url.password or cloud_url.query or cloud_url.fragment):
                raise ValueError("live cloud.base_url must be HTTPS without embedded credentials")
            if not self.cloud.model or not self.cloud.api_key_env:
                raise ValueError("live cloud requires model and api_key_env")


def load_settings(path: str | Path) -> Settings:
    path = Path(path).resolve()
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    allowed = {"gateway", "cloud", "local", "cache", "router", "observability", "context"}
    if data.keys() - allowed:
        raise ValueError("Unknown configuration section")
    settings = Settings(
        gateway=GatewaySettings(**data.get("gateway", {})),
        cloud=CloudSettings(**data.get("cloud", {})),
        local=LocalSettings(**data.get("local", {})),
        cache=CacheSettings(**data.get("cache", {})),
        router=RouterSettings(**data.get("router", {})),
        observability=ObservabilitySettings(**data.get("observability", {})),
        context=ContextSettings(**data.get("context", {})),
    )
    database = Path(settings.gateway.database)
    if settings.gateway.database != ":memory:" and not database.is_absolute():
        from dataclasses import replace
        settings = replace(settings, gateway=replace(settings.gateway, database=str(path.parent / database)))
    settings.validate()
    return settings
