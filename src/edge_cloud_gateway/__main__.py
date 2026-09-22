"""Run with `python -m edge_cloud_gateway --config config.example.toml`."""

import argparse
from pathlib import Path
import sys

import uvicorn

from .app import MissingEnvironmentVariableError, create_app
from .config import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Local/cloud gateway (mock by default)")
    parser.add_argument("--config", type=Path, default=Path("config.example.toml"))
    args = parser.parse_args()
    try:
        settings = load_settings(args.config)
        app = create_app(settings)
    except MissingEnvironmentVariableError as exc:
        print(f"Cannot start: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except (OSError, ValueError, TypeError) as exc:
        # Do not echo malformed configuration contents, URLs or credentials.
        print(f"Cannot start: {type(exc).__name__}. Check configuration and environment.", file=sys.stderr)
        raise SystemExit(2) from None
    print(f"Mode: {settings.gateway.mode}; listening on {settings.gateway.host}:{settings.gateway.port}")
    uvicorn.run(app, host=settings.gateway.host, port=settings.gateway.port, access_log=False)


if __name__ == "__main__":
    main()
