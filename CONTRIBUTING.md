# Contributing to Edge-Cloud Gateway

Thank you for your interest in contributing to Edge-Cloud Gateway.

This project is currently Experimental V1. Contributions should preserve its quality-first, conservative-routing, and offline-first principles.

## Before you start

- Search existing issues before opening a new one.
- Use synthetic or anonymized examples only.
- Never include API keys, credentials, private prompts, request logs, local databases, or other sensitive information.
- Security vulnerabilities must be reported privately according to `SECURITY.md`.

## Development setup

Requirements:

- Python 3.12
- Git

Clone and install the project:

```bash
git clone https://github.com/liufei141748-bfm/edge-cloud-gateway.git
cd edge-cloud-gateway
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[test]"
```

Run the automated tests:

```bash
.venv/bin/python -m pytest -q
```

Check installed dependencies:

```bash
.venv/bin/python -m pip check
```

## Contribution guidelines

- Keep changes focused on one clearly defined problem.
- Avoid unrelated refactoring.
- Preserve OpenAI-compatible request and response behavior.
- Preserve system and developer instructions, tool state, structured data, code, constraints, and other protected content.
- When safe context reduction cannot be demonstrated, retain the existing full-request fallback behavior.
- Do not retry or switch providers after a streamed response has started.
- Add or update regression tests for behavior changes.
- Keep the default configuration offline and free from real provider calls.
- Do not present mock or dry-run results as evidence of real-model quality, latency, token savings, or cost savings.

## Pull requests

A pull request should include:

- a concise description of the problem;
- an explanation of the proposed change;
- the tests that were run;
- relevant limitations or unresolved risks;
- confirmation that no credentials or private data are included.

By contributing, you agree that your contribution will be licensed under the project’s MIT License.
