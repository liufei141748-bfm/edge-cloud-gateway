# Security Policy

## Project status

Edge-Cloud Gateway is currently Experimental V1 and is intended for local evaluation and research.

It has not completed production hardening, client authentication, TLS termination, multi-tenant isolation, or a production-grade data-retention policy. Do not expose the service directly to a LAN or the public internet.

## Supported versions

| Version | Supported |
| --- | --- |
| 0.1.x | Best-effort security fixes |
| Earlier versions | Not supported |

## Reporting a vulnerability

Please do not report security vulnerabilities through public GitHub issues, discussions, pull requests, or comments.

Use GitHub Private Vulnerability Reporting:

https://github.com/liufei141748-bfm/edge-cloud-gateway/security/advisories/new

Include enough information to reproduce and assess the issue, while removing credentials, API keys, private prompts, personal data, production logs, and other sensitive information.

Useful details include:

- the affected version or commit;
- the affected component;
- reproduction steps using synthetic data;
- the expected and observed behavior;
- the potential security impact;
- a suggested mitigation, if available.

## Security-sensitive areas

Reports are especially useful when they involve:

- credential or authorization-header exposure;
- unintended storage or disclosure of prompt or context data;
- provider URL or redirect handling;
- request and response passthrough;
- JSON Schema validation;
- tool-call or streaming state corruption;
- unsafe routing or fallback behavior;
- local database or context-snapshot exposure;
- attempts to bypass the loopback-only network boundary.

## Response process

Security reports are reviewed on a best-effort basis. After validation, the maintainer will assess the impact, prepare a fix when appropriate, add regression coverage, and coordinate disclosure through the private advisory.

Because this is an experimental project, no guaranteed response or remediation timeline is currently offered.
