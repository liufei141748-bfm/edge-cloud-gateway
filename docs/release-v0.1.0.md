# v0.1.0 — Experimental V1

## Suggested repository metadata

- Repository name: `edge-cloud-gateway`
- Description: `Quality-first, token-efficient edge-cloud LLM gateway with automatic routing and safe local context selection.`
- Topics: `llm`, `llm-gateway`, `edge-ai`, `local-llm`, `context-optimization`, `token-efficiency`, `openai-compatible`, `ollama`, `ai-infrastructure`, `llm-routing`
- Release title: `v0.1.0 — Experimental V1`

## Release notes draft

Experimental V1 establishes a provider-configurable, quality-first edge/cloud LLM gateway while keeping the existing routing and context-protection core intact.

Highlights:

- OpenAI-compatible `POST /v1/chat/completions`, including streaming and legal tool-call responses;
- three observable routes: `direct_cloud`, `context_then_cloud`, and explicit `direct_local`;
- ordinary `model=adaptive` messages enter the Auto Router without client-specific adapters;
- provenance-tracked Raw Context and literal-slice Working Context;
- constraint protection and full-request fallback when selection is unsafe, fails, or produces no net reduction;
- configurable cloud OpenAI-compatible provider;
- configurable local Ollama or OpenAI-compatible provider;
- configurable model, Base URL, and environment-variable-based credentials;
- offline mock mode, 36-case synthetic danger A/B dry-run, and a network-blocked regression suite;
- local metrics for routing, fallback, latency, usage source, and context reduction.
- stable content-free `RoutingFeatures`, a non-bypassable `UniversalSafetyLayer`, and a thin `RuleBasedPolicy` boundary for future local policy research;
- prompt/context snapshot persistence disabled by default and available only as an explicit debug/evaluation opt-in.

Important limitations:

- Experimental, not production-ready;
- the reported **66.72% token reduction** is a simulated result from the fixed, committed fixture dry-run only; it is not a benchmark for general or real workloads and does not represent actual billing, quality, or latency;
- Chat Completions only;
- no client authentication, TLS termination, multi-tenant isolation, or automatic snapshot retention;
- local direct execution is restricted to validated standalone JSON-schema tasks;
- provider combinations are compatible at the protocol/configuration layer and have not each been authenticated against a live service;
- fixture and dry-run results do not guarantee real-model quality, latency, savings, or cost.

No real provider or paid API call is part of this release verification.
