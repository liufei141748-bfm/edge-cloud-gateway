# Routing contract — Experimental V1

## Boundary

V1 separates routing into two thin layers without changing the context-selection algorithm:

1. `UniversalSafetyLayer.assess(...)` extracts content-free structural features and authoritative protection signals.
2. `RuleBasedPolicy.decide(...)` consumes that assessment and returns a `RouteDecision`.

Future learned or hybrid policies may replace or assist `RuleBasedPolicy`, but must remain downstream of the Safety Layer. V1 implements no learned policy, scorer, training pipeline, RL, bandit, automatic tuning, telemetry upload, or shared workload dataset.

## RoutingFeatures schema

Schema version: `routing_features_v1`.

| Field | Type | Meaning |
| --- | --- | --- |
| `estimated_input_tokens` | integer | deterministic request-input estimate, not provider billing tokens |
| `block_count` | integer | Raw Context block count |
| `protected_block_count` | integer | blocks protected by non-bypassable rules |
| `protected_ratio` | number | protected blocks divided by all blocks |
| `selectable_block_count` | integer | unique blocks eligible for local selection |
| `candidate_tokens` | integer | deterministic estimate over selector-candidate text |
| `local_provider` | string | configured local adapter type |
| `local_model` | string | configured local model identifier |
| `cloud_provider` | string | configured cloud adapter type |
| `cloud_model` | string | configured cloud model identifier |
| `router_enabled` | boolean | whether automatic rule routing is enabled |

The schema contains no prompt, message, Raw Context, or Working Context text. Runtime outcome metadata is recorded separately: final route and reason, raw/working token estimates, cloud/local input/output tokens when known, compression ratio, local/cloud/total latency, fallback, and request status.

## Decision reason vocabulary

`initial_route_decision_reason` preserves the first RuleBasedPolicy result. `route_decision_reason` is the final machine-readable reason after context processing or fallback. `context_reason` records the selector/context-engine result when that path runs.

### Final or initial route decisions

| Reason | Trigger |
| --- | --- |
| `requested_cloud_model` | request model is not the local alias and no adaptive/context rule supersedes it |
| `explicit_standalone_json` | explicit local alias passes the conservative standalone JSON-schema checks |
| `local_provider_disabled` | a local-dependent route is unavailable by configuration |
| `explicit_optimize_false` | explicit full-context baseline |
| `explicit_optimize_true` | explicit context-selection request before selector outcome |
| `auto_router_disabled` | adaptive request while automatic routing is disabled |
| `context_optimization_disabled` | context optimization disabled by configuration |
| `tool_chain_protected` | tool definitions, tool calls, function calls, or tool messages require full-state pass-through |
| `opaque_or_multimodal_state_protected` | unsupported fields or non-text message content cannot be safely reconstructed |
| `auto_below_context_threshold` | adaptive input at or below the configured candidate threshold |
| `context_too_short_to_optimize` | non-auto context input at or below the threshold |
| `no_optional_context` | no block is eligible for removal |
| `context_no_selectable_blocks` | long adaptive input has neither selector candidates nor a safe rule-only reduction |
| `auto_at_or_above_context_threshold` | adaptive input is a candidate for context selection; this does not force reduction |
| `long_context_and_complex_task` | legacy context-default route remains eligible for selection |

### Local eligibility decisions

| Reason | Trigger |
| --- | --- |
| `local_unsupported_fields` | explicit local request contains unsupported request fields |
| `local_requires_standalone_text` | local JSON mode is not a single standalone user-text request |
| `local_requires_json_schema` | required JSON-schema response format is absent |
| `local_unsupported_schema` | schema is not a safe local object schema |
| `local_invalid_schema` | schema validation fails |
| `local_output_budget_exceeded` | requested output exceeds the configured local bound |
| `local_conflicting_output_limits` | both supported output-limit fields are supplied |
| `local_requires_temperature_zero` | explicit local JSON mode is non-deterministic |
| `local_input_budget_exceeded` | prepared local request exceeds the configured bound |

### Final context/fallback outcomes

| Reason | Trigger |
| --- | --- |
| `selected_original_context` | Context Selector produced a safe, net-smaller Working Context; stored as `context_reason` |
| `context_worker_failed_raw_fallback` | selector call or validation failed; full Raw Context restored |
| `context_worker_budget_exceeded` | candidates could not fit the worker budget and no net reduction resulted |
| `context_no_net_reduction` | proposed Working Context did not meet the minimum savings ratio |
| `local_failed_cloud_fallback` | explicit local execution failed before delivery and fell back once to cloud |
| `exact_local_cache_hit` | explicit opt-in exact local cache served the validated response |

Reason strings are retained for compatibility. New code should use the documented fields rather than infer semantics from human-readable logs.
