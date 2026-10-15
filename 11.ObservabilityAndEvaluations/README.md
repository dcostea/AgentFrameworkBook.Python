# Chapter 11 — Observability and evaluations (Python)

Four runnable project migrations, targeting Agent Framework **1.8.1** and OpenTelemetry **1.42.1**. Default runs are offline: simulated motors and in-memory telemetry, or dataset validation plus deterministic safety evaluation. Nothing downloads a dashboard, starts Docker, registers an identity, or sends telemetry by default.

## Run from the workspace root

```powershell
# Validate all 12 datasets / 30 records and score the three direction fixtures (offline).
.\.venv\Scripts\python.exe .\11.ObservabilityAndEvaluations\Evaluators\main.py

# Execute real reusable motor simulation tools with real middleware and local OTel.
.\.venv\Scripts\python.exe .\11.ObservabilityAndEvaluations\Observations\main.py

# Console export is a separate opt-in; still no model or remote telemetry.
.\.venv\Scripts\python.exe .\11.ObservabilityAndEvaluations\Observations\main.py --telemetry console

# Live agent requests use the existing shared SDK lifecycle/credentials.
.\.venv\Scripts\python.exe .\11.ObservabilityAndEvaluations\Observations\main.py --live --provider openai
.\.venv\Scripts\python.exe .\11.ObservabilityAndEvaluations\Observations\main.py --live --provider azure --prompt "Go back and forth 3 meters"

# Explicit-start launcher, not a .NET Aspire AppHost.
.\.venv\Scripts\python.exe .\11.ObservabilityAndEvaluations\Observations.AppHost\main.py
.\.venv\Scripts\python.exe .\11.ObservabilityAndEvaluations\Observations.AppHost\main.py --start

# Standard OTel identity adaptation — NOT Agent365 SDK export.
.\.venv\Scripts\python.exe .\11.ObservabilityAndEvaluations\ObservationsWithAgent365\main.py --tenant-id example-tenant --agent-id example-agent

# Makes provider requests: each selected metric is evaluated by a custom model judge.
.\.venv\Scripts\python.exe .\11.ObservabilityAndEvaluations\Evaluators\main.py --model-backed --dataset all --provider openai
.\.venv\Scripts\python.exe .\11.ObservabilityAndEvaluations\Evaluators\main.py --model-backed --dataset composite --provider azure

# Focused tests: no live model, identity authentication, or remote exporter.
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_observability_evaluations.py -q
```

All `main.py` files add `Path(__file__).resolve().parents[2]` to `sys.path`. Live interactive sessions stop on empty input, `exit`, or EOF. `--prompt` is a bounded single request. SDK clients are closed in `finally`, including failure paths.

## Observations: real instrumentation, safe routing

`shared.observability_examples.TelemetryRuntime` owns tracer, meter and logger providers, and flushes/shuts down all three signals. The source instrumentation scope is preserved as `OpenTelemetry.RobbyAgent`; agent name is `RobotCarDemoAgent`. AF1.8.1 renamed the older `setup_observability` API: the live process uses actual `configure_otel_providers(enable_sensitive_data=False)` with owned OTel SDK providers. Native Agent Framework agent/chat/tool instrumentation and custom middleware are not replaced with print statements.

The actual `FunctionMiddleware` runs after successful tool execution and observes `FunctionInvocationContext.result`. Python `forward`, `backward`, `turn_left`, `turn_right`, `stop` map explicitly to source `Forward`, `Backward`, `TurnLeft`, `TurnRight`, `Stop`. Every non-linear completed command resets the **tracker**, matching C#. Tracker state persists across turns in the same agent; failures/cancellation do not falsely count a successful movement. Parallel tool calls are disabled for robot sequencing; any concurrent calls would otherwise be tracked in completion order.

Signals include:

- `robot_direction_change_total`, with original `from`/`to` labels;
- `Detect Robot Direction Change` spans and `Robot direction changed` events, including previous/current/changed/agent/tool attributes;
- real tool invocation spans, completed/error counters and seconds-duration histogram;
- observable gauges for active tool calls and last linear direction (1 forward, -1 backward, 0 reset);
- structured agent start/completion/error and successful-tool log records, plus AF native GenAI duration/token-usage metrics during live agent calls.

Additional counters/gauges/histograms make the Python teaching example inspectable; they were not extra metrics in the source. No chat contents, tool argument/result strings or secrets are copied into custom telemetry. `local` stores telemetry in memory; `console` explicitly writes it to stdout. For long-running sessions use a bounded production collector rather than this demo's retained in-memory snapshot.

## Aspire purpose, not fictitious Python AppHost parity

C# `AddProject(...).WithExplicitStart()` is adapted to `--start`. The launcher uses **current `sys.executable`**, launches sibling `Observations\main.py`, inherits console IO, waits for exit and returns its status. It never launches a persistent dashboard/server or invokes a shell command.

To send traces, metrics and structured logs to an **already running** Aspire dashboard/collector, explicitly choose an OTLP mode/endpoint:

```powershell
# Receiver must already be configured separately. Original C# receiver used 19190.
.\.venv\Scripts\python.exe .\11.ObservabilityAndEvaluations\Observations.AppHost\main.py --start --otlp-endpoint http://localhost:19190
.\.venv\Scripts\python.exe .\11.ObservabilityAndEvaluations\Observations\main.py --telemetry otlp --otlp-endpoint http://localhost:4318 --otlp-protocol http/protobuf
```

The launcher can alternatively read `OTEL_EXPORTER_OTLP_ENDPOINT` or `ASPIRE_DASHBOARD_OTLP_ENDPOINT_URL`. The dashboard UI URL is not an OTLP receiver URL. gRPC passes the receiver endpoint; HTTP/protobuf appends `/v1/traces`, `/v1/metrics`, `/v1/logs` to the base receiver URL. `OTEL_EXPORTER_OTLP_HEADERS` and signal-specific header variables are handled by the official exporter SDK (for example an existing dashboard receiver API key). Don't put authentication in URLs or commit headers.

Standalone dashboard telemetry does **not** create Aspire managed resources, health probes, resource-service connections or start/stop UI controls. Original appsettings/launchSettings/aspire.config.json remain source-only .NET infrastructure configuration; no misleading Python copy is used. The original unsecured/anonymous dashboard settings are not enabled by the migration.

## Agent365: explicit supported adaptation and blocker

There **is** a first-party Python Agent365 SDK. It is not true that Agent365 has no Python equivalent. Microsoft now recommends `microsoft-opentelemetry`, replacing the older `microsoft-agents-a365-observability-*` SDK. The published PyPI `microsoft-opentelemetry==1.3.9` declares `opentelemetry-sdk~=1.44.0` (not compatible with OTel1.42.1) and its `agent-framework` extra accepts `agent-framework>=1.4.0`. The current GitHub main manifest instead raises that AF extra to `agent-framework>=1.19.0`; main-branch documentation must not be mistaken for the published wheel. The published package is missing here and cannot faithfully be validated without changing the parent's OTel1.42.1 dependency baseline. The older SDK is deprecated; we do not silently import it as parity.

`ObservationsWithAgent365` therefore implements **standard OTel identity mode only**, with a prominent runtime notice. Tenant/agent metadata are resource attributes and scoped OTel baggage; they do not obtain tokens, register agents, invoke Microsoft scopes, resolve regional A365 endpoints or export to the A365 service. `--telemetry a365` raises an explicit actionable limitation instead of silently routing to generic OTLP or declaring success.

For a future real SDK integration, first choose and validate a compatible dependency baseline separately. The official Python API is `use_microsoft_opentelemetry(enable_a365=True, a365_token_resolver=..., a365_use_s2s_endpoint=True)` plus `BaggageBuilder().tenant_id(...).agent_id(...).build()`. The source C# uses **custom-engine client credentials** for `api://9b975845-388f-4429-889e-eab1ef63949c/.default`. Microsoft documents custom-engine app registration/permissions/admin-consent requirements and a different two-step agentic identity chain for Agent365-enabled identities. These are not interchangeable, and this migration performs no identity/cloud-access changes or guessed authentication.

## Evaluator mapping and algorithm honesty

Every source test criterion is runnable with `--model-backed`; none is silently dropped. All twelve datasets are unchanged source JSONL bytes. Record loading validates required types, duplicate JSON properties/IDs, UTF-8 and physical line numbers. Direction data skips blank lines as C# does; general datasets reject them. Historical fixture models are metadata; `--provider` and optional `--judge-model` select the actual model/deployment.

The deterministic evaluator preserves safety principles: reject zero calls or unknown moves; only **Stop** clears prior linear direction; turns do not clear it; forward/backward reversal without Stop fails. It reports the first violation, boolean `DirectionChange`, source-style reason and Good/Unacceptable+Failed interpretation. **Source bug correction:** C# `toolName is "forward" or "backward"` is case-sensitive, yet the source fixtures contain `Forward`/`Backward` and expect reversals to be detected. Python case-normalizes classification while preserving original From/To spelling. This deliberate repair makes the shipped source expectations work, rather than reproducing the faulty switch. It is explicitly different from the tracker, which resets on every non-linear command.

Python/Azure first-party evaluators exist, but shared names do not prove algorithm parity. This lean migration does **not** install `azure-ai-evaluation` or claim to call its built-ins. Every non-direction result identifies provenance as **custom-openai-schema-judge (NOT Microsoft .NET/Azure evaluator algorithm)**. The following APIs are comparison references, not hidden fallbacks:

| C# criterion | Related official Python API/concept | Actual migrated behavior |
|---|---|---|
| IntentResolution | `IntentResolutionEvaluator` | Custom judge of resolving the intent; numeric 1–5 |
| Coherence | `CoherenceEvaluator` | Custom logical-order/consistency judge; numeric 1–5 |
| Relevance | `RelevanceEvaluator` | Custom direct relevance judge; numeric 1–5 |
| Fluency | `FluencyEvaluator` | Custom grammar/readability judge; source threshold **Average (3)** |
| Groundedness | `GroundednessEvaluator(response, context=groundTruth)` | Custom precision judge using supplied grounding |
| Equivalence | `SimilarityEvaluator` is related, not .NET Equivalence parity | Custom semantic-equivalence judge with reference |
| Completeness | `ResponseCompletenessEvaluator` (preview), related not identical | Custom required-element/recall judge with reference |
| Retrieval | `RetrievalEvaluator(query, context)` | Custom retrieved-chunk relevance/ranking judge; does not score final answer |
| Composite | No assumed identical composite implementation | Separate Coherence **and** Equivalence metrics; never average |
| TaskAdherence | `TaskAdherenceEvaluator`, different input/output semantics | Custom task/instruction judge over ordered actual call names/arguments |
| ToolCallAccuracy | `ToolCallAccuracyEvaluator`, service commonly 1–5, not source boolean | Custom strict boolean over tool choice, arguments, values/order/completeness |
| DirectionChange | Source-specific custom evaluator | Deterministic, offline, first unsafe reversal |

All numeric criteria except fluency pass at **Good (4)**, preserving source interpretation cutoffs rather than applying Azure default threshold 3. Composite success requires both passing; comparison with fixture `shouldPass` checks **each** metric separately (a fail/pass mixture does not count as matching a negative fixture). ToolCallAccuracy preserves a real boolean value and Good/Unacceptable interpretation, not a fabricated conversion from Azure scoring.

Judges use a strict JSON schema, integer 1–5 or boolean as appropriate, nonempty evidence/reason, no extra fields, and bounded schema retry. Refusal/truncation/invalid output and provider errors fail explicitly. Dataset `shouldPass`, `expectedBehavior`, IDs/scenarios and recorded tool results never leak into judge prompts. Model-backed tests remain probabilistic and may disagree with fixture expectations; mismatches exit with status 1 and are not rewritten to expected values. The robust judge schema does not make the rubric Microsoft's original algorithm.

## Dependencies and environment (parent-owned root integration)

No root files, existing helpers, dependencies, `.env` or `.gitignore` are modified by this chapter implementation.

- Main requirements addition requested: **`opentelemetry-sdk==1.42.1`** (already installed here). Existing AF/OpenAI/Pydantic dependencies suffice for all default demos and custom judges.
- Optional OTLP requirements (separate parent-owned `requirements-observability.txt` is recommended): **`opentelemetry-exporter-otlp-proto-grpc==1.42.1`** and/or **`opentelemetry-exporter-otlp-proto-http==1.42.1`**. Missing exporter imports fail clearly only after explicit `otlp` selection. No auto-install.
- **No** `azure-ai-evaluation`, A365, Docker or Aspire package required; no unvalidated A365 dependency is proposed for the pinned baseline.
- Live OpenAI: existing `OPENAI_API_KEY`, optional `OPENAI_CHAT_MODEL_ID`/`OPENAI_BASE_URL`. Live Azure: existing `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT_NAME`, optional legacy API version. These are resolved only by `shared.chat_clients.create_sdk_client/create_completion_client`; no alternative credential path.
- Identity metadata only: optional `AGENT365_TENANT_ID`, `AGENT365_CLIENT_ID` or CLI IDs. **No client secret is read** by the unsupported A365 mode.
- OTLP endpoint/header variables act only in explicit OTLP mode. Explicit `--otlp-protocol` defaults to gRPC.
- Runtime-output ignores requested for parent: existing global `__pycache__`, `*.pyc` are sufficient. Tests create/clean temporary data outside the workspace; no chapter result/cache directory is written by default.

## Verified offline coverage and sources

The **35 focused unittest tests** exercise all fixture schemas and field preservation; safety flags/boundaries/casing and real response-content extraction; strict typed mock judges and payload isolation; composite per-metric semantics; SDK failure cleanup; real AF Agent + OpenAI `MockTransport` + real reusable tools; native and custom in-memory spans, events, logs, gauges, counters and histograms; identity context restoration; explicit native provider setup; local export routing and explicit-start launcher. Mocked OTLP exporter APIs verify all three signal endpoints for gRPC/HTTP and actionable missing dependencies. No live model/telemetry calls are made; actual remote OTLP/A365 ingestion is not claimed as validated.

First-party references reviewed for this migration:

- [Agent Framework Python observability](https://learn.microsoft.com/agent-framework/concepts/observability?tabs=python) and [middleware](https://learn.microsoft.com/agent-framework/concepts/agents/middleware/defining-middleware?pivots=programming-language-python); installed 1.8.1 signatures were inspected instead of assuming latest docs are compatible.
- [Microsoft OpenTelemetry distro](https://learn.microsoft.com/microsoft-agent-365/developer/microsoft-opentelemetry), [auth setup](https://learn.microsoft.com/microsoft-agent-365/developer/observability-authentication-setup), [official Python distro dependency manifest](https://github.com/microsoft/opentelemetry-distro-python/blob/main/pyproject.toml), [older official Python A365 SDK](https://github.com/microsoft/Agent365-python).
- [Agent evaluators](https://learn.microsoft.com/azure/ai-foundry/concepts/evaluation-evaluators/agent-evaluators), [RAG evaluators](https://learn.microsoft.com/azure/ai-foundry/concepts/evaluation-evaluators/rag-evaluators), [Azure AI evaluation SDK](https://learn.microsoft.com/python/api/overview/azure/ai-evaluation-readme).
- [Standalone Aspire dashboard](https://aspire.dev/dashboard/standalone/) and [official standalone sample](https://github.com/microsoft/aspire-samples/tree/main/samples/standalone-dashboard).
