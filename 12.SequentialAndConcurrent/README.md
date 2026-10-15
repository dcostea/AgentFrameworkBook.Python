# Sequential and concurrent workflows (chapter 12)

Python ports of all eight `dcostea/AgentFrameworkBook` chapter 12 projects. These use real Microsoft Agent Framework `WorkflowBuilder`, `SequentialBuilder`, `ConcurrentBuilder`, `AgentExecutor`, and `Workflow.as_agent()`—not an asyncio simulation of workflow execution. Motors and sensors are **simulated hardware**.

## Dependencies and configuration

Validated with the existing Python environment: `agent-framework-core==1.8.1`, `agent-framework-openai==1.8.1`, and **`agent-framework-orchestrations==1.0.0rc3`**. The parent integration has integrated the orchestrations pin into its root dependency manifest; this chapter does not change root requirements, environment files, documentation, ignores, or existing shared helpers.

### Observability and evaluation boundary

Chapter 12 explicitly calls the installed Python `disable_instrumentation()` API before creating clients or executing graphs. This prevents Agent Framework automatic telemetry export even when shared configuration enables instrumentation, sensitive-data capture, console exporters, OTLP endpoints, or VS Code integration. No telemetry exporters/providers are configured here, and no telemetry is sent by these CLI runs. Console workflow events remain local output. The disable is process-wide and sticky; importing `build_sample()` does not change an embedding application's telemetry policy. Offline tests isolate and restore telemetry settings and prohibit external networking.

Dependency discovery at handoff: `opentelemetry-api==1.42.1` and `opentelemetry-sdk==1.42.1` are installed; **both** `opentelemetry-exporter-otlp-proto-grpc==1.42.1` and `opentelemetry-exporter-otlp-proto-http==1.42.1` are now installed too. The umbrella `opentelemetry-exporter-otlp`, `azure-monitor-opentelemetry`, `azure-monitor-opentelemetry-exporter`, and `azure-ai-evaluation` packages are absent. No packages were installed for this chapter. If the parent wants an optional OTLP requirements file, the exact protocol-specific package names above are verified against installed providers; choose the intended protocol rather than requiring both. OTLP configuration belongs to an explicitly opted-in observability command: `OTEL_EXPORTER_OTLP_ENDPOINT`/signal-specific endpoints, `OTEL_EXPORTER_OTLP_PROTOCOL`, exporter headers, and `OTEL_SERVICE_NAME`; never print exporter credentials. Sensitive data must remain disabled by default.

Python OpenTelemetry/OTLP interoperability with an Aspire dashboard is **not** native .NET Aspire API parity. This chapter does not implement or claim Agent365 SDK or .NET evaluator API parity. No model-backed evaluation is invoked; any parent evaluation integration must require a separate explicit command opt-in and a verified Python provider. The source contains no `DirectionChange` evaluator—chapter 12's safety thresholds remain exactly those listed below. See the [official Python observability guide](https://learn.microsoft.com/agent-framework/agents/observability?pivots=programming-language-python); installed 1.8.1 signatures take precedence over newer documentation.

Use the existing `shared.chat_clients` configuration and SDK lifecycle. OpenAI needs `OPENAI_API_KEY`; `OPENAI_CHAT_MODEL_ID` is optional. `--provider azure` uses `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and `AZURE_OPENAI_DEPLOYMENT_NAME`, with the existing optional API-version routing. Live runs contact the configured model and incur provider usage. No calls are made by `--help` or the offline tests.

Each entry point runs **one mission**, at most **20 workflow supersteps**, under a **120-second total default timeout** (including tools and checkpoint replay). `--timeout` and `--prompt` override those defaults. Framework function-invocation loops remain subject to this total timeout. No interactive inputs are required.

From the repository root:

```powershell
.\.venv\Scripts\python.exe .\12.SequentialAndConcurrent\SequentialOrchestration\main.py --help
.\.venv\Scripts\python.exe .\12.SequentialAndConcurrent\SequentialOrchestration\main.py --provider openai
```

Replace the project directory with any of the following exact source names:

| Project | Source graph and behavior |
| --- | --- |
| `SequentialOrchestration` | Environment → Safety → Motors. Every downstream agent receives chronological shared history, including the original mission and earlier responses. Final output contains that conversation. |
| `SequentialOrchestrationAsPipeline` | Environment → Navigator → Motors (`ProcessingPipeline`). Only the preceding agent's response is propagated; the self-contained environment prompt carries mission details. Final output contains only Motors' response. |
| `SequentialOrchestrationWithBuilder` | `RobbyMission`: Environment and Safety are intermediate outputs; only Motors is terminal. Streaming execution. |
| `SequentialOrchestrationWithCheckpoints` | `SafeExecutionWithCheckpoints`: capture the first persisted superstep whose queued messages activate Motors; finish the first run, then replay from that checkpoint on a newly instantiated `SafeExecutionFromCheckpoint` graph. |
| `SequentialOrchestrationWithIntermediateOutput` | Same selected-agent graph as WithBuilder, with buffered execution and typed intermediate events. |
| `SequentialOrchestrationWithMiddleware` | `SafeExecution`: Safety has chat-client and agent post-response checks for case-sensitive `DENIED`. An abort prevents Motors from running. |
| `ConcurrentOrchestration` | `SafetyAssessment`: Maintenance and Environment run concurrently; `SafetyAggregator` merges JSON clearances. |
| `SequenceOfConcurrentOrchestration` | `SanitizedSequentialExecution`: concurrent safety graph wrapped as `SafetyStage` agent → tool-filtering Motors, with the original mission and merged verdict preserved. |

## Source details and Python API adaptations

- Prompts, agent names, two-field JSON response schema (`clearance`, `reason`), thresholds, mission commands, and branch ordering are preserved. The worst-state score is binary: `GRANTED=0`, `DENIED=1`; any denial wins and denial reasons are joined in branch order. Missing/duplicate branches, missing assistant replies, invalid JSON, and invalid clearance values fail rather than falsely granting clearance.
- Original sensor mocks provide temperature `[-20,100)`, humidity/IR/wind `[0,100)`, all four droplet values (`None`, `Low`, `Medium`, `High`), and eight wind directions. Maintenance mocks preserve the six actions, ranges, return strings, 500-ms base delay, and 3×/5× delays. C# sensor/maintenance tool names are retained in model-visible metadata. Existing motor tools are reused only where the source attaches them.
- The concurrent source mentions `Extreme` droplets, although the source enum contains no `Extreme`, and its generated temperature/wind values cannot exceed 100. These source limitations are intentionally preserved, not silently corrected.
- At composition boundaries the exact source filter drops **whole messages** containing a function call or function result, retaining other roles/content and leaving upstream messages unchanged. The aggregator likewise excludes tool-call-bearing messages before parsing assistant JSON, preventing mixed-content tool-call pollution. Tool results still participate in their owning agent's invocation loop.
- Python rc3 orchestration constructors use named `participants=` and do not accept names. Their public graph topology is rebuilt with the installed core `WorkflowBuilder(start_executor=...)` to retain source workflow names, bounded iterations, and fan-in barriers. A typed terminal executor restores C# full-conversation versus pipeline-only final output semantics; core 1.8.1 ordinarily outputs only the last `AgentResponse`.
- AsPipeline retains its original motor instruction block, then appends a Python tool-execution clarification naming the available motor functions, requiring a real invocation/result for each movement rather than an echoed plan. Motors uses `tool_choice="required"` for the initial request and `allow_multiple_tool_calls=False` to prevent parallel movement execution; the installed framework resets required choice after the first invocation so the model can finish normally. No model output or movement parameters are fabricated by the host. With motor tools attached, per-run function middleware requires at least one successful motor invocation and rejects any failed attempt before final workflow output; prose-only completion makes the CLI exit **1**. This checks actual invocation, not whether every natural-language route line was fulfilled. Explicit `motor_tools=[]` fixtures intentionally omit execution verification. Preceding-response-only inputs and last-agent-only output are unchanged.
- Current Python events use `type`, `executor_id`, `data`, `iteration`, and `details.message`; response payloads are checked as `AgentResponse`/`AgentResponseUpdate` or message lists. `intermediate` and `output` are routed independently without reprinting final responses after streamed chunks. No deprecated C# event classes or type-selector guesses are used.
- Concurrent failure/abort helper equivalents are available in the shared chapter module, but remain disabled by default, just as in the source. Sequential middleware is enabled at both original interception layers.

## Persistent checkpoints and interruption

```powershell
.\.venv\Scripts\python.exe .\12.SequentialAndConcurrent\SequentialOrchestrationWithCheckpoints\main.py
.\.venv\Scripts\python.exe .\12.SequentialAndConcurrent\SequentialOrchestrationWithCheckpoints\main.py --interrupt
# Use the checkpoint ID printed by the interrupted run:
.\.venv\Scripts\python.exe .\12.SequentialAndConcurrent\SequentialOrchestrationWithCheckpoints\main.py --resume CHECKPOINT_ID
```

Checkpoints live in the sample's `runtime\checkpoints` directory, covered by the chapter-local `.gitignore`. No generated C# checkpoints are copied. `--interrupt` adds a deterministic pre-motor interruption after persistent safety state is captured, exits without motor execution, and permits resumption in a separate process with freshly instantiated agents/graph. `--resume` replays queued motor input and restores executor/session state; Environment and Safety are not rerun. The default source demonstration completes then replays Motors, so **simulated moves execute twice**. Python checkpoint files use the framework's trusted pickle-backed format with its restricted allowlist; never resume downloaded/untrusted state. Treat local checkpoints as private mission history.

## Offline verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_sequential_concurrent.py -q
```

The **38-test** suite uses real Workflow and Agent objects with fake `BaseChatClient` responses and OpenAI `httpx.MockTransport` (including streaming and real tool invocation), guarded against network calls. It verifies context/output ordering, concurrent overlap and aggregation failures, nested wrappers, exact tool-message filtering, granted/denied motor paths, pipeline required-tool mode and ordered execution, rejection of prose-only or failed motor execution, middleware aborts, persistent checkpoint replay and tool-result restoration on fresh graphs, intermediate events, source mock behavior, all eight entry points, and provider lifecycle. All 31 original source prompt/instruction blocks are preserved after whitespace normalization; the pipeline tool clarification is appended separately. Live provider/model quality is not tested by this suite.

Official API references: [sequential](https://learn.microsoft.com/agent-framework/workflows/orchestrations/sequential?pivots=programming-language-python), [concurrent](https://learn.microsoft.com/agent-framework/workflows/orchestrations/concurrent?pivots=programming-language-python), [checkpoints](https://learn.microsoft.com/agent-framework/workflows/checkpoints?pivots=programming-language-python), [agents in workflows](https://learn.microsoft.com/agent-framework/workflows/agents-in-workflows?pivots=programming-language-python). Installed package sources take precedence over older snippets.
