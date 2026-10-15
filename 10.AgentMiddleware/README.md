# 10. Agent middleware — Python migration

Five executable migrations of the user-owned [`dcostea/AgentFrameworkBook`](https://github.com/dcostea/AgentFrameworkBook/tree/main/10.AgentMiddleware) projects. Implementations live in `shared\agent_middleware_examples.py`; each original project has its own `main.py`.

## Run

Use the existing environment with **Agent Framework 1.8.1**. No additional packages, root configuration changes, or ignore entries are needed. Credentials and SDK ownership use the existing `shared.chat_clients` helpers: OpenAI requires `OPENAI_API_KEY`, optionally `OPENAI_CHAT_MODEL_ID`/`OPENAI_BASE_URL`; `--provider azure` uses the existing Azure endpoint, key and deployment variables. The helpers load the existing root `.env`; credentials are never embedded in samples. Running a demo contacts the configured provider; tests do not.

```powershell
.\.venv\Scripts\python.exe .\10.AgentMiddleware\AgentFunctionCallingMiddleware\main.py
.\.venv\Scripts\python.exe .\10.AgentMiddleware\AgentMiddlewareMixed\main.py
.\.venv\Scripts\python.exe .\10.AgentMiddleware\AgentResponseMiddleware\main.py
.\.venv\Scripts\python.exe .\10.AgentMiddleware\AgentSharedFunctionMiddleware\main.py
.\.venv\Scripts\python.exe .\10.AgentMiddleware\AgentWithLogging\main.py
```

All entrypoints accept `--provider openai|azure`, `--stream`, and `--scenario`. The default is the **entire original non-streaming demo**, in its original order, sharing the original sessions. Scenario filtering is optional; session state setup from skipped earlier scenarios is still applied. It does not fabricate skipped conversation history.

| Project | Default composition (outer → inner) | Default demos / scenario names |
|---|---|---|
| AgentFunctionCallingMiddleware | PreventDangerousMoves → AuditAgentFunctionCalls | `safe`: “Move forward 10 meters.”; `reversal`: “Move forward 2 meters and immediatelly move backward 2 meters.” |
| AgentMiddlewareMixed | AgentGuardrails → PersistentRemoveEmail → MovementSequenceAuditor → CaptainsLog; function Safety → Audit | `authorized`: return to original position with email; `reversal`: forward 10 then backward 10; `denied`: observer, new session |
| AgentResponseMiddleware | MovementSequenceAuditor → CaptainsLog | `captain`: forward 5; `reversal`: forward 2 then backward 2 |
| AgentSharedFunctionMiddleware | AgentGuardrails → PersistentRemoveEmail | `authorized`: forward 5 with email; `second`: left 90; `denied`: observer, same session |
| AgentWithLogging | StructuredAgentLogging around the run; StructuredFunctionLogging around each tool | `round`: exact prompt “go round.”, then prints `RobotCarAgent` |

`--scenario default` runs all listed demos. For example:

```powershell
.\.venv\Scripts\python.exe .\10.AgentMiddleware\AgentMiddlewareMixed\main.py --scenario denied
.\.venv\Scripts\python.exe .\10.AgentMiddleware\AgentFunctionCallingMiddleware\main.py --stream --scenario reversal
```

## Preserved behavior and important source distinctions

- The four middleware demos are named **MotorsAgent**, with the original description and three-line motor instructions. Logging uses **RobotCarAgent** and the original basic-moves-only instructions, reused from `shared.tool_examples.INSTRUCTIONS`. Tools come from `create_motor_tools()` and are **simulated**, not hardware. Existing `always_require` tools supplied to `build_agent` retain native approval behavior: the middleware never grants approval or invokes tools manually.
- `OperatorName` and `MissionTag` live in `AgentSession.state`, replacing StateBag. Standalone SharedFunction permits `driver` **case-insensitively**, without trimming, and defaults missing/null operators to `unknown`. **Mixed requires exact lowercase `driver`** and defaults to `none`. These are intentionally different implementations in the C# source.
- Guardrails deny before network/tool execution with the original `OperationDeniedException` message. Despite prose mentioning every user message, the source prefixes **only the last input message**, changing it to user role even if it was not user. Earlier messages are not prefixed. Mission values are `MISSION-42 | PERIMETER-SCAN` in shared/mixed, `MISSION-45 | EXPLORATION` in response, and `MISSION-99 | INSPECTION` in mixed's denied session. No session counter is implemented: the C# “counter = 2” heading is commentary, not code.
- `PersistentRemoveEmail` uses the exact source regex (including its unusual `|` in the TLD character class) and `[REDACTED-EMAIL]`, only on user input. The sanitized input is persisted by Python's normal local history provider, so subsequent turns do not send the original address. PRE/POST trace names remain `[Agent] [SharedFunction] [Guardrails]` and `[Agent] [SharedFunction] [Email]`. POST means successful complete execution, not merely stream construction.
- `CaptainsLog` prefixes every nonempty assistant text with `Stardate <UTC ISO timestamp>. Mission <tag>. Agent <name>.`, adding `Tools fired: N.` only when calls exist. Python ISO timestamps have microsecond rather than .NET's seven-digit precision. Counts are **requested calls**, including blocked ones, not successful motor actions.
- `MovementSequenceAuditor` lowercases requested function names and scans the exact joined sequence for `forward-backward` or `backward-forward`. Both `stop` **and any other intervening tool** interrupt those substrings. It alerts, never blocks. Contrary to the source's “warning footer” description, the implementation **replaces the last assistant text** with `WARNING: illegal direction '<previous text>' reversal detected in executed sequence. A stop command was missing.` Captain runs first on the way out, so its journal is inside that warning.
- Standalone `PreventDangerousMoves` compares previous/current names **case-sensitively** and only for exact agent name `MotorsAgent`. It short-circuits with `BLOCKED: cannot execute '<current>' directly after '<previous>'. Issue a stop command first.` Other agents print SKIP and pass through. A stop or turn interrupts that pair.
- **Mixed safety is different:** it lowercases and scans the **entire in-flight requested sequence**, not just previous/current. Once an illegal pair exists in visible history, subsequent tools (even stop) can be blocked. A parallel batch containing an illegal pair can have **all calls blocked**, not merely backward. Its exact result is `BLOCKED: cannot execute the command due to illegal reversal. Issue a stop command first.` This limitation is preserved, not silently repaired into a more permissive policy.
- Both actual Programs register **Safety before Audit**. Mixed's banner/comments claiming Audit wraps blocked results conflict with executable order. The implementation follows executable order: a blocked call does **not** reach Audit. Successful calls print the original `[Agent] [FunctionCall] [Audit]` identity, arguments, result, UTC time, plus measured elapsed milliseconds. Audit logs failure and rethrows; the framework's tool loop then returns `Error: Function failed.` with exception metadata to the model (its native default policy). Argument validation failures occur before FunctionMiddleware. No sentinel-based shortcuts or invented restrictions are added.

## Native Python APIs and observable adaptation

SharedFunction/Response delegates become `AgentMiddleware.process(AgentContext, call_next)`; tool delegates become `FunctionMiddleware.process(FunctionInvocationContext, call_next)`, with zero-argument async `call_next()` closures. This preserves their framework layers, not transport-only logging replacements.

Python 1.8.1's function context does not expose C# `agent` or `context.Messages`. A run-local ContextVar (`InvocationScope`) and native `ChatMiddleware` (`InvocationTranscript`) bridge the calling agent identity and per-model-round transcript. A streaming `ChatContext.stream_result_hooks` closure records the finalized pending calls before the tool loop executes them. Standalone previous/current uses conversation calls plus **earlier calls in the current model batch**, excluding the current/later calls; this defines previous-call semantics even when C# adapters expose the current batch differently. It follows requested order, not completion order of parallel motors. Mixed includes the whole pending batch. No singleton mutable history is shared between sessions/runs.

**Response `--stream` is intentionally buffered**, like C# `runStreamingFunc: null`: the inner run is non-streaming and transformed only when complete, then emits native `AgentResponseUpdate`s. Usage, response ID, timestamps, finish reason, tool calls, results, and raw representation remain on the original final response. SharedFunction and FunctionCalling use actual SSE streaming. Native result hooks run successful POST after final aggregation; native cleanup hooks run on **full consumption or ordinary stream errors**. In 1.8.1, `ResponseStream.__anext__` catches `Exception`, not `BaseException`: **`asyncio.CancelledError` does not guarantee cleanup**, even during an active pull. Pausing, breaking, or abandoning iteration also provides no deterministic cleanup guarantee. `ResponseStream` has **no `aclose` or async context-manager API**. The samples do not override cancellation or inspect private wrappers; logging's wrapper only observes ordinary exceptions and keeps these native semantics. Tests demonstrate a stream-body `RuntimeError` triggers cleanup without successful POST, whereas active cancellation does not run sample hooks. The SDK's lifetime belongs to the caller; the CLI always closes its owned SDK in `finally`, independently of stream-hook behavior.

Text replacement retains **non-text content and metadata** rather than copying C#'s destructive `Contents = [new TextContent(...)]`. This preserves function-call IDs/results, media, finish reason and usage. Python 1.8.1 omits non-streaming finish reason when mapping ChatResponse to AgentResponse; the middleware restores the upstream raw ChatResponse finish reason (without inventing one). No helper or installed package is modified.

`AgentWithLogging` uses supported Python logging middleware, not undocumented `.NET UseLogging`. The original separately built chat logger was unused by its agent, so no extra transport logger is attached here. LogRecords retain structured agent/run identity, streaming flag, tool name/call ID, elapsed milliseconds, completion metadata/usage and exception **type**. Logs omit prompt/body, credentials, tool arguments/results and exception text/traceback (which can contain secrets); normal tool-audit output masks emails and credential-shaped values. A ContextVar-scoped filter also removes the native framework's tool-failure exception text for logging-project runs only; other agents' native logs are unaffected. Python logging is the adaptation for Serilog verbose console output; console records include the structured fields as JSON and exact formatting differs.

## Offline validation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_agent_middleware.py -q
```

29 focused stdlib tests use real Agent, AsyncOpenAI and `httpx.MockTransport` with cleared environment, disabled dotenv and socket guards. They exercise all five entrypoints/default demos (both modes), order, session transformations/denials, exact warning/blocking strings, tool counts and failures, approvals, native streaming/error cleanup and cancellation limitations, concurrent run isolation, metadata aggregation, logging, and SDK ownership. No model/key is required and no live API is called.

API references: [official middleware guide](https://learn.microsoft.com/agent-framework/concepts/agents/middleware/) and [official Python API](https://learn.microsoft.com/python/api/agent-framework-core/agent_framework). Installed 1.8.1 source is authoritative where older documentation samples still mention ChatAgent or a context argument to `next`.
