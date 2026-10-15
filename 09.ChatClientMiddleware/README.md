# Chapter 09 — chat-client middleware

Each project mirrors the C# sample and retains its default prompts and policies.
Run from the workspace root with the existing `.env` configuration:

```powershell
python .\09.ChatClientMiddleware\MiddlewareBaseline\main.py
python .\09.ChatClientMiddleware\ChatClientSharedFunctionMiddleware\main.py
python .\09.ChatClientMiddleware\ChatClientResponseMiddleware\main.py
python .\09.ChatClientMiddleware\ChatClientFunctionCallingMiddleware\main.py
python .\09.ChatClientMiddleware\MiddlewareMixed\main.py
```

Every sample also accepts `--provider azure` and `--stream`.

## Policies

- **SharedFunction:** at most two requests per policy instance; email addresses
  in user text are transiently replaced with `[REDACTED-EMAIL]`. Original session
  history remains unchanged. Non-text content and function protocol data are
  preserved. This illustrative regex is not a comprehensive privacy filter.
- **Response:** a cumulative 2,000-token preflight budget. An already-exhausted
  budget returns `Token budget exhausted.` without calling the model or adding
  a timestamp. Successful calls can overshoot the budget; concurrent in-flight
  calls are not reserved in advance. Assistant text receives a UTC timestamp;
  non-text contents, tool calls, usage and response metadata are retained.
- **FunctionCalling:** only backward distances above five meters are capped;
  forward distances are unrestricted. Each simulated tool executes once and
  receives an audit entry on successful completion.
- **Baseline:** none of these policies; motor functions still auto-execute.
- **Mixed:** limit → redact → budget → timestamp → tool constrain → audit.

Policy counters last for the created chat-client pipeline's lifetime and are
shared if multiple agents reuse that client. They are deliberately not global
module state, so separate demo pipelines/tests remain independent.

## Python API differences

Agent Framework 1.8.1 uses `ChatMiddleware.process(context, call_next)` for both
C# shared-function and response interception, and `FunctionMiddleware` for
function interception. Function middleware is **chainable** in Python: constrain
arguments, then audit, then `call_next()` reaches one actual invocation. The C#
terminal-invoker/double-execution distinction does not apply to this Python API.

Standard Python `OpenAIChatCompletionClient` chat middleware sits **inside** the
function loop and runs once per model roundtrip. The focused shared/response
samples use this placement. The C# mixed builder places function invocation
innermost, so its outer policies run once per external chat call instead.
`ExternalCallChatClient` composes the official `ChatMiddlewareLayer`,
`FunctionInvocationLayer`, and raw Chat Completions provider to preserve that
placement for the mixed example. Its outer token budget counts aggregate usage
from the full tool loop, and redaction is reapplied to the next external call.
The source comments conflict on activation counts; the actual builder order is
what this port follows.

The default mixed demo's third request is blocked by its two-request cap.
Unlike its C# call sites, the console catches the policy exception and reports
it clearly, allowing the demonstration to end gracefully.

Streaming is an additional supported Python variant. Stream transforms preserve
metadata and add a timestamp before the first assistant text delta. Token usage
is recorded when the final response is finalized. Framework stream cleanup hooks
run on full consumption and ordinary stream errors. Version 1.8.1 exposes no
public close/context-manager API on `ResponseStream`; abandoning or cancelling a
stream does not guarantee these hooks or final usage accounting. The enclosing
SDK context still closes its HTTP client. These are demo budgets, not billing or
hardware-safety guarantees. Motors only print simulated moves.

## Offline tests

```powershell
python -m unittest discover -s tests -p test_chat_client_middleware.py -q
```

Tests use mocked provider transports and motor delays; they do not use secrets
or invoke live models.
