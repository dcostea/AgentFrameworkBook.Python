# Agent Framework book — Python samples

Python ports of [dcostea/AgentFrameworkBook](https://github.com/dcostea/AgentFrameworkBook),
built against the [official Microsoft Agent Framework](https://github.com/microsoft/agent-framework/tree/main/python).
Python samples mirror the book's chapter folders, with one `main.py` per C#
project. Chapters **03.ChatClients** through **14.Handoff** cover providers,
agents, tools, sessions, retrieval, MCP, middleware, evaluation, and orchestration.
Existing introductory agent samples are retained.

## Setup

Use Python 3.10–3.13. Run these commands from this workspace's root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# Only if you do not already have .env:
Copy-Item .env.example .env
```

Edit `.env` for the providers you use. Environment variables override `.env`;
credentials are never embedded in the samples. `.env` and `.venv` are ignored.
The chapter helpers load the root `.env` even when a script runs from another directory.

| .NET User Secret | Python environment variable |
| --- | --- |
| `OpenAI:ApiKey` | `OPENAI_API_KEY` |
| `OpenAI:ModelId` | `OPENAI_CHAT_MODEL_ID` |
| `AzureOpenAI:Endpoint` | `AZURE_OPENAI_ENDPOINT` |
| `AzureOpenAI:ApiKey` | `AZURE_OPENAI_API_KEY` |
| `AzureOpenAI:DeploymentName` | `AZURE_OPENAI_DEPLOYMENT_NAME` |
| `AzureOpenAI:AudioEndpoint` | `AZURE_OPENAI_AUDIO_ENDPOINT` |
| `AzureOpenAI:AudioApiKey` | `AZURE_OPENAI_AUDIO_API_KEY` |
| `AzureOpenAI:AudioDeploymentName` | `AZURE_OPENAI_AUDIO_DEPLOYMENT_NAME` |
| `Anthropic:ApiKey` | `ANTHROPIC_API_KEY` |
| `Anthropic:ModelId` | `ANTHROPIC_MODEL_ID` |

`OPENAI_CHAT_MODEL_ID` defaults to `gpt-5.4-mini`; Azure requires a deployment
name, not a model name. Optional `OPENAI_RESPONSES_MODEL_ID` and
`AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME` override the models for chapter 04.
Azure uses the `/openai/v1/` API by default: both resource URLs and v1 URLs work.
Set `AZURE_OPENAI_API_VERSION` only when using a versioned legacy endpoint.
Audio deployments have separate configuration and must support input audio.

## Chapter 03 — chat clients

| Sample (same name as the C# project) | Concept |
| --- | --- |
| [`OpenAIChatClient`](03.ChatClients/OpenAIChatClient/main.py) | Native SDK vs framework Chat Completions |
| [`AzureOpenAIChatClient`](03.ChatClients/AzureOpenAIChatClient/main.py) | Azure SDK vs framework Chat Completions |
| [`AnthropicChatClient`](03.ChatClients/AnthropicChatClient/main.py) | Anthropic provider |
| [`AzureOpenAIChatClientWithImages`](03.ChatClients/AzureOpenAIChatClientWithImages/main.py) | Binary audio (default) or image input |
| [`Chatbot`](03.ChatClients/Chatbot/main.py) | Interactive conversation history |
| [`ChatbotWithChatOptions`](03.ChatClients/ChatbotWithChatOptions/main.py) | Sampling options and typed JSON output |
| [`ChatbotWithStreaming`](03.ChatClients/ChatbotWithStreaming/main.py) | Incremental output and completed-response history |
| [`ChatbotWithTools`](03.ChatClients/ChatbotWithTools/main.py) | Automatic simulated motor tools and context reduction |
| [`ChatClientWithMiddleware`](03.ChatClients/ChatClientWithMiddleware/main.py) | Request/response logging middleware |

```powershell
python .\03.ChatClients\OpenAIChatClient\main.py
python .\03.ChatClients\AzureOpenAIChatClient\main.py
python .\03.ChatClients\AnthropicChatClient\main.py
python .\03.ChatClients\AzureOpenAIChatClientWithImages\main.py
python .\03.ChatClients\AzureOpenAIChatClientWithImages\main.py --mode image
python .\03.ChatClients\Chatbot\main.py
python .\03.ChatClients\ChatbotWithChatOptions\main.py
python .\03.ChatClients\ChatbotWithStreaming\main.py
python .\03.ChatClients\ChatbotWithTools\main.py
python .\03.ChatClients\ChatClientWithMiddleware\main.py
```

Press Enter on an empty line (or send EOF) to exit a chatbot. Tools print
simulated moves, wait one second, and return their result; they do not control hardware.
Middleware logs counts and timing without logging credentials or conversation content.

### Python adaptations

- In Agent Framework **1.8.1**, `OpenAIChatCompletionClient` is the Chat
  Completions provider; `OpenAIChatClient` is the **Responses** provider. Both
  accept an injected OpenAI SDK client for OpenAI or Azure. Dependencies are
  pinned to the API used here; older examples use different class names.
- Structured responses use Pydantic `StepsResponse` and produce
  `{"Steps": ["..."]}`. `ChatbotWithChatOptions` keeps the C# sampling and stop
  options but defaults to **512** output tokens instead of **5**, so JSON can
  finish. Set `CHATBOT_MAX_OUTPUT_TOKENS=5` to reproduce the original truncation.
  Reasoning models may reject temperature, top-p, penalties, or stop sequences;
  choose a compatible deployment for this particular example.
- Streaming uses the framework's final response to preserve messages and
  metadata. Each turn gets a fresh stream; previous output is not accumulated.
- Python's `SlidingWindowStrategy(keep_last_groups=10)` replaces the C#
  `MessageCountingChatReducer(10)`. It bounds the context sent to the model by
  **message groups**, preserving system instructions and linked tool calls/results.
  The console's local history remains available; this is not a ten-message
  in-memory cap.
- The C# image sample actually defaults to MP3 audio. That behavior is retained,
  with `--mode image` enabling its image alternative. `audio/mp3` is used because
  the current Python adapter needs it to emit OpenAI's `input_audio` format.
  Original `path.jpg`, `road_and_tree.png`, and `Task.mp3` assets are included.

## Chapter 04 — responses and self-hosted clients

| Sample | Concept |
| --- | --- |
| [`OpenAIResponses`](04.ChatClients/OpenAIResponses/main.py) | Native SDK vs framework Responses API |
| [`AzureOpenAIResponses`](04.ChatClients/AzureOpenAIResponses/main.py) | Azure Responses API |
| [`OpenAIResponsesWithBackgroundResponses`](04.ChatClients/OpenAIResponsesWithBackgroundResponses/main.py) | Background creation and continuation-token polling |
| [`AzureOpenAIResponsesWithBackgroundResponses`](04.ChatClients/AzureOpenAIResponsesWithBackgroundResponses/main.py) | Azure background responses |
| [`OllamaChatClient`](04.ChatClients/OllamaChatClient/main.py) | Local Ollama provider |
| [`OnnxChatClient`](04.ChatClients/OnnxChatClient/main.py) | Local ONNX text generation with timing |
| [`OnnxChatClientMultiModal`](04.ChatClients/OnnxChatClientMultiModal/main.py) | Local ONNX image understanding with timing |

```powershell
python .\04.ChatClients\OpenAIResponses\main.py
python .\04.ChatClients\AzureOpenAIResponses\main.py
python .\04.ChatClients\OpenAIResponsesWithBackgroundResponses\main.py
python .\04.ChatClients\AzureOpenAIResponsesWithBackgroundResponses\main.py
python .\04.ChatClients\OllamaChatClient\main.py
python .\04.ChatClients\OnnxChatClient\main.py
python .\04.ChatClients\OnnxChatClientMultiModal\main.py
```

Background examples use `background=True`, `store=True`, then poll with an empty
message list and `continuation_token`, at most **60 times**, one second apart.
They report immediate completion, final output, failures, and bounded timeouts.
A timeout leaves the response running and reports its ID for later retrieval or
cancellation. Use a model, deployment, and region supporting background Responses;
background mode stores provider-side response data and is unsuitable for
zero-data-retention configurations. SDK-vs-framework examples deliberately make
**two requests**, like the originals, and may incur API charges.

For Ollama, install/start Ollama separately and pull your chosen model. Configure
`OLLAMA_HOST` (default `http://localhost:11434`) and `OLLAMA_MODEL` (default
`gemma4:e4b`, as in the C# project). Scripts do not start servers or download models.

### ONNX setup

ONNX Runtime GenAI is optional and installed separately:

```powershell
python -m pip install -r requirements-onnx.txt
```

The optional runtime is pinned to `onnxruntime-genai==0.17.1`. Set
`ONNX_MODEL_PATH` to a local text model directory containing `genai_config.json`.
Use single quotes around Windows paths in `.env` to preserve backslashes, for
example `ONNX_MODEL_PATH='C:\Models\phi-4-mini-onnx'`.
Set `ONNX_MULTIMODAL_MODEL_PATH` to a compatible **vision**
model directory, not a text-only Phi-4 Mini model. `ONNX_PROVIDER` defaults to
`cpu`; CUDA requires the matching GPU runtime, drivers, and a CUDA-compatible
model. Do not install CPU and GPU runtime variants together.
`ONNX_IMAGE_PATH` optionally replaces the bundled JPEG. Supported image model
families are Phi-3/3.5 Vision (`phi3v`) and Phi-4 Multimodal (`phi4mm`), with
complete tokenizer and processor assets. This port demonstrates images, not audio.
Text prompts use the tokenizer's native chat template (or the model's
`chat_template.jinja`); a Phi-only fallback is available. An optional
`ONNX_CHAT_TEMPLATE` overrides this for a single-user prompt and must contain
`{prompt}`, for example `ONNX_CHAT_TEMPLATE='<|user|>{prompt}<|end|><|assistant|>'`.
The adapter uses ONNX Runtime GenAI directly because Agent Framework Python
has no built-in equivalent of C# `OnnxRuntimeGenAIChatClient`. It exposes framework
messages and streaming responses, reporting load time, first/last token latency,
and actual generated-token throughput. It generates at most 512 new tokens.

## Chapter 05 — agents and sessions

| Sample | Concept |
| --- | --- |
| [`Agent`](05.Agents/Agent/main.py) | Basic agent and independent runs |
| [`AgentWithChatMessages`](05.Agents/AgentWithChatMessages/main.py) | Explicit chat-message input |
| [`AgentWithChatMessagesAndMultiModal`](05.Agents/AgentWithChatMessagesAndMultiModal/main.py) | Robot commands from binary image content |
| [`AgentWithStructuredOutput`](05.Agents/AgentWithStructuredOutput/main.py) | Typed JSON response |
| [`AgentWithStreaming`](05.Agents/AgentWithStreaming/main.py) | Streaming agent response |
| [`AgentWithSession`](05.Agents/AgentWithSession/main.py) | Conversation history scoped to a session |
| [`AgentWithSerialization`](05.Agents/AgentWithSerialization/main.py) | Save and restore a Python agent session |
| [`MultipleAgentsWithSameSession`](05.Agents/MultipleAgentsWithSameSession/main.py) | Share session history between agents |
| [`AgentWithInServiceConversation`](05.Agents/AgentWithInServiceConversation/main.py) | Provider-side conversation state |
| [`AgentWithBackgroundResponses`](05.Agents/AgentWithBackgroundResponses/main.py) | Agent background execution and continuation |

```powershell
python .\05.Agents\Agent\main.py
python .\05.Agents\AgentWithSession\main.py
python .\05.Agents\AgentWithSerialization\main.py
python .\05.Agents\AgentWithStructuredOutput\main.py
python .\05.Agents\AgentWithInServiceConversation\main.py
python .\05.Agents\AgentWithBackgroundResponses\main.py
```

Python's `AgentSession` replaces C#'s session object. Serialized sessions use the
Python framework's schema, not the C# state-bag schema; generated C# session
files are not interchangeable. `AgentWithSerialization` writes sample-local
`agent_session.json`, which is ignored by Git and contains plaintext history.
Cloud-stored conversation and background examples require models and endpoints
supporting the Responses API. See the [chapter 05 guide](05.Agents/README.md)
for provider selection, session behavior, and multimodal/structured-output notes.

## Chapter 06 — tools

| Sample | Concept |
| --- | --- |
| [`AgentWithTools`](06.Tools/AgentWithTools/main.py) | Automatic simulated motor function invocation |
| [`AgentWithToolMode`](06.Tools/AgentWithToolMode/main.py) | Auto, disabled, required, or specific tool choice |
| [`AgentWithMultipleToolCalls`](06.Tools/AgentWithMultipleToolCalls/main.py) | Single-call rounds vs multiple calls per response |
| [`AgentWithManualToolInvocation`](06.Tools/AgentWithManualToolInvocation/main.py) | Explicit dispatch and correlated tool results |
| [`AgentWithFunctionApproval`](06.Tools/AgentWithFunctionApproval/main.py) | Human approval for backward and stop moves |

```powershell
python .\06.Tools\AgentWithTools\main.py
python .\06.Tools\AgentWithToolMode\main.py
python .\06.Tools\AgentWithToolMode\main.py --mode none
python .\06.Tools\AgentWithToolMode\main.py --mode required
python .\06.Tools\AgentWithToolMode\main.py --mode backward
python .\06.Tools\AgentWithMultipleToolCalls\main.py
python .\06.Tools\AgentWithMultipleToolCalls\main.py --parallel
python .\06.Tools\AgentWithManualToolInvocation\main.py
python .\06.Tools\AgentWithFunctionApproval\main.py
```

Motor tools are simulations only. Their Python names are `forward`, `backward`,
`turn_left`, `turn_right`, and `stop`. They print, wait one second, and return a
result; they never connect to hardware. Required selection applies to the first
model roundtrip; the framework then resets it to avoid an endless tool loop.
`--parallel` permits several function calls in one model response; it does not
make any hardware sequencing guarantees.

Manual invocation disables the framework's automatic loop, dispatches the batch
explicitly, and sends results back with the original call IDs and session.
Human approval uses `approval_mode="always_require"` for `backward` and `stop`:
Enter or `Y` approves, `N` (or EOF) denies. Python may wrap an entire mixed tool
batch in approval requests; the sample prompts only for the protected moves and
allows ordinary simulated moves as before. Approval and manual loops are bounded
to 20 rounds and propagate invalid calls rather than executing arbitrary code.

## Chapter 07 — memories and retrieval

| Sample | Concept |
| --- | --- |
| [`AgentWithInMemoryChatHistoryProvider`](07.Memories/AgentWithInMemoryChatHistoryProvider/main.py) | Session-local history provider |
| [`AgentWithFileBasedChatHistoryProvider`](07.Memories/AgentWithFileBasedChatHistoryProvider/main.py) | File-backed chat history |
| [`AgentWithCustomFileBasedContextProvider`](07.Memories/AgentWithCustomFileBasedContextProvider/main.py) | File instructions and historical context |
| [`AgentWithInMemoryChatMessagesAndInMemoryContextProvider`](07.Memories/AgentWithInMemoryChatMessagesAndInMemoryContextProvider/main.py) | Context instructions alongside history |
| [`AgentWithInMemoryVectorStoreProvider`](07.Memories/AgentWithInMemoryVectorStoreProvider/main.py) | Embedding-based memory retrieval |
| [`AgentWithTextSearchProvider`](07.Memories/AgentWithTextSearchProvider/main.py) | Custom keyword search and citations |
| [`AgentWithVectorStoreChatHistoryProvider`](07.Memories/AgentWithVectorStoreChatHistoryProvider/main.py) | Chat history backed by an in-memory vector store |

```powershell
python .\07.Memories\AgentWithInMemoryChatHistoryProvider\main.py
python .\07.Memories\AgentWithFileBasedChatHistoryProvider\main.py
python .\07.Memories\AgentWithCustomFileBasedContextProvider\main.py
python .\07.Memories\AgentWithInMemoryChatMessagesAndInMemoryContextProvider\main.py
python .\07.Memories\AgentWithInMemoryVectorStoreProvider\main.py
python .\07.Memories\AgentWithTextSearchProvider\main.py
python .\07.Memories\AgentWithVectorStoreChatHistoryProvider\main.py
```

All chapter 07 samples accept `--provider openai` (default) or `--provider azure`.
For example:

```powershell
python .\07.Memories\AgentWithInMemoryVectorStoreProvider\main.py --provider azure
```

Semantic retrieval uses actual provider embeddings and cosine similarity in a
small in-memory store. Configure `OPENAI_EMBEDDING_MODEL_ID` or
`AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` with a model/deployment supporting
**1536 dimensions**; `text-embedding-3-small` is an example. Embedding requests
may incur API charges. No database server or additional packages are needed.

The custom providers use Python's `ContextProvider` and `InMemoryHistoryProvider`
APIs rather than the C# provider classes. File-backed outputs live in each
sample's `runtime` folder and are ignored by Git. They contain conversation
content in plaintext; do not treat them as a secure secrets store. Python and
C# serialized histories are not interchangeable.

`AgentWithVectorStoreChatHistoryProvider` follows its source's actual behavior:
it stores timestamped messages in a process-local collection and retrieves the
**earliest five** messages. Despite the name, this particular C# example has no
embeddings or similarity search. `AgentWithInMemoryVectorStoreProvider` is the
separate semantic-search example. Custom keyword search remains keyword search.
See the [chapter 07 guide](07.Memories/README.md) for source-specific reducer,
keyword matching, and persistence details.

To resume file history after restarting, use the session ID printed by the first
run; no serialized session snapshot is required:

```powershell
python .\07.Memories\AgentWithFileBasedChatHistoryProvider\main.py --provider openai
python .\07.Memories\AgentWithFileBasedChatHistoryProvider\main.py --provider openai --session-id "<printed-id>"
```

## Chapter 08 — Model Context Protocol (MCP)

| Sample | Concept |
| --- | --- |
| [`MCPServerWithStdio`](08.MCP/MCPServerWithStdio/main.py) | Local stdio server with motor tools, prompts, resources, and maintenance |
| [`MCPClientWithStdio`](08.MCP/MCPClientWithStdio/main.py) | Spawn and connect to the sibling Python MCP server |
| [`MCPServerWithHttp`](08.MCP/MCPServerWithHttp/main.py) | Local Streamable HTTP MCP server |
| [`MCPClientWithHttp`](08.MCP/MCPClientWithHttp/main.py) | HTTP tools, resources, prompts, and agent integration |

Servers expose simulated functions only and require no model credentials.
Clients use the root `.env` for model-provider credentials. The stdio server
keeps logs on stderr so stdout remains valid JSON-RPC. HTTP defaults to loopback;
see the [chapter 08 guide](08.MCP/README.md) for startup, endpoint configuration,
protocol demonstrations, and task support.

```powershell
# Terminal 1: leave this server running while the HTTP client is used.
python .\08.MCP\MCPServerWithHttp\main.py --port 3001
# Terminal 2:
python .\08.MCP\MCPClientWithHttp\main.py --port 3001 --provider openai
# Stdio starts/stops its own sibling server; no separate server terminal needed.
python .\08.MCP\MCPClientWithStdio\main.py --provider openai
```

The default HTTP endpoint is `http://127.0.0.1:3001/mcp`; use `MCP_PORT` or matching
`--port` values to change it. Stdio first demonstrates protocol operations, then
asks whether to run the optional model-powered agent. The stdio maintenance
sample supports genuine experimental MCP task creation, polling and cancellation;
Python negotiates the SDK's `2025-11-25` protocol rather than the newer C# protocol.
HTTP intentionally does not enable Tasks. Dependencies are pinned for these APIs.
The current MCP SDK's SSE path can emit AnyIO receive-stream cleanup warnings;
these reproduce without the sample adapter. See the chapter guide for evidence
and limitations. The sample closes its client sessions and test listeners.

## Chapter 09 — chat-client middleware

| Sample | Concept |
| --- | --- |
| [`MiddlewareBaseline`](09.ChatClientMiddleware/MiddlewareBaseline/main.py) | Comparison without policy middleware |
| [`ChatClientSharedFunctionMiddleware`](09.ChatClientMiddleware/ChatClientSharedFunctionMiddleware/main.py) | Request limit and transient email redaction |
| [`ChatClientResponseMiddleware`](09.ChatClientMiddleware/ChatClientResponseMiddleware/main.py) | Token budget and response timestamps |
| [`ChatClientFunctionCallingMiddleware`](09.ChatClientMiddleware/ChatClientFunctionCallingMiddleware/main.py) | Backward-distance cap and function audit |
| [`MiddlewareMixed`](09.ChatClientMiddleware/MiddlewareMixed/main.py) | Ordered prepare/response/function pipeline |

```powershell
python .\09.ChatClientMiddleware\MiddlewareBaseline\main.py
python .\09.ChatClientMiddleware\ChatClientSharedFunctionMiddleware\main.py
python .\09.ChatClientMiddleware\ChatClientResponseMiddleware\main.py --stream
python .\09.ChatClientMiddleware\ChatClientFunctionCallingMiddleware\main.py
python .\09.ChatClientMiddleware\MiddlewareMixed\main.py --provider azure
```

The source defaults are retained: two requests, a 2,000-token budget, and a
five-meter cap on backward moves only. These are illustrative policies, not
billing, privacy-compliance, or hardware-safety guarantees. Python function
middleware is chainable, unlike the C# terminal invoker. The mixed port explicitly
preserves its source's outer-policy placement around the full tool loop. See the
[chapter 09 guide](09.ChatClientMiddleware/README.md) for streaming and activation-count
adaptations.

## Chapter 10 — agent middleware

| Sample | Concept |
| --- | --- |
| [`AgentSharedFunctionMiddleware`](10.AgentMiddleware/AgentSharedFunctionMiddleware/main.py) | Agent-level request preparation and restrictions |
| [`AgentResponseMiddleware`](10.AgentMiddleware/AgentResponseMiddleware/main.py) | Agent response interception |
| [`AgentFunctionCallingMiddleware`](10.AgentMiddleware/AgentFunctionCallingMiddleware/main.py) | Function arguments and audited invocation |
| [`AgentMiddlewareMixed`](10.AgentMiddleware/AgentMiddlewareMixed/main.py) | Composed agent/function pipeline |
| [`AgentWithLogging`](10.AgentMiddleware/AgentWithLogging/main.py) | Agent and function logging |

```powershell
python .\10.AgentMiddleware\AgentSharedFunctionMiddleware\main.py
python .\10.AgentMiddleware\AgentResponseMiddleware\main.py
python .\10.AgentMiddleware\AgentFunctionCallingMiddleware\main.py
python .\10.AgentMiddleware\AgentMiddlewareMixed\main.py
python .\10.AgentMiddleware\AgentWithLogging\main.py
```

Agent middleware wraps an agent run and can access its session; function
middleware wraps each simulated function invocation. All chapter 10 samples
accept `--provider openai|azure`, `--stream`, and project-specific `--scenario`
choices (`--help` lists them). Default runs preserve the original demonstrations.
Response interception buffers its streaming variant; other examples stream
natively. `AgentWithLogging` uses DEBUG console logging with structured fields.
The [chapter 10 guide](10.AgentMiddleware/README.md) documents the original
policies, ordering, and Python-specific behavior.

In framework 1.8.1, stream cleanup hooks run on full consumption or ordinary
stream errors; cancellation or abandonment does not guarantee final hooks or
usage accounting. Enclosing SDK context managers still close HTTP clients.

## Chapter 11 — observability and evaluations

| Sample | Concept |
| --- | --- |
| [`Observations`](11.ObservabilityAndEvaluations/Observations/main.py) | Agent telemetry and direction-change tracking |
| [`Observations.AppHost`](11.ObservabilityAndEvaluations/Observations.AppHost/main.py) | Dashboard-compatible observation launcher |
| [`ObservationsWithAgent365`](11.ObservabilityAndEvaluations/ObservationsWithAgent365/main.py) | Agent 365 integration/adaptation boundary |
| [`Evaluators`](11.ObservabilityAndEvaluations/Evaluators/main.py) | Source datasets, deterministic and model-backed evaluations |

```powershell
# Offline local telemetry and dataset validation; no model credentials required.
python .\11.ObservabilityAndEvaluations\Observations\main.py
python .\11.ObservabilityAndEvaluations\ObservationsWithAgent365\main.py
python .\11.ObservabilityAndEvaluations\Evaluators\main.py
# Explicit model requests:
python .\11.ObservabilityAndEvaluations\Observations\main.py --live --prompt "Go round."
python .\11.ObservabilityAndEvaluations\Evaluators\main.py --dataset coherence --model-backed
# Optional exporters; configure/start a collector separately.
python -m pip install -r requirements-observability.txt
```

Default telemetry stays in memory. Use `--telemetry console` for local console
export, or explicitly select `--telemetry otlp --otlp-endpoint <collector-url>`
for network export; `--otlp-protocol` selects `grpc` or `http/protobuf` (the
`OTEL_EXPORTER_OTLP_PROTOCOL` environment variable is not used by these runners).
OTLP supports endpoint/header environment settings; do not store collector
credentials in source or print authentication headers.
`Observations.AppHost` is a bounded Python launcher for a separately configured
collector/dashboard, **not** a .NET Aspire host. The Agent 365 sample exposes a
standard-OTel identity adaptation, not native Agent 365 ingestion/authentication;
selecting `--telemetry a365` fails explicitly. Its optional `AGENT365_TENANT_ID`
and `AGENT365_CLIENT_ID` fields are non-secret identity metadata only.
AppHost starts Observations only with `--start`; leave `OTEL_EXPORTER_OTLP_ENDPOINT`
and `ASPIRE_DASHBOARD_OTLP_ENDPOINT_URL` unset for an offline launch.
Model-backed evaluation uses documented custom Python judge prompts, not the
.NET evaluator algorithms; deterministic direction-change scoring retains its
source rules. The twelve source JSONL datasets are included.

See the [chapter 11 guide](11.ObservabilityAndEvaluations/README.md) for telemetry
configuration, evaluation commands, and .NET/Python integration limits.
Model-based scoring may incur provider charges. Telemetry can contain sensitive
conversation/tool data; configure exports deliberately rather than enabling them
in an environment containing secrets or personal data.

## Chapter 12 — sequential and concurrent workflows

| Sample | Concept |
| --- | --- |
| [`SequentialOrchestration`](12.SequentialAndConcurrent/SequentialOrchestration/main.py) | Agent sequence with accumulated context |
| [`SequentialOrchestrationAsPipeline`](12.SequentialAndConcurrent/SequentialOrchestrationAsPipeline/main.py) | Each stage consumes the previous stage's output |
| [`SequentialOrchestrationWithBuilder`](12.SequentialAndConcurrent/SequentialOrchestrationWithBuilder/main.py) | Explicit workflow builder |
| [`SequentialOrchestrationWithCheckpoints`](12.SequentialAndConcurrent/SequentialOrchestrationWithCheckpoints/main.py) | Save and resume execution checkpoints |
| [`SequentialOrchestrationWithIntermediateOutput`](12.SequentialAndConcurrent/SequentialOrchestrationWithIntermediateOutput/main.py) | Observe intermediate stages |
| [`SequentialOrchestrationWithMiddleware`](12.SequentialAndConcurrent/SequentialOrchestrationWithMiddleware/main.py) | Agent/chat middleware in a sequence |
| [`ConcurrentOrchestration`](12.SequentialAndConcurrent/ConcurrentOrchestration/main.py) | Concurrent safety assessment and aggregation |
| [`SequenceOfConcurrentOrchestration`](12.SequentialAndConcurrent/SequenceOfConcurrentOrchestration/main.py) | Concurrent assessment followed by a sequential mission |

```powershell
python .\12.SequentialAndConcurrent\SequentialOrchestration\main.py
python .\12.SequentialAndConcurrent\SequentialOrchestrationAsPipeline\main.py
python .\12.SequentialAndConcurrent\SequentialOrchestrationWithBuilder\main.py
python .\12.SequentialAndConcurrent\SequentialOrchestrationWithIntermediateOutput\main.py
python .\12.SequentialAndConcurrent\SequentialOrchestrationWithMiddleware\main.py
python .\12.SequentialAndConcurrent\ConcurrentOrchestration\main.py
python .\12.SequentialAndConcurrent\SequenceOfConcurrentOrchestration\main.py
```

These samples use the real Python workflow/orchestration builders from the pinned
`agent-framework-orchestrations` package. Sequential history and previous-output-only
pipelines remain distinct; concurrent branches overlap before aggregation.
All runners accept `--provider openai|azure`.

```powershell
$sample = '.\12.SequentialAndConcurrent\SequentialOrchestrationWithCheckpoints\main.py'
python $sample
python $sample --interrupt
python $sample --resume "<printed-checkpoint-id>"
```

The checkpoint sample completes and then replays the Motors stage by default.
`--interrupt` saves a checkpoint before Motors and exits; `--resume` continues
from the specified checkpoint. Checkpoints are stored under the sample's
`runtime\checkpoints` folder and are Git-ignored. They contain readable mission
history and pickle-backed state: keep them private and resume **trusted files only**.
A replay can repeat tool actions; all tools here are simulated.
See the [chapter 12 guide](12.SequentialAndConcurrent/README.md) for graph structure,
checkpoint commands, aggregation semantics, and API differences.

## Chapter 13 — group chat

| Sample | Concept |
| --- | --- |
| [`GroupChatOrchestration`](13.GroupChat/GroupChatOrchestration/main.py) | Managed discussion and approval-based termination |
| [`GroupChatOrchestrationWithCustomManager`](13.GroupChat/GroupChatOrchestrationWithCustomManager/main.py) | Custom investigation/calibration manager |
| [`GroupChatOrchestrationWithHitl`](13.GroupChat/GroupChatOrchestrationWithHitl/main.py) | Human-in-the-loop group decisions |

```powershell
python .\13.GroupChat\GroupChatOrchestration\main.py
python .\13.GroupChat\GroupChatOrchestrationWithCustomManager\main.py
python .\13.GroupChat\GroupChatOrchestrationWithHitl\main.py
```

Group-chat managers select speakers, track decisions, and stop according to the
source's conditions, with a ten-turn bound. Output identifies each speaker and
reports when the turn limit is reached. All runners accept `--provider openai|azure`.
In the HITL example, `y`/`yes` approves; `n`/`no`, empty input, and EOF deny.
Other input supplies denial feedback. The Navigator is advisory: human denial
keeps motor tools unavailable. Approval is never fabricated or silently bypassed.
The [chapter 13 guide](13.GroupChat/README.md) documents round limits, human input,
manager state, and Python workflow behavior.

## Chapter 14 — handoff

| Sample | Concept |
| --- | --- |
| [`HandoffOrchestration`](14.Handoff/HandoffOrchestration/main.py) | Sensor routing criteria in the router's instructions |
| [`HandoffOrchestrationAndHandoffReasons`](14.Handoff/HandoffOrchestrationAndHandoffReasons/main.py) | Per-edge handoff descriptions |

```powershell
python .\14.Handoff\HandoffOrchestration\main.py
python .\14.Handoff\HandoffOrchestration\main.py --scenario fire
python .\14.Handoff\HandoffOrchestrationAndHandoffReasons\main.py --scenario rain --provider azure
```

Both examples use native `HandoffBuilder` workflows. A model chooses a single
specialist from simulated sensor readings, then hazard branches converge on the
sole motor authority. `--scenario random|safe|fire|rain` selects simulated readings;
it does not replace model routing with a Python conditional. The
[chapter 14 guide](14.Handoff/README.md) explains tool-evidence transfer, terminal
output, and native pauses when a handoff is not completed. No hardware is controlled.

## Existing introductory agents

| Sample | Concept |
| --- | --- |
| [`Agent`](Agent/main.py) | Basic client/tool example |
| [`AgentWithChatMessages`](AgentWithChatMessages/main.py) | Lists of chat messages |
| [`AgentWithChatMessagesAndMultiModal`](AgentWithChatMessagesAndMultiModal/main.py) | Text and binary image content |
| [`AgentWithStructuredOutput`](AgentWithStructuredOutput/main.py) | Typed JSON output |
| [`DeclarativeAgent`](DeclarativeAgent/main.py) | YAML agent specification and tool |

These existing samples continue to use `shared/config.py`. The chapter samples
reuse `shared/chat_clients.py` for provider configuration and SDK lifecycle.
Conversation loops, polling, tools, middleware, and local inference support live
in `shared/`; motor and color helpers are reused.

## Offline validation

```powershell
python -m unittest discover -s tests -v
```

The chapters 03–14 regression suite passes **312 tests**; all **74 entrypoints**
(69 chapter ports and five introductory samples) also import with dummy credentials
and network access blocked. Tests mock provider transports and local model generation;
no API credentials, API charges, Ollama server, or ONNX model downloads are needed.
MCP tests use bounded local subprocesses and loopback listeners, which close after tests.
The pinned dependencies emit some experimental/deprecation warnings and the documented
MCP SSE receive-stream warnings; these are not suppressed. Live inference still
requires the relevant credentials, model/deployment, or local runtime.

## Live checks and Python interview preparation

See the [individual live-run report](LIVE-RUN-REPORT.md) for verified provider/tool
execution, unavailable integrations, model-quality differences, and fixes found
by running the scripts. Offline test success is not proof of live provider behavior.
The [4–8-hour .NET-to-Python interview plan](PYTHON-INTERVIEW-PLAN.md) prioritizes
Python semantics and backend practice over revisiting familiar programming concepts.

API references: [OpenAI providers](https://learn.microsoft.com/agent-framework/agents/providers/openai),
[official Python samples](https://github.com/microsoft/agent-framework/tree/main/python/samples),
and [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai).
