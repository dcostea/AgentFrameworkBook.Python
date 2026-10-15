# Individual sample execution report

**Run date:** 4 October 2026. **Interpreter:** existing Python 3.13 virtual environment. **Live model:** configured OpenAI `gpt-5.4-mini`.

## Bottom line

- All **69 migrated entrypoints** were attempted individually. MCP servers were launched as actual processes and checked through their real protocol clients, not left waiting without a client.
- **56** completed the available live/local integrations after fixes and reruns.
- **12** are blocked by missing credentials or local model runtimes. Their original provider was retained; they were not falsely tested by substituting OpenAI.
- **1** evaluator completed all 30 records but disagreed with three expected labels, so its intentional exit code was 1. This is not an all-green evaluation.
- The **five original introductory scripts** also executed successfully, including the original multimodal and declarative examples.
- Final regression: **312 offline tests passed**; all **74 entrypoints imported** with dummy credentials and network blocked; `pip check` reported no broken requirements.

**Execution success is not a guarantee of model-answer correctness.** Outputs vary between runs. These are simulated robot examples, not validated physical navigation/safety software. A live semantic-memory answer misidentified the second-last movement, and some generated routes were geometrically questionable even when the tool chain executed. Deterministic routing/validation is needed for production guarantees.

## Individual outcomes

PASS means the relevant script/protocol ran with actual configured integrations and observable completion; it does not mean every optional CLI mode or every possible branch was tested. BLOCKED records a real launch and missing prerequisite, not a code success.

| Chapter | Completed | Blocked | Quality disagreement |
| --- | ---: | ---: | ---: |
| 03 | 2 | 7 | 0 |
| 04 | 2 | 5 | 0 |
| 05 | 10 | 0 | 0 |
| 06 | 5 | 0 | 0 |
| 07 | 7 | 0 | 0 |
| 08 | 4 | 0 | 0 |
| 09 | 5 | 0 | 0 |
| 10 | 5 | 0 | 0 |
| 11 | 3 | 0 | 1 |
| 12 | 8 | 0 | 0 |
| 13 | 3 | 0 | 0 |
| 14 | 2 | 0 | 0 |

| Script | Outcome | Evidence / prerequisite |
| --- | --- | --- |
| `03.ChatClients\AnthropicChatClient\main.py` | BLOCKED | `ANTHROPIC_API_KEY` absent. |
| `03.ChatClients\AzureOpenAIChatClientWithImages\main.py` | BLOCKED | Azure audio endpoint/key/deployment missing (default audio mode); Azure vision mode also unavailable. |
| `03.ChatClients\AzureOpenAIChatClient\main.py` | BLOCKED | Azure endpoint/key/deployment absent; this sample defaults to Azure. |
| `03.ChatClients\ChatClientWithMiddleware\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `03.ChatClients\ChatbotWithChatOptions\main.py` | BLOCKED | Azure endpoint/key/deployment absent; this sample defaults to Azure. |
| `03.ChatClients\ChatbotWithStreaming\main.py` | BLOCKED | Azure endpoint/key/deployment absent; this sample defaults to Azure. |
| `03.ChatClients\ChatbotWithTools\main.py` | BLOCKED | Azure endpoint/key/deployment absent; this sample defaults to Azure. |
| `03.ChatClients\Chatbot\main.py` | BLOCKED | Azure endpoint/key/deployment absent; this sample defaults to Azure. |
| `03.ChatClients\OpenAIChatClient\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `04.ChatClients\AzureOpenAIResponsesWithBackgroundResponses\main.py` | BLOCKED | Azure endpoint/key/deployment absent; this sample defaults to Azure. |
| `04.ChatClients\AzureOpenAIResponses\main.py` | BLOCKED | Azure endpoint/key/deployment absent; this sample defaults to Azure. |
| `04.ChatClients\OllamaChatClient\main.py` | BLOCKED | Ollama executable exists, but its loopback API is not running; required model availability unverified. |
| `04.ChatClients\OnnxChatClientMultiModal\main.py` | BLOCKED | Local ONNX model path/export missing; no model download performed. |
| `04.ChatClients\OnnxChatClient\main.py` | BLOCKED | Local ONNX model path/export missing; no model download performed. |
| `04.ChatClients\OpenAIResponsesWithBackgroundResponses\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `04.ChatClients\OpenAIResponses\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `05.Agents\AgentWithBackgroundResponses\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `05.Agents\AgentWithChatMessagesAndMultiModal\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `05.Agents\AgentWithChatMessages\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `05.Agents\AgentWithInServiceConversation\main.py` | PASS | Actual Responses conversation created/continued; provider cleanup reported Deleted: True. |
| `05.Agents\AgentWithSerialization\main.py` | PASS | Session serialized/deserialized and continued; generated snapshot removed afterward. |
| `05.Agents\AgentWithSession\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `05.Agents\AgentWithStreaming\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `05.Agents\AgentWithStructuredOutput\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `05.Agents\Agent\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `05.Agents\MultipleAgentsWithSameSession\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `06.Tools\AgentWithFunctionApproval\main.py` | PASS | Protected stop/backward calls explicitly approved and executed. |
| `06.Tools\AgentWithManualToolInvocation\main.py` | PASS | Correlated model tool calls explicitly invoked; final assistant response completed. |
| `06.Tools\AgentWithMultipleToolCalls\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `06.Tools\AgentWithToolMode\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `06.Tools\AgentWithTools\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `07.Memories\AgentWithCustomFileBasedContextProvider\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `07.Memories\AgentWithFileBasedChatHistoryProvider\main.py` | PASS | History written and resumed in a new process; generated file removed afterward. |
| `07.Memories\AgentWithInMemoryChatHistoryProvider\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `07.Memories\AgentWithInMemoryChatMessagesAndInMemoryContextProvider\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `07.Memories\AgentWithInMemoryVectorStoreProvider\main.py` | PASS | Real embeddings/retrieval/auditor completed with process-only `OPENAI_EMBEDDING_MODEL_ID=text-embedding-3-small` override. |
| `07.Memories\AgentWithTextSearchProvider\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `07.Memories\AgentWithVectorStoreChatHistoryProvider\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `08.MCP\MCPClientWithHttp\main.py` | PASS | Actual MCP process/protocol discovery/tool/resources/prompts and live client agent exercised; helpers stopped. |
| `08.MCP\MCPClientWithStdio\main.py` | PASS | Actual MCP process/protocol discovery/tool/resources/prompts and live client agent exercised; helpers stopped. Experimental Tasks/progress also exercised. |
| `08.MCP\MCPServerWithHttp\main.py` | PASS | Actual MCP process/protocol discovery/tool/resources/prompts and live client agent exercised; helpers stopped. |
| `08.MCP\MCPServerWithStdio\main.py` | PASS | Actual MCP process/protocol discovery/tool/resources/prompts and live client agent exercised; helpers stopped. Experimental Tasks/progress also exercised. |
| `09.ChatClientMiddleware\ChatClientFunctionCallingMiddleware\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `09.ChatClientMiddleware\ChatClientResponseMiddleware\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `09.ChatClientMiddleware\ChatClientSharedFunctionMiddleware\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `09.ChatClientMiddleware\MiddlewareBaseline\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `09.ChatClientMiddleware\MiddlewareMixed\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `10.AgentMiddleware\AgentFunctionCallingMiddleware\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `10.AgentMiddleware\AgentMiddlewareMixed\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `10.AgentMiddleware\AgentResponseMiddleware\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `10.AgentMiddleware\AgentSharedFunctionMiddleware\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `10.AgentMiddleware\AgentWithLogging\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `11.ObservabilityAndEvaluations\Evaluators\main.py` | QUALITY MISMATCH | All 30 records scored; 27 matched fixture labels, 3 disagreed; exit 1 is intentional. |
| `11.ObservabilityAndEvaluations\Observations.AppHost\main.py` | PASS | Real prompt/tool calls plus 13 local spans/5 logs; identity/launcher adaptation only where applicable. |
| `11.ObservabilityAndEvaluations\ObservationsWithAgent365\main.py` | PASS | Real prompt/tool calls plus 13 local spans/5 logs; identity/launcher adaptation only where applicable. |
| `11.ObservabilityAndEvaluations\Observations\main.py` | PASS | Real prompt/tool calls plus 13 local spans/5 logs; identity/launcher adaptation only where applicable. |
| `12.SequentialAndConcurrent\ConcurrentOrchestration\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `12.SequentialAndConcurrent\SequenceOfConcurrentOrchestration\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `12.SequentialAndConcurrent\SequentialOrchestrationAsPipeline\main.py` | PASS | After fix, actual motor calls executed; prose-only completion is rejected. |
| `12.SequentialAndConcurrent\SequentialOrchestrationWithBuilder\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `12.SequentialAndConcurrent\SequentialOrchestrationWithCheckpoints\main.py` | PASS | Default replay, interrupt-before-Motors and fresh-process resume exercised; generated state removed afterward. |
| `12.SequentialAndConcurrent\SequentialOrchestrationWithIntermediateOutput\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `12.SequentialAndConcurrent\SequentialOrchestrationWithMiddleware\main.py` | PASS | Default rain denial correctly aborted; deterministic safe sensors plus real model completed seven motor calls. |
| `12.SequentialAndConcurrent\SequentialOrchestration\main.py` | PASS | Live configured OpenAI execution completed; structured/stream/tool output checked where applicable. |
| `13.GroupChat\GroupChatOrchestrationWithCustomManager\main.py` | PASS | Two fresh sensor pairs observed; both denial and successful calibration/approval paths exercised. |
| `13.GroupChat\GroupChatOrchestrationWithHitl\main.py` | PASS | First proposal denied, revised proposal approved; no Motors calls before human approval. |
| `13.GroupChat\GroupChatOrchestration\main.py` | PASS | After fix, revised plan received Navigator approval and actual Motors execution completed. |
| `14.Handoff\HandoffOrchestrationAndHandoffReasons\main.py` | PASS | Both variants checked on safe/fire/rain; hazards stopped with no forward/backward/turn calls. |
| `14.Handoff\HandoffOrchestration\main.py` | PASS | Both variants checked on safe/fire/rain; hazards stopped with no forward/backward/turn calls. |

## Original introductory samples

| Script | Outcome |
| --- | --- |
| `AgentWithChatMessagesAndMultiModal\main.py` | PASS |
| `AgentWithChatMessages\main.py` | PASS |
| `AgentWithStructuredOutput\main.py` | PASS |
| `Agent\main.py` | PASS |
| `DeclarativeAgent\main.py` | PASS |

## Fixes discovered only by actual execution

1. **Previous-output pipeline:** Motors sometimes echoed its plan without calling tools. Added explicit invocation guidance, required initial tool selection and completed-call accounting; incomplete execution now fails rather than appearing successful.
2. **Group chat:** planning omitted numeric tool parameters, so the Navigator repeatedly rejected indistinguishable plans. Added text-only schema references and numeric revision guidance; tools still remain unavailable before approval. Separately, custom management could count replayed sensor readings as a second observation. Only fresh successful tool evidence now counts.
3. **Handoff:** a fire specialist announced a transfer in text without invoking `handoff_to_MotorsAgent`. Instructions/descriptions now require the actual routing tool. Motors instructions explicitly recognize plain-text transferred hazard evidence and stop rather than executing the mission. Native missing-handoff pauses remain honest, and the CLI exits nonzero without terminal output.

No fabricated success, silent auto-approval, alternate provider substitution or untrusted checkpoint loading was used. New regression checks cover the coupled changes.

## Evaluator details

The custom Python judges completed all twelve source datasets, 30 records total. Disagreements occurred for `robby-intent-003`, `robby-coherence-001`, and `robby-groundedness-001`. Deterministic direction-change fixtures matched. The judges are not the .NET evaluator algorithms; do not adjust source expected labels just to make an LLM judge pass. Model-backed results are probabilistic and should be reviewed.

## What you need to configure

- Azure samples need `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT_NAME`; audio requires the separate `AZURE_OPENAI_AUDIO_*` settings. No Azure key was invented or copied from OpenAI.
- Anthropic needs `ANTHROPIC_API_KEY` and a suitable model.
- Ollama needs an already-running server and an installed model matching `OLLAMA_MODEL`.
- ONNX samples need complete compatible local exports in `ONNX_MODEL_PATH` / `ONNX_MULTIMODAL_MODEL_PATH`; the installed package alone is not a model.
- Semantic memory was verified using a temporary embedding-model setting. To run it yourself with OpenAI, add `OPENAI_EMBEDDING_MODEL_ID=text-embedding-3-small` to your root `.env` or process environment. Your existing `.env` was not edited.

## Beginner run commands

Run from this workspace root; no virtual-environment activation is required with an explicit interpreter path:

```powershell
.\.venv\Scripts\python.exe .\05.Agents\AgentWithStructuredOutput\main.py
.\.venv\Scripts\python.exe .\06.Tools\AgentWithManualToolInvocation\main.py
.\.venv\Scripts\python.exe .\13.GroupChat\GroupChatOrchestrationWithHitl\main.py --provider openai
.\.venv\Scripts\python.exe .\14.Handoff\HandoffOrchestrationAndHandoffReasons\main.py --scenario rain
```

The first two do not require console input. HITL requires an explicit approval; `y` approves, `n`/empty input/EOF denies. Live observation samples without `--prompt` are interactive: EOF can exit with no model call. For a bounded single call use:

```powershell
.\.venv\Scripts\python.exe .\11.ObservabilityAndEvaluations\Observations\main.py --live --telemetry local --prompt "Go forward 2 meters, turn left 90 degrees, then stop."
```

These commands may incur API charges. Start with the offline tests if you only want to practice Python syntax/debugging.

## Validation boundaries and cleanup

- Real model prompts/media were sent only under your live-run authorization. No external telemetry collector or Agent 365 ingestion was contacted. Local spans/logs were verified with SDK instrumentation enabled; disabling the SDK yields zero spans even if the model executes.
- Agent 365 is a standard-OTel identity adaptation; AppHost is a bounded observation launcher. Neither is native .NET feature parity.
- Optional streaming/approval/tool modes were not exhaustively live-tested. Actual defaults were retained except explicit live/stream/scenario options documented above.
- MCP loopback readiness used a real initialize/list-tools/tool-call exchange. All owned servers/client transports were stopped. Generated session/history/checkpoint files from this validation were removed; original fixtures were retained.
- Pinned third-party SDK experimental/deprecation/SSE cleanup warnings remain documented; installed package code was not patched.
- Redacted execution logs and structured per-run JSON are stored in this session's artifact folder outside the project. Some initial artifact logs were overwritten before unique-variant names were added; final report selections point at completion runs, and initial JSON retains output excerpts.

Next: follow [the four-hour core / eight-hour extended study plan](PYTHON-INTERVIEW-PLAN.md).
