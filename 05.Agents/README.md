# Chapter 05 — Agents

Python migrations of all ten projects in `dcostea/AgentFrameworkBook/05.Agents`, using `agent-framework-core` and `agent-framework-openai` **1.8.1**. The similarly named introductory examples at the repository root are separate and unchanged.

Run from the repository root, for example:

```powershell
.\.venv\Scripts\python.exe .\05.Agents\Agent\main.py
.\.venv\Scripts\python.exe .\05.Agents\AgentWithSerialization\main.py
```

All examples insert the repository root into `sys.path`, resolve data/output paths relative to their own `main.py`, and close the asynchronous OpenAI SDK client on success or failure. They use the existing `shared.chat_clients` factories, not the legacy configuration adapter.

| Project | Behavior preserved |
| --- | --- |
| `Agent` | Two stateless runs; the second cannot recall the first response. Named `RobotCarAgent`. |
| `AgentWithBackgroundResponses` | Responses API background execution, session, initial/final status, continuation-token polling. |
| `AgentWithChatMessages` | Explicit user/assistant/user history with the original eight moves; append the new assistant reply. |
| `AgentWithChatMessagesAndMultiModal` | Original local map sent as image content; alternative original image/audio assets are opt-in. |
| `AgentWithInServiceConversation` | Create server conversation, two turns, paginated history, delete in `finally`. |
| `AgentWithSerialization` | Run, serialize local session to UTF-8 JSON, reload, and ask about the earlier moves. |
| `AgentWithSession` | Two runs sharing one in-memory session. |
| `AgentWithStreaming` | Print text as it arrives and report input/output/total usage. |
| `AgentWithStructuredOutput` | Typed `Steps` array of `Move` and nullable `Value`; temperature 0.4, output limit 300. |
| `MultipleAgentsWithSameSession` | `MotorsAgent` acts; `AuditorAgent` receives the same history and explains the moves in fewer than 50 words. |

## Configuration and model requirements

The default provider is OpenAI, using `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, `OPENAI_CHAT_MODEL_ID`, and optional `OPENAI_RESPONSES_MODEL_ID` (otherwise the chat model). Each asynchronous `main(provider="azure")` also supports the existing Azure factory mapping: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT_NAME`, optional `AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME`, and optional `AZURE_OPENAI_API_VERSION`. No credentials are serialized.

Use a vision-capable model for images, an audio-input-capable chat model for MP3, a model accepting JSON-schema structured output and temperature for `AgentWithStructuredOutput`, and a Responses-capable deployment for the two Responses examples. Provider/model capability errors are surfaced rather than silently switching API types. For Azure audio, the existing factory uses `AZURE_OPENAI_AUDIO_ENDPOINT`, `AZURE_OPENAI_AUDIO_API_KEY`, and `AZURE_OPENAI_AUDIO_DEPLOYMENT_NAME`; for OpenAI set `OPENAI_CHAT_MODEL_ID` to the desired audio-capable model.

The multimodal default matches the active C# sample: `Data\Map.png` with `image/png`. The other assets were commented alternatives in C#; activate them explicitly:

```powershell
.\.venv\Scripts\python.exe .\05.Agents\AgentWithChatMessagesAndMultiModal\main.py --asset path.jpg
.\.venv\Scripts\python.exe .\05.Agents\AgentWithChatMessagesAndMultiModal\main.py --asset road_and_tree.png
.\.venv\Scripts\python.exe .\05.Agents\AgentWithChatMessagesAndMultiModal\main.py --asset Task.mp3
```

`road_and_tree.png` is intentionally labeled `image/png`, correcting the C# commented example's `image/jpeg`. Local assets replace the old optional remote URLs, so no downloads are required. MP3 uses `audio/mp3` because the pinned Python chat adapter recognizes that alias to produce OpenAI's `input_audio`/`mp3` payload; C# used `audio/mpeg`. The default image example does not unnecessarily read the unused audio file. The sample constructs one framework message with text and media; the 1.8.1 chat adapter serializes those parts as adjacent user messages.

## Sessions and service storage

Local sessions use Python's real `AgentSession.to_dict()` / `AgentSession.from_dict()` schema. `AgentWithSerialization\agent_session.json` is created at runtime from the current Python response, not copied from the generated C# fixture. The cosmetic C# two-second serialization delay is omitted. Treat the generated file as plaintext conversation data; keep secrets out of prompts and do not commit it. Integration should ignore this exact runtime path:

```text
05.Agents/AgentWithSerialization/agent_session.json
```

The two-agent example explicitly shares the `shared_history` provider namespace, so both agents see the motors response but retain their own instructions and author names.

The in-service example uses `sdk.conversations.create`, `sdk.conversations.items.list`, and `sdk.conversations.delete`. `agent.get_session(service_session_id=conversation.id)` makes the framework translate the `conv_*` ID into the Responses API's `conversation` field, without resending local history. The Python 1.8.1 adapter omits instructions when a conversation ID is present; the conversation is therefore seeded with the robot system instructions. Deletion runs even if a turn or history listing fails. Server-side storage requires a supporting provider; deleting a conversation is not a guarantee that all associated response items are erased under the provider's retention policies.

Background runs send `background=True, store=True` on creation, then call `agent.run([], session=session, options={"continuation_token": ...})` for retrieval, not new completions. Polling is bounded to 60 one-second attempts by default, with configurable `max_attempts` and `delay` in `run_example`. A timeout identifies the response so it can be retrieved or cancelled with the provider SDK. Terminal failed, cancelled, or incomplete responses raise an error. These operations store data with the provider and may incur costs when run normally.

## Offline tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_agents.py -q
```

The tests import all ten scripts and use the real framework and asynchronous OpenAI SDK with `httpx.MockTransport`. No model, network, or real credential is used. The root dependency/configuration/documentation files are intentionally outside this chapter migration's write scope; no additional packages are required.

## API references

- [Microsoft Agent Framework — sessions, service IDs, and serialization](https://learn.microsoft.com/agent-framework/concepts/agents/conversations/session?pivots=programming-language-python)
- [Microsoft Agent Framework — OpenAI integration](https://learn.microsoft.com/agent-framework/integrations/by-component/model-providers/openai?pivots=programming-language-python)

The exact continuation, MIME conversion, and service-conversation behavior was also verified against the installed 1.8.1 `Agent`, `AgentSession`, `OpenAIChatClient`, and `OpenAIChatCompletionClient` source and the installed OpenAI SDK conversations API.
