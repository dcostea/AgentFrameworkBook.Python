# Memories — Python ports

All seven folders retain their C# project names. Run a sample from any working
folder using its absolute path, or from the repository root:

```powershell
.\.venv\Scripts\python.exe .\07.Memories\AgentWithCustomFileBasedContextProvider\main.py
```

Every entry point accepts `--provider openai` (default) or `--provider azure`.
Chat configuration and SDK lifecycle use `shared.chat_clients`: OpenAI uses
`OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, and `OPENAI_CHAT_MODEL_ID`; Azure
uses `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and
`AZURE_OPENAI_DEPLOYMENT_NAME`, with the helper's optional API-version routing.
There are no extra chapter-specific package dependencies: the installed
Agent Framework core/OpenAI integration and OpenAI SDK suffice. Stores and
cosine calculations use the Python standard library, not an external database.

## Behavior preserved

| Project | Memory behavior |
| --- | --- |
| `AgentWithCustomFileBasedContextProvider` | Reloads guideline instructions and the deliberate date-seed conversation before every invocation. Prints the simulated June 6–7 weather update afterward without changing either asset. Keeps new conversation turns in session memory. |
| `AgentWithFileBasedChatHistoryProvider` | Atomically replaces UTF-8 JSON containing only new user/assistant turns appended to existing history. Each source/session pair has a separate file. |
| `AgentWithInMemoryChatHistoryProvider` | Before retrieval, mutates session history to keep the first system message and the last three non-system messages, excluding function calls/results. New turns are appended afterward, so stored count may temporarily exceed three. |
| `AgentWithInMemoryChatMessagesAndInMemoryContextProvider` | Adds `Respond with JSON` and a system message from the optional guidelines file; keeps conversation in `InMemoryHistoryProvider`. A missing file supplies no context. |
| `AgentWithInMemoryVectorStoreProvider` | Weather, motors and auditor agents share embedded memory and one conversation session. Stores actual input/output embeddings, performs cosine top-three retrieval, and searches across sessions for `User_1`. |
| `AgentWithTextSearchProvider` | Four active robot policies; exact keyword-substring matching, source names/links and stable match-count ordering. Searches before each invocation and retains up to six recent user query texts; logs retention after each invocation. |
| `AgentWithVectorStoreChatHistoryProvider` | Full serialized messages in a process-local metadata collection. Loads the **earliest five by timestamp**, not nearest vectors or most recent messages. Like the original, this sample does not require embeddings. |

The custom retrieval/context examples explicitly compose an
`InMemoryHistoryProvider` so follow-up prompts retain the C# agent's ordinary
conversation history. Retrieved messages and deliberate seed messages are not
appended into that history or re-embedded on each run. The keyword provider's
recent query memory includes users only, matching the C# provider's default
role filter; it is separate from the full conversation history.

### Durable file resume

The file-history sample prints a session ID. Reuse it to start a fresh Python
process and ask the follow-up against the existing transcript:

```powershell
.\.venv\Scripts\python.exe .\07.Memories\AgentWithFileBasedChatHistoryProvider\main.py --session-id <printed-id>
```

Writable JSON lives only in that sample's `runtime` directory; the root ignore
configuration should include `07.Memories/**/runtime/`. The committed C# run dump
is intentionally **not** copied or used as writable initial history. The
read-only context provider's explicit date seed is copied separately and
converted from C# `Role`/`Contents` text JSON into Python `Message` objects.
File history saves complete Python messages (including role, author, ID,
metadata and content), and a session ID is sufficient for cold reload.

Atomic replacement protects the previous file if a write fails. This is a
sequential teaching sample, not a concurrent multi-process persistence service.
History is plaintext; use appropriate local file permissions.

### Semantic embeddings

Only `AgentWithInMemoryVectorStoreProvider` needs an embedding model:

- OpenAI: `OPENAI_EMBEDDING_MODEL_ID`.
- Azure: `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` (the embedding deployment, not
  the chat deployment).

The model must support `dimensions=1536` and `encoding_format="float"`, for
example `text-embedding-3-small`. The SDK client is borrowed from the existing
chat helper, so the embedding call uses the same provider endpoint/credentials
and is closed with that client. The implementation fails explicitly on missing
configuration or invalid embedding responses; it never falls back to keyword
retrieval. No live model or embedding calls were used to validate this migration.

Both in-memory collection stores lose their contents on process exit. A new
metadata-history provider can resume with the same session ID only if it shares
the original store. Serializing an `AgentSession` alone cannot preserve external
in-memory collection records. `InMemoryHistoryProvider` messages, in contrast,
can round-trip with `AgentSession.to_dict()` / `AgentSession.from_dict()`.

The keyword adapter intentionally preserves the original quirks: split only on
space, period, comma, colon and semicolon (not newline); a stored keyword must
contain the query token, not the reverse; repeated tokens increase the count;
and commented-out weather entries stay disabled. It does not stem words or
search arbitrary document text. The declared text-search `.csproj` asset is not
read by the source program and was not invented for the Python version.

## Offline validation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_memories.py -q
```

Tests drive the real `Agent` pipeline with offline chat/embedding doubles and
cover context hooks, reduction, file cold resume and atomic failure cleanup,
source/session isolation, full message serialization, cosine ranking versus
keyword matching, embedding request options, all seven imports and both
provider lifecycle paths. Tests block `.env` loading and do not read credentials.

API references used for the port:

- [Python context providers](https://learn.microsoft.com/agent-framework/concepts/agents/conversations/context-providers?pivots=programming-language-python)
- [Official custom history sample](https://github.com/microsoft/agent-framework/blob/main/python/samples/02-agents/conversations/custom_history_provider.py)
- [Official collection history sample](https://github.com/microsoft/agent-framework/blob/main/python/samples/02-agents/conversations/vector_store_history_provider.py)

The installed `agent-framework-core`/`agent-framework-openai` 1.8.1 APIs are the
runtime authority (`ContextProvider(source_id)`, keyword-only `before_run` and
`after_run`, `SessionContext`, `HistoryProvider` and `AgentSession`). Custom
providers deliberately avoid depending on newer vector integrations or extra
packages shown in current upstream examples.
