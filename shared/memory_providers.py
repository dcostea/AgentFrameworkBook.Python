"""Chapter 07 providers for Agent Framework's session/context pipeline.

File history is durable; both collection stores below are process-local. Semantic
memory uses actual SDK embeddings, not the keyword search adapter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any
from uuid import uuid4

from agent_framework import (
    AgentSession, Content, ContextProvider, HistoryProvider,
    InMemoryHistoryProvider, Message, SessionContext,
)

LOGGER = logging.getLogger(__name__)
ROBOT_INSTRUCTIONS = """You are an AI assistant controlling a robot car capable of performing basic moves: forward, backward, turn left, turn right, and stop.
You have to break down the provided complex commands into the basic moves you know.
Respond only with the moves and their parameters (angle or distance), without any additional explanations."""
ROBOT_PROMPTS = (
    'Complex command:\n"There is a tree directly in front of the car. Avoid it and then return to the original path."',
    "What was your second last basic move?",
)
MORE_WEATHER = """June 6, 2025 - Morning: 13°C, partly cloudy, wind 9 km/h, dry. Afternoon: 21°C, sunny, wind 11 km/h, dry roads. Night: 12°C, clear, wind 6 km/h, calm.
June 7, 2025 - Morning: 14°C, sunny, wind 7 km/h, dry. Afternoon: 23°C, mostly sunny, wind 10 km/h, no rain. Night: 13°C, few clouds, wind 5 km/h, calm."""


def load_messages(path: Path) -> list[Message]:
    """Read Python messages or the source sample's deliberately seeded C# text JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("Message data must be a JSON array")
    messages = []
    for item in data:
        if "Role" not in item:
            messages.append(Message.from_dict(item))
            continue
        contents = []
        for content in item.get("Contents", []):
            if content.get("$type") != "text":
                raise ValueError("Only text C# seed contents are supported")
            contents.append(Content.from_text(content.get("Text", "")))
        properties = dict(item.get("AdditionalProperties") or {})
        if item.get("CreatedAt"):
            properties["created_at"] = item["CreatedAt"]
        messages.append(Message(
            role=item["Role"], contents=contents,
            author_name=item.get("AuthorName"), message_id=item.get("MessageId"),
            additional_properties=properties,
        ))
    return messages


def atomic_write_json(path: Path, value: Any) -> None:
    """Replace a complete UTF-8 document, retaining the old file on write failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _session_key(session_id: str | None) -> str:
    if not session_id:
        raise ValueError("An explicit session ID is required")
    return session_id


class CustomFileBasedContextProvider(ContextProvider):
    """Reload guidelines and a read-only seeded conversation before every run."""

    def __init__(self, instructions_path: Path, history_path: Path, source_id="file-context"):
        super().__init__(source_id)
        self.instructions_path = Path(instructions_path)
        self.history_path = Path(history_path)

    async def before_run(self, *, agent, session: AgentSession, context: SessionContext, state):
        context.extend_instructions(self.source_id, self.instructions_path.read_text(encoding="utf-8-sig"))
        context.extend_messages(self, load_messages(self.history_path))

    async def after_run(self, *, agent, session: AgentSession, context: SessionContext, state):
        instructions = self.instructions_path.read_text(encoding="utf-8-sig")
        print("[after_run] Simulated file update (not written to disk):")
        print(instructions + "\n" + MORE_WEATHER)


class FileBasedContextProvider(ContextProvider):
    """Optional file context: JSON instruction plus a system guidelines message."""

    def __init__(self, path: Path, source_id="guidelines"):
        super().__init__(source_id)
        self.path = Path(path)

    async def before_run(self, *, agent, session: AgentSession, context: SessionContext, state):
        if self.path.exists():
            context.extend_instructions(self.source_id, "Respond with JSON")
            context.extend_messages(self, [Message(
                "system", [self.path.read_text(encoding="utf-8-sig")],
            )])


class MessageCountingHistoryProvider(InMemoryHistoryProvider):
    """C# MessageCountingChatReducer: first system plus last N non-tool messages.

    Reduction happens before retrieval and updates stored session history, not
    after saving a turn, matching the source's default reducer trigger.
    """

    def __init__(self, target_count=3, source_id="reduced-history"):
        if target_count <= 0:
            raise ValueError("target_count must be positive")
        super().__init__(source_id)
        self.target_count = target_count

    async def get_messages(self, session_id, *, state=None, **kwargs):
        messages = await super().get_messages(session_id, state=state)
        system = next((m for m in messages if m.role == "system"), None)
        ordinary = [m for m in messages if m.role != "system" and not any(
            c.type in {"function_call", "function_result"} for c in m.contents
        )]
        reduced = ([system] if system is not None else []) + ordinary[-self.target_count:]
        if state is not None:
            state["messages"] = reduced
        return reduced


class CustomFileBasedChatHistoryProvider(HistoryProvider):
    """One atomic JSON document per provider source and session.

    Stores new input/output messages only (never the reloaded context). The
    session ID is sufficient for cold resume; no serialized SDK state is needed.
    Intended for sequential sample use, not concurrent multi-process writers.
    """

    def __init__(self, storage_path: Path, source_id="file-history"):
        super().__init__(source_id)
        self.storage_path = Path(storage_path)

    def path_for(self, session_id: str) -> Path:
        scope = json.dumps([self.source_id, _session_key(session_id)], ensure_ascii=False)
        return self.storage_path / (hashlib.sha256(scope.encode("utf-8")).hexdigest() + ".json")

    async def get_messages(self, session_id, *, state=None, **kwargs):
        path = self.path_for(session_id)
        if not path.exists():
            return []
        document = json.loads(path.read_text(encoding="utf-8"))
        if (document.get("version"), document.get("source_id"), document.get("session_id")) != (
            1, self.source_id, session_id,
        ):
            raise ValueError("History file does not match the provider/session scope")
        return [Message.from_dict(item) for item in document["messages"]]

    async def save_messages(self, session_id, messages: Sequence[Message], *, state=None, **kwargs):
        if not messages:
            return
        existing = await self.get_messages(session_id)
        atomic_write_json(self.path_for(session_id), {
            "version": 1, "source_id": self.source_id, "session_id": session_id,
            "messages": [message.to_dict() for message in [*existing, *messages]],
        })


@dataclass(frozen=True)
class ChatHistoryItem:
    key: str
    source_id: str
    session_id: str
    timestamp: str
    serialized_message: str
    message_text: str
    sequence: int


class InMemoryChatHistoryStore:
    """Metadata collection, deliberately without embeddings or disk persistence."""

    def __init__(self):
        self.records: dict[tuple[str, str, str], ChatHistoryItem] = {}
        self._sequence = 0

    def upsert(self, source_id: str, session_id: str, message: Message) -> None:
        key = message.message_id or str(uuid4())
        scope = (source_id, session_id, key)
        previous = self.records.get(scope)
        self._sequence += 1
        self.records[scope] = ChatHistoryItem(
            key=key, source_id=source_id, session_id=session_id,
            timestamp=previous.timestamp if previous else datetime.now(timezone.utc).isoformat(),
            serialized_message=json.dumps(message.to_dict(), ensure_ascii=False),
            message_text=message.text,
            sequence=previous.sequence if previous else self._sequence,
        )

    def get(self, source_id: str, session_id: str) -> list[Message]:
        rows = sorted((row for row in self.records.values()
                       if row.source_id == source_id and row.session_id == session_id),
                      key=lambda row: (row.timestamp, row.sequence))
        return [Message.from_dict(json.loads(row.serialized_message)) for row in rows]


class VectorStoreChatHistoryProvider(HistoryProvider):
    """Faithful metadata-history port: earliest top N by timestamp, not similarity.

    Resume works with a new provider/session sharing the SAME in-memory store.
    A newly constructed store is empty, even after a session snapshot restore.
    """

    def __init__(self, store: InMemoryChatHistoryStore, top_results=10, source_id="collection-history"):
        if top_results <= 0:
            raise ValueError("top_results must be positive")
        super().__init__(source_id)
        self.store = store
        self.top_results = top_results

    async def all_messages(self, session_id: str) -> list[Message]:
        return self.store.get(self.source_id, _session_key(session_id))

    async def get_messages(self, session_id, *, state=None, **kwargs):
        return (await self.all_messages(session_id))[:self.top_results]

    async def save_messages(self, session_id, messages: Sequence[Message], *, state=None, **kwargs):
        session_id = _session_key(session_id)
        for message in messages:
            self.store.upsert(self.source_id, session_id, message)


def embedding_model_for(provider: str) -> str:
    names = {"openai": "OPENAI_EMBEDDING_MODEL_ID", "azure": "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"}
    if provider not in names:
        raise ValueError(f"Unknown provider: {provider}")
    value = os.environ.get(names[provider], "").strip()
    if not value:
        raise RuntimeError(f"{names[provider]} is required (use an embedding model supporting dimensions)")
    return value


def _vector(values: Sequence[float], dimensions: int) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != dimensions or not all(math.isfinite(value) for value in result):
        raise ValueError(f"Expected {dimensions} finite embedding values")
    return result


class OpenAIEmbeddingGenerator:
    """Borrow the caller-owned OpenAI/Azure SDK client and request float vectors."""

    def __init__(self, sdk, model: str, dimensions=1536):
        if not model or dimensions <= 0:
            raise ValueError("An embedding model and positive dimensions are required")
        self.sdk, self.model, self.dimensions = sdk, model, dimensions

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        response = await self.sdk.embeddings.create(
            input=list(texts), model=self.model, dimensions=self.dimensions, encoding_format="float",
        )
        rows = sorted(response.data, key=lambda row: row.index)
        if [row.index for row in rows] != list(range(len(texts))):
            raise ValueError("Embedding response indices do not match the input batch")
        return [_vector(row.embedding, self.dimensions) for row in rows]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Cosine vectors must have the same nonzero dimension")
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


@dataclass(frozen=True)
class MemoryItem:
    key: str
    source_id: str
    user_id: str
    session_id: str
    vector: tuple[float, ...]
    serialized_message: dict[str, Any]

    @property
    def message(self) -> Message:
        return Message.from_dict(self.serialized_message)


class InMemoryVectorStore:
    """Actual cosine-ranked vectors, scoped by provider/user and optionally session."""

    def __init__(self, dimensions=1536):
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        self.records: dict[tuple[str, str, str, str], MemoryItem] = {}

    def upsert(self, source_id, user_id, session_id, message: Message, vector: Sequence[float]):
        key = message.message_id or str(uuid4())
        row = MemoryItem(key, source_id, user_id, session_id, _vector(vector, self.dimensions), message.to_dict())
        self.records[(source_id, user_id, session_id, key)] = row

    def search(self, vector, *, source_id, user_id, top_k=3, session_id=None):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        vector = _vector(vector, self.dimensions)
        candidates = [(row, cosine_similarity(vector, row.vector)) for row in self.records.values()
                      if row.source_id == source_id and row.user_id == user_id
                      and (session_id is None or row.session_id == session_id)]
        return sorted(candidates, key=lambda pair: pair[1], reverse=True)[:top_k]


class ChatHistoryMemoryProvider(ContextProvider):
    """Embedded input/output memory shared by weather, motors and auditor agents.

    Storage gets a per-session ID; retrieval spans all sessions for User_1, as
    specified by the C# sample. Only new external messages are embedded/stored.
    """

    def __init__(self, store: InMemoryVectorStore, embeddings: OpenAIEmbeddingGenerator,
                 user_id="User_1", top_results=3, source_id="RobotCarKnowledge"):
        if top_results <= 0 or embeddings.dimensions != store.dimensions:
            raise ValueError("Positive top_results and matching embedding/store dimensions are required")
        super().__init__(source_id)
        self.store, self.embeddings = store, embeddings
        self.user_id, self.top_results = user_id, top_results

    def _storage_session(self, state):
        return state.setdefault("storage_session_id", str(uuid4()))

    async def before_run(self, *, agent, session: AgentSession, context: SessionContext, state):
        self._storage_session(state)
        query = "\n".join(message.text for message in context.input_messages if message.text.strip())
        if not query:
            return
        vector = (await self.embeddings.embed([query]))[0]
        results = self.store.search(vector, source_id=self.source_id, user_id=self.user_id,
                                    top_k=self.top_results)
        context.metadata[self.source_id] = [
            {"key": row.key, "score": score, "session_id": row.session_id,
             "role": row.message.role, "author_name": row.message.author_name}
            for row, score in results
        ]
        texts = [row.message.text for row, _ in results if row.message.text.strip()]
        if texts:
            context.extend_messages(self, [Message("user", [
                "## Memories\nConsider the following memories when answering user questions:\n" + "\n".join(texts),
            ])])

    async def after_run(self, *, agent, session: AgentSession, context: SessionContext, state):
        messages = [message for message in context.get_messages(
            sources=set(), include_input=True, include_response=True,
        ) if message.text.strip()]
        vectors = await self.embeddings.embed([message.text for message in messages])
        if len(vectors) != len(messages):
            raise ValueError("Embedding batch size does not match message count")
        storage_session = self._storage_session(state)
        for message, vector in zip(messages, vectors):
            self.store.upsert(self.source_id, self.user_id, storage_session, message, vector)


@dataclass(frozen=True)
class SearchItem:
    key: str
    text: str
    source_name: str
    keywords: Sequence[str]


@dataclass(frozen=True)
class TextSearchResult:
    key: str
    text: str
    source_name: str
    source_link: str
    match_count: int


class CustomKeywordSearchAdapter:
    """Exact C# matching: stored keyword CONTAINS query token, with stable ties.

    Splits only space/dot/comma/colon/semicolon, counts repeated query tokens,
    and searches keywords rather than document text. No stemming or embeddings.
    """

    def __init__(self, items: Sequence[SearchItem], top_results=5):
        if top_results <= 0:
            raise ValueError("top_results must be positive")
        self.items, self.top_results = list(items), top_results

    def search(self, query: str) -> list[TextSearchResult]:
        tokens = [token for token in re.split(r"[ .,:;]+", query.lower()) if token]
        results = []
        for item in self.items:
            count = sum(any(token in keyword.lower() for keyword in item.keywords) for token in tokens)
            if count:
                results.append(TextSearchResult(item.key, item.text, item.source_name,
                                                ", ".join(item.keywords), count))
        return sorted(results, key=lambda result: result.match_count, reverse=True)[:self.top_results]


class TextSearchProvider(ContextProvider):
    """Before-run retrieval plus six recent USER query texts (not chat history)."""

    def __init__(self, adapter: CustomKeywordSearchAdapter, recent_message_memory_limit=6,
                 source_id="keyword-search"):
        if recent_message_memory_limit < 0:
            raise ValueError("recent_message_memory_limit cannot be negative")
        super().__init__(source_id)
        self.adapter = adapter
        self.recent_message_memory_limit = recent_message_memory_limit

    async def before_run(self, *, agent, session: AgentSession, context: SessionContext, state):
        current = [message.text for message in context.input_messages if message.text.strip()]
        query = "\n".join([*state.get("recent_messages", []), *current])
        results = self.adapter.search(query)
        context.metadata[self.source_id] = [result.__dict__.copy() for result in results]
        if results:
            parts = ["## Additional Context\nConsider the following information from source documents when responding to the user:"]
            for result in results:
                parts.extend([f"SourceDocName: {result.source_name}", f"SourceDocLink: {result.source_link}",
                              f"Contents: {result.text}", "----"])
            parts.append("Include citations to the source document with document name and link if document name and link is available.")
            context.extend_messages(self, [Message("user", ["\n".join(parts)])])

    async def after_run(self, *, agent, session: AgentSession, context: SessionContext, state):
        limit = self.recent_message_memory_limit
        new_texts = [message.text for message in context.input_messages
                     if message.role == "user" and message.text.strip()]
        if limit:
            state["recent_messages"] = [*state.get("recent_messages", []), *new_texts][-limit:]
        LOGGER.info("TextSearchProvider after_run: retained %d recent user messages; received %d response messages",
                    len(state.get("recent_messages", [])), len(context.response.messages) if context.response else 0)


async def run_prompts(agent, session: AgentSession, prompts: Sequence[str]) -> None:
    for prompt in prompts:
        print(f"USER: {prompt}")
        response = await agent.run(prompt, session=session)
        print(f"ASSISTANT: {response.text}")


def print_history(messages: Sequence[Message]) -> None:
    print("HISTORY:")
    for message in messages:
        source = message.additional_properties.get("_attribution", {}).get("source_id", "conversation")
        print(f"[{source}] {message.role} ({message.author_name or 'unnamed'}): {message.text}")
