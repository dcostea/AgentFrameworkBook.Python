"""Offline chapter 07 regressions; no credentials or live model calls.

Run: .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -p test_memories.py -q
"""

import asyncio
from contextlib import redirect_stdout
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

with patch("dotenv.load_dotenv", return_value=False):
    from agent_framework import Agent, AgentSession, BaseChatClient, ChatResponse, Content, InMemoryHistoryProvider, Message, SessionContext
    from shared import memory_providers as mp

SAMPLES = (
    "AgentWithCustomFileBasedContextProvider",
    "AgentWithFileBasedChatHistoryProvider",
    "AgentWithInMemoryChatHistoryProvider",
    "AgentWithInMemoryChatMessagesAndInMemoryContextProvider",
    "AgentWithInMemoryVectorStoreProvider",
    "AgentWithTextSearchProvider",
    "AgentWithVectorStoreChatHistoryProvider",
)


def load_sample(name):
    spec = importlib.util.spec_from_file_location("offline_memories_" + name, ROOT / "07.Memories" / name / "main.py")
    module = importlib.util.module_from_spec(spec)
    original_path = sys.path[:]
    try:
        with patch("dotenv.load_dotenv", return_value=False):
            spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
    return module


def message(role, text, **kwargs):
    return Message(role, [text], **kwargs)


class OfflineChatClient(BaseChatClient):
    def __init__(self, responses=None):
        super().__init__()
        self.calls = []
        self.responses = list(responses or [])

    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        if stream:
            raise AssertionError("Samples do not use streaming")
        self.calls.append((copy.deepcopy(list(messages)), dict(options)))
        result = self.responses.pop(0) if self.responses else message("assistant", f"answer-{len(self.calls)}")
        return ChatResponse(messages=[result])


class OfflineEmbeddings:
    dimensions = 2

    def __init__(self):
        self.calls = []

    async def embed(self, texts):
        self.calls.append(list(texts))
        return [(1.0, 0.0) if "weather" in text.lower() or "temperature" in text.lower()
                else (0.0, 1.0) for text in texts]


class FileContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_custom_context_seed_instructions_after_hook_and_no_duplication(self):
        sample = load_sample(SAMPLES[0])
        provider = mp.CustomFileBasedContextProvider(sample.DATA / "robot-car-guidelines.txt", sample.DATA / "robot-car-chat-history.json")
        before_bytes = [path.read_bytes() for path in (provider.instructions_path, provider.history_path)]
        client = OfflineChatClient()
        agent = sample.build_agent(client)
        session = agent.create_session()
        output = io.StringIO()
        with redirect_stdout(output):
            await mp.run_prompts(agent, session, (*mp.ROBOT_PROMPTS, "wind speed?"))
        self.assertIn("30, 45, and 60 degrees", client.calls[0][1]["instructions"])
        self.assertIn("June 2, 2025", client.calls[0][1]["instructions"])
        self.assertEqual([(m.role, m.text) for m in client.calls[0][0]][:2],
                         [("user", "What day is today?"), ("assistant", "Today is June 2, 2025")])
        self.assertEqual(sum(m.text == "What day is today?" for m in client.calls[2][0]), 1)
        self.assertEqual(len(session.state["conversation"]["messages"]), 6)
        self.assertNotIn("What day is today?", [m.text for m in session.state["conversation"]["messages"]])
        self.assertEqual([path.read_bytes() for path in (provider.instructions_path, provider.history_path)], before_bytes)
        self.assertEqual(output.getvalue().count("Simulated file update"), 3)
        self.assertIn("June 7, 2025", output.getvalue())
        self.assertNotIn("June 7, 2025", client.calls[2][1]["instructions"])

    async def test_optional_context_missing_then_loaded_and_reloaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guidelines.txt"
            client = OfflineChatClient()
            agent = Agent(client=client, context_providers=[mp.FileBasedContextProvider(path), InMemoryHistoryProvider("history")])
            session = agent.create_session()
            await agent.run("first", session=session)
            self.assertEqual([m.role for m in client.calls[0][0]], ["user"])
            path.write_text("turn only 30°", encoding="utf-8")
            await agent.run("second", session=session)
            self.assertIn("Respond with JSON", client.calls[1][1]["instructions"])
            self.assertEqual([(m.role, m.text) for m in client.calls[1][0]][0], ("system", "turn only 30°"))
            path.write_text("updated 45°", encoding="utf-8")
            await agent.run("third", session=session)
            self.assertEqual(client.calls[2][0][0].text, "updated 45°")
            self.assertEqual(len(session.state["history"]["messages"]), 6)
            self.assertTrue(all(m.role != "system" for m in session.state["history"]["messages"]))

    async def test_csharp_seed_loader_maps_metadata_and_rejects_nontext(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed.json"
            seed = [{"Role": "assistant", "AuthorName": "Robot", "MessageId": "a1", "CreatedAt": "2025-06-02T14:00:00Z",
                     "Contents": [{"$type": "text", "Text": "22°C"}], "AdditionalProperties": {"source": "seed"}}]
            path.write_text(json.dumps(seed), encoding="utf-8-sig")
            loaded = mp.load_messages(path)[0]
            self.assertEqual((loaded.role, loaded.text, loaded.author_name, loaded.message_id), ("assistant", "22°C", "Robot", "a1"))
            self.assertEqual(loaded.additional_properties, {"source": "seed", "created_at": "2025-06-02T14:00:00Z"})
            path.write_text(json.dumps([loaded.to_dict()]), encoding="utf-8")
            self.assertEqual(mp.load_messages(path)[0].to_dict(), loaded.to_dict())
            seed[0]["Contents"][0]["$type"] = "image"
            path.write_text(json.dumps(seed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Only text"):
                mp.load_messages(path)
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "array"):
                mp.load_messages(path)


class InMemoryHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_reducer_preserves_first_system_excludes_tools_and_keeps_three(self):
        provider = mp.MessageCountingHistoryProvider(3)
        state = {"messages": [message("system", "first-system"), message("user", "old"),
                 message("system", "second-system"), message("user", "u1"), message("assistant", "a1"),
                 Message("assistant", [Content.from_function_call(call_id="c", name="move", arguments={})]),
                 Message("tool", [Content.from_function_result(call_id="c", result="moved")]), message("user", "u2")]}
        reduced = await provider.get_messages("session", state=state)
        self.assertEqual([m.text for m in reduced], ["first-system", "u1", "a1", "u2"])
        self.assertIs(state["messages"], reduced)

    async def test_real_agent_reduces_before_retrieval_not_after_store(self):
        client = OfflineChatClient()
        agent = load_sample(SAMPLES[2]).build_agent(client)
        session = agent.create_session()
        for prompt in ("one", "two", "three"):
            await agent.run(prompt, session=session)
        self.assertEqual([m.text for m in client.calls[2][0]], ["answer-1", "two", "answer-2", "three"])
        self.assertEqual(len(session.state["reduced-history"]["messages"]), 5)
        self.assertEqual([m.text for m in session.state["reduced-history"]["messages"]],
                         ["answer-1", "two", "answer-2", "three", "answer-3"])

    async def test_memory_is_per_source_and_session_and_snapshot_resumes(self):
        provider = InMemoryHistoryProvider("first")
        client = OfflineChatClient()
        agent = Agent(client=client, context_providers=[provider])
        original = AgentSession(session_id="original")
        await agent.run("remember", session=original)
        restored = AgentSession.from_dict(json.loads(json.dumps(original.to_dict())))
        await agent.run("resume", session=restored)
        self.assertEqual([m.text for m in client.calls[1][0]], ["remember", "answer-1", "resume"])
        await agent.run("isolated", session=AgentSession(session_id="different"))
        self.assertEqual([m.text for m in client.calls[2][0]], ["isolated"])
        self.assertEqual(await InMemoryHistoryProvider("second").get_messages("original", state=original.state.get("second")), [])
        self.assertEqual(len(original.state["first"]["messages"]), 2)
        self.assertEqual(len(restored.state["first"]["messages"]), 4)


class DurableHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_agent_load_save_cold_resume_without_replaying_loaded_history(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = mp.CustomFileBasedChatHistoryProvider(Path(directory))
            client = OfflineChatClient()
            agent = Agent(client=client, context_providers=[provider])
            session = AgentSession(session_id="resume-me")
            await agent.run("22°C in București", session=session)
            await agent.run("follow-up", session=session)
            cold_provider = mp.CustomFileBasedChatHistoryProvider(Path(directory))
            cold_agent = Agent(client=client, context_providers=[cold_provider])
            await cold_agent.run("recall", session=AgentSession(session_id="resume-me"))
            self.assertEqual([m.text for m in client.calls[2][0]],
                             ["22°C in București", "answer-1", "follow-up", "answer-2", "recall"])
            saved = await cold_provider.get_messages("resume-me")
            self.assertEqual([m.role for m in saved], ["user", "assistant"] * 3)
            self.assertEqual(len(saved), 6)
            self.assertIn("București", provider.path_for("resume-me").read_text(encoding="utf-8"))
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    async def test_scopes_isolated_even_for_pathlike_session_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            first = mp.CustomFileBasedChatHistoryProvider(Path(directory), "one")
            second = mp.CustomFileBasedChatHistoryProvider(Path(directory), "two")
            await first.save_messages("..\\escape", [message("user", "first")])
            await first.save_messages("other", [message("user", "other")])
            await second.save_messages("..\\escape", [message("user", "second")])
            self.assertEqual([m.text for m in await first.get_messages("..\\escape")], ["first"])
            self.assertEqual([m.text for m in await first.get_messages("other")], ["other"])
            self.assertEqual([m.text for m in await second.get_messages("..\\escape")], ["second"])
            self.assertEqual(first.path_for("..\\escape").parent, Path(directory))
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 3)
            with self.assertRaises(ValueError):
                await first.get_messages(None)

    async def test_full_message_roundtrip_and_repeated_text_are_not_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = mp.CustomFileBasedChatHistoryProvider(Path(directory))
            original = Message("assistant", ["move", Content.from_function_call(call_id="x", name="move", arguments={"angle": 30})],
                               message_id="msg-1", author_name="Robot", additional_properties={"metadata": {"angle": 30}})
            await provider.save_messages("s", [original, message("user", "same")])
            await provider.save_messages("s", [message("user", "same")])
            loaded = await provider.get_messages("s")
            self.assertEqual(loaded[0].to_dict(), original.to_dict())
            self.assertEqual([m.text for m in loaded], ["move", "same", "same"])

    async def test_atomic_replace_failure_keeps_old_document_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = mp.CustomFileBasedChatHistoryProvider(Path(directory))
            await provider.save_messages("s", [message("user", "old")])
            path = provider.path_for("s")
            original = path.read_bytes()
            with patch.object(mp.os, "replace", side_effect=OSError("offline write failure")):
                with self.assertRaisesRegex(OSError, "offline write"):
                    await provider.save_messages("s", [message("assistant", "new")])
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    async def test_scope_mismatch_and_invalid_json_fail_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = mp.CustomFileBasedChatHistoryProvider(Path(directory))
            await provider.save_messages("s", [message("user", "old")])
            path = provider.path_for("s")
            document = json.loads(path.read_text(encoding="utf-8"))
            document["session_id"] = "wrong"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scope"):
                await provider.get_messages("s")
            path.write_text("{broken", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                await provider.save_messages("s", [message("assistant", "new")])
            self.assertEqual(path.read_text(encoding="utf-8"), "{broken")


class CollectionHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_timestamp_earliest_top_five_full_messages_and_isolation(self):
        store = mp.InMemoryChatHistoryStore()
        provider = mp.VectorStoreChatHistoryProvider(store, top_results=5)
        rows = [message("user" if i % 2 == 0 else "assistant", f"m{i}", message_id=f"id{i}",
                        author_name="Robot" if i % 2 else "User", additional_properties={"index": i}) for i in range(8)]
        await provider.save_messages("s", rows)
        await provider.save_messages("other", [message("user", "private")])
        self.assertEqual([m.text for m in await provider.get_messages("s")], ["m0", "m1", "m2", "m3", "m4"])
        all_messages = await provider.all_messages("s")
        self.assertEqual([m.to_dict() for m in all_messages], [m.to_dict() for m in rows])
        self.assertTrue(all(row.timestamp and row.message_text and row.serialized_message for row in store.records.values()))
        self.assertEqual(await mp.VectorStoreChatHistoryProvider(store, source_id="different").get_messages("s"), [])

    async def test_same_store_resume_and_new_store_loss_are_explicit(self):
        store = mp.InMemoryChatHistoryStore()
        provider = mp.VectorStoreChatHistoryProvider(store)
        await provider.save_messages("resume", [message("user", "first", message_id="stable"), message("assistant", "second")])
        resumed = mp.VectorStoreChatHistoryProvider(store)
        self.assertEqual([m.text for m in await resumed.get_messages("resume")], ["first", "second"])
        self.assertEqual(await mp.VectorStoreChatHistoryProvider(mp.InMemoryChatHistoryStore()).get_messages("resume"), [])
        await provider.save_messages("resume", [message("user", "updated", message_id="stable")])
        self.assertEqual([m.text for m in await provider.all_messages("resume")], ["updated", "second"])
        self.assertEqual(len(store.records), 2)

    async def test_real_agent_stores_user_and_assistant_without_context_reappend(self):
        client = OfflineChatClient()
        agent = load_sample(SAMPLES[6]).build_agent(client)
        session = agent.create_session()
        await agent.run("one", session=session)
        await agent.run("two", session=session)
        self.assertEqual([m.text for m in client.calls[1][0]], ["one", "answer-1", "two"])
        history = await agent.context_providers[0].all_messages(session.session_id)
        self.assertEqual([(m.role, m.text) for m in history],
                         [("user", "one"), ("assistant", "answer-1"), ("user", "two"), ("assistant", "answer-2")])


class VectorTests(unittest.TestCase):
    def test_cosine_scoring_ranking_top_k_is_not_keyword_search(self):
        self.assertAlmostEqual(mp.cosine_similarity((1, 0), (3, 4)), 0.6)
        self.assertEqual(mp.cosine_similarity((1, 0), (0, 0)), 0)
        self.assertEqual(mp.cosine_similarity((1, 0), (-1, 0)), -1)
        store = mp.InMemoryVectorStore(2)
        for text, vector in (("hot afternoon", (1, 0)), ("temperature keyword only", (-1, 0)), ("unrelated", (0, 1))):
            store.upsert("source", "user", "s", message("user", text), vector)
        results = store.search((1, 0), source_id="source", user_id="user", top_k=2)
        self.assertEqual([row.message.text for row, _ in results], ["hot afternoon", "unrelated"])
        self.assertEqual([score for _, score in results], [1.0, 0.0])
        adapter = mp.CustomKeywordSearchAdapter([mp.SearchItem("key", "temperature keyword only", "doc", ("temperature",))])
        self.assertEqual(adapter.search("temperature")[0].text, "temperature keyword only")

    def test_vector_scope_and_session_filter_and_dimension_validation(self):
        store = mp.InMemoryVectorStore(2)
        for source, user, session, text in (("s", "u", "a", "a"), ("s", "u", "b", "b"), ("s", "private", "a", "private"), ("other", "u", "a", "other")):
            store.upsert(source, user, session, message("assistant", text), (1, 0))
        self.assertEqual([r.message.text for r, _ in store.search((1, 0), source_id="s", user_id="u")], ["a", "b"])
        self.assertEqual([r.message.text for r, _ in store.search((1, 0), source_id="s", user_id="u", session_id="b")], ["b"])
        for values in ((1,), (float("nan"), 0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                store.search(values, source_id="s", user_id="u")
        with self.assertRaises(ValueError):
            store.search((1, 0), source_id="s", user_id="u", top_k=0)
        with self.assertRaises(ValueError):
            mp.cosine_similarity((1,), (1, 2))


class EmbeddingTests(unittest.IsolatedAsyncioTestCase):
    async def test_sdk_batch_model_options_dimensions_and_index_order(self):
        sdk = SimpleNamespace(embeddings=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(data=[
            SimpleNamespace(index=1, embedding=[0, 1]), SimpleNamespace(index=0, embedding=[1, 0]),
        ]))))
        generator = mp.OpenAIEmbeddingGenerator(sdk, "configured-embedding", dimensions=2)
        vectors = await generator.embed(["first", "second"])
        self.assertEqual(vectors, [(1.0, 0.0), (0.0, 1.0)])
        sdk.embeddings.create.assert_awaited_once_with(input=["first", "second"], model="configured-embedding", dimensions=2, encoding_format="float")
        self.assertEqual(await generator.embed([]), [])
        self.assertEqual(sdk.embeddings.create.await_count, 1)

    async def test_bad_embedding_response_rejected(self):
        sdk = SimpleNamespace(embeddings=SimpleNamespace(create=AsyncMock()))
        generator = mp.OpenAIEmbeddingGenerator(sdk, "embedding", 2)
        for rows in ([SimpleNamespace(index=1, embedding=[1, 0])], [SimpleNamespace(index=0, embedding=[1])],
                     [SimpleNamespace(index=0, embedding=[float("inf"), 0])]):
            with self.subTest(rows=rows):
                sdk.embeddings.create.return_value = SimpleNamespace(data=rows)
                with self.assertRaises(ValueError):
                    await generator.embed(["one"])

    async def test_embedding_env_model_requirements_without_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            for provider, name in (("openai", "OPENAI_EMBEDDING_MODEL_ID"), ("azure", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")):
                with self.assertRaisesRegex(RuntimeError, name):
                    mp.embedding_model_for(provider)
                with patch.dict(os.environ, {name: " configured "}):
                    self.assertEqual(mp.embedding_model_for(provider), "configured")
            with self.assertRaises(ValueError):
                mp.embedding_model_for("unknown")

    async def test_semantic_memory_real_agent_before_after_embeddings_and_full_metadata(self):
        embeddings = OfflineEmbeddings()
        store = mp.InMemoryVectorStore(2)
        provider = mp.ChatHistoryMemoryProvider(store, embeddings, top_results=1)
        client = OfflineChatClient()
        agent = Agent(client=client, context_providers=[provider])
        session = agent.create_session()
        report = message("user", "weather: 22°C", message_id="report", author_name="System", additional_properties={"kind": "report"})
        await agent.run(report, session=session)
        await agent.run("temperature?", session=session)
        self.assertEqual(embeddings.calls[0], ["weather: 22°C"])
        self.assertEqual(embeddings.calls[1], ["weather: 22°C", "answer-1"])
        self.assertEqual(embeddings.calls[2], ["temperature?"])
        self.assertEqual(embeddings.calls[3], ["temperature?", "answer-2"])
        self.assertEqual(len(store.records), 4)
        self.assertIn("## Memories", client.calls[1][0][0].text)
        self.assertIn("weather: 22°C", client.calls[1][0][0].text)
        self.assertEqual(client.calls[1][0][-1].text, "temperature?")
        stored_report = next(row.message for row in store.records.values() if row.key == "report")
        self.assertEqual(stored_report.to_dict(), report.to_dict())
        self.assertEqual({row.message.role for row in store.records.values()}, {"user", "assistant"})
        self.assertEqual({row.session_id for row in store.records.values()}, {session.state[provider.source_id]["storage_session_id"]})

    async def test_semantic_memory_search_cross_session_same_user_not_other_source_user(self):
        store = mp.InMemoryVectorStore(2)
        embeddings = OfflineEmbeddings()
        provider = mp.ChatHistoryMemoryProvider(store, embeddings)
        store.upsert(provider.source_id, "User_1", "old-session", message("user", "weather remembered"), (1, 0))
        store.upsert(provider.source_id, "User_2", "old-session", message("user", "private weather"), (1, 0))
        store.upsert("other-source", "User_1", "old-session", message("user", "wrong source"), (1, 0))
        session = AgentSession()
        context = SessionContext(session_id=session.session_id, input_messages=[message("user", "temperature?")])
        await provider.before_run(agent=None, session=session, context=context, state={})
        self.assertEqual(context.context_messages[provider.source_id][0].text,
                         "## Memories\nConsider the following memories when answering user questions:\nweather remembered")
        self.assertEqual(context.metadata[provider.source_id][0]["score"], 1)
        self.assertEqual(context.metadata[provider.source_id][0]["session_id"], "old-session")

    async def test_weather_motors_auditor_share_memory_and_inmemory_history(self):
        sample = load_sample(SAMPLES[4])
        embeddings = OfflineEmbeddings()
        client = OfflineChatClient()
        weather, motors, auditor = sample.build_agents(client, embeddings)
        self.assertIs(weather.context_providers[0], motors.context_providers[0])
        self.assertIs(motors.context_providers[0], auditor.context_providers[0])
        session = weather.create_session()
        await weather.run("weather report", session=session)
        await motors.run("temperature?", session=session)
        await auditor.run("audit", session=session)
        self.assertEqual(len(session.state["conversation"]["messages"]), 6)
        self.assertEqual(len(weather.context_providers[0].store.records), 6)
        self.assertEqual([m.author_name for m in session.state["conversation"]["messages"] if m.role == "assistant"],
                         ["WeatherAgent", "MotorsAgent", "RobotCarAgent"])
        report = (sample.DATA / "weather-report.txt").read_text(encoding="utf-8")
        self.assertIn("June 2, 2025 - Morning: 15°C", report)
        self.assertIn("Afternoon: 22°C", report)


class KeywordTests(unittest.IsolatedAsyncioTestCase):
    def test_exact_matching_direction_delimiters_repeated_tokens_ranking_and_metadata(self):
        sample = load_sample(SAMPLES[5])
        adapter = mp.CustomKeywordSearchAdapter(sample.KNOWLEDGE)
        results = adapter.search("move move turn")
        self.assertEqual([row.key for row in results], ["2", "1", "3", "4"])
        self.assertEqual([row.match_count for row in results], [3, 2, 2, 2])
        self.assertEqual(adapter.search("mo")[0].key, "1")
        self.assertEqual(adapter.search("moves backward"), [])
        self.assertEqual(adapter.search("weather temperature wind"), [])
        self.assertEqual(adapter.search("turn:angle;emergency")[0].key, "2")
        self.assertEqual(adapter.search("turn\nangle"), [])
        self.assertEqual(adapter.search("turn\nno match"), [])
        self.assertEqual(adapter.search("turn.\nno match")[0].key, "2")
        self.assertEqual(results[0].source_name, "RobotCar Turning Policy")
        self.assertEqual(results[0].source_link, "command, move, turn, angle")
        self.assertEqual(len(mp.CustomKeywordSearchAdapter(sample.KNOWLEDGE, 2).search("move")), 2)

    async def test_before_injects_source_metadata_and_after_logs_recent_users_only(self):
        adapter = mp.CustomKeywordSearchAdapter([mp.SearchItem("1", "turn at 30°", "Policy", ("turn",))])
        provider = mp.TextSearchProvider(adapter, recent_message_memory_limit=6)
        client = OfflineChatClient()
        agent = Agent(client=client, context_providers=[provider])
        session = agent.create_session()
        with self.assertLogs(mp.LOGGER, level="INFO") as captured:
            await agent.run("turn.", session=session)
        wire = client.calls[0][0]
        self.assertEqual(wire[0].role, "user")
        self.assertIn("SourceDocName: Policy", wire[0].text)
        self.assertIn("SourceDocLink: turn", wire[0].text)
        self.assertIn("Contents: turn at 30°", wire[0].text)
        self.assertEqual(session.state[provider.source_id]["recent_messages"], ["turn."])
        self.assertIn("received 1 response messages", captured.output[0])
        context = SessionContext(session_id=session.session_id, input_messages=[message("user", "no match")])
        await provider.before_run(agent=agent, session=session, context=context, state=session.state[provider.source_id])
        self.assertEqual(context.metadata[provider.source_id][0]["key"], "1")
        for i in range(8):
            await agent.run(f"query-{i}", session=session)
        self.assertEqual(session.state[provider.source_id]["recent_messages"], [f"query-{i}" for i in range(2, 8)])
        self.assertTrue(all(not text.startswith("answer") for text in session.state[provider.source_id]["recent_messages"]))
        self.assertNotIn("SourceDoc", " ".join(session.state[provider.source_id]["recent_messages"]))
        other = AgentSession()
        context = SessionContext(session_id=other.session_id, input_messages=[message("user", "no match")])
        await provider.before_run(agent=agent, session=other, context=context, state={})
        self.assertEqual(context.context_messages, {})
        self.assertEqual(context.metadata[provider.source_id], [])

    async def test_sample_rag_does_not_duplicate_retrieved_context_into_chat_history(self):
        client = OfflineChatClient()
        agent = load_sample(SAMPLES[5]).build_agent(client)
        session = agent.create_session()
        await agent.run("move.", session=session)
        await agent.run("move.", session=session)
        stored = session.state["conversation"]["messages"]
        self.assertEqual([m.text for m in stored], ["move.", "answer-1", "move.", "answer-2"])
        self.assertEqual(sum("SourceDoc" in m.text for m in client.calls[1][0]), 1)


class SampleTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_seven_import_without_sdk_construction_and_keep_exact_names(self):
        with patch("shared.chat_clients.create_sdk_client", side_effect=AssertionError("No SDK construction on import")), \
             patch("shared.chat_clients.create_completion_client", side_effect=AssertionError("No model selection on import")):
            for name in SAMPLES:
                with self.subTest(sample=name):
                    sample = load_sample(name)
                    self.assertTrue(asyncio.iscoroutinefunction(sample.main))
                    self.assertTrue(callable(getattr(sample, "build_agent", getattr(sample, "build_agents", None))))
                    self.assertEqual(Path(sample.__file__).parent.name, name)

    async def test_all_entrypoints_use_shared_client_lifecycle_for_both_providers(self):
        for name in SAMPLES:
            for provider in ("openai", "azure"):
                with self.subTest(sample=name, provider=provider), tempfile.TemporaryDirectory() as directory:
                    sample = load_sample(name)
                    client = OfflineChatClient()
                    sdk = Mock()
                    manager = SimpleNamespace(__aenter__=AsyncMock(return_value=sdk), __aexit__=AsyncMock(return_value=False))
                    class SDKContext:
                        async def __aenter__(self):
                            return await manager.__aenter__()
                        async def __aexit__(self, *args):
                            return await manager.__aexit__(*args)
                    patches = [patch.object(sample, "create_sdk_client", return_value=SDKContext()),
                               patch.object(sample, "create_completion_client", return_value=client)]
                    with patches[0] as sdk_factory, patches[1] as completion_factory, redirect_stdout(io.StringIO()):
                        if name == SAMPLES[1]:
                            original_build = sample.build_agent
                            with patch.object(sample, "build_agent", side_effect=lambda c: original_build(c, Path(directory))):
                                await sample.main(provider)
                        elif name == SAMPLES[4]:
                            with patch.object(sample, "embedding_model_for", return_value="embedding"), \
                                 patch.object(sample, "OpenAIEmbeddingGenerator", return_value=OfflineEmbeddings()) as embedding_factory:
                                await sample.main(provider)
                            embedding_factory.assert_called_once_with(sdk, "embedding", dimensions=1536)
                        else:
                            await sample.main(provider)
                    sdk_factory.assert_called_once_with(provider)
                    completion_factory.assert_called_once_with(sdk, provider)
                    manager.__aexit__.assert_awaited_once()
                    self.assertGreaterEqual(len(client.calls), 2)

    async def test_file_entrypoint_resumes_followup_only_and_closes_sdk_on_failure(self):
        sample = load_sample(SAMPLES[1])
        client = OfflineChatClient()
        sdk = AsyncMock()
        sdk.__aenter__.return_value = sdk
        with tempfile.TemporaryDirectory() as directory:
            original = sample.build_agent
            with patch.object(sample, "create_sdk_client", return_value=sdk), \
                 patch.object(sample, "create_completion_client", return_value=client), \
                 patch.object(sample, "build_agent", side_effect=lambda c: original(c, Path(directory))), redirect_stdout(io.StringIO()):
                await sample.main(session_id="persisted")
                await sample.main(session_id="persisted")
            self.assertEqual(len(client.calls), 3)
            self.assertEqual(client.calls[2][0][-1].text, mp.ROBOT_PROMPTS[1])
            self.assertEqual(len(await mp.CustomFileBasedChatHistoryProvider(Path(directory)).get_messages("persisted")), 6)
        failing_sdk = AsyncMock()
        failing_sdk.__aenter__.return_value = failing_sdk
        with patch.object(sample, "create_sdk_client", return_value=failing_sdk), \
             patch.object(sample, "create_completion_client", side_effect=RuntimeError("offline failure")):
            with self.assertRaisesRegex(RuntimeError, "offline failure"):
                await sample.main()
        failing_sdk.__aexit__.assert_awaited_once()

    async def test_no_generated_csharp_history_dump_copied_and_assets_exist(self):
        chapter = ROOT / "07.Memories"
        self.assertEqual(list(chapter.rglob("*.cs")), [])
        self.assertEqual(list(chapter.rglob("*.csproj")), [])
        self.assertEqual(list((chapter / SAMPLES[1]).glob("robot-car-chat-history_*.json")), [])
        self.assertTrue((chapter / SAMPLES[0] / "Data" / "robot-car-chat-history.json").is_file())
        self.assertTrue((chapter / SAMPLES[0] / "Data" / "robot-car-guidelines.txt").is_file())
        self.assertTrue((chapter / SAMPLES[3] / "Data" / "robot-car-guidelines.txt").is_file())
        self.assertTrue((chapter / SAMPLES[4] / "Data" / "weather-report.txt").is_file())


if __name__ == "__main__":
    unittest.main()
