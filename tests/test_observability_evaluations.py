"""Offline chapter 11 tests: real AF tools/middleware and in-memory OTel providers."""

import asyncio
import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from openai import AsyncOpenAI

with patch.dict(os.environ, {}, clear=True), patch("dotenv.load_dotenv", return_value=False):
    from shared import evaluation_examples as evaluations
    from shared import observability_examples as observations
    from shared import chat_clients, tool_examples
from agent_framework import AgentResponse, Content, Message, FunctionInvocationContext, FunctionMiddleware, tool
from agent_framework import observability as native_obs
from opentelemetry import baggage, trace
import opentelemetry.metrics._internal as metric_globals
import opentelemetry._logs._internal as log_globals

DATA = evaluations.DATA_DIRECTORY


def raw_record(dataset="coherence"):
    return json.loads((DATA / f"{dataset}.jsonl").read_text(encoding="utf-8-sig").splitlines()[0])


def judge_response(value=4, reason="Evidence from the response"):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
        message=SimpleNamespace(content=json.dumps({"value": value, "reason": reason}), refusal=None))])


def flatten_metrics(runtime):
    data = runtime.metric_reader.get_metrics_data()
    return {metric.name: metric for resource in data.resource_metrics
            for scope in resource.scope_metrics for metric in scope.metrics} if data else {}


class DatasetTests(unittest.TestCase):
    def write_dataset(self, text):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "sample.jsonl"
        path.write_text(text, encoding="utf-8")
        return path

    def test_all_twelve_datasets_have_original_ids_schemas_and_counts(self):
        expected = {name: 2 for name in evaluations.DATASETS}
        expected.update({"intent-resolution": 3, "direction-change": 3, "task-adherence": 4, "tool-call-accuracy": 4})
        self.assertEqual(len(expected), 12)
        self.assertEqual(sum(expected.values()), 30)
        ids = set()
        for name, count in expected.items():
            with self.subTest(dataset=name):
                records = evaluations.load_records(DATA / f"{name}.jsonl")
                self.assertEqual(len(records), count)
                self.assertTrue(all(r.id not in ids for r in records))
                ids.update(r.id for r in records)
                self.assertEqual(records[0].agent.model_name, "gpt-5.2")
                self.assertIn("Forward", records[0].agent.tools)
                self.assertEqual(records[0].tool_calls, tuple(records[0].tool_calls))

    def test_loader_preserves_every_fixture_field(self):
        for name in evaluations.DATASETS:
            originals = [json.loads(line) for line in (DATA / f"{name}.jsonl").read_text(encoding="utf-8-sig").splitlines()]
            for original, record in zip(originals, evaluations.load_records(DATA / f"{name}.jsonl")):
                with self.subTest(id=record.id):
                    self.assertEqual(record.user_input, original["userInput"])
                    self.assertEqual(record.final_response, original["finalResponse"])
                    self.assertEqual(record.expected_behavior, original["expectedBehavior"])
                    self.assertEqual(record.agent.instructions, original["agent"]["instructions"])
                    self.assertEqual([(c.name, c.arguments, c.result) for c in record.tool_calls],
                                     [(c["name"], c["arguments"], c["result"]) for c in original["toolCalls"]])
                    if isinstance(record, evaluations.EvaluationRecord):
                        self.assertEqual(record.ground_truth, original.get("groundTruth"))
                        self.assertEqual(list(record.retrieved_context_chunks) if record.retrieved_context_chunks is not None else None,
                                         original.get("retrievedContextChunks"))
                        self.assertEqual(record.should_pass, original["shouldPass"])

    def test_bom_and_case_insensitive_schema(self):
        raw = raw_record()
        raw = {key.upper(): value for key, value in raw.items()}
        path = self.write_dataset("\ufeff" + json.dumps(raw) + "\n")
        self.assertEqual(evaluations.load_records(path)[0].id, "robby-coherence-001")

    def test_malformed_null_blank_and_empty_report_context(self):
        for text in ("{bad json}", "null", "[]", "\n", "", "NaN", "Infinity"):
            with self.subTest(text=text):
                with self.assertRaisesRegex(evaluations.DatasetError, "sample.jsonl"):
                    evaluations.load_records(self.write_dataset(text))

    def test_duplicate_json_fields_case_fields_and_ids_rejected(self):
        raw = raw_record()
        for text in ('{"id":"a","id":"b"}', json.dumps({**raw, "ID": "extra"}),
                     json.dumps(raw) + "\n" + json.dumps(raw)):
            with self.subTest(text=text[:50]):
                with self.assertRaisesRegex(evaluations.DatasetError, "[Dd]uplicate"):
                    evaluations.load_records(self.write_dataset(text))

    def test_schema_type_errors_and_missing_context(self):
        for field, value in (("shouldPass", "true"), ("agent", None), ("toolCalls", {}),
                             ("retrievedContextChunks", [1]), ("groundTruth", 3), ("id", "")):
            with self.subTest(field=field):
                raw = raw_record()
                raw[field] = value
                with self.assertRaises(evaluations.DatasetError):
                    evaluations.parse_record(raw)
        record = evaluations.parse_record(raw_record())
        with self.assertRaisesRegex(evaluations.DatasetError, "requires ground_truth"):
            evaluations.judge_payload(record, "groundedness")

    def test_direction_loader_skips_blank_lines_with_physical_line_errors(self):
        raw = raw_record("direction-change")
        path = self.write_dataset("\n" + json.dumps(raw) + "\n\n")
        self.assertEqual(len(evaluations.load_records(path, direction=True)), 1)
        path.write_text("\nnull", encoding="utf-8")
        with self.assertRaisesRegex(evaluations.DatasetError, "line 2"):
            evaluations.load_records(path, direction=True)


class DirectionTests(unittest.TestCase):
    def test_all_direction_fixtures_match_flags_from_to_value_and_failed(self):
        for record in evaluations.load_records(DATA / "direction-change.jsonl"):
            with self.subTest(id=record.id):
                names = [call.name for call in record.tool_calls]
                analysis = evaluations.analyze_direction_change_safety(names)
                self.assertEqual((analysis.changed, analysis.from_direction, analysis.to_direction),
                                 (record.expected_changed, record.expected_from, record.expected_to))
                metric = evaluations.DirectionChangeEvaluator().evaluate(names).get("DirectionChange")
                self.assertEqual(metric.value, not record.expected_changed)
                self.assertEqual(metric.failed, record.expected_changed)
                self.assertEqual(metric.rating, evaluations.Rating.UNACCEPTABLE if record.expected_changed else evaluations.Rating.GOOD)

    def test_stop_only_resets_safety_evaluator_and_turns_do_not(self):
        sequences = [(["Forward", "Stop", "Backward"], True),
                     (["forward", "turnleft", "backward"], False),
                     (["Backward", "TurnRight", "Forward"], False),
                     (["Stop"], True), (["Forward", "Forward"], True)]
        for names, safe in sequences:
            with self.subTest(names=names):
                self.assertEqual(evaluations.analyze_direction_change_safety(names).is_safe, safe)

    def test_direction_evaluator_extracts_real_response_calls_across_messages(self):
        response = AgentResponse(messages=[
            Message("assistant", [Content.from_text("Ignored response text"),
                Content.from_function_call("call1", "forward", arguments={"distance": 2})]),
            Message("tool", [Content.from_function_result("call1", result="moved")]),
            Message("assistant", [Content.from_function_call("call2", "turn_left", arguments={"angle": 90}),
                Content.from_function_call("call3", "backward", arguments={"distance": 2})])])
        metric = evaluations.DirectionChangeEvaluator().evaluate_response(response).get("DirectionChange")
        self.assertFalse(metric.value)
        self.assertIn("Forward followed by Backward", metric.reason)

    def test_case_normalization_corrects_source_switch_without_altering_output(self):
        analysis = evaluations.analyze_direction_change_safety(["fOrWaRd", "BACKWARD"])
        self.assertFalse(analysis.is_safe)
        self.assertEqual((analysis.from_direction, analysis.to_direction), ("fOrWaRd", "BACKWARD"))
        self.assertIn("without Stop", analysis.reason)

    def test_no_calls_unknown_calls_and_types(self):
        self.assertEqual(evaluations.analyze_direction_change_safety([]).reason,
                         "No tool calls were found in Robby's response.")
        analysis = evaluations.analyze_direction_change_safety(["Jump", "Fly", "jump", "Forward"])
        self.assertFalse(analysis.changed)
        self.assertEqual(analysis.reason, "Unknown move calls found: Jump, Fly.")
        with self.assertRaises(TypeError):
            evaluations.analyze_direction_change_safety([None])


class JudgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.create = AsyncMock(return_value=judge_response())
        self.sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))
        self.judge = evaluations.ModelJudge(self.sdk, "offline-model")
        for target in ("socket.create_connection", "socket.socket.connect", "socket.socket.connect_ex"):
            guard = patch(target, side_effect=AssertionError("Network is forbidden in tests"))
            guard.start()
            self.addCleanup(guard.stop)

    async def test_numeric_rating_boundaries_preserve_fluency_average_and_others_good(self):
        for dataset in ("fluency", "coherence"):
            record = evaluations.load_records(DATA / f"{dataset}.jsonl")[0]
            for value in range(1, 6):
                with self.subTest(dataset=dataset, value=value):
                    self.create.return_value = judge_response(value)
                    result = await self.judge.evaluate(record, dataset)
                    metric = result.metrics[0]
                    self.assertEqual(metric.value, value)
                    self.assertEqual(metric.passed, value >= (3 if dataset == "fluency" else 4))
                    self.assertIn("NOT Microsoft", metric.provenance)

    async def test_boolean_tool_call_metric_has_source_boolean_not_numeric_output(self):
        record = evaluations.load_records(DATA / "tool-call-accuracy.jsonl")[0]
        for value in (True, False):
            with self.subTest(value=value):
                self.create.return_value = judge_response(value)
                metric = (await self.judge.evaluate(record, "tool-call-accuracy")).metrics[0]
                self.assertIs(metric.value, value)
                self.assertIs(metric.passed, value)
                self.assertEqual(metric.rating, evaluations.Rating.GOOD if value else evaluations.Rating.UNACCEPTABLE)

    async def test_schema_validation_retries_typed_errors_and_exhaustion(self):
        record = evaluations.load_records(DATA / "coherence.jsonl")[0]
        for invalid in (True, 0, 6, 3.5, "4", None):
            with self.subTest(invalid=invalid):
                self.create.side_effect = [judge_response(invalid), judge_response(4)]
                result = await self.judge.evaluate(record, "coherence")
                self.assertTrue(result.passed)
        self.create.side_effect = [judge_response(4, ""), judge_response(4, "")]
        with self.assertRaisesRegex(evaluations.JudgeResponseError, "after 2"):
            await self.judge.evaluate(record, "coherence")

    async def test_refused_truncated_extra_fields_and_invalid_json_fail_explicitly(self):
        record = evaluations.load_records(DATA / "coherence.jsonl")[0]
        variants = []
        response = judge_response(); response.choices[0].finish_reason = "length"; variants.append(response)
        response = judge_response(); response.choices[0].message.refusal = "refused"; variants.append(response)
        response = judge_response(); response.choices[0].message.content = '{"value":4,"reason":"ok","extra":1}'; variants.append(response)
        response = judge_response(); response.choices[0].message.content = "not-json"; variants.append(response)
        for response in variants:
            with self.subTest(response=response):
                self.create.side_effect = None
                self.create.return_value = response
                with self.assertRaises(evaluations.JudgeResponseError):
                    await self.judge.evaluate(record, "coherence")

    async def test_expectations_and_metadata_never_leak_to_judge(self):
        for dataset in evaluations.CRITERIA:
            record = evaluations.load_records(DATA / f"{dataset}.jsonl")[0]
            with self.subTest(dataset=dataset):
                self.create.return_value = judge_response(True if dataset == "tool-call-accuracy" else 4)
                await self.judge.evaluate(record, dataset)
                request = self.create.call_args.kwargs
                payload = json.loads(request["messages"][1]["content"])
                for key in ("id", "source", "scenario", "expectedBehavior", "shouldPass", "modelName"):
                    self.assertNotIn(key, payload)
                self.assertNotIn(record.expected_behavior, json.dumps(payload))
                self.assertTrue(request["response_format"]["json_schema"]["strict"])
                if dataset == "retrieval":
                    self.assertNotIn("response", payload)
                    self.assertEqual(payload["context"], list(record.retrieved_context_chunks))
                if dataset in {"task-adherence", "tool-call-accuracy"}:
                    self.assertNotIn("response", payload)
                    self.assertTrue(all("result" not in call for call in payload["function_calls"]))
                    self.assertEqual({t["name"] for t in payload["tool_definitions"]}, set(evaluations.PYTHON_TOOL_NAMES))

    async def test_composite_preserves_individual_metrics_and_source_assertions(self):
        record = evaluations.load_records(DATA / "composite.jsonl")[0]
        self.create.side_effect = [judge_response(4), judge_response(2)]
        result = await evaluations.evaluate_record(record, "composite", self.judge)
        self.assertEqual([m.name for m in result.metrics], ["Coherence", "Equivalence"])
        self.assertFalse(result.passed)
        self.assertFalse(result.matches_expected(False))
        self.assertFalse(result.matches_expected(True))
        with self.assertRaises(ValueError):
            evaluations.EvaluationResult((result.metrics[0], result.metrics[0]))

    async def test_provider_errors_are_not_silently_replaced_with_approximate_scores(self):
        record = evaluations.load_records(DATA / "coherence.jsonl")[0]
        self.create.side_effect = RuntimeError("provider failed")
        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            await self.judge.evaluate(record, "coherence")
        self.assertEqual(self.create.await_count, 1)
        with self.assertRaisesRegex(RuntimeError, "opt-in"):
            await evaluations.evaluate_record(record, "coherence")

    async def test_runner_closes_sdk_even_when_judge_fails(self):
        sdk = SimpleNamespace(close=AsyncMock())
        with patch("shared.chat_clients.create_sdk_client", return_value=sdk), \
             patch("shared.chat_clients.model_for", return_value="offline-model"), \
             patch.object(evaluations.ModelJudge, "evaluate", side_effect=RuntimeError("judge failed")), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "judge failed"):
                await evaluations.evaluation_main(["--dataset", "coherence", "--model-backed"])
        sdk.close.assert_awaited_once()


class TelemetryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        env = patch.dict(os.environ, {"OPENAI_CHAT_MODEL_ID": "offline-model"}, clear=True)
        env.start(); self.addCleanup(env.stop)
        for target in ("socket.create_connection", "socket.socket.connect", "socket.socket.connect_ex"):
            guard = patch(target, side_effect=AssertionError("Network is forbidden in tests"))
            guard.start(); self.addCleanup(guard.stop)
        self.output = io.StringIO()
        capture = contextlib.redirect_stdout(self.output)
        capture.__enter__(); self.addCleanup(capture.__exit__, None, None, None)
        self.runtime = observations.TelemetryRuntime()
        self.addCleanup(self.runtime.close)
        self.tracker = observations.DirectionChangeTracker(self.runtime)

    async def invoke(self, name, *, error=None, during=None):
        async def function():
            return name + " result"
        wrapped = tool(function, name=name)
        invocation = FunctionInvocationContext(wrapped, {})
        async def next_call():
            if during:
                during()
            if error:
                raise error
            invocation.result = await wrapped.invoke(arguments={}, skip_parsing=True)
        await self.tracker.process(invocation, next_call)
        return invocation.result

    async def test_actual_function_middleware_direction_counter_span_tags_events(self):
        self.assertIsInstance(self.tracker, FunctionMiddleware)
        for name in ("backward", "forward", "forward"):
            self.assertEqual(await self.invoke(name), name + " result")
        exported = self.runtime.span_exporter.get_finished_spans()
        detected = [s for s in exported if s.name == "Detect Robot Direction Change"]
        self.assertEqual(len(detected), 3)
        changed = detected[1]
        self.assertEqual(changed.attributes["robot.direction.previous"], "Backward")
        self.assertEqual(changed.attributes["robot.direction.current"], "Forward")
        self.assertTrue(changed.attributes["robot.direction.changed"])
        self.assertEqual([e.name for e in changed.events], ["Robot direction changed"])
        self.assertTrue(all(s.parent is not None for s in detected))
        self.assertEqual(len({s.context.trace_id for s in exported}), 3)
        metric = flatten_metrics(self.runtime)["robot_direction_change_total"]
        self.assertEqual(metric.data.data_points[0].value, 1)
        self.assertEqual(metric.data.data_points[0].attributes, {"from": "Backward", "to": "Forward"})
        self.assertEqual(len(self.runtime.log_exporter.get_finished_logs()), 3)

    async def test_tracker_resets_on_turn_unlike_safety_evaluator(self):
        for reset in ("stop", "turn_left", "turn_right", "unknown"):
            with self.subTest(reset=reset):
                self.tracker.previous = None
                before = self.tracker.direction_changes
                for name in ("forward", reset, "backward"):
                    await self.invoke(name)
                self.assertEqual(self.tracker.direction_changes, before)
                self.assertEqual(self.tracker.previous, "Backward")

    async def test_gauges_histograms_success_error_and_cancellation(self):
        self.tracker.clock = iter([1.0, 1.25, 2.0, 2.5, 3.0, 3.75]).__next__
        await self.invoke("forward", during=lambda: self.assertEqual(
            flatten_metrics(self.runtime)["robot_active_tool_calls"].data.data_points[0].value, 1))
        for error in (RuntimeError("failed"), asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(type(error)):
                    await self.invoke("backward", error=error)
                self.assertEqual(self.tracker.previous, "Forward")
        data = flatten_metrics(self.runtime)
        self.assertEqual(data["robot_active_tool_calls"].data.data_points[0].value, 0)
        self.assertEqual(data["robot_linear_direction"].data.data_points[0].value, 1)
        self.assertEqual(sum(p.count for p in data["robot_tool_duration"].data.data_points), 3)
        self.assertAlmostEqual(sum(p.sum for p in data["robot_tool_duration"].data.data_points), 1.5)
        self.assertEqual(sum(p.value for p in data["robot_tool_errors_total"].data.data_points), 2)
        self.assertEqual(sum(p.value for p in data["robot_tool_calls_total"].data.data_points), 1)
        self.assertFalse(self.tracker.direction_changes)

    async def test_default_simulation_invokes_real_reusable_tools(self):
        with patch("shared.motor_tools._DELAY_SECONDS", 0):
            count = await observations.run_simulation(self.tracker)
        self.assertEqual(count, 1)
        data = flatten_metrics(self.runtime)
        self.assertEqual(sum(p.value for p in data["robot_tool_calls_total"].data.data_points), 6)
        self.assertEqual(self.tracker.previous, "Backward")
        self.assertIn("MOTORS", self.output.getvalue())

    async def test_identity_baggage_context_and_resources_restore(self):
        self.assertIsNone(baggage.get_baggage("tenant.id"))
        with observations.identity_scope("tenant-example", "agent-example"):
            self.assertEqual(baggage.get_baggage("tenant.id"), "tenant-example")
            with observations.identity_scope("nested"):
                self.assertEqual(baggage.get_baggage("tenant.id"), "nested")
                self.assertEqual(baggage.get_baggage("agent.id"), "agent-example")
            self.assertEqual(baggage.get_baggage("tenant.id"), "tenant-example")
        self.assertIsNone(baggage.get_baggage("tenant.id"))
        self.assertEqual(self.runtime.tracer_provider.resource.attributes["service.name"], "observations")

    async def test_live_real_agent_completion_tools_and_native_spans_export_in_memory(self):
        replies = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call1", "type": "function", "function": {"name": "backward", "arguments": '{"distance":3}'}}]},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call2", "type": "function", "function": {"name": "forward", "arguments": '{"distance":3}'}}]},
            {"role": "assistant", "content": "Done"}]
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            message = replies.pop(0)
            return httpx.Response(200, json={"id": "completion", "object": "chat.completion", "created": 1,
                "model": "offline-model", "choices": [{"index": 0, "message": message,
                    "finish_reason": "tool_calls" if "tool_calls" in message else "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            async with AsyncOpenAI(api_key="offline-placeholder", base_url="https://offline.invalid/v1/", http_client=http, max_retries=0) as sdk:
                settings = copy.copy(native_obs.OBSERVABILITY_SETTINGS)
                settings.enable_instrumentation = True
                settings.enable_sensitive_data = False
                settings._executed_setup = True
                with patch.object(trace, "_TRACER_PROVIDER", self.runtime.tracer_provider), \
                     patch.object(metric_globals, "_METER_PROVIDER", self.runtime.meter_provider), \
                     patch.object(log_globals, "_LOGGER_PROVIDER", self.runtime.logger_provider), \
                     patch.object(native_obs, "OBSERVABILITY_SETTINGS", settings), \
                     patch("shared.motor_tools._DELAY_SECONDS", 0):
                    agent, tracker = observations.build_observed_agent(sdk, self.runtime)
                    response = await agent.run("Go back and forth 3 meters", session=agent.create_session())
        self.assertEqual(response.text, "Done")
        self.assertEqual(tracker.direction_changes, 1)
        self.assertEqual(len(requests), 3)
        self.assertFalse(requests[0]["parallel_tool_calls"])
        spans = self.runtime.span_exporter.get_finished_spans()
        self.assertTrue(any(s.name.startswith("invoke_agent") for s in spans), [s.name for s in spans])
        self.assertTrue(any(s.name.startswith("chat") for s in spans), [s.name for s in spans])
        self.assertEqual(len({s.context.trace_id for s in spans}), 1)
        metrics = flatten_metrics(self.runtime)
        self.assertIn("gen_ai.client.token.usage", metrics)
        self.assertIn("gen_ai.client.operation.duration", metrics)
        self.assertTrue(all("Go back" not in str(s.attributes) for s in spans))
        logs = self.runtime.log_exporter.get_finished_logs()
        self.assertEqual([record.log_record.body for record in logs],
                         ["Agent run started", "Robot tool completed", "Robot tool completed", "Agent run completed"])
        self.assertTrue(all("Go back" not in str(record.log_record.attributes) for record in logs))

    async def test_local_mode_ignores_environment_exporters_and_close_is_idempotent(self):
        with patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": "https://never.invalid", "ENABLE_CONSOLE_EXPORTERS": "true"}):
            with observations.TelemetryRuntime(mode="local") as runtime:
                with runtime.tracer.start_as_current_span("local-only"):
                    runtime.logger.info("safe local event")
                self.assertEqual(runtime.summary(), {"spans": 1, "logs": 1, "export_mode": "local"})
            runtime.close()
            self.assertTrue(runtime._closed)

    async def test_otlp_missing_dependency_is_actionable_and_never_auto_installed(self):
        import builtins
        original = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.startswith("opentelemetry.exporter.otlp"):
                raise ModuleNotFoundError("optional exporter absent")
            return original(name, *args, **kwargs)
        with patch.object(builtins, "__import__", side_effect=guarded_import):
            with self.assertRaisesRegex(RuntimeError, "opentelemetry-exporter-otlp-proto-grpc==1.42.1"):
                observations.TelemetryRuntime(mode="otlp", endpoint="http://localhost:4317")

    async def test_otlp_protocol_routes_all_three_signals_using_exporter_api_without_network(self):
        import sys
        from types import ModuleType
        from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
        from opentelemetry.sdk.metrics.export import MetricExporter, MetricExportResult
        from opentelemetry.sdk._logs.export import LogRecordExporter, LogRecordExportResult
        created = []
        class Span(SpanExporter):
            def __init__(self, **kwargs): created.append(("traces", kwargs))
            def export(self, spans): return SpanExportResult.SUCCESS
            def shutdown(self): pass
        class Metric(MetricExporter):
            def __init__(self, **kwargs):
                super().__init__(); created.append(("metrics", kwargs))
            def export(self, metrics_data, timeout_millis=10000, **kwargs): return MetricExportResult.SUCCESS
            def force_flush(self, timeout_millis=10000): return True
            def shutdown(self, timeout_millis=30000, **kwargs): pass
        class Log(LogRecordExporter):
            def __init__(self, **kwargs): created.append(("logs", kwargs))
            def export(self, batch): return LogRecordExportResult.SUCCESS
            def shutdown(self): pass
        modules = {}
        for protocol in ("grpc", "http"):
            base = "opentelemetry.exporter.otlp.proto." + protocol
            for module_name, attribute, implementation in (("trace_exporter", "OTLPSpanExporter", Span),
                    ("metric_exporter", "OTLPMetricExporter", Metric), ("_log_exporter", "OTLPLogExporter", Log)):
                module = ModuleType(base + "." + module_name)
                setattr(module, attribute, implementation)
                modules[module.__name__] = module
        with patch.dict(sys.modules, modules):
            for protocol in ("grpc", "http/protobuf"):
                with self.subTest(protocol=protocol):
                    created.clear()
                    with observations.TelemetryRuntime(mode="otlp", endpoint="http://collector.invalid:4317", protocol=protocol) as runtime:
                        with runtime.tracer.start_as_current_span("mock-otlp-export"):
                            runtime.logger.info("mock log")
                        runtime.meter.create_counter("mock_counter").add(1)
                    self.assertEqual({signal for signal, _ in created}, {"traces", "metrics", "logs"})
                    for signal, kwargs in created:
                        suffix = "" if protocol == "grpc" else "/v1/" + signal
                        self.assertEqual(kwargs["endpoint"], "http://collector.invalid:4317" + suffix)
                        self.assertEqual(kwargs["timeout"], 5)

    async def test_native_install_respects_explicit_routing_without_sticky_test_globals(self):
        with patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": "https://never.invalid", "VS_CODE_EXTENSION_PORT": "9999"}), \
             patch.object(native_obs, "configure_otel_providers") as setup, \
             patch.object(observations.trace, "set_tracer_provider") as set_trace, \
             patch.object(observations.metrics, "set_meter_provider") as set_meter, \
             patch.object(observations, "set_logger_provider") as set_logs:
            def assert_clean(**kwargs):
                self.assertNotIn("OTEL_EXPORTER_OTLP_ENDPOINT", os.environ)
                self.assertNotIn("VS_CODE_EXTENSION_PORT", os.environ)
                self.assertEqual(kwargs, {"enable_sensitive_data": False, "enable_console_exporters": False})
            setup.side_effect = assert_clean
            self.runtime.install()
            self.assertEqual(os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"], "https://never.invalid")
            set_trace.assert_called_once_with(self.runtime.tracer_provider)
            set_meter.assert_called_once_with(self.runtime.meter_provider)
            set_logs.assert_called_once_with(self.runtime.logger_provider)

    async def test_a365_is_explicitly_unsupported_and_endpoint_validation(self):
        with self.assertRaisesRegex(RuntimeError, "NOT Agent365"):
            observations.TelemetryRuntime(mode="a365")
        for endpoint in ("", "file:///tmp", "https://user:secret@example.com", "http://localhost:4317?token=x"):
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(ValueError):
                    observations.validate_endpoint(endpoint)
        self.assertEqual(observations.validate_endpoint("http://localhost:4317/"), "http://localhost:4317")


class CliTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_evaluator_runner_loads_all_and_never_constructs_sdk(self):
        with patch("shared.chat_clients.create_sdk_client", side_effect=AssertionError("Offline runner created SDK")), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            result = await evaluations.evaluation_main([])
        self.assertEqual(result, 0)
        self.assertEqual(output.getvalue().count("SKIPPED"), 11)
        self.assertEqual(output.getvalue().count('"matches_expected": true'), 3)

    async def test_apphost_explicit_start_no_automatic_server(self):
        with patch.object(observations.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()):
            result = observations.apphost_main([])
        self.assertEqual(result, 0)
        run.assert_not_called()

    async def test_apphost_uses_current_interpreter_sibling_endpoint_and_waits(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(observations.subprocess, "run", return_value=SimpleNamespace(returncode=7)) as run, \
             contextlib.redirect_stdout(io.StringIO()):
            result = observations.apphost_main(["--start", "--otlp-endpoint", "http://localhost:19190", "--live", "--prompt", "Stop"])
        self.assertEqual(result, 7)
        command = run.call_args.args[0]
        self.assertEqual(command[0], observations.sys.executable)
        self.assertEqual(Path(command[1]).parent.name, "Observations")
        self.assertIn("otlp", command)
        self.assertIn("http://localhost:19190", command)
        self.assertEqual(run.call_args.kwargs["env"]["OTEL_SERVICE_NAME"], "observations")
        self.assertFalse(run.call_args.kwargs["check"])

    async def test_all_four_project_entrypoints_exist_and_use_required_parent_path(self):
        chapter = DATA.parent.parent
        for project in ("Evaluators", "Observations", "Observations.AppHost", "ObservationsWithAgent365"):
            with self.subTest(project=project):
                text = (chapter / project / "main.py").read_text()
                self.assertIn("parents[2]", text)
                compile(text, str(chapter / project / "main.py"), "exec")


if __name__ == "__main__":
    unittest.main()
