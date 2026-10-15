"""Chapter 11 OpenTelemetry examples: local by default, exporters explicitly opted in."""

import argparse
import logging
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

from agent_framework import Agent, AgentContext, AgentMiddleware, FunctionInvocationContext, FunctionMiddleware
from opentelemetry import baggage, context, metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.metrics import Observation
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor, InMemoryLogRecordExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from .evaluation_examples import PYTHON_TOOL_NAMES

SOURCE_NAME = "OpenTelemetry.RobbyAgent"
AGENT_NAME = "RobotCarDemoAgent"
INSTRUCTIONS = ("You are an AI assistant controlling a robot car. "
                "The available robot car permitted moves are forward, backward, turn left, turn right, and stop.")
A365_LIMITATION = (
    "Agent365 export is not implemented in this pinned example. The official microsoft-opentelemetry "
    "Python distro exists, but the published 1.3.9 package requires OTel~=1.44.0 "
    "(this workspace uses AF1.8.1/OTel1.42.1). Its published AF extra accepts AF>=1.4.0; "
    "the GitHub main extra targets AF>=1.19.0. Use --telemetry local/console/otlp for standard OTel identity "
    "metadata; that is NOT Agent365 ingestion or authentication. See the chapter README."
)


def validate_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("OTLP endpoint must be an http(s) URL without embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("OTLP endpoint must not include a query or fragment")
    return endpoint.rstrip("/")


class TelemetryRuntime:
    """Owns all three OTel providers and their bounded exporter lifecycle.

    Local providers are not installed globally unless install() is called by a CLI.
    Tests inject these providers and temporarily bind globals without sticky OTel setters.
    """

    def __init__(self, *, service_name="observations", mode="local", endpoint=None, protocol="grpc",
                 identity=None):
        if mode == "a365":
            raise RuntimeError(A365_LIMITATION)
        if mode not in {"local", "console", "otlp"}:
            raise ValueError(f"Unknown telemetry mode: {mode}")
        if protocol not in {"grpc", "http/protobuf"}:
            raise ValueError("OTLP protocol must be grpc or http/protobuf")
        self.mode = mode
        self.identity = dict(identity or {})
        resource = Resource.create({"service.name": service_name, "service.version": "1.0.0",
                                    **self.identity})
        self.span_exporter = InMemorySpanExporter()
        self.metric_reader = InMemoryMetricReader()
        self.log_exporter = InMemoryLogRecordExporter()
        spans, logs, readers = [self.span_exporter], [self.log_exporter], [self.metric_reader]
        if mode == "console":
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            from opentelemetry.sdk.metrics.export import ConsoleMetricExporter
            from opentelemetry.sdk._logs.export import ConsoleLogRecordExporter
            spans.append(ConsoleSpanExporter())
            logs.append(ConsoleLogRecordExporter())
            readers.append(PeriodicExportingMetricReader(ConsoleMetricExporter(), export_interval_millis=5000))
        elif mode == "otlp":
            endpoint = validate_endpoint(endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", ""))
            try:
                if protocol == "grpc":
                    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
                    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
                    options = {"endpoint": endpoint}
                    span_endpoint = metric_endpoint = log_endpoint = options
                else:
                    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
                    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
                    span_endpoint = {"endpoint": endpoint + "/v1/traces"}
                    metric_endpoint = {"endpoint": endpoint + "/v1/metrics"}
                    log_endpoint = {"endpoint": endpoint + "/v1/logs"}
            except ImportError as error:
                suffix = "grpc" if protocol == "grpc" else "http"
                raise RuntimeError(f"OTLP export requires optional opentelemetry-exporter-otlp-proto-{suffix}==1.42.1; "
                                   "ask the parent to add/install it first, or select --telemetry local") from error
            # Explicit endpoints avoid accidental routing; SDK still reads OTEL_EXPORTER_OTLP_HEADERS.
            spans.append(OTLPSpanExporter(**span_endpoint, timeout=5))
            logs.append(OTLPLogExporter(**log_endpoint, timeout=5))
            readers.append(PeriodicExportingMetricReader(OTLPMetricExporter(**metric_endpoint, timeout=5),
                                                        export_interval_millis=5000))
        self.tracer_provider = TracerProvider(resource=resource)
        for exporter in spans:
            processor = BatchSpanProcessor(exporter) if mode == "otlp" and exporter is not self.span_exporter else SimpleSpanProcessor(exporter)
            self.tracer_provider.add_span_processor(processor)
        self.meter_provider = MeterProvider(metric_readers=readers, resource=resource)
        self.logger_provider = LoggerProvider(resource=resource)
        for exporter in logs:
            self.logger_provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
        self.tracer = self.tracer_provider.get_tracer(SOURCE_NAME)
        self.meter = self.meter_provider.get_meter(SOURCE_NAME)
        self.logger = logging.getLogger(f"chapter11.{service_name}.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.handler = LoggingHandler(level=logging.INFO, logger_provider=self.logger_provider)
        self.logger.addHandler(self.handler)
        self._closed = False

    def install(self):
        from agent_framework.observability import configure_otel_providers

        # Native AF setup reads environment exporter settings implicitly. The runtime's
        # explicit mode owns routing, so suppress those settings only during setup.
        keys = [key for key in os.environ if key.startswith("OTEL_EXPORTER_OTLP") or
                key in {"VS_CODE_EXTENSION_PORT", "ENABLE_CONSOLE_EXPORTERS"}]
        saved = {key: os.environ.pop(key) for key in keys}
        try:
            configure_otel_providers(enable_sensitive_data=False, enable_console_exporters=False)
        finally:
            os.environ.update(saved)
        trace.set_tracer_provider(self.tracer_provider)
        metrics.set_meter_provider(self.meter_provider)
        set_logger_provider(self.logger_provider)

    def flush(self):
        self.tracer_provider.force_flush(timeout_millis=5000)
        self.meter_provider.force_flush(timeout_millis=5000)
        self.logger_provider.force_flush(timeout_millis=5000)

    def close(self):
        if not self._closed:
            try:
                self.flush()
            finally:
                self.logger.removeHandler(self.handler)
                self.handler.close()
                try:
                    self.tracer_provider.shutdown()
                finally:
                    try:
                        self.meter_provider.shutdown()
                    finally:
                        self.logger_provider.shutdown()
                        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def summary(self) -> dict:
        return {"spans": len(self.span_exporter.get_finished_spans()),
                "logs": len(self.log_exporter.get_finished_logs()), "export_mode": self.mode}


@contextmanager
def identity_scope(tenant_id: str | None = None, agent_id: str | None = None):
    """Standard OTel baggage only; this does not register/authenticate an A365 agent."""
    current = context.get_current()
    for key, value in (("tenant.id", tenant_id), ("agent.id", agent_id)):
        if value:
            current = baggage.set_baggage(key, value, context=current)
    token = context.attach(current)
    try:
        yield
    finally:
        context.detach(token)


class DirectionChangeTracker(FunctionMiddleware):
    """Post-invocation direction tracking, matching Observations' reset-on-any-turn.

    This is deliberately different from the safety evaluator, where ONLY Stop
    clears a direction. Python function names map explicitly to source names.
    Concurrent tool calls are tracked in completion order, as in the source.
    """

    def __init__(self, runtime: TelemetryRuntime, *, agent_name=AGENT_NAME, clock=time.perf_counter):
        self.runtime, self.agent_name, self.clock = runtime, agent_name, clock
        self.previous = None
        self.direction_changes = 0
        self.active_calls = 0
        self._lock = threading.RLock()
        self.counter = runtime.meter.create_counter("robot_direction_change_total", unit="1",
            description="Total times the robot changed direction between forward and backward")
        self.calls = runtime.meter.create_counter("robot_tool_calls_total", unit="1")
        self.errors = runtime.meter.create_counter("robot_tool_errors_total", unit="1")
        self.duration = runtime.meter.create_histogram("robot_tool_duration", unit="s")
        self.active_gauge = runtime.meter.create_observable_gauge("robot_active_tool_calls", [self._observe_active], unit="1")
        self.direction_gauge = runtime.meter.create_observable_gauge("robot_linear_direction", [self._observe_direction], unit="1",
            description="Forward=1, backward=-1, reset=0")

    def _observe_active(self, _):
        with self._lock:
            return [Observation(self.active_calls, {"agent.name": self.agent_name})]

    def _observe_direction(self, _):
        with self._lock:
            return [Observation({"Forward": 1, "Backward": -1}.get(self.previous, 0),
                                {"agent.name": self.agent_name})]

    async def process(self, invocation: FunctionInvocationContext, call_next):
        python_name = invocation.function.name
        source_name = {value: key for key, value in PYTHON_TOOL_NAMES.items()}.get(python_name, python_name)
        tags = {"agent.name": self.agent_name, "robot.tool.name": source_name}
        started = self.clock()
        with self._lock:
            self.active_calls += 1
        try:
            with self.runtime.tracer.start_as_current_span("Robot Tool Invocation", attributes=tags):
                try:
                    await call_next()
                except BaseException:
                    self.errors.add(1, tags)
                    raise
                self.calls.add(1, tags)
                self.runtime.logger.info("Robot tool completed", extra={"robot.tool.name": source_name,
                                         "agent.name": self.agent_name})
                with self._lock:
                    if source_name not in {"Forward", "Backward"}:
                        self.previous = None
                        return
                    changed = self.previous is not None and self.previous != source_name
                    direction_tags = {**tags, "robot.direction.previous": self.previous or "none",
                                      "robot.direction.current": source_name, "robot.direction.changed": changed}
                    with self.runtime.tracer.start_as_current_span("Detect Robot Direction Change", attributes=direction_tags) as span:
                        if changed:
                            self.counter.add(1, {"from": self.previous, "to": source_name})
                            self.direction_changes += 1
                            span.add_event("Robot direction changed")
                        self.previous = source_name
        finally:
            self.duration.record(max(0, self.clock() - started), tags)
            with self._lock:
                self.active_calls -= 1


class AgentRunLogging(AgentMiddleware):
    """Structured run lifecycle logs without prompts, responses or credentials."""

    def __init__(self, runtime: TelemetryRuntime):
        self.logger = runtime.logger

    async def process(self, invocation: AgentContext, call_next):
        attributes = {"agent.name": invocation.agent.name or AGENT_NAME}
        self.logger.info("Agent run started", extra=attributes)
        try:
            await call_next()
        except BaseException as error:
            self.logger.error("Agent run failed", extra={**attributes, "error.type": type(error).__name__})
            raise
        self.logger.info("Agent run completed", extra=attributes)


def build_observed_agent(sdk, runtime, *, provider="openai", tools=None):
    from .chat_clients import create_completion_client
    from .tool_examples import create_motor_tools

    tracker = DirectionChangeTracker(runtime)
    agent = Agent(client=create_completion_client(sdk, provider), name=AGENT_NAME,
                  instructions=INSTRUCTIONS, tools=create_motor_tools() if tools is None else tools,
                  middleware=[AgentRunLogging(runtime), tracker], default_options={"parallel_tool_calls": False})
    return agent, tracker


async def run_simulation(tracker: DirectionChangeTracker):
    from .tool_examples import create_motor_tools

    tools = {function.name: function for function in create_motor_tools()}
    sequence = [("backward", {"distance": 3}), ("forward", {"distance": 3}), ("stop", {}),
                ("forward", {"distance": 2}), ("turn_left", {"angle": 90}), ("backward", {"distance": 2})]
    for name, arguments in sequence:
        invocation = FunctionInvocationContext(function=tools[name], arguments=arguments)

        async def invoke():
            invocation.result = await invocation.function.invoke(arguments=invocation.arguments,
                                                                 context=invocation)

        await tracker.process(invocation, invoke)
    return tracker.direction_changes


async def observation_main(argv=None, *, identity_mode=False) -> int:
    parser = argparse.ArgumentParser(description="OpenTelemetry robot demo; defaults to offline local telemetry.")
    parser.add_argument("--provider", choices=("openai", "azure"), default="openai")
    parser.add_argument("--telemetry", choices=("local", "console", "otlp", "a365"), default="local")
    parser.add_argument("--otlp-endpoint", help="Explicitly opt into export with --telemetry otlp")
    parser.add_argument("--otlp-protocol", choices=("grpc", "http/protobuf"), default="grpc")
    parser.add_argument("--live", action="store_true", help="Opt into model-backed interactive agent calls")
    parser.add_argument("--prompt", help="Single live request instead of the input loop")
    parser.add_argument("--tenant-id", help="Non-secret standard OTel identity metadata (not authentication)")
    parser.add_argument("--agent-id", help="Non-secret standard OTel identity metadata (not authentication)")
    args = parser.parse_args(argv)
    if args.prompt and not args.live:
        parser.error("--prompt requires --live")
    if args.otlp_endpoint and args.telemetry != "otlp":
        parser.error("--otlp-endpoint requires --telemetry otlp")
    if args.telemetry == "a365":
        raise RuntimeError(A365_LIMITATION)
    tenant = args.tenant_id or (os.environ.get("AGENT365_TENANT_ID") if identity_mode else None)
    agent_id = args.agent_id or (os.environ.get("AGENT365_CLIENT_ID") if identity_mode else None)
    identity = {key: value for key, value in (("tenant.id", tenant), ("agent.id", agent_id)) if value}
    if identity_mode:
        print("STANDARD OTEL IDENTITY MODE - NOT Agent365 SDK integration/ingestion")
    sdk = None
    with TelemetryRuntime(mode=args.telemetry, endpoint=args.otlp_endpoint,
                          protocol=args.otlp_protocol, identity=identity) as runtime:
        runtime.install()
        try:
            if not args.live:
                tracker = DirectionChangeTracker(runtime)
                with identity_scope(tenant, agent_id), runtime.tracer.start_as_current_span("Offline Robot Simulation"):
                    await run_simulation(tracker)
                print(f"Direction changes: {tracker.direction_changes}")
            else:
                from .chat_clients import create_sdk_client
                sdk = create_sdk_client(args.provider)
                agent, tracker = build_observed_agent(sdk, runtime, provider=args.provider)
                session = agent.create_session()
                while True:
                    try:
                        user_input = args.prompt if args.prompt is not None else input("You (or 'exit' to quit): ")
                    except EOFError:
                        break
                    if not user_input.strip() or user_input.casefold() == "exit":
                        break
                    with identity_scope(tenant, agent_id), runtime.tracer.start_as_current_span("Robot Conversation", attributes=identity):
                        response = await agent.run(user_input, session=session)
                    print(f"RESPONSE: {response.text}")
                    if args.prompt is not None:
                        break
            runtime.flush()
            print(runtime.summary())
            return 0
        finally:
            if sdk is not None:
                await sdk.close()


def apphost_main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Dashboard-compatible bounded Python launcher; not a .NET Aspire AppHost.")
    parser.add_argument("--start", action="store_true", help="Explicitly start Observations, matching WithExplicitStart")
    parser.add_argument("--otlp-endpoint", help="Preconfigured separate dashboard/collector receiver URL")
    parser.add_argument("--otlp-protocol", choices=("grpc", "http/protobuf"), default="grpc")
    parser.add_argument("--provider", choices=("openai", "azure"), default="openai")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--prompt")
    args = parser.parse_args(argv)
    endpoint = args.otlp_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or os.environ.get("ASPIRE_DASHBOARD_OTLP_ENDPOINT_URL")
    print("Python launcher only: no dashboard download, Docker, resource service or persistent server.")
    if not args.start:
        print("Observations has explicit start enabled. Pass --start to run it.")
        return 0
    if args.prompt and not args.live:
        parser.error("--prompt requires --live")
    target = Path(__file__).resolve().parents[1] / "11.ObservabilityAndEvaluations" / "Observations" / "main.py"
    command = [sys.executable, str(target), "--provider", args.provider]
    if endpoint:
        command += ["--telemetry", "otlp", "--otlp-endpoint", validate_endpoint(endpoint),
                    "--otlp-protocol", args.otlp_protocol]
    if args.live:
        command += ["--live"]
    if args.prompt:
        command += ["--prompt", args.prompt]
    environment = os.environ.copy()
    environment["OTEL_SERVICE_NAME"] = "observations"
    # subprocess.run waits for the child, inherits Ctrl+C, and leaves no persistent server.
    return subprocess.run(command, env=environment, check=False).returncode
