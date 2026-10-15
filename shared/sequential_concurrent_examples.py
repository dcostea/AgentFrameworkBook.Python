"""Chapter 12: source-faithful sequential and concurrent Agent Framework graphs."""

import argparse
import asyncio
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from agent_framework import (
    Agent, AgentExecutor, AgentExecutorResponse, AgentMiddleware, AgentResponse,
    AgentResponseUpdate, BaseChatClient, ChatMiddleware, ChatMiddlewareLayer, Executor,
    FileCheckpointStorage, FunctionMiddleware, Message, Workflow, WorkflowBuilder,
    WorkflowCheckpoint, WorkflowContext, WorkflowEvent, handler, tool,
)
from agent_framework.exceptions import AgentFrameworkException
from agent_framework.observability import disable_instrumentation
from agent_framework.orchestrations import ConcurrentBuilder, SequentialBuilder
from openai import OpenAIError
from pydantic import BaseModel, ConfigDict

from .chat_clients import create_completion_client, create_sdk_client
from .tool_examples import create_motor_tools

SAMPLES = (
    "SequentialOrchestration", "SequentialOrchestrationAsPipeline",
    "SequentialOrchestrationWithBuilder", "SequentialOrchestrationWithCheckpoints",
    "SequentialOrchestrationWithIntermediateOutput", "SequentialOrchestrationWithMiddleware",
    "ConcurrentOrchestration", "SequenceOfConcurrentOrchestration",
)
MISSION_PROMPT = """# MISSION COMMAND: Exploration Trip

There is a tree directly in front of the car. Avoid it and then come back to the original path."""
ASSESSMENT_PROMPT = """MISSION COMMAND: Exploration Trip

Assess the environment conditions and ensure safety clearance."""
MIXED_PROMPT = """MISSION COMMAND: Exploration Trip
There is a tree directly in front of the car. Avoid the tree and continue the exploration."""
ENVIRONMENT_INSTRUCTIONS = """## PERSONA
You are the EnvironmentAgent that reads sensors.

## ACTIONS
Call SensorTools to read temperature, humidity, rain drops, and wind speed.

## OUTPUT TEMPLATE
Respond only with the environment report, and make the rain status easy to identify."""
SAFETY_INSTRUCTIONS = """## PERSONA
You are the SafetyAgent that grants or denies mission clearance.

## ACTIONS
Grant clearance unless the droplet level is Medium or High (rain detected), otherwise deny.

## OUTPUT TEMPLATE
Respond with GRANTED or DENIED and a brief reason."""
MOTORS_INSTRUCTIONS = """## PERSONA
You are the MotorsAgent that executes movement commands.

## ACTIONS
If clearance is DENIED, call the Stop tool and then respond with "Mission stopped due to unsafe conditions."
Otherwise, break the mission into moves (forward, backward, turn left, turn right, stop) and execute them using MotorTools.

## OUTPUT TEMPLATE
Respond only with the executed movement sequence."""
PIPELINE_ENVIRONMENT_INSTRUCTIONS = """## PERSONA
You are the EnvironmentAgent that starts a mission by reading sensors and interpreting the mission command.

## ACTIONS
Call SensorTools to read temperature, humidity, rain drops, and wind speed.
Extract the destination, obstacles, and required outcome from the mission command.
Include all mission details in your report, including the original command, because the next agent receives only your response.

## OUTPUT TEMPLATE
Respond only with a self-contained report containing:
- Mission objective
- Known obstacles
- Temperature
- Humidity
- Rain droplet level
- Wind speed"""
NAVIGATOR_INSTRUCTIONS = """## PERSONA
You are the NavigatorAgent that transforms an environment report into a route plan.

## ACTIONS
Use only the received environment report.
Plan a safe route that completes the mission objective and avoids every known obstacle.
Express the route as an ordered list using only: forward, backward, turn left, turn right, and stop.

## OUTPUT TEMPLATE
Respond only with the ordered movement plan, with one movement per line."""
PIPELINE_MOTORS_INSTRUCTIONS = """## PERSONA
You are the MotorsAgent that executes an ordered movement plan.

## ACTIONS
Execute each received movement in order using MotorTools.
Do not reinterpret the mission or alter the route plan.

## OUTPUT TEMPLATE
Respond only with the executed movement sequence."""
PIPELINE_TOOL_ADAPTATION = """## PYTHON TOOL EXECUTION
MotorTools means the available forward, backward, turn_left, turn_right, and stop functions.
Invoke the matching available function for each movement, in the received order, using its parameter schema.
Wait for each tool result before invoking the next movement. Do not just repeat the plan as text.
Only report movements whose tool calls actually succeeded; never claim execution without a tool result."""
MAINTENANCE_INSTRUCTIONS = """## PERSONA
You are the MaintenanceAgent that monitors maintenance conditions.

## ACTIONS
Activate maintenance protocols for dangerous conditions detection.

## SAFETY THRESHOLDS
Grant clearance unless ANY of the following hard limits are exceeded:
- Battery health below 30%
- Tire pressure below 28 PSI or above 36 PSI
- Motor efficiency below 30%

## OUTPUT TEMPLATE
Respond with the maintenance clearance using the requested JSON schema."""
CONCURRENT_ENVIRONMENT_INSTRUCTIONS = """## PERSONA
You are the EnvironmentAgent that reads sensors.

## ACTIONS
Read sensors for temperature, humidity, rain drops, and wind speed.

## SAFETY THRESHOLDS
Grant clearance unless ANY of the following hard limits are exceeded:
- Temperature above 100 Celsius
- Humidity above 80%
- Droplet level is Extreme
- Wind speed above 100 kmph

## OUTPUT TEMPLATE
Respond with the environment clearance using the requested JSON schema."""


class DropletLevel(str, Enum):
    NONE = "None"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class Direction(str, Enum):
    NORTH = "North"
    EAST = "East"
    SOUTH = "South"
    WEST = "West"
    NORTH_WEST = "NorthWest"
    NORTH_EAST = "NorthEast"
    SOUTH_WEST = "SouthWest"
    SOUTH_EAST = "SouthEast"


class SensorTools:
    """The six 00.Shared SensorTools mocks, including exclusive upper bounds."""

    def __init__(self, *, rng=None, delay: float = 0.5):
        self.rng = rng if rng is not None else random.Random()
        self.delay = delay

    async def _read(self, label: str, value, unit: str = ""):
        print(f"[{datetime.now():%I:%M:%S:%f}] SENSORS: READING {label}: {value}{unit}")
        await asyncio.sleep(self.delay)
        return value

    async def read_temperature(self) -> int:
        """Use thermal sensors to detect abnormal heat levels."""
        return await self._read("Temperature", self.rng.randrange(-20, 100), " Celsius degrees.")

    async def read_infrared_radiation(self) -> int:
        """Confirm the presence of flames via IR sensors."""
        return await self._read("Infrared Radiation", self.rng.randrange(0, 100))

    async def read_humidity(self) -> int:
        """Check local humidity as a precursor to rain detection (fires often reduce local moisture)."""
        return await self._read("Humidity", self.rng.randrange(0, 100), " %")

    async def read_droplet_level(self) -> str:
        """Use optical or capacitive rain sensors to measure the presence and intensity of raindrops on surfaces like windshields or body panels."""
        return await self._read("Droplet Level", self.rng.choice(list(DropletLevel)).value)

    async def read_wind_speed(self) -> int:
        """Reads and returns the wind speed in kmph."""
        return await self._read("Wind speed", self.rng.randrange(0, 100), " kmph")

    async def read_wind_direction(self) -> str:
        """Reads and returns the wind direction. The wind direction (output) is like North, NorthWest, etc."""
        return await self._read("Wind direction", self.rng.choice(list(Direction)).value)

    def as_tools(self) -> list:
        return [tool(fn, name=name) for fn, name in (
            (self.read_droplet_level, "ReadDropletLevelAsync"),
            (self.read_humidity, "ReadHumidityAsync"),
            (self.read_infrared_radiation, "ReadInfraredRadiationAsync"),
            (self.read_temperature, "ReadTemperatureAsync"),
            (self.read_wind_direction, "ReadWindDirectionAsync"),
            (self.read_wind_speed, "ReadWindSpeedAsync"),
        )]


class MaintenanceTools:
    """The six 00.Shared maintenance mocks with the original delay multipliers."""

    def __init__(self, *, rng=None, delay: float = 0.5):
        self.rng = rng if rng is not None else random.Random()
        self.delay = delay

    async def _act(self, action: str, result: str, multiplier: int = 1) -> str:
        print(f"[{datetime.now():%I:%M:%S:%f}] MAINTENANCE: {action}")
        await asyncio.sleep(self.delay * multiplier)
        return result

    async def calibrate_sensors(self) -> str:
        """Calibrates all sensors on the robot car."""
        return await self._act("CALIBRATING sensors...", "All sensors have been calibrated.")

    async def check_motors(self) -> str:
        """Checks the motors of the robot car."""
        status = self.rng.randrange(0, 100)
        return await self._act(f"CHECKING motors. Status: {status}%", f"Motors checked. Status: {status}% efficiency.")

    async def check_tire_pressure(self) -> str:
        """Checks the tire pressure of the robot car."""
        pressure = self.rng.randrange(30, 35)
        return await self._act(f"CHECKING tire pressure: {pressure} PSI", f"Tire pressure is {pressure} PSI.")

    async def clean_solar_panels(self) -> str:
        """Cleans the solar panels of the robot car."""
        return await self._act("CLEANING solar panels...", "Solar panels have been cleaned.", 3)

    async def check_battery_health(self) -> str:
        """Checks the battery health of the robot car."""
        health = self.rng.randrange(0, 100)
        return await self._act(f"CHECKING battery health: {health}%", f"Battery health is at {health}%.")

    async def update_firmware(self) -> str:
        """Updates the firmware of the robot car."""
        return await self._act("UPDATING firmware...", "Firmware has been updated to the latest version.", 5)

    def as_tools(self) -> list:
        return [tool(fn, name=name) for fn, name in (
            (self.update_firmware, "UpdateFirmwareAsync"),
            (self.check_battery_health, "CheckBatteryHealthAsync"),
            (self.clean_solar_panels, "CleanSolarPanelsAsync"),
            (self.check_tire_pressure, "CheckTirePressureAsync"),
            (self.check_motors, "CheckMotorsAsync"),
            (self.calibrate_sensors, "CalibrateSensorsAsync"),
        )]


class ClearanceState(str, Enum):
    GRANTED = "GRANTED"
    DENIED = "DENIED"


class ClearanceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clearance: ClearanceState
    reason: str


CLEARANCE_SCORE = {ClearanceState.GRANTED: 0, ClearanceState.DENIED: 1}


def without_tool_call_messages(messages: Sequence[Message]) -> list[Message]:
    # The C# boundary drops the entire message, not just its tool contents.
    return [message for message in messages
            if not any(content.type in {"function_call", "function_result"} for content in message.contents)]


class ToolCallFiltering(AgentMiddleware):
    async def process(self, context, call_next):
        context.messages = without_tool_call_messages(context.messages)
        await call_next()


class MissionAbort(RuntimeError):
    pass


def _check_clearance(response, *, structured: bool = False):
    if structured:
        try:
            denied = ClearanceResponse.model_validate_json(response.text).clearance == ClearanceState.DENIED
        except ValueError as exc:
            raise MissionAbort("CRITICAL FAILURE (invalid JSON)") from exc
    else:
        denied = "DENIED" in response.text
    if denied:
        print("\n*** MISSION ABORT ***")
        raise MissionAbort("MISSION ABORT")
    return response


class MissionAbortAgent(AgentMiddleware):
    def __init__(self, *, structured: bool = False, trace: list | None = None):
        self.structured = structured
        self.trace = trace

    async def process(self, context, call_next):
        if self.trace is not None:
            self.trace.append("agent:pre")
        await call_next()

        def post(response):
            if self.trace is not None:
                self.trace.append("agent:post")
            return _check_clearance(response, structured=self.structured)

        if context.stream:
            context.stream_result_hooks.append(post)
        else:
            post(context.result)


class MiddlewareChatClient(ChatMiddlewareLayer, BaseChatClient):
    """Apply response middleware around an existing client's full invocation loop."""

    def __init__(self, client, *, middleware):
        self.inner_client = client
        super().__init__(middleware=middleware)

    def _inner_get_response(self, *, messages, stream, options, **kwargs):
        return self.inner_client.get_response(messages, stream=stream, options=options, client_kwargs=kwargs)


class MissionAbortChat(ChatMiddleware):
    def __init__(self, *, trace: list | None = None):
        self.trace = trace

    async def process(self, context, call_next):
        if self.trace is not None:
            self.trace.append("chat:pre")
        await call_next()

        def post(response):
            if self.trace is not None:
                self.trace.append("chat:post")
            return _check_clearance(response)

        if context.stream:
            context.stream_result_hooks.append(post)
        else:
            post(context.result)


class SimulateAgentFailure(ChatMiddleware):
    async def process(self, context, call_next):
        mode = " (STREAMING)" if context.stream else ""
        print(f"\n*** SIMULATE AGENT FAILURE{mode} ***")
        raise RuntimeError("(streaming chat response middleware)" if context.stream else "(chat response middleware)")


class SimulateToolFailure(FunctionMiddleware):
    async def process(self, context, call_next):
        print("\n*** SIMULATE TOOL FAILURE ***")
        raise RuntimeError("(function calling middleware)")


class PipelineMotorCalls(FunctionMiddleware):
    def __init__(self, tool_names):
        self.tool_names = tool_names
        self.attempted = 0
        self.completed = 0

    async def process(self, context, call_next):
        is_motor = context.function.name in self.tool_names
        if is_motor:
            self.attempted += 1
        await call_next()
        if is_motor:
            self.completed += 1

    def verify(self, response):
        if not self.completed or self.completed != self.attempted:
            raise RuntimeError(
                f"Incomplete pipeline motor execution: {self.completed} successful motor calls "
                f"out of {self.attempted} attempted; prose is not execution."
            )
        return response


class RequirePipelineMotorExecution(AgentMiddleware):
    def __init__(self, tools):
        self.tool_names = frozenset(function.name for function in tools)

    async def process(self, context, call_next):
        calls = PipelineMotorCalls(self.tool_names)
        context.client_kwargs["middleware"] = [*context.client_kwargs.get("middleware", []), calls]
        await call_next()
        # Streaming responses may omit tool-result contents; verify actual invocation middleware instead.
        if context.stream:
            context.stream_result_hooks.append(calls.verify)
        else:
            calls.verify(context.result)


class SimulatedInterruption(RuntimeError):
    pass


class InterruptBeforeMotors(AgentMiddleware):
    async def process(self, context, call_next):
        raise SimulatedInterruption("Interrupted before MotorsAgent; resume the persisted checkpoint.")


def aggregate_clearances(results: list[AgentExecutorResponse]) -> AgentResponse:
    expected = {"MaintenanceAgent", "EnvironmentAgent"}
    if len(results) != 2 or {result.executor_id for result in results} != expected:
        raise ValueError("Safety assessment requires one result from each maintenance/environment branch.")
    responses = []
    for result in results:
        messages = [message for message in without_tool_call_messages(result.agent_response.messages)
                    if message.role == "assistant" and message.text.strip()]
        if not messages:
            raise ValueError(f"Incomplete safety response from {result.executor_id}.")
        for message in messages:
            responses.append(ClearanceResponse.model_validate_json(message.text))
    worst = max((response.clearance for response in responses), key=CLEARANCE_SCORE.__getitem__)
    reasons = [response.reason for response in responses if response.clearance == ClearanceState.DENIED]
    summary = f"DENIED: {' '.join(reasons)}" if worst == ClearanceState.DENIED else "GRANTED: All clearances were granted."
    return AgentResponse(messages=[Message("assistant", [summary], author_name="SafetyAggregator")])


class SafetyAggregator(Executor):
    @handler
    async def aggregate(self, results: list[AgentExecutorResponse], ctx: WorkflowContext[None, AgentResponse]) -> None:
        await ctx.yield_output(aggregate_clearances(results))


class ConversationOutput(Executor):
    def __init__(self, *, pipeline: bool = False):
        super().__init__(id="WorkflowConversation")
        self.pipeline = pipeline

    @handler
    async def complete(self, result: AgentExecutorResponse, ctx: WorkflowContext[None, AgentResponse]) -> None:
        messages = result.agent_response.messages if self.pipeline else result.full_conversation
        await ctx.yield_output(AgentResponse(messages=list(messages)))


def _named_graph(graph: Workflow, name: str, *, output: list[Executor], intermediate=(), storage=None) -> Workflow:
    # rc3 orchestration builders have no name/max_iterations arguments. Rebuild their
    # public topology with WorkflowBuilder, preserving fan-in barriers and node IDs.
    builder = WorkflowBuilder(
        start_executor=graph.executors[graph.start_executor_id], name=name, max_iterations=20,
        checkpoint_storage=storage, output_from=output, intermediate_output_from=list(intermediate),
    )
    for group in graph.edge_groups:
        if group.type == "InternalEdgeGroup":
            continue
        edges = group.edges
        if group.type == "FanOutEdgeGroup":
            builder.add_fan_out_edges(graph.executors[edges[0].source_id], [graph.executors[e.target_id] for e in edges])
        elif group.type == "FanInEdgeGroup":
            builder.add_fan_in_edges([graph.executors[e.source_id] for e in edges], graph.executors[edges[0].target_id])
        elif group.type == "SingleEdgeGroup":
            builder.add_edge(graph.executors[edges[0].source_id], graph.executors[edges[0].target_id])
        else:
            raise TypeError(f"Unsupported source topology: {group.type}")
    return builder.build()


def build_safety_workflow(client, *, mixed: bool = False, sensor_tools=None, maintenance_tools=None) -> Workflow:
    maintenance_prompt = MAINTENANCE_INSTRUCTIONS
    environment_prompt = CONCURRENT_ENVIRONMENT_INSTRUCTIONS
    if mixed:
        maintenance_prompt = maintenance_prompt.replace(
            "Activate maintenance protocols for dangerous conditions detection.",
            "Call MaintenanceTools to activate maintenance protocols for dangerous conditions detection.",
        )
        environment_prompt = environment_prompt.replace(
            "Read sensors for temperature, humidity, rain drops, and wind speed.",
            "Call SensorTools to read sensors for temperature, humidity, rain drops, and wind speed.",
        )
    options = {"response_format": ClearanceResponse}
    agents = [
        Agent(client=client, name="MaintenanceAgent", instructions=maintenance_prompt, default_options=options,
              tools=MaintenanceTools().as_tools() if maintenance_tools is None else maintenance_tools),
        Agent(client=client, name="EnvironmentAgent", instructions=environment_prompt, default_options=options,
              tools=SensorTools().as_tools() if sensor_tools is None else sensor_tools),
    ]
    aggregator = SafetyAggregator(id="SafetyAggregator")
    graph = ConcurrentBuilder(participants=agents).with_aggregator(aggregator).build()
    return _named_graph(graph, "SafetyAssessment", output=[aggregator])


def build_sample(sample: str, client, *, sensor_tools=None, motor_tools=None, maintenance_tools=None,
                 checkpoint_storage=None, interrupt: bool = False, resuming: bool = False) -> Workflow:
    if sample not in SAMPLES:
        raise ValueError(f"Unknown chapter 12 sample: {sample}")
    if sample == "ConcurrentOrchestration":
        return build_safety_workflow(client, sensor_tools=sensor_tools, maintenance_tools=maintenance_tools)
    pipeline = sample == "SequentialOrchestrationAsPipeline"
    selected = sample in {"SequentialOrchestrationWithBuilder", "SequentialOrchestrationWithIntermediateOutput"}
    motor_middleware = [InterruptBeforeMotors()] if interrupt else []
    if sample == "SequenceOfConcurrentOrchestration":
        safety = build_safety_workflow(client, mixed=True, sensor_tools=sensor_tools, maintenance_tools=maintenance_tools)
        participants = [safety.as_agent(name="SafetyStage")]
        motor_middleware.append(ToolCallFiltering())
        name = "SanitizedSequentialExecution"
    else:
        environment = Agent(
            client=client, name="EnvironmentAgent",
            instructions=PIPELINE_ENVIRONMENT_INSTRUCTIONS if pipeline else ENVIRONMENT_INSTRUCTIONS,
            tools=SensorTools().as_tools() if sensor_tools is None else sensor_tools,
        )
        safety_client = client
        safety_middleware = []
        if sample == "SequentialOrchestrationWithMiddleware":
            safety_client = MiddlewareChatClient(client, middleware=[MissionAbortChat()])
            safety_middleware = [MissionAbortAgent()]
        safety = Agent(client=safety_client, name="NavigatorAgent" if pipeline else "SafetyAgent",
                       instructions=NAVIGATOR_INSTRUCTIONS if pipeline else SAFETY_INSTRUCTIONS,
                       middleware=safety_middleware)
        participants = [environment, safety]
        name = {
            "SequentialOrchestrationAsPipeline": "ProcessingPipeline",
            "SequentialOrchestrationWithBuilder": "RobbyMission",
            "SequentialOrchestrationWithCheckpoints": "SafeExecutionFromCheckpoint" if resuming else "SafeExecutionWithCheckpoints",
            "SequentialOrchestrationWithMiddleware": "SafeExecution",
        }.get(sample, sample)
    movement_tools = create_motor_tools() if motor_tools is None else motor_tools
    if pipeline and movement_tools:
        motor_middleware.append(RequirePipelineMotorExecution(movement_tools))
    motors = Agent(client=client, name="MotorsAgent",
                   instructions=f"{PIPELINE_MOTORS_INSTRUCTIONS}\n\n{PIPELINE_TOOL_ADAPTATION}" if pipeline else MOTORS_INSTRUCTIONS,
                   tools=movement_tools, middleware=motor_middleware,
                   default_options={"tool_choice": "required", "allow_multiple_tool_calls": False} if pipeline else None)
    executors = [AgentExecutor(agent, context_mode="last_agent" if pipeline else "full")
                 for agent in [*participants, motors]]
    if selected:
        graph = SequentialBuilder(participants=executors, output_from=[executors[-1]],
                                  intermediate_output_from=executors[:-1]).build()
        return _named_graph(graph, name, output=[executors[-1]], intermediate=executors[:-1])
    output = ConversationOutput(pipeline=pipeline)
    graph = SequentialBuilder(participants=[*executors, output], chain_only_agent_responses=pipeline,
                              output_from=[output]).build()
    return _named_graph(graph, name, output=[output], storage=checkpoint_storage)


def output_messages(data) -> list[Message]:
    if isinstance(data, AgentResponse):
        return list(data.messages)
    if isinstance(data, Message):
        return [data]
    if isinstance(data, list) and all(isinstance(message, Message) for message in data):
        return list(data)
    raise TypeError(f"Expected AgentResponse, Message, or list[Message], not {type(data).__name__}.")


def print_event(event: WorkflowEvent, *, all_messages: bool = False) -> None:
    if event.type == "executor_completed":
        print(f"[EXECUTOR] {event.executor_id} completed.")
    elif event.type == "superstep_completed":
        print(f"[SUPER STEP] {event.iteration} completed.")
    elif event.type in {"output", "intermediate"}:
        label = "WORKFLOW OUTPUT" if event.type == "output" else "INTERMEDIATE OUTPUT"
        if isinstance(event.data, AgentResponseUpdate):
            if event.data.text:
                print(f"[{label}] {event.executor_id}: {event.data.text}")
            return
        messages = output_messages(event.data)
        if all_messages:
            text = "\n".join(f"{message.role}: {message.text}" for message in messages
                             if message.role != "tool" and message.text)
        else:
            text = messages[-1].text if messages else ""
        if text.strip():
            print(f"[{label}] {text}")
    elif event.type in {"error", "failed", "executor_failed"}:
        text = event.details.message if event.details is not None else str(event.data)
        print(f"[WORKFLOW ERROR] {event.executor_id or ''}: {text}")


def pending_executor_ids(workflow: Workflow, checkpoint: WorkflowCheckpoint) -> set[str]:
    pending = set()
    for source, messages in checkpoint.messages.items():
        for message in messages:
            if message.target_id:
                pending.add(message.target_id.removeprefix("internal:"))
            else:
                pending.update(edge.target_id.removeprefix("internal:")
                               for group in workflow.edge_groups for edge in group.edges if edge.source_id == source)
    return pending


@dataclass
class CheckpointDemoResult:
    checkpoint_id: str
    first_events: list[WorkflowEvent]
    resumed_events: list[WorkflowEvent]
    interrupted: bool = False


async def run_checkpoint_demo(build: Callable[..., Workflow], storage: FileCheckpointStorage, *, prompt=MISSION_PROMPT,
                              interrupt: bool = False, resume_id: str | None = None, emit: Callable = print_event):
    first_events = []
    resumed_events = []
    checkpoint_id = resume_id
    if resume_id is None:
        workflow = build(checkpoint_storage=storage, interrupt=interrupt)
        try:
            async for event in workflow.run(prompt, stream=True):
                first_events.append(event)
                emit(event)
                if event.type == "superstep_completed" and checkpoint_id is None:
                    checkpoint = await storage.get_latest(workflow_name=workflow.name)
                    if checkpoint is not None and any(name.startswith("MotorsAgent")
                                                      for name in pending_executor_ids(workflow, checkpoint)):
                        checkpoint_id = checkpoint.checkpoint_id
                        print(f"*** CHECKPOINT CREATED *** {checkpoint_id}")
        except SimulatedInterruption:
            if not interrupt or checkpoint_id is None:
                raise
            print(f"*** INTERRUPTED *** Resume with --resume {checkpoint_id}")
            return CheckpointDemoResult(checkpoint_id, first_events, [], interrupted=True)
        if checkpoint_id is None:
            raise RuntimeError("No checkpoint activating MotorsAgent was persisted.")
    print(f"*** RESUMING FROM CHECKPOINT *** {checkpoint_id}")
    resumed = build(checkpoint_storage=storage, resuming=True)
    async for event in resumed.run(checkpoint_id=checkpoint_id, checkpoint_storage=storage, stream=True):
        resumed_events.append(event)
        emit(event)
    return CheckpointDemoResult(checkpoint_id, first_events, resumed_events)


def argument_parser(sample: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"{sample}: one mission, at most 20 workflow supersteps, 120-second default timeout; simulated hardware only.",
    )
    parser.add_argument("--provider", choices=("openai", "azure"), default="openai")
    default_prompt = ASSESSMENT_PROMPT if sample == "ConcurrentOrchestration" else (
        MIXED_PROMPT if sample == "SequenceOfConcurrentOrchestration" else MISSION_PROMPT)
    parser.add_argument("--prompt", default=default_prompt, help="Override the source mission command.")
    parser.add_argument("--timeout", type=float, default=120, help="Positive total timeout in seconds (default: 120).")
    if sample == "SequentialOrchestrationWithCheckpoints":
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--interrupt", action="store_true", help="Persist state, interrupt before motors, and exit without replay.")
        group.add_argument("--resume", metavar="CHECKPOINT_ID", help="Resume trusted local runtime/checkpoints state on a fresh graph.")
    return parser


async def run_sample(sample: str, args, *, sample_dir: Path) -> None:
    # Disable framework auto-export even when shared environment settings contain OTLP endpoints.
    disable_instrumentation()
    async with create_sdk_client(args.provider) as sdk:
        client = create_completion_client(sdk, args.provider)
        if sample == "SequentialOrchestrationWithCheckpoints":
            storage = FileCheckpointStorage(sample_dir / "runtime" / "checkpoints", allowed_checkpoint_types=[])
            await run_checkpoint_demo(lambda **kwargs: build_sample(sample, client, **kwargs), storage,
                                      prompt=args.prompt, interrupt=args.interrupt, resume_id=args.resume)
            return
        workflow = build_sample(sample, client)
        print(f"[WORKFLOW] {workflow.name}")
        streaming = sample in {"SequentialOrchestrationAsPipeline", "SequentialOrchestrationWithBuilder"}
        if streaming:
            async for event in workflow.run(args.prompt, stream=True):
                print_event(event, all_messages=sample == "SequentialOrchestrationAsPipeline")
        else:
            for event in await workflow.run(args.prompt):
                print_event(event)


async def run_bounded(sample: str, args, *, sample_dir: Path) -> None:
    await asyncio.wait_for(run_sample(sample, args, sample_dir=sample_dir), args.timeout)


def main(sample: str, *, sample_dir: Path, argv=None) -> int:
    parser = argument_parser(sample)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    try:
        asyncio.run(run_bounded(sample, args, sample_dir=sample_dir))
    except (AgentFrameworkException, OpenAIError, OSError, RuntimeError, ValueError, TimeoutError) as exc:
        print(f"[WORKFLOW ERROR] {exc}")
        return 1
    return 0
