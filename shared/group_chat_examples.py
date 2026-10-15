"""Native, bounded group-chat workflows for the three chapter 13 projects."""

import argparse
import asyncio
import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from agent_framework import (
    Agent, AgentExecutor, AgentExecutorResponse, AgentResponse, AgentResponseUpdate,
    AgentSession, ChatContext, ChatMiddleware, Executor, Message, ResponseStream,
    Workflow, WorkflowContext, handler, tool,
)
from agent_framework.orchestrations import GroupChatBuilder, GroupChatOrchestrator, GroupChatState
# These participant envelopes/registry are not re-exported by orchestrations 1.0.0rc3.
from agent_framework_orchestrations._group_chat import (
    GroupChatParticipantMessage, GroupChatRequestMessage, GroupChatResponseMessage,
    ParticipantRegistry,
)
from pydantic import BaseModel

from .chat_clients import create_completion_client, create_sdk_client
from .tool_examples import create_motor_tools

MAX_ITERATIONS = 10
MISSION = '''# MISSION COMMAND: Exploration Trip
There is a tree directly in front of the car. Avoid it and then come back to the original path.
The distance to the tree is 50 meters.'''
INVESTIGATION_PROMPT = "Check Robby's environment and decide whether it is safe."

NAVIGATOR_INSTRUCTIONS = '''You are NavigatorAgent. Approve or deny proposed move sequences.
Prefer turning angles of 30, 45, or 60 degrees for efficient pathing. Use 90 degrees only when geometry requires it.
Respond APPROVED if the sequence is optimal; otherwise DENIED with concrete angle or path improvements.
Return exactly one line: APPROVED: <optimality tips> or DENIED: <optimality tips>.'''
MOTOR_INSTRUCTIONS = '''You are MotorsAgent. Permitted moves: forward, backward, turn left, turn right, stop.
Break the mission into a move sequence, submit it to NavigatorAgent for approval, and end your turn.
Every turn must specify numeric angle degrees for each turn and integer distance meters for each move.
Do not submit bare move names or leave NavigatorAgent to guess parameters. On DENIED, incorporate its
concrete angle/path feedback into the numeric parameters of the revised proposal.
NavigatorAgent's most recent verdict is the ONLY one that matters for execution.
If DENIED, revise based on feedback and resubmit; do not use motor tools.
If APPROVED, execute that approved sequence using MotorTools, then respond EXECUTED: <one-line summary>.
Never execute a proposed, revised, or denied sequence without an explicit NavigatorAgent APPROVED decision.'''
HITL_MOTOR_INSTRUCTIONS = '''You are MotorsAgent. Permitted moves: forward, backward, turn left, turn right, stop.
Break the mission into a move sequence and state it clearly, then end your turn.
NavigatorAgent offers advisory APPROVED or DENIED feedback; it never blocks execution.
HumanAgent's most recent verdict is the ONLY one that matters for execution.
If HUMAN DENIED, revise using the feedback and resubmit; do not use motor tools.
If HUMAN APPROVED, execute that approved sequence using MotorTools, then respond EXECUTED: <one-line summary>.
Never execute without the most recent HumanAgent APPROVED. NavigatorAgent cannot override or delay human approval.'''
ENVIRONMENT_INSTRUCTIONS = '''You are EnvironmentAgent. Collect Robby's environmental readings.
On every turn, read temperature and humidity using each tool exactly once.
These two integer measurements form ONE observation, not two observations.
Report only this turn's observation. Do not repeat earlier readings or assess safety.
Output: Observation: temperature = <value> Celsius; humidity = <value>%.'''
MAINTENANCE_INSTRUCTIONS = '''You are MaintenanceAgent. Maintain Robby's sensors.
Call your calibration tool and wait for its result. Confirm calibration only if the tool reports success.
If the tool fails, report failure instead of claiming calibration is complete. Do not repeat sensor readings.'''
SAFETY_INSTRUCTIONS = '''You are SafetyAgent. Decide whether Robby's readings are safe.
One observation requires a NEW successful integer temperature/humidity tool pair in one EnvironmentAgent turn.
Repeated reports and old tool results are NOT additional observations. Use the latest InvestigationManager state
for the verified sensor values, completed observation count, and calibration status. A report without a verified
pair is not evidence. With zero verified observations request CALIBRATION for a fresh observation.
Your previous decisions are not measurements.
CALIBRATION requests work; it never confirms that maintenance happened.
Apply these rules IN ORDER using actual readings from ALL completed observations:
1. If ANY temperature is greater than 90 Celsius or ANY humidity is greater than 95%, respond DENIED.
2. With at least two observations, if ANY temperature is greater than 60 Celsius or ANY humidity is greater
   than 80%, respond DENIED. Check BOTH observations, not only the latest. Do not apply these lower limits
   when the completed observation count is less than two.
3. With only one observation, respond CALIBRATION and request a second observation.
4. If calibration is not complete, respond CALIBRATION and request sensor calibration.
5. Otherwise APPROVED requires the manager to confirm calibration complete AND MaintenanceAgent to have
   actually reported successful calibration.
Values EQUAL to a limit do not exceed it. Verify every numeric comparison. For DENIED cite an actual offending
measurement and its limit. Never invent readings or calibration evidence.
Return exactly one line: APPROVED: <reason>, DENIED: <reason>, or CALIBRATION: <reason>.'''


class ExecutionDecision(str, Enum):
    APPROVED = "APPROVED"
    DENIED = "DENIED"
    EXECUTED = "EXECUTED"


class SafetyDecision(str, Enum):
    APPROVED = "APPROVED"
    DENIED = "DENIED"
    CALIBRATION = "CALIBRATION"


class ExecutionVerdict(BaseModel):
    decision: ExecutionDecision
    reason: str


class SafetyVerdict(BaseModel):
    decision: SafetyDecision
    reason: str


def parse_decision(text: str, decision_type=ExecutionDecision, *, human: bool = False):
    """Read a leading decision (or typed JSON); never accept a keyword in its reason."""
    text = text.strip()
    if text.startswith("{"):
        try:
            value = json.loads(text)
            return decision_type(value["decision"])
        except (ValueError, KeyError, TypeError):
            return None
    if human:
        if not text.upper().startswith("HUMAN "):
            return None
        text = text[6:].lstrip()
    keyword = text.split(":", 1)[0].strip().upper()
    try:
        return decision_type(keyword)
    except ValueError:
        return None


def normalize_input(messages) -> list[Message]:
    if messages is None:
        return []
    if isinstance(messages, str):
        return [Message("user", [messages])]
    if isinstance(messages, Message):
        return [messages]
    return list(messages)


def participant_messages(messages: Sequence[Message], name: str) -> list[Message]:
    return [message for message in messages if message.role == "assistant" and message.author_name == name
            and message.text.strip()]


def latest_message(messages: Sequence[Message], name: str) -> Message | None:
    return next(iter(reversed(participant_messages(messages, name))), None)


def plain_history(messages: Sequence[Message]) -> list[Message]:
    """Do not replay unmatched tool calls/results into a different actor's provider request."""
    return [Message(message.role, [message.text], author_name=message.author_name)
            for message in messages if message.text.strip() and message.role != "tool"]


@dataclass
class ApprovedTerminationManager:
    participants: tuple[str, ...] = ("MotorsAgent", "NavigatorAgent")
    maximum_iteration_count: int = MAX_ITERATIONS
    approver_name: str = "NavigatorAgent"
    iteration_count: int = 0
    last_agent: str | None = None
    history: list[Message] = field(default_factory=list)

    def __post_init__(self):
        if self.maximum_iteration_count < 1:
            raise ValueError("maximum_iteration_count must be positive")
        if not self.participants or self.approver_name not in self.participants:
            raise ValueError("The approver must be a participant")

    def select_next(self, state: GroupChatState) -> str:
        self.last_agent = self.participants[state.current_round % len(self.participants)]
        self.iteration_count = state.current_round + 1
        return self.last_agent

    def should_terminate(self, messages: list[Message]) -> bool:
        self.history = list(messages)
        if self.iteration_count >= self.maximum_iteration_count:
            return True
        last = next((message for message in reversed(messages)
                     if message.role == "assistant" and message.author_name in self.participants), None)
        if last is None or last.author_name != "MotorsAgent":
            return False
        approval = latest_message(messages, self.approver_name)
        return (parse_decision(last.text) == ExecutionDecision.EXECUTED and approval is not None
                and parse_decision(approval.text, human=self.approver_name == "HumanAgent")
                == ExecutionDecision.APPROVED)


class GatedMotorAgent:
    """SupportsAgentRun adapter: no motor tools before the authoritative actor approves."""

    def __init__(self, agent: Agent, *, approver_name: str, motor_tools: Sequence):
        self.agent = agent
        self.id, self.name, self.description = agent.id, agent.name, agent.description
        self.approver_name = approver_name
        self.motor_tools = list(motor_tools)
        self._last_review = None

    def create_session(self, *, session_id=None):
        return self.agent.create_session(session_id=session_id)

    def get_session(self, service_session_id, *, session_id=None):
        return self.agent.get_session(service_session_id, session_id=session_id)

    def run(self, messages=None, *, stream=False, session=None, function_invocation_kwargs=None, client_kwargs=None):
        messages = normalize_input(messages)
        approval = latest_message(messages, self.approver_name)
        approved = (approval is not None and approval is not self._last_review
                    and parse_decision(approval.text, human=self.approver_name == "HumanAgent")
                    == ExecutionDecision.APPROVED)
        # Consume the grant for this turn, so a later revised proposal cannot reuse it.
        if approval is not None:
            self._last_review = approval
        options = {"response_format": None, "tool_choice": "auto" if approved else "none"}
        if approved:
            instruction = ("Execution phase: the authoritative reviewer approved the most recent proposal. "
                           "Execute that exact reviewed sequence with the available motor tools, not another proposal. "
                           "Only report EXECUTED after the tools actually complete successfully.")
        else:
            schemas = [motor.to_json_schema_spec() for motor in self.motor_tools
                       if hasattr(motor, "to_json_schema_spec")]
            instruction = ("Planning/revision phase: do NOT execute. Motor tools are intentionally unavailable. "
                           "Propose a complete move sequence with actual integer arguments: angle in degrees for "
                           "each turn_left/turn_right and distance in meters for each forward/backward. "
                           "Use the latest authoritative feedback to revise the numeric angles/distances. "
                           "The following schemas are planning reference only, NOT permission to call tools:\n"
                           + json.dumps(schemas))
        return self.agent.run(
            [*messages, Message("user", [instruction])], stream=stream, session=session,
            tools=self.motor_tools if approved else [], options=options,
            function_invocation_kwargs=function_invocation_kwargs, client_kwargs=client_kwargs,
        )


class HumanAgent:
    """A real SupportsAgentRun participant, with normal messages and native streaming."""

    id = "human-agent"
    name = "HumanAgent"
    description = "Human operator: the authoritative approval for the latest proposed motor sequence."

    def __init__(self, read_input: Callable[[str], str] = input):
        self.read_input = read_input

    def create_session(self, *, session_id=None):
        return AgentSession(session_id=session_id)

    def get_session(self, service_session_id, *, session_id=None):
        return AgentSession(service_session_id=service_session_id, session_id=session_id)

    async def _respond(self, messages: list[Message]) -> AgentResponse:
        executed = latest_message(messages, "MotorsAgent")
        if executed is not None and parse_decision(executed.text) == ExecutionDecision.EXECUTED:
            text = "HUMAN APPROVED: sequence approved already."
        else:
            proposal = latest_message(messages, "MotorsAgent")
            prompt = f"\nProposed sequence: {proposal.text if proposal else '(no proposal)'}\nHUMAN: Approve? [y/n, or feedback]: "

            def read_answer():
                try:
                    return self.read_input(prompt)
                except (EOFError, StopIteration):
                    return ""

            answer = await asyncio.to_thread(read_answer)
            answer = (answer or "").strip()
            approved = answer.lower() in {"y", "yes"} and proposal is not None
            if approved:
                text = "HUMAN APPROVED: the proposed move sequence looks good. Execute it now."
            else:
                feedback = answer if answer.lower() not in {"", "n", "no"} else "please revise the proposed move sequence."
                text = f"HUMAN DENIED: {feedback}"
        return AgentResponse(messages=[Message("assistant", [text], author_name=self.name)])

    def run(self, messages=None, *, stream=False, session=None, function_invocation_kwargs=None, client_kwargs=None):
        messages = normalize_input(messages)
        if not stream:
            return self._respond(messages)
        response = None

        async def updates():
            nonlocal response
            response = await self._respond(messages)
            message = response.messages[0]
            yield AgentResponseUpdate(contents=message.contents, role=message.role, author_name=self.name,
                                      message_id=message.message_id)

        return ResponseStream(updates(), finalizer=lambda _: response)


class HumanAgentExecutor(Executor):
    """Native group-chat executor, keeping the console intervention inside the workflow graph."""

    def __init__(self, agent: HumanAgent):
        super().__init__(id=agent.name)
        self.agent = agent
        self.history: list[Message] = []
        self.session = agent.create_session()

    @handler
    async def receive_history(self, message: GroupChatParticipantMessage,
                              ctx: WorkflowContext[GroupChatResponseMessage, AgentResponse | AgentResponseUpdate]):
        self.history.extend(message.messages)

    @handler
    async def request_decision(self, request: GroupChatRequestMessage,
                               ctx: WorkflowContext[GroupChatResponseMessage, AgentResponse | AgentResponseUpdate]):
        if ctx.is_streaming():
            stream = self.agent.run(self.history, stream=True, session=self.session)
            async for update in stream:
                await ctx.yield_output(update)
            response = await stream.get_final_response()
        else:
            response = await self.agent.run(self.history, session=self.session)
            await ctx.yield_output(response)
        self.history.extend(response.messages)
        await ctx.send_message(GroupChatResponseMessage(response.messages[-1]))


class SensorTurnPolicy(ChatMiddleware):
    """Require each sensor on this turn, not results retained in an agent session."""

    def __init__(self, sensor_tools):
        self.sensor_tools = list(sensor_tools)
        self.baseline_length = None

    def begin_turn(self):
        self.baseline_length = None

    async def process(self, context: ChatContext, call_next):
        if self.baseline_length is None:
            self.baseline_length = len(context.messages)
        current = context.messages[self.baseline_length:]
        calls = {content.call_id: content.name for message in current for content in message.contents
                 if content.type == "function_call"}
        attempted = {calls[content.call_id] for message in current for content in message.contents
                     if content.type == "function_result" and content.call_id in calls}
        pending = [sensor for sensor in self.sensor_tools if sensor.name not in attempted]
        context.options = {**(context.options or {}), "tools": pending,
                           "tool_choice": "required" if pending else "none", "response_format": None}
        await call_next()


class FreshSensorAgent:
    """Reset sensor requirements for every real EnvironmentAgent participant turn."""

    def __init__(self, agent: Agent, policy: SensorTurnPolicy):
        self.agent, self.policy = agent, policy
        self.id, self.name, self.description = agent.id, agent.name, agent.description

    def create_session(self, *, session_id=None):
        return self.agent.create_session(session_id=session_id)

    def get_session(self, service_session_id, *, session_id=None):
        return self.agent.get_session(service_session_id, session_id=session_id)

    def run(self, messages=None, *, stream=False, session=None, function_invocation_kwargs=None, client_kwargs=None):
        self.policy.begin_turn()
        instruction = ("Collect a NEW independent observation now. Invoke both sensor tools exactly once for "
                       "this turn; never reuse readings or tool results from earlier turns. Report only the NEW pair.")
        return self.agent.run([*normalize_input(messages), Message("user", [instruction])], stream=stream,
                              session=session, options={"response_format": None},
                              function_invocation_kwargs=function_invocation_kwargs, client_kwargs=client_kwargs)


class InvestigationManager(GroupChatOrchestrator):
    """BaseGroupChatOrchestrator subclass preserving source routing and raw calibration evidence."""

    def __init__(self, participants: Sequence[Executor], *, maximum_iteration_count=MAX_ITERATIONS,
                 sensor_names=("read_temperature", "read_humidity")):
        if maximum_iteration_count < 1:
            raise ValueError("maximum_iteration_count must be positive")
        self.maximum_iteration_count = maximum_iteration_count
        self.observation_count = 0
        self.observations: list[tuple[int, int]] = []
        self.sensor_names = tuple(sensor_names)
        self._seen_sensor_calls: set[str] = set()
        self.calibrated = False
        self.last_agent: str | None = None
        self.raw_history: list[Message] = []
        super().__init__("InvestigationManager", ParticipantRegistry(participants), self.select_next,
                         max_rounds=maximum_iteration_count, termination_condition=self.should_terminate)

    @property
    def history(self):
        return self._get_conversation()

    @property
    def iteration_count(self):
        return self._round_index

    def select_next(self, state: GroupChatState) -> str:
        if self.last_agent is None:
            next_agent = "EnvironmentAgent"
        elif self.last_agent in {"EnvironmentAgent", "MaintenanceAgent"}:
            next_agent = "SafetyAgent"
        elif self.observation_count < 2:
            next_agent = "EnvironmentAgent"
        elif not self.calibrated:
            next_agent = "MaintenanceAgent"
        else:
            next_agent = "EnvironmentAgent"
        self.last_agent = next_agent
        return next_agent

    def should_terminate(self, messages: list[Message]) -> bool:
        if self.iteration_count >= self.maximum_iteration_count:
            return True
        last = latest_message(messages, "SafetyAgent")
        decision = parse_decision(last.text, SafetyDecision) if last else None
        return self.last_agent == "SafetyAgent" and (
            decision == SafetyDecision.DENIED or (self.calibrated and self.observation_count >= 2
                                                  and decision == SafetyDecision.APPROVED))

    def _process_participant_response(self, response: AgentExecutorResponse | GroupChatResponseMessage):
        messages = super()._process_participant_response(response)
        self.raw_history.extend(messages)
        if self.last_agent == "EnvironmentAgent":
            calls = {content.call_id: content.name for message in messages for content in message.contents
                     if content.type == "function_call"}
            values = {}
            for message in messages:
                for content in message.contents:
                    if (content.type != "function_result" or content.exception is not None
                            or content.call_id in self._seen_sensor_calls):
                        continue
                    name = calls.get(content.call_id)
                    if name not in self.sensor_names:
                        continue
                    try:
                        value = json.loads(content.result)
                    except (ValueError, TypeError):
                        continue
                    if type(value) is int:
                        values[name] = (value, content.call_id)
            if len(self.sensor_names) == 2 and all(name in values for name in self.sensor_names):
                self.observations.append(tuple(values[name][0] for name in self.sensor_names))
                self.observation_count += 1
            else:
                messages = [Message("assistant", ["Unverified Environment report: no NEW successful integer "
                                                   "temperature/humidity tool pair was collected this turn."],
                                    author_name="EnvironmentAgent")]
            self._seen_sensor_calls.update(call_id for call_id, name in calls.items() if name in self.sensor_names)
        elif self.last_agent == "MaintenanceAgent":
            # Inspect this turn's raw evidence before the native orchestrator strips tool envelopes.
            self.calibrated = any(content.type == "function_result" and content.exception is None
                                  and bool(content.result) for message in messages for content in message.contents)
        evidence = "\n".join(f"Observation {index}: temperature = {temperature} Celsius; humidity = {humidity}%."
                             for index, (temperature, humidity) in enumerate(self.observations, 1))
        state = Message("user", [f"InvestigationManager state:\nCompleted observations: {self.observation_count}\n"
                                 f"Calibration completed: {self.calibrated}\nVerified sensor observations:\n{evidence}"],
                        author_name="InvestigationManager")
        return [*plain_history(messages), state]

    async def on_checkpoint_save(self):
        return {"observation_count": self.observation_count, "observations": self.observations,
                "sensor_names": self.sensor_names, "seen_sensor_calls": sorted(self._seen_sensor_calls),
                "calibrated": self.calibrated,
                "last_agent": self.last_agent, "round_index": self._round_index,
                "raw_history": [message.to_dict() for message in self.raw_history],
                "conversation": [message.to_dict() for message in self._get_conversation()]}

    async def on_checkpoint_restore(self, state):
        self.observation_count = state.get("observation_count", 0)
        self.observations = [tuple(observation) for observation in state.get("observations", [])]
        self.sensor_names = tuple(state.get("sensor_names", self.sensor_names))
        self._seen_sensor_calls = set(state.get("seen_sensor_calls", []))
        self.calibrated = state.get("calibrated", False)
        self.last_agent = state.get("last_agent")
        self._round_index = state.get("round_index", 0)
        self.raw_history = [Message.from_dict(message) for message in state.get("raw_history", [])]
        self._full_conversation = [Message.from_dict(message) for message in state.get("conversation", [])]


async def read_temperature() -> int:
    """Use thermal sensors to detect abnormal heat levels (simulated integer Celsius)."""
    value = random.randrange(-20, 100)
    print(f"SENSORS: READING Temperature: {value} Celsius degrees.")
    await asyncio.sleep(0.5)
    return value


async def read_humidity() -> int:
    """Read local humidity (simulated integer percent)."""
    value = random.randrange(0, 100)
    print(f"SENSORS: READING Humidity: {value}%.")
    await asyncio.sleep(0.5)
    return value


async def calibrate_sensors() -> str:
    """Calibrate all sensors on the robot car (simulation; random readings are unchanged)."""
    print("MAINTENANCE: CALIBRATING sensors...")
    await asyncio.sleep(0.5)
    return "All sensors have been calibrated."


@dataclass
class GroupChatSample:
    workflow: Workflow
    manager: ApprovedTerminationManager | InvestigationManager
    output_agent: str
    prompt: str

    @property
    def last_response(self):
        message = latest_message(self.manager.history, self.output_agent)
        return message.text if message else ""


def build_execution(sdk, *, provider="openai", hitl=False, read_input=input,
                    maximum_iteration_count=MAX_ITERATIONS, motor_tools=None, response_format=None) -> GroupChatSample:
    client = create_completion_client(sdk, provider)
    approver = "HumanAgent" if hitl else "NavigatorAgent"
    motors = GatedMotorAgent(
        Agent(client, name="MotorsAgent", instructions=HITL_MOTOR_INSTRUCTIONS if hitl else MOTOR_INSTRUCTIONS,
              tools=[], default_options={"response_format": None}),
        approver_name=approver, motor_tools=create_motor_tools() if motor_tools is None else motor_tools,
    )
    navigator = Agent(client, name="NavigatorAgent", instructions=NAVIGATOR_INSTRUCTIONS,
                      default_options={"response_format": response_format})
    participants = [motors, navigator]
    if hitl:
        participants.append(HumanAgentExecutor(HumanAgent(read_input)))
    manager = ApprovedTerminationManager(tuple(participant.name if not isinstance(participant, Executor)
                                              else participant.id for participant in participants),
                                         maximum_iteration_count, approver)
    workflow = GroupChatBuilder(participants=participants, selection_func=manager.select_next,
                                termination_condition=manager.should_terminate, max_rounds=maximum_iteration_count,
                                output_from=["MotorsAgent"], intermediate_output_from="all_other").build()
    return GroupChatSample(workflow, manager, "MotorsAgent", MISSION)


def build_investigation(sdk, *, provider="openai", maximum_iteration_count=MAX_ITERATIONS,
                        sensor_tools=None, calibration_tool=None, response_format=None) -> GroupChatSample:
    client = create_completion_client(sdk, provider)
    sensors = [tool(read_temperature), tool(read_humidity)] if sensor_tools is None else list(sensor_tools)
    sensor_policy = SensorTurnPolicy(sensors)
    sensor_client = create_completion_client(sdk, provider, middleware=[sensor_policy])
    agents = [
        FreshSensorAgent(Agent(sensor_client, name="EnvironmentAgent", instructions=ENVIRONMENT_INSTRUCTIONS,
                               tools=sensors, default_options={"response_format": None}), sensor_policy),
        Agent(client, name="MaintenanceAgent", instructions=MAINTENANCE_INSTRUCTIONS,
              tools=[tool(calibrate_sensors)] if calibration_tool is None else [calibration_tool],
              default_options={"response_format": None}),
        Agent(client, name="SafetyAgent", instructions=SAFETY_INSTRUCTIONS,
              default_options={"response_format": response_format}),
    ]
    executors = [AgentExecutor(agent, id=agent.name) for agent in agents]
    manager = InvestigationManager(executors, maximum_iteration_count=maximum_iteration_count,
                                   sensor_names=tuple(sensor.name for sensor in sensors))
    workflow = GroupChatBuilder(participants=executors, orchestrator=manager, output_from=["SafetyAgent"],
                                intermediate_output_from="all_other").build()
    return GroupChatSample(workflow, manager, "SafetyAgent", INVESTIGATION_PROMPT)


async def run_streaming(sample: GroupChatSample):
    """Consume native workflow events; actor routing stays entirely in GroupChatBuilder."""
    stream = sample.workflow.run(sample.prompt, stream=True, include_status_events=True)
    last_author = None
    failure = None
    async for event in stream:
        if event.type in {"intermediate", "output"} and isinstance(event.data, AgentResponseUpdate):
            author = event.data.author_name or event.executor_id
            if author != last_author:
                print(f"\n[{author}]: ", end="", flush=True)
                last_author = author
            print(event.data.text, end="", flush=True)
        elif event.type == "executor_completed":
            print(f"\n[EXECUTOR] {event.executor_id} completed.")
        elif event.type in {"error", "executor_failed"}:
            failure = event.data
    result = await stream.get_final_response()
    if failure is not None:
        raise RuntimeError(f"[WORKFLOW ERROR] {failure}")
    print(f"\n[WORKFLOW OUTPUT] {sample.last_response}")
    if sample.manager.iteration_count >= sample.manager.maximum_iteration_count:
        print(f"[TURN LIMIT] Stopped after {sample.manager.maximum_iteration_count} participant turns.")
    return result


async def run_example(project: str, *, provider="openai", read_input=input):
    if project not in {"GroupChatOrchestration", "GroupChatOrchestrationWithCustomManager",
                       "GroupChatOrchestrationWithHitl"}:
        raise ValueError(f"Unknown group chat project: {project}")
    sdk = create_sdk_client(provider)
    try:
        sample = (build_investigation(sdk, provider=provider)
                  if project == "GroupChatOrchestrationWithCustomManager" else
                  build_execution(sdk, provider=provider, hitl=project == "GroupChatOrchestrationWithHitl",
                                  read_input=read_input))
        return await run_streaming(sample)
    finally:
        await sdk.close()


def provider_from_cli(argv=None):
    parser = argparse.ArgumentParser(description="Chapter 13 native group chat (10 participant turns maximum).")
    parser.add_argument("--provider", choices=("openai", "azure"), default="openai")
    return parser.parse_args(argv).provider
