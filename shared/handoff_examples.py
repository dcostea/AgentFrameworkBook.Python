"""Chapter 14 sensor-driven handoffs with a single simulated motor authority."""

import asyncio
from dataclasses import dataclass
from enum import Enum
import json
from random import choice, randrange
from typing import Annotated

from agent_framework import Agent, AgentMiddleware, FunctionMiddleware, Message, WorkflowViz, tool
from agent_framework.orchestrations import HandoffBuilder
from pydantic import Field

from .chat_clients import create_completion_client
from .color_helper import Color, print_colored_line
from .tool_examples import create_motor_tools

PROMPT = """# MISSION COMMAND: Exploration Trip

There is a tree directly in front of the car. Avoid it and then come back to the original path. The distance to the tree is 50 meters."""
ROUTING_REASONS = {
    "FireDetectorAgent": "Transfer if the temperature is above 60°C (possible fire).",
    "RainDetectorAgent": "Transfer if the droplet level is High (rain detected).",
    "MotorsAgent": "Transfer if conditions are safe (temperature normal and no heavy rain).",
}
ENVIRONMENT_INSTRUCTIONS = """## PERSONA
You are the EnvironmentAgent. Read the weather conditions using SensorTools, including
read_temperature and read_droplet_level, then hand off to exactly one agent.
Invoke the selected handoff_to_* tool to transfer control. Saying that you are
handing off in ordinary text does not transfer control."""
MOTOR_INSTRUCTIONS = """## PERSONA
You are the MotorsAgent, responsible for car movements using MotorTools.
The permitted movements are: move forward, turn left, turn right, and stop.

## ACTIONS
Completed sensor and specialist tool results are shared as plain-text assistant
messages headed "Completed tool results". They are real evidence, not hypothetical
instructions, even though they are not tool-role messages.
- If FireDetectorAgent or RainDetectorAgent handed control to you, or any completed
  evidence contains "Fire alarm sounded", "Water sprinkler started", or "Wipers have started":
  invoke the stop tool and summarize the hazard. Do not execute the obstacle-avoidance
  mission or invoke forward/backward/turn tools after a hazard handoff.
- Only a safe handoff directly from EnvironmentAgent permits carrying out the user's
  mission with the available movement tools, then summarizing."""


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


@dataclass
class SensorReadings:
    temperature: int | None = None
    infrared_radiation: int | None = None
    humidity: int | None = None
    droplet_level: DropletLevel | None = None
    wind_speed: int | None = None
    wind_direction: Direction | None = None

    @classmethod
    def scenario(cls, name: str):
        if name == "random":
            return cls()
        if name not in {"safe", "fire", "rain"}:
            raise ValueError(f"Unknown sensor scenario: {name}")
        return cls(temperature=85 if name == "fire" else 25, infrared_radiation=80 if name == "fire" else 10,
                   humidity=80 if name == "rain" else 40,
                   droplet_level=DropletLevel.HIGH if name == "rain" else DropletLevel.NONE,
                   wind_speed=12, wind_direction=Direction.NORTH)


class EnvironmentTools:
    def __init__(self, readings: SensorReadings | None = None, *, delay: float = 0.5):
        self.readings = readings or SensorReadings()
        self.delay = delay

    async def _read(self, field, value):
        print_colored_line(f"SENSORS: READING {field}: {value}", Color.BLUE)
        await asyncio.sleep(self.delay)
        return value

    async def read_temperature(self) -> int:
        """Use thermal sensors to detect abnormal heat levels."""
        value = self.readings.temperature
        return await self._read("Temperature", value if value is not None else randrange(-20, 100))

    async def read_infrared_radiation(self) -> int:
        """Confirm the presence of flames via IR sensors."""
        value = self.readings.infrared_radiation
        return await self._read("Infrared Radiation", value if value is not None else randrange(100))

    async def read_humidity(self) -> int:
        """Check local humidity as a precursor to rain detection (fires often reduce local moisture)."""
        value = self.readings.humidity
        return await self._read("Humidity", value if value is not None else randrange(100))

    async def read_droplet_level(self) -> DropletLevel:
        """Measure the presence and intensity of raindrops on surfaces."""
        value = self.readings.droplet_level
        return await self._read("Droplet Level", value if value is not None else choice(list(DropletLevel)))

    async def read_wind_speed(self) -> int:
        """Read wind speed in kmph."""
        value = self.readings.wind_speed
        return await self._read("Wind speed", value if value is not None else randrange(100))

    async def read_wind_direction(self) -> Direction:
        """Read wind direction, such as North or NorthWest."""
        value = self.readings.wind_direction
        return await self._read("Wind direction", value if value is not None else choice(list(Direction)))

    def tools(self):
        return [tool(function) for function in (self.read_droplet_level, self.read_humidity, self.read_infrared_radiation,
                                               self.read_temperature, self.read_wind_direction, self.read_wind_speed)]


class DetectorTools:
    def __init__(self, *, fire_delay: float = 0.5, rain_delay: float = 1.0):
        self.fire_delay = fire_delay
        self.rain_delay = rain_delay

    async def sound_alarm(self) -> str:
        """Trigger an audible or visual alarm to warn nearby humans or systems."""
        print_colored_line("FIRE DETECTOR: Sounding fire alarm!", Color.BLUE)
        await asyncio.sleep(self.fire_delay)
        return "Fire alarm sounded."

    async def start_water_sprinkle(self) -> str:
        """Activate sprinklers to suppress the fire."""
        print_colored_line("FIRE DETECTOR: Starting water sprinkle.", Color.BLUE)
        await asyncio.sleep(self.fire_delay)
        return "Water sprinkler started — fire suppression in progress."

    async def stop_water_sprinkle(self) -> str:
        """Stop sprinklers when fire is extinguished."""
        print_colored_line("FIRE DETECTOR: Stopping water sprinkle.", Color.BLUE)
        await asyncio.sleep(self.fire_delay)
        return "Water sprinkler stopped."

    async def start_wipers(self, droplet_level: Annotated[DropletLevel, Field(description="Droplet level detected by the sensor.")]) -> str:
        """Start wipers for the detected droplet level, from light drizzle to heavy rain."""
        print_colored_line(f"RAIN DETECTOR: Starting wipers for droplet level: {droplet_level.value}.", Color.BLUE)
        await asyncio.sleep(self.rain_delay)
        return f"Wipers have started for droplet level: {droplet_level.value}."

    async def stop_wipers(self) -> str:
        """Turn off wipers once no droplets are detected for a predefined duration."""
        print_colored_line("RAIN DETECTOR: Stopping wipers.", Color.BLUE)
        await asyncio.sleep(self.rain_delay)
        return "Wipers have stopped."

    def fire_tools(self):
        return [tool(function) for function in (self.start_water_sprinkle, self.stop_water_sprinkle, self.sound_alarm)]

    def rain_tools(self):
        return [tool(function) for function in (self.stop_wipers, self.start_wipers)]


_EVIDENCE_STATE = "handoff_tool_evidence"


class RecordToolEvidence(FunctionMiddleware):
    async def process(self, context, call_next):
        await call_next()
        if context.session is not None and not context.function.name.startswith("handoff_to_"):
            result = context.result
            if isinstance(result, list):
                result = " ".join(content.text for content in result if content.type == "text")
            if not isinstance(result, str):
                result = json.dumps(result, default=str)
            context.session.state.setdefault(_EVIDENCE_STATE, []).append(f"{context.function.name}: {result}")


class ToolEvidenceMiddleware(AgentMiddleware):
    """Expose plain-text tool evidence because Python handoffs strip tool-control messages."""

    async def process(self, context, call_next):
        if context.session is not None:
            context.session.state[_EVIDENCE_STATE] = []
        await call_next()

        def add_evidence(response):
            evidence = context.session.state.pop(_EVIDENCE_STATE, []) if context.session is not None else []
            return self.add_evidence(response, context.agent.name, evidence)

        if context.stream:
            context.stream_result_hooks.append(add_evidence)
        else:
            add_evidence(context.result)

    @staticmethod
    def add_evidence(response, author, recorded=()):
        calls = {content.call_id: content.name for message in response.messages for content in message.contents
                 if content.type == "function_call"}
        evidence = list(recorded)
        for message in response.messages:
            if message.role == "assistant" and not message.author_name:
                message.author_name = author
            for content in message.contents:
                if content.type != "function_result" or content.result is None:
                    continue
                name = calls.get(content.call_id, "tool")
                if name.startswith("handoff_to_"):
                    continue
                result = content.result if isinstance(content.result, str) else json.dumps(content.result, default=str)
                entry = f"{name}: {result}"
                if entry not in evidence:
                    evidence.append(entry)
        if evidence:
            # Keep the final handoff result in last position for the framework's handoff detector.
            response.messages.insert(max(len(response.messages)-1, 0), Message("assistant", [
                "Completed tool results:\n" + "\n".join(evidence),
            ], author_name=author))
        return response


def motor_completed(messages: list[Message]) -> bool:
    return bool(messages and messages[-1].role == "assistant" and messages[-1].author_name == "MotorsAgent")


def build_agents(sdk, *, provider="openai", reasons=False, readings=None, sensor_delay=0.5,
                 fire_delay=0.5, rain_delay=1.0):
    client = create_completion_client(sdk, provider)
    sensors = EnvironmentTools(readings, delay=sensor_delay)
    detectors = DetectorTools(fire_delay=fire_delay, rain_delay=rain_delay)
    env_instructions = ENVIRONMENT_INSTRUCTIONS
    if not reasons:
        env_instructions += "\n\n## ROUTING\nChoose the target agent based on the sensor readings:\n" + "\n".join(
            f"- {target}: {reason}" for target, reason in ROUTING_REASONS.items())
    specifications = [
        ("EnvironmentAgent", env_instructions, sensors.tools()),
        ("FireDetectorAgent", """You are the FireDetectorAgent. You were invoked because the temperature is dangerously high.
Call sound_alarm and start_water_sprinkle. After both tools finish, invoke the
handoff_to_MotorsAgent tool so MotorsAgent stops the car. Do not finish with a text-only
handoff announcement: ordinary text does not transfer control.""", detectors.fire_tools()),
        ("RainDetectorAgent", """You are the RainDetectorAgent. Handle any rain condition using the available tools.
After the rain tools finish, invoke the handoff_to_MotorsAgent tool so MotorsAgent stops
the car. Do not finish with a text-only handoff announcement: ordinary text does not
transfer control.""", detectors.rain_tools()),
        ("MotorsAgent", MOTOR_INSTRUCTIONS, create_motor_tools()),
    ]
    return {
        name: Agent(client=client, id=name, name=name, description=f"{name} for the robot car.", instructions=instructions,
                    tools=functions, middleware=[ToolEvidenceMiddleware(), RecordToolEvidence()],
                    require_per_service_call_history_persistence=True)
        for name, instructions, functions in specifications
    }


def build_workflow(sdk, *, provider="openai", reasons=False, readings=None, sensor_delay=0.5,
                   fire_delay=0.5, rain_delay=1.0):
    agents = build_agents(sdk, provider=provider, reasons=reasons, readings=readings, sensor_delay=sensor_delay,
                          fire_delay=fire_delay, rain_delay=rain_delay)
    environment, fire, rain, motors = (agents[n] for n in ("EnvironmentAgent", "FireDetectorAgent", "RainDetectorAgent", "MotorsAgent"))
    builder = HandoffBuilder(name="RobotCarHandoff", participants=list(agents.values()), output_from=[motors],
                             termination_condition=motor_completed).with_start_agent(environment)
    if reasons:
        for target in (fire, rain, motors):
            builder.add_handoff(environment, [target], description=ROUTING_REASONS[target.name])
    else:
        builder.add_handoff(environment, [fire, rain, motors])
    builder.add_handoff(fire, [motors], description="Transfer control after fire suppression so MotorsAgent stops the car.")
    builder.add_handoff(rain, [motors], description="Transfer control after rain handling so MotorsAgent stops the car.")
    return builder.build()


async def run_workflow(workflow, prompt=PROMPT):
    print(WorkflowViz(workflow).to_mermaid())
    response_stream = workflow.run(prompt, stream=True, include_status_events=True)
    async for event in response_stream:
        if event.type == "handoff_sent":
            print_colored_line(f"[HANDOFF] {event.data.source} → {event.data.target}", Color.CYAN)
        elif event.type == "executor_completed":
            print_colored_line(f"[EXECUTOR] {event.executor_id} completed.", Color.WHITE)
        elif event.type == "output":
            data = event.data
            if hasattr(data, "contents"):
                if data.text:
                    print(data.text, end="", flush=True)
            else:
                messages = data if isinstance(data, list) else getattr(data, "messages", [])
                text = messages[-1].text if messages else getattr(data, "text", str(data))
                print_colored_line(f"\n[WORKFLOW OUTPUT] {text}", Color.YELLOW)
        elif event.type in {"error", "executor_failed"}:
            print_colored_line(f"[WORKFLOW ERROR] {event.data}", Color.RED)
    result = await response_stream.get_final_response()
    output_text = "".join(getattr(output, "text", "") for output in result.get_outputs() if hasattr(output, "contents"))
    if output_text:
        print_colored_line(f"\n[WORKFLOW OUTPUT] {output_text}", Color.YELLOW)
    for request in result.get_request_info_events():
        print_colored_line(f"[WORKFLOW PAUSED] {request.executor_id or request.source_executor_id}: handoff was not completed.", Color.YELLOW)
    return result
