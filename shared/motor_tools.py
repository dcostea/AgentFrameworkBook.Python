"""Robot car motor tools - Python mirror of the C# AITools/MotorTools.cs.

Each function is an async "basic move" the robot car can perform. The
docstrings and parameter annotations act as the tool descriptions that the
agent reads, mirroring the [Description] attributes in the C# version.
"""

import asyncio
from datetime import datetime
from typing import Annotated

from .color_helper import Color, print_colored_line

_DELAY_SECONDS = 1.0  # mock an action taking ~1s, like the C# Delay = 1000ms


def _timestamp() -> str:
	return datetime.now().strftime("%H:%M:%S:%f")[:-3]


async def backward(
	distance: Annotated[int, "The distance (in meters) to move the robot car backward."],
) -> str:
	"""Basic command: Moves the robot car backward."""
	print_colored_line(f"[{_timestamp()}] MOTORS: Backward: {distance}m", Color.BLUE)
	await asyncio.sleep(_DELAY_SECONDS)
	return f"moved backward for {distance} meters."


async def forward(
	distance: Annotated[int, "The distance (in meters) to move the robot car forward."],
) -> str:
	"""Basic command: Moves the robot car forward."""
	print_colored_line(f"[{_timestamp()}] MOTORS: Forward: {distance}m", Color.BLUE)
	await asyncio.sleep(_DELAY_SECONDS)
	return f"moved forward for {distance} meters."


async def stop() -> str:
	"""Basic command: Stops the robot car."""
	print_colored_line(f"[{_timestamp()}] MOTORS: Stop", Color.BLUE)
	await asyncio.sleep(_DELAY_SECONDS)
	return "stopped."


async def turn_left(
	angle: Annotated[int, "The angle (in degrees) to turn the robot car anticlockwise."],
) -> str:
	"""Basic command: Turns the robot car anticlockwise."""
	print_colored_line(f"[{_timestamp()}] MOTORS: TurnLeft: {angle}deg", Color.BLUE)
	await asyncio.sleep(_DELAY_SECONDS)
	return f"turned anticlockwise {angle} degrees."


async def turn_right(
	angle: Annotated[int, "The angle (in degrees) to turn the robot car clockwise."],
) -> str:
	"""Basic command: Turns the robot car clockwise."""
	print_colored_line(f"[{_timestamp()}] MOTORS: TurnRight: {angle}deg", Color.BLUE)
	await asyncio.sleep(_DELAY_SECONDS)
	return f"turned clockwise {angle} degrees."


def as_ai_tools() -> list:
	"""Return all motor tools - mirror of the C# MotorTools.AsAITools()."""
	return [turn_right, turn_left, stop, forward, backward]
