"""AgentWithStructuredOutput - Python mirror of the C#
AgentWithStructuredOutput/Program.cs.

Configures the agent with options (temperature, max tokens) and a Pydantic
response_format so the model returns typed, structured JSON.
"""

import asyncio
import sys
from pathlib import Path

# Allow importing the sibling "shared" package (Agents/shared).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel  # noqa: E402

from shared import create_chat_client  # noqa: E402


class StepItem(BaseModel):
	move: str
	value: str | None = None


class StepsResponse(BaseModel):
	steps: list[StepItem]


async def main() -> None:
	agent = create_chat_client().create_agent(
		name="RobotCarAgent",
		description="An agent that assists a robot with the basic moves.",
		instructions="""
You are an AI assistant controlling a robot car capable of performing basic moves: forward, backward, turn left, turn right, and stop.
You have to break down the provided complex commands into the basic moves you know.
Respond only with the moves and their parameters (angle or distance), without any additional explanations.
""",
		temperature=0.4,
		max_tokens=300,
		response_format=StepsResponse,
	)

	query = """
Complex command:
"There is a tree directly in front of the car. Avoid it and then return to the original path."
"""
	response = await agent.run(query)

	if response.value:
		print(response.value.model_dump_json(indent=2))
	else:
		print(f"Failed to parse response: {response.text}")


if __name__ == "__main__":
	asyncio.run(main())
