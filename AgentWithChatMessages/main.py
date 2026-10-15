"""AgentWithChatMessages - Python mirror of the C# AgentWithChatMessages/Program.cs.

Instead of a single string, the agent runs against a list of chat messages
that already contains a prior user/assistant exchange.
"""

import asyncio
import sys
from pathlib import Path

# Allow importing the sibling "shared" package (Agents/shared).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_framework import Message  # noqa: E402

from shared import create_chat_client  # noqa: E402


async def main() -> None:
	agent = create_chat_client().create_agent(
		instructions="""
You are an AI assistant controlling a robot car capable of performing basic moves: forward, backward, turn left, turn right, and stop.
You have to break down the provided complex commands into the basic moves you know.
Use a JSON array like [move1, move2, move3] for the response.
Respond only with the moves and their parameters (angle or distance), without any additional explanations.
""",
	)

	conversation = [
		Message(
			role="user",
			contents=["""
Complex command:
"There is a tree directly in front of the car. Avoid it and then return to the original path.
"""],
		),
		Message(
			role="assistant",
			contents=["""
```json
[{"move": "stop"}, {"move": "turn", "direction": "right", "angle": 90}, {"move": "forward", "distance": "5" }, {"move": "turn", "direction": "left", "angle": 90}, {"move": "forward", "distance": "5"}, {"move": "turn", "direction": "left", "angle": 90}, {"move": "forward", "distance": "5"}, {"move": "turn", "direction": "right", "angle": 90}]
```
"""],
		),
		Message(
			role="user",
			contents=["""
What was your second last basic move?
"""],
		),
	]

	response = await agent.run(conversation)
	print(response.text)

	conversation.append(Message(role="assistant", contents=[response.text]))


if __name__ == "__main__":
	asyncio.run(main())
