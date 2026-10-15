"""AgentWithChatMessagesAndMultiModal - Python mirror of the C#
AgentWithChatMessagesAndMultiModal/Program.cs.

Sends a message that mixes text with binary image content (and shows the
commented-out URI / audio alternatives, just like the C# sample).
"""

import asyncio
import sys
from pathlib import Path

# Allow importing the sibling "shared" package (Agents/shared).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_framework import Content, Message  # noqa: E402

from shared import create_chat_client  # noqa: E402

_DATA_DIR = Path(__file__).resolve().parent / "data"


async def main() -> None:
	agent = create_chat_client().create_agent(
		instructions="""
You are an AI assistant controlling a robot car capable of performing basic moves: forward, backward, turn left, turn right, and stop.
You have to break down the provided complex commands into the basic moves you know.
Use a JSON array like [move1, move2, move3] for the response.
Respond only with the moves and their parameters (angle or distance), without any additional explanations.
""",
	)

	audio_bytes = (_DATA_DIR / "Task.mp3").read_bytes()  # noqa: F841
	image_bytes = (_DATA_DIR / "Map.png").read_bytes()

	conversation = [
		# new(Role.USER, [
		#     TextContent("Look at the image of the map and proceed safely."),
		#     UriContent("http://apexcode.ro/path.jpg", "image/jpeg"),
		# ]),
		# new(Role.USER, [
		#     TextContent("Look at the image of the map and proceed safely."),
		#     UriContent("http://apexcode.ro/road_and_tree.png", "image/jpeg"),
		# ]),
		Message(
			role="user",
			contents=[
				Content.from_text("Look at the image of the map and proceed safely."),
				Content.from_data(data=image_bytes, media_type="image/jpeg"),
			],
		),
		# new(Role.USER, [
		#     UriContent("https://apexcode.ro/task.mp3", "audio/mpeg"),
		# ]),
		# new(Role.USER, [
		#     DataContent(audio_bytes, "audio/mpeg"),
		# ]),
	]

	response = await agent.run(conversation)
	print(response.text)


if __name__ == "__main__":
	asyncio.run(main())
