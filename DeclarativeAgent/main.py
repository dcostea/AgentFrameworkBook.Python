"""DeclarativeAgent - Python mirror of the C# DeclarativeAgent/Program.cs.

The C# sample loads an agent definition from `Agents/MotorsAgent.yaml` and
creates it with a `ChatClientPromptAgentFactory`, wiring up the `backward`
function tool.

NOTE: In the current Python Agent Framework, the declarative `AgentFactory`
builds its own client (e.g. from Foundry credentials) and does not expose a
documented way to inject a pre-built OpenAIChatClient together with local
function tools, the way C#'s `ChatClientPromptAgentFactory` does. To keep this
sample reliable and provider-consistent with the others, we read the same YAML
file at runtime and construct the agent from its values, registering the
`backward` tool explicitly. The YAML stays the single source of truth for the
name, description, instructions and model options.
"""

import asyncio
import sys
from pathlib import Path

import yaml

# Allow importing the sibling "shared" package (Agents/shared).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import create_chat_client  # noqa: E402
from shared.motor_tools import backward  # noqa: E402

_YAML_PATH = Path(__file__).resolve().parent / "agents" / "motors_agent.yaml"


async def main() -> None:
	spec = yaml.safe_load(_YAML_PATH.read_text(encoding="utf-8"))
	options = spec.get("model", {}).get("options", {})

	# Create the agent from the YAML definition, registering the backward tool.
	agent = create_chat_client().create_agent(
		name=spec["name"],
		description=spec.get("description"),
		instructions=spec["instructions"],
		temperature=options.get("temperature"),
		top_p=options.get("topP"),
		tools=[backward],
	)

	query = """
Complex command:
"There is a tree directly in front of the car. Avoid it and then return to the original path."
"""
	response = await agent.run(query)
	print(response.text)


if __name__ == "__main__":
	asyncio.run(main())
