"""Weather, motors and auditor share real cosine-ranked User_1 memory.

Requires OPENAI_EMBEDDING_MODEL_ID or AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
with a model supporting 1536 dimensions (e.g. text-embedding-3-small).
Embeddings use the same caller-owned SDK client/credentials as chat.
The vector store remains in memory; no database service is required.
"""

import argparse
import asyncio
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework import Agent, InMemoryHistoryProvider, Message
from shared.chat_clients import create_completion_client, create_sdk_client
from shared.memory_providers import (
    ChatHistoryMemoryProvider, InMemoryVectorStore, OpenAIEmbeddingGenerator,
    ROBOT_INSTRUCTIONS, ROBOT_PROMPTS, embedding_model_for, print_history, run_prompts,
)

DATA = Path(__file__).resolve().parent / "Data"
WEATHER_INSTRUCTIONS = "You are an AI assistant collecting weather information.\nRespond with echo of collected information."
AUDITOR_INSTRUCTIONS = "You are an AI auditor observing the robot car moves.\nYou resume explaining briefly the robot car moves."


def build_agents(client, embeddings, store=None):
    memory = ChatHistoryMemoryProvider(
        store if store is not None else InMemoryVectorStore(embeddings.dimensions), embeddings,
    )
    history = InMemoryHistoryProvider("conversation")
    return tuple(Agent(
        client=client, name=name, instructions=instructions, description=description,
        context_providers=[memory, history],
    ) for name, instructions, description in (
        ("WeatherAgent", WEATHER_INSTRUCTIONS, "An agent that feeds weather information."),
        ("MotorsAgent", ROBOT_INSTRUCTIONS, "An agent that assists a robot with the basic moves."),
        ("RobotCarAgent", AUDITOR_INSTRUCTIONS, "An auditor agent that observes the robot car moves."),
    ))


async def main(provider="openai"):
    model = embedding_model_for(provider)
    async with create_sdk_client(provider) as sdk:
        embeddings = OpenAIEmbeddingGenerator(sdk, model, dimensions=1536)
        weather, motors, auditor = build_agents(create_completion_client(sdk, provider), embeddings)
        session = weather.create_session()
        report = Message("user", [DATA.joinpath("weather-report.txt").read_text(encoding="utf-8")],
                         message_id=str(uuid4()), author_name="System")
        print(f"USER: {report.text}")
        response = await weather.run(report, session=session)
        print(f"ASSISTANT: {response.text}")
        await run_prompts(motors, session, (*ROBOT_PROMPTS, "Today is 2nd of June, 2PM. What is the temperature?"))
        print_history(session.state["conversation"]["messages"])
        response = await auditor.run("Audit the actions performed by the agents.", session=session)
        print(f"AUDITOR: {response.text}")
        print("Vector memory is in process only and searches across sessions for User_1.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openai", "azure"), default="openai")
    asyncio.run(main(parser.parse_args().provider))
