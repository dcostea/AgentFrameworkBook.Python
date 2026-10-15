"""MCP robot simulations and client demonstrations for chapter 08."""

import asyncio
import logging
import os
from pathlib import Path
import sys
from typing import Annotated
from urllib.parse import unquote

import anyio
from mcp import types
from mcp.client.stdio import StdioServerParameters
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.prompts.base import UserMessage
from pydantic import Field

MOTOR_DELAY = 0.1
DIAGNOSTIC_DELAY = 2.0
DEFAULT_PORT = 3001
BIO_URI = "resource://mcp/bio"
GREET_TEMPLATE = "resource://mcp/greet/{name}"
ACTION = "There is a tree directly in front of the car. Avoid it and then return to the original path."
MESSAGE_PROMPT = f'## Context\nComplex command:\n"{ACTION}"'
STRING_PROMPT = f'## Context\nComplex command: \n"{ACTION}"'
DIAGNOSTIC_RESULT = "Diagnostics complete. All 4 motors passed."
AGENT_INSTRUCTIONS = """You are an AI assistant controlling a robot car capable of performing basic moves: forward, backward, turn left, turn right, and stop.
You have to break down the provided complex commands into the basic moves you know. Run the basic moves using the corresponding tools.
Respond only with the moves and their parameters (angle or distance), without any additional explanations."""

Distance = Annotated[int, Field(strict=True, ge=-(2**31), le=2**31 - 1,
                               description="The distance (in meters) to move the robot car.")]
Angle = Annotated[int, Field(strict=True, ge=-(2**31), le=2**31 - 1,
                            description="The angle (in ° / degrees) to turn the robot car.")]


def port_number(value):
    port = int(value)
    if not 1 <= port <= 65535:
        raise ValueError("MCP port must be between 1 and 65535")
    return port


def configured_port():
    return port_number(os.environ.get("MCP_PORT", str(DEFAULT_PORT)))


def http_url(port=DEFAULT_PORT):
    return f"http://127.0.0.1:{port_number(port)}/mcp"


def stdio_parameters(client_file):
    server = Path(client_file).resolve().parent.parent / "MCPServerWithStdio" / "main.py"
    return StdioServerParameters(
        command=sys.executable, args=["-u", str(server)],
        env={"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"},
    )


def configure_server_logging():
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")


class MotorsServer(FastMCP):
    """FastMCP plus the pinned SDK's low-level, experimental Tasks adapter."""

    def __init__(self, *, tasks_enabled=False, host="127.0.0.1", port=DEFAULT_PORT,
                 motor_delay=MOTOR_DELAY, diagnostic_delay=DIAGNOSTIC_DELAY):
        self.tasks_enabled = tasks_enabled
        self.motor_delay = motor_delay
        self.diagnostic_delay = diagnostic_delay
        self.task_scopes = {}
        super().__init__("Motors Server", host=host, port=port,
                         stateless_http=True, streamable_http_path="/mcp")
        # FastMCP 1.27.2 has no public version or Tasks registration arguments.
        self._mcp_server.version = "1.0.0"
        self._mcp_server.call_tool(validate_input=True)(self.dispatch_tool)
        if tasks_enabled:
            support = self._mcp_server.experimental.enable_tasks()
            cancel_handler = self._mcp_server.request_handlers[types.CancelTaskRequest]

            async def cancel_task(request):
                result = await cancel_handler(request)
                scope = self.task_scopes.get(request.params.taskId)
                if scope is not None:
                    scope.cancel()
                return result

            self._mcp_server.request_handlers[types.CancelTaskRequest] = cancel_task
            self.task_store = support.store
        register_server_content(self)

    async def list_tools(self):
        tools = await super().list_tools()
        if self.tasks_enabled:
            for tool in tools:
                tool.execution = types.ToolExecution(taskSupport="optional")
        return tools

    async def dispatch_tool(self, name, arguments):
        experimental = self._mcp_server.request_context.experimental
        experimental.validate_task_mode("optional" if self.tasks_enabled else "forbidden")
        if not experimental.is_task:
            return await self.call_tool(name, arguments)

        async def work(task):
            with anyio.CancelScope() as scope:
                self.task_scopes[task.task_id] = scope
                try:
                    state = await self.task_store.get_task(task.task_id)
                    if state.status == "cancelled":
                        scope.cancel()
                    content = await self.call_tool(name, arguments)
                finally:
                    self.task_scopes.pop(task.task_id, None)
            if scope.cancel_called:
                # Refresh the SDK task context so it does not auto-complete a cancelled task.
                await task.update_status("Cancelled", notify=False)
                return types.CallToolResult(content=[], isError=True)
            return types.CallToolResult(content=list(content))

        return await experimental.run_task(work)


def register_server_content(server):
    def annotations(title, *, readonly=False, destructive=True, idempotent=False):
        return types.ToolAnnotations(title=title, readOnlyHint=readonly,
                                     destructiveHint=destructive, idempotentHint=idempotent,
                                     openWorldHint=False)

    async def move(log, result):
        print(f"MOTORS: {log}", file=sys.stderr, flush=True)
        await asyncio.sleep(server.motor_delay)
        return result

    @server.tool(name="backward", title="Move Backward", meta={"category": "motor"},
                 annotations=annotations("Move Backward"), structured_output=False)
    async def backward(distance: Annotated[Distance, Field(
        description="The distance (in meters) to move the robot car backward.",
    )]) -> str:
        """Basic command: Moves the robot car backward."""
        return await move(f"Backward: {distance}m", f"moved backward for {distance} meters.")

    @server.tool(name="forward", title="Move Forward", annotations=annotations("Move Forward"),
                 structured_output=False)
    async def forward(distance: Annotated[Distance, Field(
        description="The distance (in meters) to move the robot car forward.",
    )]) -> str:
        """Basic command: Moves the robot car forward."""
        return await move(f"Forward: {distance}m", f"moved forward for {distance} meters.")

    @server.tool(name="stop", title="Stop", annotations=annotations("Stop", destructive=False, idempotent=True),
                 structured_output=False)
    async def stop() -> str:
        """Basic command: Stops the robot car."""
        return await move("Stop", "stopped.")

    @server.tool(name="turn_left", title="Turn Left", annotations=annotations("Turn Left"),
                 structured_output=False)
    async def turn_left(angle: Annotated[Angle, Field(
        description="The angle (in ° / degrees) to turn the robot car anticlockwise.",
    )]) -> str:
        """Basic command: Turns the robot car anticlockwise."""
        return await move(f"TurnLeft: {angle}°", f"turned anticlockwise {angle}°.")

    @server.tool(name="turn_right", title="Turn Right", annotations=annotations("Turn Right"),
                 structured_output=False)
    async def turn_right(angle: Annotated[Angle, Field(
        description="The angle (in ° / degrees) to turn the robot car clockwise.",
    )]) -> str:
        """Basic command: Turns the robot car clockwise."""
        return await move(f"TurnRight: {angle}°", f"turned clockwise {angle}°.")

    @server.tool(name="run_diagnostics", title="Run Diagnostics",
                 annotations=annotations("Run Diagnostics", readonly=True, destructive=False, idempotent=True),
                 structured_output=False)
    async def run_diagnostics() -> str:
        """Runs a full diagnostics check on all robot car motors. This is a long-running operation."""
        await asyncio.sleep(server.diagnostic_delay)
        return DIAGNOSTIC_RESULT

    progress_description = (
        "Runs a full diagnostics check on all robot car motors and reports progress notifications."
        if server.tasks_enabled else
        "Runs a full diagnostics check with progress on all robot car motors. "
        "This is a long-running operation that reports progress notifications."
    )

    @server.tool(name="run_diagnostics_with_progress", title="Run Diagnostics with Progress",
                 description=progress_description,
                 annotations=annotations("Run Diagnostics with Progress", readonly=True,
                                         destructive=False, idempotent=True), structured_output=False)
    async def run_diagnostics_with_progress(ctx: Context) -> str:
        for motor in range(1, 5):
            await asyncio.sleep(server.diagnostic_delay)
            if server.tasks_enabled:
                await ctx.report_progress(motor, total=4)
            else:
                # FastMCP 1.27.2 omits request routing in report_progress; stateless HTTP needs it.
                meta = ctx.request_context.meta
                if meta is not None and meta.progressToken is not None:
                    await ctx.session.send_progress_notification(
                        meta.progressToken, motor, total=4, related_request_id=ctx.request_id,
                    )
        return DIAGNOSTIC_RESULT

    @server.resource(BIO_URI, name="bio", mime_type="text/plain",
                     description="A static resource: returns a fixed bio for the robot")
    def bio() -> str:
        return "My name is Robby, the robot."

    @server.resource(GREET_TEMPLATE, name="greet", mime_type="text/plain",
                     description="A template resource: returns a personalised greeting for the given name")
    def greet(name: str, ctx: Context) -> str:
        client = ctx.session.client_params
        version = client.clientInfo.version if client is not None else "none"
        return f"Hello, {unquote(name)}! I am Robby, the robot. Version: {version}"

    @server.prompt(name="string_prompt", description="A string prompt without arguments")
    def string_prompt() -> str:
        return STRING_PROMPT

    @server.prompt(name="message_prompt", description="A message prompt without arguments")
    def message_prompt() -> list[UserMessage]:
        return [UserMessage(MESSAGE_PROMPT)]

    @server.prompt(name="parametrized_message_prompt", description="A message prompt with arguments")
    def parametrized_message_prompt(
        action: Annotated[str, Field(description="The complex action to be performed")],
    ) -> list[UserMessage]:
        return [UserMessage(f"## Context\nComplex command:\n{action}")]


async def list_available(session):
    tools = await session.list_tools()
    print("TOOLS AVAILABLE:")
    for tool in tools.tools:
        print(f"  {tool.name} {tool.inputSchema.get('properties', {})}")
    for heading, method, field in (
        ("PROMPTS AVAILABLE:", session.list_prompts, "prompts"),
        ("RESOURCES AVAILABLE:", session.list_resources, "resources"),
        ("RESOURCE TEMPLATES AVAILABLE:", session.list_resource_templates, "resourceTemplates"),
    ):
        print(f"\n{heading}")
        for item in getattr(await method(), field):
            print(f"  {item.name} {item.model_dump_json(by_alias=True, exclude_none=True)}")
    print()
    return tools.tools


def text_content(result):
    if result.isError:
        raise RuntimeError("MCP tool failed: " + " ".join(c.text for c in result.content if c.type == "text"))
    return "\n".join(c.text for c in result.content if c.type == "text")


async def call_with_polling(session, *, timeout=30):
    """Real tools/call + tasks/get + tasks/result, not a local asyncio stand-in."""
    task_id = None
    try:
        async with asyncio.timeout(timeout):
            created = await session.experimental.call_tool_as_task("run_diagnostics")
            task_id = created.task.taskId
            async for status in session.experimental.poll_task(task_id):
                if status.status in ("failed", "cancelled"):
                    raise RuntimeError(f"Diagnostics task {status.status}: {status.statusMessage}")
            return await session.experimental.get_task_result(task_id, types.CallToolResult)
    except TimeoutError:
        if task_id is not None:
            try:
                async with asyncio.timeout(2):
                    await session.experimental.cancel_task(task_id)
            except Exception:
                logging.getLogger(__name__).warning("Could not cancel timed-out diagnostics task", exc_info=True)
        raise


async def demonstrate(session, *, stdio=False):
    await list_available(session)
    if stdio:
        print("1. SIMPLE CALL: turn_left")
    result = await session.call_tool("turn_left", {"angle": 99})
    print(f"{'SIMPLE CALL RESULT' if stdio else 'TOOL RESPONSE'}: {text_content(result)}\n")
    if stdio:
        print("2. CALL WITH PROGRESS: run_diagnostics_with_progress")

        async def progress(current, total, message):
            print(f"  PROGRESS: {current:g}/{total:g} motors checked")

        result = await session.call_tool("run_diagnostics_with_progress", progress_callback=progress)
        print(f"PROGRESS CALL RESULT: {text_content(result)}\n")
        print("3. CALL WITH POLLING: run_diagnostics (Python SDK experimental Tasks)")
        result = await call_with_polling(session)
        print(f"POLLING CALL RESULT: {text_content(result)}\n")

    simple = await session.get_prompt("message_prompt")
    template = await session.get_prompt("parametrized_message_prompt", {"action": ACTION})
    simple_text = next(m.content.text for m in simple.messages if m.role == "user" and m.content.type == "text")
    prompt = next(m.content.text for m in template.messages if m.role == "user" and m.content.type == "text")
    print(f"SIMPLE PROMPT RESPONSE: {simple_text}")
    print(f"PROMPT TEMPLATE (PARAMETRIZED) RESPONSE: {prompt}\n")
    bio = await session.read_resource(BIO_URI)
    greet = await session.read_resource("resource://mcp/greet/Robby")
    print(f"RESOURCE RESPONSE: {bio.contents[0].text}")
    print(f"TEMPLATE RESOURCE RESPONSE: {greet.contents[0].text}\n")
    return prompt


def should_run_agent(read_input=input):
    try:
        return read_input("Run the AI agent? [y/N] ").strip().lower() == "y"
    except EOFError:
        return False


async def run_robot_agent(mcp_tool, prompt, provider="openai"):
    from agent_framework import Agent
    from shared.chat_clients import create_completion_client, create_sdk_client

    async with create_sdk_client(provider) as sdk:
        async with mcp_tool:
            agent = Agent(client=create_completion_client(sdk, provider),
                          instructions=AGENT_INSTRUCTIONS, tools=[mcp_tool])
            async with agent:
                print("AGENT RESPONSE:")
                response = await agent.run(prompt)
                print(response.text)
                return response
