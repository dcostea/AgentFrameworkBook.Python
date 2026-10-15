"""Credential-free MCP exchanges and Agent Framework tool invocation regression tests."""

import asyncio
from contextlib import asynccontextmanager, redirect_stderr, redirect_stdout
from datetime import timedelta
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from mcp import ClientSession, types
from mcp.client.stdio import get_default_environment, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import McpError
from openai import AsyncOpenAI
import uvicorn

with patch("dotenv.load_dotenv", return_value=False):
    from agent_framework import MCPStdioTool, MCPStreamableHTTPTool
    from shared import chat_clients, mcp_examples as examples

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ("MCPClientWithHttp", "MCPClientWithStdio", "MCPServerWithHttp", "MCPServerWithStdio")
TOOLS = {"forward", "backward", "stop", "turn_left", "turn_right", "run_diagnostics", "run_diagnostics_with_progress"}
BASE_ENV = get_default_environment()
ORIGINAL_HTTP_SEND = httpx.AsyncHTTPTransport.handle_async_request


async def loopback_only(transport, request):
    if request.url.host not in {"127.0.0.1", "localhost", "::1"}:
        raise AssertionError(f"External network forbidden: {request.url.host}")
    return await ORIGINAL_HTTP_SEND(transport, request)
MOVES = (
    ("forward", {"distance": 3}, "moved forward for 3 meters.", "Forward: 3m"),
    ("backward", {"distance": -2}, "moved backward for -2 meters.", "Backward: -2m"),
    ("stop", {}, "stopped.", "Stop"),
    ("turn_left", {"angle": 99}, "turned anticlockwise 99°.", "TurnLeft: 99°"),
    ("turn_right", {"angle": 0}, "turned clockwise 0°.", "TurnRight: 0°"),
)


def load_sample(name):
    spec = importlib.util.spec_from_file_location(f"mcp_{name}", ROOT / "08.MCP" / name / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def subprocess_parameters(*, fast=True):
    params = examples.stdio_parameters(ROOT / "08.MCP" / "MCPClientWithStdio" / "main.py")
    if fast:
        # Execute the real entry point with injected simulation delays, not a fake protocol server.
        script = (
            f"import runpy,sys; sys.path.insert(0,{str(ROOT)!r}); "
            f"ns=runpy.run_path({params.args[-1]!r}); original=ns['build_server']; "
            "ns['main'].__globals__['build_server']=lambda: original(motor_delay=0,diagnostic_delay=0.04); "
            "ns['main']()"
        )
        params.args = ["-u", "-c", script]
    return params


@asynccontextmanager
async def stdio_session(*, fast=True):
    with open(os.devnull, "w") as errlog:
        async with stdio_client(subprocess_parameters(fast=fast), errlog=errlog) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=10),
                                     client_info=types.Implementation(name="Offline Motors Client", version="test-1")) as session:
                async with asyncio.timeout(10):
                    initialized = await session.initialize()
                yield session, initialized


@asynccontextmanager
async def http_server():
    module = load_sample("MCPServerWithHttp")
    mcp = module.build_server(motor_delay=0, diagnostic_delay=0.02)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(mcp.streamable_http_app(), host="127.0.0.1", port=port,
                                           log_level="critical", access_log=False, lifespan="on"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if task.done():
                    await task
                    raise AssertionError("HTTP test server exited before readiness")
                await asyncio.sleep(0.01)
        yield port, mcp
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=5)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            sock.close()


@asynccontextmanager
async def http_session():
    async with http_server() as (port, _):
        async with httpx.AsyncClient(timeout=10, trust_env=False) as http:
            async with streamable_http_client(examples.http_url(port), http_client=http) as (read, write, _):
                async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=10),
                                         client_info=types.Implementation(name="Offline Motors Client", version="test-1")) as session:
                    initialized = await session.initialize()
                    yield session, initialized


def model_payload(*, call=False):
    message = {"role": "assistant", "content": "Done."}
    if call:
        message = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "motor-1", "type": "function", "function": {
                "name": "turn_left", "arguments": '{"angle":45}',
            }},
        ]}
    return {
        "id": "mcp-offline", "object": "chat.completion", "created": 1, "model": "test-model",
        "choices": [{"index": 0, "finish_reason": "tool_calls" if call else "stop", "message": message}],
    }


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        asyncio.get_running_loop().set_debug(False)

    def setUp(self):
        self.enterContext(patch("dotenv.load_dotenv", return_value=False))
        self.enterContext(patch.dict(os.environ, BASE_ENV, clear=True))
        self.enterContext(patch.object(httpx.AsyncHTTPTransport, "handle_async_request", loopback_only))
        self.addCleanup(logging.disable, logging.root.manager.disable)
        logging.disable(logging.CRITICAL)
        self.output = io.StringIO()
        self.errors = io.StringIO()
        self.enterContext(redirect_stdout(self.output))
        self.enterContext(redirect_stderr(self.errors))

    async def test_all_four_entrypoints_import_without_startup_or_credentials(self):
        with patch.object(uvicorn.Server, "serve", new=AsyncMock(side_effect=AssertionError("No startup on import"))), \
             patch.object(chat_clients, "create_sdk_client", side_effect=AssertionError("No credentials on import")):
            for name in PROJECTS:
                with self.subTest(project=name):
                    module = load_sample(name)
                    self.assertTrue(callable(module.main))
                    self.assertIn("parents[2]", Path(module.__file__).read_text(encoding="utf-8"))
        self.assertEqual(self.output.getvalue(), "")

    async def test_source_latency_and_http_tasks_difference(self):
        stdio = load_sample("MCPServerWithStdio").build_server()
        http = load_sample("MCPServerWithHttp").build_server()
        self.assertEqual((stdio.motor_delay, http.motor_delay), (0.1, 0.1))
        self.assertEqual((stdio.diagnostic_delay, http.diagnostic_delay), (2.0, 2.0))
        self.assertTrue(stdio.tasks_enabled)
        self.assertFalse(http.tasks_enabled)
        self.assertEqual(http.settings.host, "127.0.0.1")
        self.assertTrue(http.settings.stateless_http)
        self.assertEqual(http.settings.port, 3001)
        self.assertEqual(stdio.task_scopes, {})
        self.assertIsNone(http._mcp_server.experimental.task_support)

    async def test_both_transports_initialize_and_list_exact_protocol_surface(self):
        for transport in (stdio_session, http_session):
            with self.subTest(transport=transport.__name__):
                async with transport() as (session, initialized):
                    self.assertEqual(initialized.serverInfo.name, "Motors Server")
                    self.assertEqual(initialized.serverInfo.version, "1.0.0")
                    self.assertEqual(initialized.protocolVersion, "2025-11-25")
                    self.assertEqual(initialized.capabilities.tasks is not None, transport is stdio_session)
                    tools = {t.name: t for t in (await session.list_tools()).tools}
                    self.assertEqual(set(tools), TOOLS)
                    self.assertEqual(tools["forward"].inputSchema["required"], ["distance"])
                    self.assertEqual(tools["turn_left"].inputSchema["required"], ["angle"])
                    self.assertEqual(tools["run_diagnostics_with_progress"].inputSchema["properties"], {})
                    self.assertTrue(all(t.outputSchema is None for t in tools.values()))
                    prompts = {p.name: p for p in (await session.list_prompts()).prompts}
                    self.assertEqual(set(prompts), {"string_prompt", "message_prompt", "parametrized_message_prompt"})
                    self.assertEqual([(a.name, a.required, a.description) for a in prompts["parametrized_message_prompt"].arguments],
                                     [("action", True, "The complex action to be performed")])
                    resources = (await session.list_resources()).resources
                    self.assertEqual([(r.name, str(r.uri), r.mimeType) for r in resources],
                                     [("bio", examples.BIO_URI, "text/plain")])
                    templates = (await session.list_resource_templates()).resourceTemplates
                    self.assertEqual([(r.name, r.uriTemplate, r.mimeType) for r in templates],
                                     [("greet", examples.GREET_TEMPLATE, "text/plain")])

    async def test_tool_titles_annotations_and_transport_parity(self):
        servers = [load_sample(name).build_server() for name in ("MCPServerWithStdio", "MCPServerWithHttp")]
        definitions = [{t.name: t for t in await s.list_tools()} for s in servers]
        for tools in definitions:
            self.assertEqual(tools["backward"].meta, {"category": "motor"})
            for name, title in (("forward", "Move Forward"), ("backward", "Move Backward"),
                                ("turn_left", "Turn Left"), ("turn_right", "Turn Right"), ("stop", "Stop"),
                                ("run_diagnostics", "Run Diagnostics"),
                                ("run_diagnostics_with_progress", "Run Diagnostics with Progress")):
                with self.subTest(tool=name):
                    tool = tools[name]
                    self.assertEqual(tool.title, title)
                    hints = tool.annotations
                    self.assertEqual(hints.title, title)
                    maintenance = name.startswith("run_diagnostics")
                    self.assertEqual(hints.readOnlyHint, maintenance)
                    self.assertEqual(hints.destructiveHint, not maintenance and name != "stop")
                    self.assertEqual(hints.idempotentHint, maintenance or name == "stop")
                    self.assertFalse(hints.openWorldHint)
        for name in TOOLS:
            self.assertEqual(definitions[0][name].inputSchema, definitions[1][name].inputSchema)
            self.assertEqual(definitions[0][name].annotations, definitions[1][name].annotations)
            self.assertEqual(definitions[0][name].execution.taskSupport, "optional")
            self.assertIsNone(definitions[1][name].execution)

    async def test_motor_results_and_stderr_only_logs_with_injected_delays(self):
        for project in ("MCPServerWithStdio", "MCPServerWithHttp"):
            server = load_sample(project).build_server()
            with patch.object(examples.asyncio, "sleep", new=AsyncMock()) as delay:
                for name, args, expected, log in MOVES:
                    with self.subTest(project=project, tool=name):
                        result = await server.call_tool(name, args)
                        self.assertEqual([c.text for c in result], [expected])
                        self.assertIn(f"MOTORS: {log}\n", self.errors.getvalue())
                self.assertEqual([c.args for c in delay.await_args_list], [(0.1,)] * 5)
        self.assertEqual(self.output.getvalue(), "")
        self.assertNotIn("\x1b", self.errors.getvalue())
        self.assertEqual(self.errors.getvalue().count("MOTORS:"), 10)

    async def test_real_stdio_subprocess_calls_every_motor_with_plain_text_results(self):
        async with stdio_session(fast=False) as (session, _):
            for name, args, expected, _ in MOVES:
                with self.subTest(tool=name):
                    result = await session.call_tool(name, args)
                    self.assertFalse(result.isError)
                    self.assertEqual(examples.text_content(result), expected)
                    self.assertIsNone(result.structuredContent)

    async def test_protocol_input_boundaries_and_unknown_tool(self):
        for transport in (stdio_session, http_session):
            with self.subTest(transport=transport.__name__):
                async with transport() as (session, _):
                    for args in ({}, {"distance": "5"}, {"distance": 1.5}, {"distance": True},
                                 {"distance": 2**31}, {"distance": -(2**31) - 1}):
                        with self.subTest(invalid=args):
                            result = await session.call_tool("forward", args)
                            self.assertTrue(result.isError)
                            self.assertIn("validation", result.content[0].text.lower())
                    for value in (-(2**31), 2**31 - 1):
                        result = await session.call_tool("forward", {"distance": value})
                        self.assertEqual(examples.text_content(result), f"moved forward for {value} meters.")
                    result = await session.call_tool("unknown_motor", {})
                    self.assertTrue(result.isError)
                    self.assertIn("unknown", result.content[0].text.lower())

    async def test_prompts_preserve_roles_text_template_and_required_action(self):
        async with stdio_session() as (session, _):
            for name, args, expected in (("string_prompt", None, examples.STRING_PROMPT),
                                          ("message_prompt", None, examples.MESSAGE_PROMPT),
                                          ("parametrized_message_prompt", {"action": 'Turn 90°\nthen "stop".'},
                                           '## Context\nComplex command:\nTurn 90°\nthen "stop".')):
                with self.subTest(prompt=name):
                    result = await session.get_prompt(name, args)
                    self.assertEqual(len(result.messages), 1)
                    self.assertEqual(result.messages[0].role, "user")
                    self.assertEqual(result.messages[0].content.text, expected)
            with self.assertRaises(McpError):
                await session.get_prompt("parametrized_message_prompt")

    async def test_resources_preserve_uri_mime_greeting_client_version(self):
        for transport, version in ((stdio_session, "test-1"), (http_session, "none")):
            with self.subTest(transport=transport.__name__):
                async with transport() as (session, _):
                    bio = await session.read_resource(examples.BIO_URI)
                    self.assertEqual(bio.contents[0].text, "My name is Robby, the robot.")
                    self.assertEqual(str(bio.contents[0].uri), examples.BIO_URI)
                    uri = "resource://mcp/greet/Robby%20Junior"
                    greet = await session.read_resource(uri)
                    self.assertEqual(str(greet.contents[0].uri), uri)
                    self.assertEqual(greet.contents[0].mimeType, "text/plain")
                    self.assertEqual(greet.contents[0].text,
                                     f"Hello, Robby Junior! I am Robby, the robot. Version: {version}")
                    with self.assertRaises(McpError):
                        await session.read_resource("resource://mcp/not_found")

    async def test_maintenance_delays_and_exact_progress_counts(self):
        server = load_sample("MCPServerWithStdio").build_server()
        ctx = Mock(report_progress=AsyncMock())
        with patch.object(examples.asyncio, "sleep", new=AsyncMock()) as delay, \
             patch.object(server, "get_context", return_value=ctx):
            result = await server.call_tool("run_diagnostics", {})
            self.assertEqual(result[0].text, examples.DIAGNOSTIC_RESULT)
            self.assertEqual(ctx.report_progress.await_count, 0)
            result = await server.call_tool("run_diagnostics_with_progress", {})
            self.assertEqual(result[0].text, examples.DIAGNOSTIC_RESULT)
            self.assertEqual([c.args for c in delay.await_args_list], [(2.0,)] * 5)
            self.assertEqual([(c.args, c.kwargs) for c in ctx.report_progress.await_args_list],
                             [((n,), {"total": 4}) for n in range(1, 5)])

    async def test_protocol_progress_notifications_both_transports(self):
        for transport in (stdio_session, http_session):
            with self.subTest(transport=transport.__name__):
                async with transport() as (session, _):
                    progress = []

                    async def report(current, total, message):
                        progress.append((current, total, message))

                    result = await session.call_tool("run_diagnostics_with_progress", progress_callback=report)
                    self.assertEqual(examples.text_content(result), examples.DIAGNOSTIC_RESULT)
                    self.assertEqual(progress, [(n, 4, None) for n in range(1, 5)])
                    self.assertEqual((await session.list_tools()).tools[0].execution is not None,
                                     transport is stdio_session)

    async def test_stdio_tasks_transition_working_completed_result_and_ordinary_call(self):
        async with stdio_session() as (session, _):
            ordinary = await session.call_tool("run_diagnostics")
            self.assertEqual(examples.text_content(ordinary), examples.DIAGNOSTIC_RESULT)
            self.assertEqual((await session.experimental.list_tasks()).tasks, [])
            created = await session.experimental.call_tool_as_task("run_diagnostics")
            self.assertEqual(created.task.status, "working")
            self.assertEqual(created.task.pollInterval, 500)
            statuses = []
            async for status in session.experimental.poll_task(created.task.taskId):
                statuses.append(status.status)
            self.assertEqual(statuses[0], "working")
            self.assertEqual(statuses[-1], "completed")
            result = await session.experimental.get_task_result(created.task.taskId, types.CallToolResult)
            self.assertEqual(examples.text_content(result), examples.DIAGNOSTIC_RESULT)
            self.assertEqual([t.taskId for t in (await session.experimental.list_tasks()).tasks], [created.task.taskId])
            self.assertEqual((await session.experimental.get_task(created.task.taskId)).status, "completed")
            with self.assertRaises(McpError):
                await session.experimental.cancel_task(created.task.taskId)

    async def test_stdio_task_cancellation_and_session_scoping(self):
        async with stdio_session() as (session, _):
            created = await session.experimental.call_tool_as_task("run_diagnostics_with_progress")
            cancelled = await session.experimental.cancel_task(created.task.taskId)
            self.assertEqual(cancelled.status, "cancelled")
            await asyncio.sleep(0.2)
            self.assertEqual((await session.experimental.get_task(created.task.taskId)).status, "cancelled")
            self.assertEqual([t.taskId for t in (await session.experimental.list_tasks()).tasks], [created.task.taskId])
            with self.assertRaises(McpError):
                await session.experimental.get_task("a-different-scope:unknown")

    async def test_http_has_ordinary_diagnostics_but_rejects_task_requests(self):
        async with http_session() as (session, initialized):
            self.assertIsNone(initialized.capabilities.tasks)
            result = await session.call_tool("run_diagnostics")
            self.assertEqual(examples.text_content(result), examples.DIAGNOSTIC_RESULT)
            with self.assertRaises(Exception):
                await session.experimental.call_tool_as_task("run_diagnostics")
            result = await session.call_tool("stop")
            self.assertEqual(examples.text_content(result), "stopped.")

    async def test_http_server_shutdown_releases_ephemeral_listener(self):
        async with http_server() as (port, _):
            async with httpx.AsyncClient(trust_env=False) as http:
                response = await http.post(examples.http_url(port), json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                    "protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"},
                }}, headers={"Accept": "application/json, text/event-stream"})
                self.assertEqual(response.status_code, 200)
                self.assertIn("Motors Server", response.text)
        with socket.socket() as probe:
            probe.settimeout(0.2)
            self.assertNotEqual(probe.connect_ex(("127.0.0.1", port)), 0)


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        asyncio.get_running_loop().set_debug(False)

    def setUp(self):
        self.enterContext(patch("dotenv.load_dotenv", return_value=False))
        self.enterContext(patch.dict(os.environ, {**BASE_ENV, "OPENAI_CHAT_MODEL_ID": "test-model",
                                               "AZURE_OPENAI_DEPLOYMENT_NAME": "test-model"}, clear=True))
        self.enterContext(patch.object(httpx.AsyncHTTPTransport, "handle_async_request", loopback_only))
        self.addCleanup(logging.disable, logging.root.manager.disable)
        logging.disable(logging.CRITICAL)
        self.output = io.StringIO()
        self.enterContext(redirect_stdout(self.output))

    def mock_sdk(self):
        requests = []

        def handle(request):
            requests.append(json.loads(request.content))
            self.assertLessEqual(len(requests), 2, "Unexpected model request")
            return httpx.Response(200, json=model_payload(call=len(requests) == 1))

        sdk = AsyncOpenAI(api_key="offline-placeholder", max_retries=0,
                          http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        self.addAsyncCleanup(sdk.close)
        return sdk, requests

    async def test_stdio_executable_absolute_sibling_path_and_utf8(self):
        params = examples.stdio_parameters(ROOT / "08.MCP" / "MCPClientWithStdio" / "main.py")
        self.assertEqual(params.command, sys.executable)
        self.assertTrue(Path(params.args[-1]).is_absolute())
        self.assertEqual(Path(params.args[-1]), ROOT / "08.MCP" / "MCPServerWithStdio" / "main.py")
        self.assertEqual(params.env["PYTHONIOENCODING"], "utf-8")
        tool = load_sample("MCPClientWithStdio").create_mcp_tool(Mock())
        self.assertIsInstance(tool, MCPStdioTool)
        self.assertEqual(tool.command, sys.executable)
        self.assertFalse(tool.load_prompts_flag)

    async def test_port_configuration_and_http_mcp_tool(self):
        self.assertEqual(examples.configured_port(), 3001)
        with patch.dict(os.environ, {"MCP_PORT": "3101"}):
            self.assertEqual(examples.configured_port(), 3101)
        for invalid in ("zero", 0, -1, 65536):
            with self.subTest(port=invalid), self.assertRaises(ValueError):
                examples.port_number(invalid)
        self.assertEqual(examples.http_url(3101), "http://127.0.0.1:3101/mcp")
        tool = load_sample("MCPClientWithHttp").create_mcp_tool(Mock(), 3101)
        self.assertIsInstance(tool, MCPStreamableHTTPTool)
        self.assertEqual(tool.url, "http://127.0.0.1:3101/mcp")
        self.assertFalse(tool.load_prompts_flag)

    async def test_agent_choice_exact_yes_default_no_eof(self):
        for answer, expected in (("", False), ("n", False), (" Y ", True), ("yes", False)):
            with self.subTest(answer=answer):
                self.assertEqual(examples.should_run_agent(lambda _: answer), expected)
        self.assertFalse(examples.should_run_agent(Mock(side_effect=EOFError)))

    async def test_stdio_client_lists_requests_progress_polling_and_skips_model(self):
        module = load_sample("MCPClientWithStdio")
        with patch.object(module, "stdio_parameters", return_value=subprocess_parameters()), \
             patch.object(chat_clients, "create_sdk_client", side_effect=AssertionError("No model credentials permitted")):
            await module.main(read_input=lambda _: "")
        output = self.output.getvalue()
        for heading in ("TOOLS AVAILABLE:", "PROMPTS AVAILABLE:", "RESOURCES AVAILABLE:", "RESOURCE TEMPLATES AVAILABLE:"):
            self.assertIn(heading, output)
        self.assertLess(output.index("1. SIMPLE CALL:"), output.index("2. CALL WITH PROGRESS:"))
        self.assertLess(output.index("2. CALL WITH PROGRESS:"), output.index("3. CALL WITH POLLING:"))
        self.assertIn("SIMPLE CALL RESULT: turned anticlockwise 99°.", output)
        self.assertIn("POLLING CALL RESULT: " + examples.DIAGNOSTIC_RESULT, output)
        self.assertEqual(output.count("motors checked"), 4)
        self.assertIn("MCP demos complete; skipping the LLM agent.", output)
        self.assertIn("Version: 1.0.0", output)

    async def test_agent_framework_model_calls_real_mcp_tools_over_both_transports(self):
        for transport, provider in ((stdio_session, "openai"), (http_session, "azure")):
            with self.subTest(transport=transport.__name__, provider=provider):
                async with transport() as (session, _):
                    sdk, requests = self.mock_sdk()
                    if transport is stdio_session:
                        tool = load_sample("MCPClientWithStdio").create_mcp_tool(session)
                    else:
                        tool = load_sample("MCPClientWithHttp").create_mcp_tool(session, 3001)
                    with patch.object(chat_clients, "create_sdk_client", return_value=sdk) as factory:
                        response = await examples.run_robot_agent(tool, "Turn left 45 degrees", provider)
                    factory.assert_called_once_with(provider)
                    self.assertEqual(response.text, "Done.")
                    self.assertEqual(len(requests), 2)
                    self.assertEqual({t["function"]["name"] for t in requests[0]["tools"]}, TOOLS)
                    self.assertEqual(requests[0]["messages"][0]["content"], examples.AGENT_INSTRUCTIONS)
                    result = [m for m in requests[1]["messages"] if m["role"] == "tool"]
                    self.assertEqual(len(result), 1)
                    self.assertEqual(result[0]["tool_call_id"], "motor-1")
                    self.assertIn("turned anticlockwise 45°.", result[0]["content"])
                    self.assertFalse(tool.is_connected)
                    self.assertTrue(sdk.is_closed())
                    # The caller owns the injected MCP SDK session and it remains usable.
                    self.assertEqual(examples.text_content(await session.call_tool("stop")), "stopped.")

    async def test_http_client_runs_agent_without_stdio_demos_or_question(self):
        module = load_sample("MCPClientWithHttp")
        sdk, requests = self.mock_sdk()
        async with http_server() as (port, _):
            with patch.object(chat_clients, "create_sdk_client", return_value=sdk), \
                 patch("builtins.input", side_effect=AssertionError("HTTP source has no question")):
                response = await module.main("openai", port)
        self.assertEqual(response.text, "Done.")
        output = self.output.getvalue()
        self.assertIn("TOOL RESPONSE: turned anticlockwise 99°.", output)
        self.assertIn("AGENT RESPONSE:", output)
        self.assertNotIn("CALL WITH PROGRESS", output)
        self.assertNotIn("CALL WITH POLLING", output)
        user_messages = [m for m in requests[0]["messages"] if m["role"] == "user"]
        self.assertEqual(user_messages[0]["content"], f"## Context\nComplex command:\n{examples.ACTION}")

    async def test_polling_helper_uses_sdk_results_and_cancels_on_timeout(self):
        async with stdio_session() as (session, _):
            result = await examples.call_with_polling(session)
            self.assertEqual(examples.text_content(result), examples.DIAGNOSTIC_RESULT)
            with self.assertRaises(TimeoutError):
                await examples.call_with_polling(session, timeout=0.02)
            tasks = (await session.experimental.list_tasks()).tasks
            self.assertEqual({t.status for t in tasks}, {"completed", "cancelled"})

    async def test_tool_errors_are_not_printed_as_success(self):
        result = types.CallToolResult(isError=True, content=[types.TextContent(type="text", text="bad motor input")])
        with self.assertRaisesRegex(RuntimeError, "MCP tool failed: bad motor input"):
            examples.text_content(result)


if __name__ == "__main__":
    unittest.main()
