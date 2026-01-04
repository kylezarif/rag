"""
Minimal MCP client to exercise MCP servers without an LLM.

Usage:
    uv run mcp_client.py ./mcp_weather.py
Then issue commands like:
    tool get_forecast {"latitude": 32.7767, "longitude": -96.7970}
    tool get_alerts {"state": "TX"}
Type 'help' for commands, 'quit' to exit.
"""

import asyncio
import json
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class SimpleMCPClient:
    def __init__(self):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()

    async def connect(self, server_script_path: str):
        is_python = server_script_path.endswith(".py")
        if not is_python:
            raise ValueError("Server script must be a .py file")

        params = StdioServerParameters(command="uv", args=["run", server_script_path])
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

    async def list_tools(self):
        if not self.session:
            return []
        resp = await self.session.list_tools()
        return resp.tools

    async def call_tool(self, name: str, args: dict):
        if not self.session:
            raise RuntimeError("Session not connected")
        return await self.session.call_tool(name, args)

    async def close(self):
        await self.exit_stack.aclose()


async def repl(server_path: str):
    client = SimpleMCPClient()
    await client.connect(server_path)
    tools = await client.list_tools()
    print("Connected. Tools available:")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")

    print("\nCommands:")
    print("  tool <name> <json_args>")
    print("  tools  (list tools)")
    print("  quit   (exit)")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.lower() in {"quit", "exit"}:
            break
        if line.lower() == "tools":
            tools = await client.list_tools()
            for tool in tools:
                print(f"- {tool.name}: {tool.description}")
            continue
        if line.startswith("tool "):
            _, rest = line.split(" ", 1)
            parts = rest.split(" ", 1)
            if len(parts) != 2:
                print("Usage: tool <name> <json_args>")
                continue
            name, arg_str = parts
            try:
                args = json.loads(arg_str)
            except json.JSONDecodeError as exc:
                print(f"Invalid JSON: {exc}")
                continue
            result = await client.call_tool(name, args)
            print("Result:", result.content)
            continue
        print("Unknown command. Type 'tools' or 'quit'.")

    await client.close()


async def main():
    if len(sys.argv) < 2:
        print("Usage: uv run mcp_client.py <server_script.py>")
        sys.exit(1)
    await repl(sys.argv[1])


if __name__ == "__main__":
    asyncio.run(main())
