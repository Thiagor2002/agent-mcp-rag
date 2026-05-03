#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MCP (Model Context Protocol) 客户端模块
负责管理与 MCP Server 的通信，包括：
- 启动/停止 MCP Server 子进程
- 获取工具列表
- 执行远程工具调用

参考：https://modelcontextprotocol.io/docs/concepts/architecture
"""

import asyncio
import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .utils import log_title


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    """MCP 工具描述"""
    name: str
    description: str
    input_schema: Dict[str, Any]

    def to_openai_format(self) -> Dict[str, Any]:
        """转换为 OpenAI function calling 格式"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


# ---------------------------------------------------------------------------
# MCPClient 主类
# ---------------------------------------------------------------------------

class MCPClient:
    """
    MCP 客户端，管理与 MCP Server 的 stdio 通信

    支持通过 uvx / npx 启动 MCP Server 进程，
    通过 stdin/stdout JSON-RPC 协议进行通信。

    Usage:
        client = MCPClient("fetch", "uvx", ["mcp-server-fetch"])
        await client.init()
        tools = client.get_tools()
        result = await client.call_tool("fetch", {"url": "https://example.com"})
        await client.close()
    """

    def __init__(
        self,
        name: str,
        command: str,
        args: List[str],
        env: Optional[Dict[str, str]] = None,
    ):
        """
        初始化 MCP 客户端

        Args:
            name: 客户端名称标识
            command: 启动 MCP Server 的命令 (uvx / npx / python)
            args: 命令行参数列表
            env: 额外的环境变量
        """
        self.name = name
        self.command = command
        self.args = args
        self.env = env

        self._process: Optional[asyncio.subprocess.Process] = None
        self._tools: List[Tool] = []
        self._request_id = 0
        self._reader_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def init(self) -> None:
        """启动 MCP Server 并获取工具列表"""
        await self._connect()

    async def close(self) -> None:
        """关闭与 MCP Server 的连接"""
        if self._process:
            try:
                self._process.stdin.close()
            except Exception:
                pass
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            except Exception:
                pass
            self._process = None

    # ------------------------------------------------------------------
    # 工具管理
    # ------------------------------------------------------------------

    def get_tools(self) -> List[Dict[str, Any]]:
        """获取所有工具的 OpenAI 兼容格式定义"""
        return [t.to_openai_format() for t in self._tools]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        调用 MCP Server 上的工具

        Args:
            name: 工具名称
            arguments: 工具参数

        Returns:
            工具执行结果
        """
        if not self._process:
            raise RuntimeError("MCP 客户端未连接，请先调用 init()")

        return await self._call_tool(name, arguments)

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    async def _connect(self) -> None:
        """启动 MCP Server 进程并初始化连接"""
        try:
            # 构建完整命令
            cmd = [self.command] + self.args

            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, **(self.env or {})},
            )

            # 发送 initialize 请求
            init_result = await self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
            })
            self._server_info = init_result

            # 发送 initialized 通知
            await self._send_notification("notifications/initialized", {})

            # 获取工具列表
            tools_result = await self._send_request("tools/list", {})
            self._tools = [
                Tool(
                    name=t["name"],
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema", {"type": "object", "properties": {}}),
                )
                for t in tools_result.get("tools", [])
            ]

            log_title("TOOLS")
            print(f"[{self.name}] 已连接，可用工具: {[t.name for t in self._tools]}")

        except Exception as e:
            await self.close()
            raise RuntimeError(f"连接 MCP Server '{self.name}' 失败: {e}") from e

    async def _send_request(self, method: str, params: Dict[str, Any]) -> Any:
        """发送 JSON-RPC 请求并等待响应"""
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }

        return await self._send_and_receive(request)

    async def _send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """发送 JSON-RPC 通知（无需响应）"""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        await self._send_raw(notification)

    async def _call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """调用工具"""
        result = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        # 解析结果内容
        content = result.get("content", [])
        if isinstance(content, list):
            return [c.get("text", str(c)) for c in content]
        return content

    async def _send_and_receive(self, data: Dict[str, Any]) -> Any:
        """发送数据并等待响应"""
        if not self._process or not self._process.stdin:
            raise RuntimeError("进程未启动")

        async with self._reader_lock:
            await self._send_raw(data)
            return await self._receive_response()

    async def _send_raw(self, data: Dict[str, Any]) -> None:
        """向进程发送原始 JSON"""
        if not self._process or not self._process.stdin:
            raise RuntimeError("进程未启动")
        line = json.dumps(data, ensure_ascii=False)
        self._process.stdin.write((line + "\n").encode())
        await self._process.stdin.drain()

    async def _receive_response(self) -> Any:
        """从进程读取响应"""
        if not self._process or not self._process.stdout:
            raise RuntimeError("进程未启动")

        while True:
            line = await self._process.stdout.readline()
            if not line:
                raise RuntimeError("MCP Server 连接已关闭")

            try:
                data = json.loads(line.decode().strip())
                # 跳过通知（无 id 的消息）
                if "id" not in data:
                    continue
                if "error" in data:
                    raise RuntimeError(
                        f"MCP 错误: {data['error'].get('message', str(data['error']))}"
                    )
                return data.get("result", data)
            except json.JSONDecodeError:
                continue
