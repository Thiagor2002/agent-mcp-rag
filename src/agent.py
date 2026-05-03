#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Agent 核心模块
==============

整合 CRAG 检索 + MCP 工具调用 + CLI 编码 + LLM 推理的增强型 Agent。

流程:
    User Input → CRAG 检索 → 构建上下文 → LLM 推理
        ↓
    Tool Call? → MCP / CLI 执行 → 工具结果 → LLM 继续推理
        ↓
    Final Response

支持:
- CRAG: 纠正性检索增强生成
- MCP: 多服务器工具调用
- CLI: 轻量级命令执行（减少 token 消耗）
- 多模型路由: 一般推理 / 代码任务
"""

import json
from typing import List, Optional, Dict, Any

from .mcp_client import MCPClient
from .chat_openai import ChatOpenAI, ChatResponse
from .coder_cli import CoderCLI
from .utils import log_title


class Agent:
    """
    增强型 LLM Agent (CRAG + MCP + CLI Coder)

    Usage:
        agent = Agent(model="Qwen/Qwen2.5-7B-Instruct", mcp_clients=[fetch_mcp])
        await agent.init()
        result = await agent.invoke("帮我查资料并生成报告")
        await agent.close()
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        mcp_clients: Optional[List[MCPClient]] = None,
        coder_cli: Optional[CoderCLI] = None,
        system_prompt: str = "",
        context: str = "",
        api_key: str = "",
        base_url: str = "",
    ):
        self.model = model
        self.mcp_clients = mcp_clients or []
        self.coder_cli = coder_cli
        self.system_prompt = system_prompt
        self.context = context
        self._api_key = api_key
        self._base_url = base_url
        self.llm: Optional[ChatOpenAI] = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def init(self) -> None:
        """初始化 MCP 客户端、工具和 LLM"""
        log_title("AGENT INIT")

        # 初始化 MCP 客户端
        for client in self.mcp_clients:
            await client.init()

        # 收集所有可用工具
        tools = []
        for client in self.mcp_clients:
            tools.extend(client.get_tools())

        # 添加 CLI Coder 工具
        if self.coder_cli:
            tools.extend(self.coder_cli.get_tools())

        # 初始化 LLM
        self.llm = ChatOpenAI(
            model=self.model,
            system_prompt=self.system_prompt,
            tools=tools,
            context=self.context,
            api_key=self._api_key,
            base_url=self._base_url,
        )

        print(f"  已加载 {len(tools)} 个工具")

    async def close(self) -> None:
        """清理所有连接"""
        for client in self.mcp_clients:
            await client.close()

    # ------------------------------------------------------------------
    # 核心循环
    # ------------------------------------------------------------------

    async def invoke(
        self, prompt: str, max_tool_rounds: int = 10
    ) -> Optional[str]:
        """
        执行 Agent 推理循环

        循环: LLM 推理 → 工具调用 → 工具执行 → 结果反馈 → 继续推理

        Args:
            prompt: 用户输入
            max_tool_rounds: 最大工具调用轮数

        Returns:
            最终回复内容
        """
        if not self.llm:
            raise RuntimeError("Agent 未初始化，请先调用 init()")

        response = await self.llm.chat(prompt)
        round_count = 0

        while response.has_tool_calls and round_count < max_tool_rounds:
            round_count += 1
            for tc in response.tool_calls:
                result = await self._execute_tool(tc.function_name, tc.function_args)
                self.llm.append_tool_result(tc.id, result)

            response = await self.llm.chat()

        return response.content if response else None

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    async def _execute_tool(self, name: str, args_json: str) -> str:
        """查找并执行匹配的工具"""
        log_title(f"TOOL: {name}")
        print(f"  参数: {args_json[:200]}")

        try:
            args = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError:
            args = {}

        # 1. 尝试 MCP 工具
        for client in self.mcp_clients:
            for tool in client.get_tools():
                if tool.get("name") == name:
                    try:
                        result = await client.call_tool(name, args)
                        return json.dumps(result, ensure_ascii=False, default=str)
                    except Exception as e:
                        return f"MCP工具错误: {e}"

        # 2. 尝试 CLI Coder 工具
        if self.coder_cli:
            for tool in self.coder_cli.get_tools():
                if tool.get("name") == name:
                    try:
                        return await self.coder_cli.execute(name, **args)
                    except Exception as e:
                        return f"CLI工具错误: {e}"

        return f"未找到工具: {name}"
