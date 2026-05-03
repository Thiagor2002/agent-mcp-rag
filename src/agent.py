#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Agent 模块
核心智能体，整合 LLM 推理、MCP 工具调用和 RAG 上下文，
实现自主对话-工具调用循环。

架构：
    User Prompt → Agent → LLM (思考) → Tool Call? → MCP (执行)
                                ↑                    ↓
                                +—— Tool Result ←————+
                                ↓
                            Final Response
"""

import json
from typing import List, Optional

from .mcp_client import MCPClient
from .chat_openai import ChatOpenAI, ChatResponse
from .utils import log_title


class Agent:
    """
    增强型 LLM Agent

    整合 MCP 工具调用能力，支持多轮对话和自主工具使用。

    Usage:
        agent = Agent("Qwen/Qwen2.5-7B-Instruct", [fetch_mcp, file_mcp])
        await agent.init()
        result = await agent.invoke("帮我查一下今天的新闻")
        await agent.close()
    """

    def __init__(
        self,
        model: str,
        mcp_clients: List[MCPClient],
        system_prompt: str = "",
        context: str = "",
    ):
        """
        初始化 Agent

        Args:
            model: LLM 模型名称
            mcp_clients: MCP 客户端列表
            system_prompt: 系统提示词
            context: RAG 检索到的上下文
        """
        self.model = model
        self.mcp_clients = mcp_clients
        self.system_prompt = system_prompt
        self.context = context
        self.llm: Optional[ChatOpenAI] = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def init(self) -> None:
        """初始化所有 MCP 客户端并构建 LLM 实例"""
        log_title("TOOLS")
        for client in self.mcp_clients:
            await client.init()

        # 收集所有 MCP 工具
        tools = []
        for client in self.mcp_clients:
            tools.extend(client.get_tools())

        # 初始化 LLM，注入工具定义和上下文
        self.llm = ChatOpenAI(
            model=self.model,
            system_prompt=self.system_prompt,
            tools=tools,
            context=self.context,
        )

    async def close(self) -> None:
        """清理所有 MCP 连接"""
        for client in self.mcp_clients:
            await client.close()

    # ------------------------------------------------------------------
    # 核心循环
    # ------------------------------------------------------------------

    async def invoke(self, prompt: str) -> Optional[str]:
        """
        执行 Agent，运行 思考→行动 循环直到 LLM 不再需要工具

        Args:
            prompt: 用户输入

        Returns:
            LLM 最终回复内容
        """
        if not self.llm:
            raise RuntimeError("Agent 未初始化，请先调用 init()")

        # 首轮对话
        response = await self.llm.chat(prompt)

        # 工具调用循环
        while response.has_tool_calls:
            for tc in response.tool_calls:
                result = await self._execute_tool(tc.function_name, tc.function_args)
                self.llm.append_tool_result(tc.id, result)

            # 继续对话，让 LLM 处理工具结果
            response = await self.llm.chat()

        return response.content if response else None

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    async def _execute_tool(self, name: str, args_json: str) -> str:
        """
        查找并执行匹配的 MCP 工具

        Args:
            name: 工具名称
            args_json: JSON 格式的参数

        Returns:
            工具执行结果的字符串表示
        """
        log_title("TOOL USE")
        print(f"调用工具: {name}")
        print(f"参数: {args_json}")

        # 查找对应的 MCP 客户端
        for client in self.mcp_clients:
            for tool in client.get_tools():
                if tool.get("name") == name:
                    try:
                        args = json.loads(args_json) if args_json else {}
                        result = await client.call_tool(name, args)
                        result_str = json.dumps(result, ensure_ascii=False)
                        print(f"结果: {result_str}")
                        return result_str
                    except json.JSONDecodeError:
                        return f"参数解析失败: {args_json}"
                    except Exception as e:
                        return f"工具执行错误: {e}"

        return f"未找到工具: {name}"
