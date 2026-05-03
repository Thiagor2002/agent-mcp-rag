#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM 聊天接口模块
================

提供 OpenAI 兼容 API 的统一聊天接口，支持:
- 多提供商: SiliconFlow, MiniMax, OpenAI
- 多模型路由: cheap (一般推理) / coder (代码生成)
- 流式输出 / 非流式输出
- 工具调用 (Function Calling)

用法:
    client = ChatOpenAI(model="Qwen/Qwen2.5-7B-Instruct")
    response = await client.chat("你好")
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import openai
from dotenv import load_dotenv

from .utils import log_title, log_section

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """LLM 工具调用"""
    id: str
    function_name: str
    function_args: str


@dataclass
class ChatResponse:
    """LLM 聊天响应"""
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass
class Message:
    """聊天消息"""
    role: str
    content: str = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        msg: Dict[str, Any] = {"role": self.role}
        if self.content:
            msg["content"] = self.content
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg


# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
CODER_MODEL = os.getenv("CODER_MODEL", "deepseek-ai/DeepSeek-V3")


def _get_env(key: str, fallback: str = "") -> str:
    """多源环境变量解析"""
    env_map = {
        "api_key": ["OPENAI_API_KEY", "SILICONFLOW_API_KEY", "MINIMAX_API_KEY"],
        "base_url": ["OPENAI_BASE_URL", "SILICONFLOW_BASE_URL", "MINIMAX_BASE_URL"],
    }
    if key in env_map:
        for k in env_map[key]:
            val = os.getenv(k, "")
            if val:
                return val
        return fallback
    return os.getenv(key, fallback)


# ---------------------------------------------------------------------------
# ChatOpenAI 主类
# ---------------------------------------------------------------------------

class ChatOpenAI:
    """
    OpenAI 兼容 API 聊天客户端，支持多模型路由

    - cheap_model: 一般推理任务（便宜、低延迟）
    - coder_model: 代码生成任务（更强的推理能力）

    Usage:
        client = ChatOpenAI()  # 使用默认配置
        response = await client.chat("Write a Python sort function", use_coder=True)
    """

    def __init__(
        self,
        model: str = "",
        cheap_model: str = "",
        coder_model: str = "",
        system_prompt: str = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        context: str = "",
        api_key: str = "",
        base_url: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        self.model = model or DEFAULT_MODEL
        self.cheap_model = cheap_model or os.getenv("CHEAP_MODEL", self.model)
        self.coder_model = coder_model or CODER_MODEL
        self.tools = tools or []
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.messages: List[Message] = []

        _key = api_key or _get_env("api_key")
        _base = base_url or _get_env("base_url", DEFAULT_BASE_URL)

        if not _key:
            raise ValueError("未设置 API Key。请设置环境变量 OPENAI_API_KEY")

        self._client = openai.OpenAI(api_key=_key, base_url=_base)

        if system_prompt:
            self.messages.append(Message(role="system", content=system_prompt))
        if context:
            self.messages.append(Message(role="system", content=f"参考上下文:\n{context}"))

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    async def chat(
        self,
        prompt: Optional[str] = None,
        use_coder: bool = False,
        temperature: Optional[float] = None,
    ) -> ChatResponse:
        """
        发送聊天消息

        Args:
            prompt: 用户提示词
            use_coder: 是否使用 coder 模型（用于代码任务）
            temperature: 覆盖默认温度参数
        """
        model = self.coder_model if use_coder else self.model
        return await self._do_chat(model, prompt, temperature)

    async def chat_cheap(self, prompt: Optional[str] = None) -> ChatResponse:
        """使用便宜的模型进行一般推理"""
        return await self._do_chat(self.cheap_model, prompt)

    async def chat_code(self, prompt: Optional[str] = None) -> ChatResponse:
        """使用 coder 模型进行代码生成"""
        return await self._do_chat(self.coder_model, prompt)

    def append_tool_result(self, tool_call_id: str, result: str) -> None:
        """追加工具执行结果到消息历史"""
        self.messages.append(Message(
            role="tool", content=str(result), tool_call_id=tool_call_id
        ))

    def clear_messages(self) -> None:
        """清空消息历史"""
        self.messages.clear()

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    async def _do_chat(
        self,
        model: str,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> ChatResponse:
        log_title("CHAT")
        if prompt:
            self.messages.append(Message(role="user", content=prompt))

        response = await self._stream(model, temperature)

        assistant_msg = Message(role="assistant", content=response.content)
        if response.tool_calls:
            assistant_msg.tool_calls = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function_name, "arguments": tc.function_args}}
                for tc in response.tool_calls
            ]
        self.messages.append(assistant_msg)
        return response

    async def _stream(
        self, model: str, temperature: Optional[float] = None
    ) -> ChatResponse:
        log_section("RESPONSE")
        api_messages = [m.to_dict() for m in self.messages]
        api_tools = self._format_tools()

        stream = self._client.chat.completions.create(
            model=model,
            messages=api_messages,
            stream=True,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=self.max_tokens,
            tools=api_tools,
            tool_choice="auto" if api_tools else None,
        )

        content_parts: List[str] = []
        buffers: Dict[int, ToolCall] = {}

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue
            if delta.content:
                content_parts.append(delta.content)
                print(delta.content, end="", flush=True)
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in buffers:
                        buffers[idx] = ToolCall(id="", function_name="", function_args="")
                    b = buffers[idx]
                    if tc_delta.id:
                        b.id += tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            b.function_name += tc_delta.function.name
                        if tc_delta.function.arguments:
                            b.function_args += tc_delta.function.arguments

        print()
        return ChatResponse(
            content="".join(content_parts),
            tool_calls=list(buffers.values()),
        )

    def _format_tools(self) -> Optional[List[Dict[str, Any]]]:
        if not self.tools:
            return None
        return [{
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("inputSchema", t.get("parameters", {})),
            }
        } for t in self.tools]
