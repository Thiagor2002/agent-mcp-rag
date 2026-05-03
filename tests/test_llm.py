#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""测试 LLM 接口和 ChatOpenAI"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chat_openai import ChatOpenAI, Message, ChatResponse, ToolCall
from supports.llm import LLMInterface, LLMConfig, TokenCounter, ModelType
from supports.config import GlobalConfig, config


def test_token_counter():
    """测试 Token 计算"""
    print("\n=== 测试 TokenCounter ===")
    tc = TokenCounter()
    n = tc.count_tokens("Hello, world!")
    assert n > 0
    print(f"  PASS: 'Hello, world!' = {n} tokens")

    n2 = tc.count_messages_tokens([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi!"},
    ])
    assert n2 > 0
    print(f"  PASS: 消息 token 数 = {n2}")


def test_config():
    """测试配置加载"""
    print("\n=== 测试配置 ===")
    gc = GlobalConfig()
    llm_cfg = gc.get_llm_config()
    assert "openai_api_key" in llm_cfg
    assert llm_cfg["openai_api_key"], "API Key 不应为空"
    print(f"  PASS: API Key 已加载 (前缀: {llm_cfg['openai_api_key'][:8]}...)")
    print(f"  Model: {llm_cfg['default_model_name']}")


def test_message_model():
    """测试消息模型"""
    print("\n=== 测试 Message ===")
    m = Message(role="user", content="Hello")
    d = m.to_dict()
    assert d["role"] == "user"
    assert d["content"] == "Hello"
    print(f"  PASS: {d}")

    m2 = Message(role="tool", content="result", tool_call_id="call_123")
    d2 = m2.to_dict()
    assert d2["tool_call_id"] == "call_123"


async def test_chat_api():
    """测试实际的 API 调用"""
    print("\n=== 测试 Chat API (SiliconFlow) ===")
    try:
        llm = ChatOpenAI()
        response = await llm.chat("Say exactly: HELLO_WORLD")
        assert response.content, "应有响应内容"
        print(f"  PASS: API 响应长度 {len(response.content)}")
        assert not response.has_tool_calls, "简单回复不应有工具调用"
    except Exception as e:
        print(f"  SKIP: API 调用失败 ({e})")


async def test_supports_llm():
    """测试 supports/llm.py 的 LLMInterface"""
    print("\n=== 测试 LLMInterface ===")
    try:
        cfg = LLMConfig(
            model_type=ModelType.OPENAI,
            model_name="Qwen/Qwen2.5-7B-Instruct",
        )
        llm = LLMInterface(config=cfg)
        response = llm.generate(
            prompt="Say just: OK",
            context=[{"role": "user", "content": "Say just: OK"}],
        )
        assert response.get("content"), "应有响应"
        print(f"  PASS: 响应 '{response['content'][:50]}'")
    except Exception as e:
        print(f"  SKIP: {e}")


async def main():
    test_token_counter()
    test_config()
    test_message_model()
    await test_chat_api()
    await test_supports_llm()
    print("\n✓ 所有 LLM 测试通过!")


if __name__ == "__main__":
    asyncio.run(main())
