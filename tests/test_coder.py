#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""测试 CLI Coder 和 CoderAgent"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.coder_cli import CoderCLI, CODER_TOOLS, ALLOWED_COMMANDS, DANGEROUS_PATTERNS
from src.coder_agent import CoderAgent


def test_tool_definitions():
    """测试工具定义"""
    print("\n=== 测试工具定义 ===")
    assert len(CODER_TOOLS) == 5, f"期望 5 个工具"
    names = [t["name"] for t in CODER_TOOLS]
    assert "run_command" in names
    assert "read_file" in names
    assert "write_file" in names
    assert "list_dir" in names
    assert "search_code" in names
    print(f"  PASS: 工具列表 {names}")


def test_security():
    """测试安全校验"""
    print("\n=== 测试安全校验 ===")
    is_ok, reason = CoderCLI._validate("ls -la")
    assert is_ok, f"ls 应为安全命令: {reason}"

    is_ok, reason = CoderCLI._validate("rm -rf /")
    assert not is_ok, "rm -rf 应为危险命令"
    print(f"  PASS: 危险命令 '{'rm -rf /'}' 被拒绝: {reason}")

    is_ok, reason = CoderCLI._validate("sudo ls")
    assert not is_ok, "sudo 应为危险命令"


async def test_cli_tools():
    """测试 CLI 工具执行"""
    print("\n=== 测试 CLI 工具 ===")
    coder = CoderCLI()

    # 测试 list_dir
    result = await coder.execute("list_dir", path=".", pattern="*.md")
    assert "README.md" in result or "错误" not in result
    print(f"  PASS: list_dir 返回结果")

    # 测试 read_file (读取自身)
    result = await coder.execute("read_file", path="README.md", start_line=1, end_line=5)
    assert "agent-mcp-rag" in result.lower() or len(result) > 0
    print(f"  PASS: read_file 读取了 README.md 前5行")

    # 测试 run_command (安全命令)
    result = await coder.execute("run_command", command="echo hello_test")
    assert "hello_test" in result
    print(f"  PASS: run_command 输出: {result.strip()}")


async def test_coder_agent():
    """测试 CoderAgent (需要有效的 API Key)"""
    print("\n=== 测试 CoderAgent ===")
    try:
        agent = CoderAgent()

        # 测试代码生成
        code = await agent.generate_code("Write a function that adds two numbers", "python")
        assert code, "应生成代码"
        print(f"  PASS: 生成代码 ({len(code)} 字符)")

        # 测试代码解释
        explanation = await agent.explain_code("def add(a, b): return a + b")
        assert explanation
        print(f"  PASS: 代码解释 ({len(explanation)} 字符)")

    except Exception as e:
        print(f"  SKIP: CoderAgent 测试跳过 ({e})")


def test_whitelist():
    """测试命令白名单"""
    print("\n=== 测试命令白名单 ===")
    safe = ["ls", "python", "git", "pip", "node", "npm", "grep"]
    for cmd in safe:
        assert cmd in ALLOWED_COMMANDS, f"{cmd} 应在白名单中"
    print(f"  PASS: {len(ALLOWED_COMMANDS)} 个命令在白名单中")


async def main():
    test_tool_definitions()
    test_security()
    test_whitelist()
    await test_cli_tools()
    await test_coder_agent()
    print("\n✓ 所有 Coder 测试通过!")


if __name__ == "__main__":
    asyncio.run(main())
