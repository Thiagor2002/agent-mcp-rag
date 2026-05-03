#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主入口 - Augmented LLM Chatbot Agent (Chat + MCP + CRAG + CLI)
==============================================================

完整演示:
  1. CRAG 检索: 从本地知识库纠正性检索
  2. MCP 工具: 网页抓取 + 文件系统操作
  3. CLI Coder: 轻量级命令执行
  4. Agent 循环: 多轮工具调用自主推理

任务: 从知识库中检索用户信息，创作故事并保存为 Markdown
"""

import asyncio
import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_PATH))

from src.agent import Agent
from src.mcp_client import MCPClient
from src.crag_retriever import CRAGRetriever
from src.coder_cli import CoderCLI
from src.utils import log_title, ensure_dir

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUT_PATH = ROOT_PATH / "output"
KNOWLEDGE_PATH = ROOT_PATH / "knowledge"

TASK = f"""
告诉我Antonette的信息，先从我给你的context中找到相关信息，
总结后创作一个关于她的故事。把故事和她的基本信息保存到
{OUT_PATH}/antonette.md，输出一个漂亮的 Markdown 文件。
"""


# ---------------------------------------------------------------------------
# MCP 客户端
# ---------------------------------------------------------------------------

def _create_mcp_clients() -> list:
    ensure_dir(OUT_PATH)
    fetch_mcp = MCPClient("fetch", "uvx", ["mcp-server-fetch"])
    file_mcp = MCPClient(
        "filesystem", "npx",
        ["-y", "@modelcontextprotocol/server-filesystem", str(OUT_PATH)],
    )
    return [fetch_mcp, file_mcp]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

async def run_crag_retrieval(task: str) -> str:
    """使用 CRAG 从知识库检索相关信息"""
    log_title("CRAG RETRIEVAL")

    retriever = CRAGRetriever()
    loaded = retriever.load_knowledge_dir(str(KNOWLEDGE_PATH))
    print(f"  已加载 {loaded} 个知识文档")

    result = retriever.retrieve(task, top_k=5, enable_correction=True)
    return result.corrected_context


async def main():
    print("=" * 60)
    print("  Augmented LLM Agent (CRAG + MCP + CLI Coder)")
    print("=" * 60)

    # Step 1: CRAG 检索
    context = await run_crag_retrieval(TASK)

    # Step 2: 初始化 Agent
    mcp_clients = _create_mcp_clients()
    coder = CoderCLI(str(ROOT_PATH))

    agent = Agent(
        model=MODEL,
        mcp_clients=mcp_clients,
        coder_cli=coder,
        context=context,
    )

    try:
        await agent.init()
        # Step 3: 执行任务
        await agent.invoke(TASK)
        log_title("DONE")
        print(f"  输出文件: {OUT_PATH / 'antonette.md'}")
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
