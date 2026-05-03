#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主入口模块 - Augmented LLM Chatbot Agent (Chat + MCP + RAG)

演示完整的 RAG + MCP Agent 工作流：
1. 从本地知识库中检索相关信息（RAG）
2. 将检索到的上下文注入 LLM
3. LLM 通过 MCP 工具完成实际任务（如获取网页、保存文件）

任务示例：从上下文中找到 Antonette 的信息，创作故事并保存为 Markdown 文件
"""

import asyncio
import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径
ROOT_PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_PATH))

from src.agent import Agent
from src.mcp_client import MCPClient
from src.embedding_retriever import EmbeddingRetriever
from src.utils import log_title, ensure_dir

# ---------------------------------------------------------------------------
# 配置常量
# ---------------------------------------------------------------------------

URL = "https://news.ycombinator.com/"
OUT_PATH = ROOT_PATH / "output"
KNOWLEDGE_PATH = ROOT_PATH / "knowledge"

# 使用硅基流动最便宜的模型
MODEL = "Qwen/Qwen2.5-7B-Instruct"

TASK = f"""
告诉我Antonette的信息，先从我给你的context中找到相关信息，
总结后创作一个关于她的故事。把故事和她的基本信息保存到
{OUT_PATH}/antonette.md，输出一个漂亮的 Markdown 文件。
"""


# ---------------------------------------------------------------------------
# MCP 客户端初始化
# ---------------------------------------------------------------------------

def _create_mcp_clients() -> list:
    """创建并配置 MCP 客户端"""
    ensure_dir(OUT_PATH)

    fetch_mcp = MCPClient(
        name="fetch",
        command="uvx",
        args=["mcp-server-fetch"],
    )
    file_mcp = MCPClient(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(OUT_PATH)],
    )
    return [fetch_mcp, file_mcp]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

async def retrieve_context(task: str) -> str:
    """
    RAG 检索：从知识库中找到与任务最相关的文档

    Args:
        task: 任务描述，用作检索查询

    Returns:
        拼接后的上下文字符串
    """
    log_title("RAG RETRIEVAL")

    retriever = EmbeddingRetriever()

    # 加载本地知识库
    if KNOWLEDGE_PATH.exists():
        for md_file in KNOWLEDGE_PATH.glob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            if content.strip():
                await retriever.embed_document(content)
                print(f"  已索引: {md_file.name}")
    else:
        print(f"  知识库目录不存在: {KNOWLEDGE_PATH}")

    # 检索相关文档
    docs = await retriever.retrieve(task, top_k=3)

    log_title("CONTEXT")
    context = "\n\n---\n\n".join(docs)
    print(context[:1000] + ("..." if len(context) > 1000 else ""))
    return context


async def main() -> None:
    """主函数：RAG 检索 → Agent 执行 → 保存结果"""

    # Step 1: RAG 检索上下文
    context = await retrieve_context(TASK)

    # Step 2: 创建 Agent 并执行任务
    mcp_clients = _create_mcp_clients()
    agent = Agent(
        model=MODEL,
        mcp_clients=mcp_clients,
        context=context,
    )

    try:
        await agent.init()
        await agent.invoke(TASK)
        log_title("DONE")
        print(f"任务完成，输出文件: {OUT_PATH / 'antonette.md'}")
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
