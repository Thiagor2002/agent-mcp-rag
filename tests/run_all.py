#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""运行所有测试"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_crag import (
    test_vector_store, test_relevance_grader, test_crag_engine,
    test_crag_retriever, test_crag_stats,
)
from tests.test_llm import (
    test_token_counter, test_config, test_message_model,
)

# 导出测试函数供外部使用
__all__ = [
    "test_vector_store", "test_relevance_grader", "test_crag_engine",
    "test_crag_retriever", "test_crag_stats",
    "test_token_counter", "test_config", "test_message_model",
]


async def run_all():
    print("=" * 60)
    print("  Agent MCP RAG - 全部测试")
    print("=" * 60)

    # Unit tests (no API needed)
    test_vector_store()
    test_relevance_grader()
    test_crag_engine()
    test_crag_retriever()
    test_crag_stats()
    test_token_counter()
    test_config()
    test_message_model()

    # Coder tests (no API needed for most)
    from tests.test_coder import (
        test_tool_definitions, test_security, test_whitelist,
    )
    test_tool_definitions()
    test_security()
    test_whitelist()

    print("\n" + "=" * 60)
    print("  所有单元测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all())
