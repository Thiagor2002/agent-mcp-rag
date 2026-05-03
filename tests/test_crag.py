#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""测试 CRAG 引擎和检索器"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supports.crag import (
    CRAGEngine, LightweightVectorStore, RelevanceGrader,
    KnowledgeSearch, RelevanceLevel, RetrievalDoc,
)
from src.crag_retriever import CRAGRetriever


def test_vector_store():
    """测试轻量级向量存储"""
    print("\n=== 测试 LightweightVectorStore ===")
    vs = LightweightVectorStore()

    vs.add("Python is a programming language for AI and data science")
    vs.add("Cooking pasta requires boiling water and adding salt")
    vs.add("Machine learning uses Python for training models")

    assert vs.size == 3, f"期望 3 个文档，实际 {vs.size}"

    results = vs.search("Python AI machine learning", top_k=2)
    assert len(results) > 0, "搜索应返回结果"
    assert "Python" in results[0][0], f"最相关文档应包含 'Python'"
    print(f"  PASS: 存储 {vs.size} 文档, 搜索返回 {len(results)} 结果")


def test_relevance_grader():
    """测试相关性评估器"""
    print("\n=== 测试 RelevanceGrader ===")
    grader = RelevanceGrader()

    docs = [
        ("Antonette Williams is a software engineer in San Francisco.", 0.8),
        ("Cooking Italian pasta is a popular hobby.", 0.1),
    ]
    query = "Who is Antonette and where does she work?"

    graded = grader.grade(query, docs)
    assert len(graded) == 2
    assert graded[0].relevance in (RelevanceLevel.CORRECT, RelevanceLevel.AMBIGUOUS)
    print(f"  PASS: '{query[:30]}...' → 第一个文档: {graded[0].relevance.value}")


def test_crag_engine():
    """测试完整 CRAG 引擎"""
    print("\n=== 测试 CRAGEngine ===")
    engine = CRAGEngine()

    engine.index_documents([
        "Antonette Williams, 28, software engineer at Synthwave Systems in San Francisco. She studied at Stanford.",
        "Bob Chen, 35, data scientist at ByteDance in Beijing. NLP and recommendation systems expert.",
        "Carla Martinez, 31, product designer at Figma in Barcelona. UI/UX and design systems.",
        "Python programming guide: decorators, generators, async/await patterns.",
        "Machine learning basics: supervised learning, neural networks, and transformers.",
    ])

    # 测试简单 RAG (无纠正)
    result = engine.process("Antonette software engineer", top_k=3, enable_correction=False)
    assert len(result.docs) > 0, "应检索到文档"
    assert any("Antonette" in d.content for d in result.docs), "应包含 Antonette"
    print(f"  PASS (RAG): 检索到 {len(result.docs)} 个文档")

    # 测试 CRAG (带纠正)
    result2 = engine.process("Antonette software engineer", top_k=3, enable_correction=True)
    assert len(result2.action_log) > 0, "应有操作日志"
    assert result2.corrected_context, "应有纠正后上下文"
    print(f"  PASS (CRAG): {len(result2.action_log)} 条操作日志")
    for log in result2.action_log:
        print(f"    {log}")


def test_crag_retriever():
    """测试应用层 CRAG 检索器"""
    print("\n=== 测试 CRAGRetriever ===")
    retriever = CRAGRetriever()
    retriever.add_documents([
        "Antonette Williams is a 28-year-old software engineer in San Francisco.",
        "She works at Synthwave Systems and contributes to open source projects.",
        "Her hobbies include rock climbing and electronic music production.",
    ])

    context = retriever.get_context("Who is Antonette?", top_k=2, enable_correction=True)
    assert "Antonette" in context
    print(f"  PASS: 上下文长度 {len(context)} 字符")


def test_crag_stats():
    """测试引擎统计"""
    print("\n=== 测试 CRAG 统计 ===")
    engine = CRAGEngine()
    engine.index_documents(["Doc A", "Doc B", "Doc C"])
    stats = engine.get_stats()
    assert stats["total_docs"] == 3
    print(f"  PASS: 统计数据 {stats}")


if __name__ == "__main__":
    test_vector_store()
    test_relevance_grader()
    test_crag_engine()
    test_crag_retriever()
    test_crag_stats()
    print("\n✓ 所有 CRAG 测试通过!")
