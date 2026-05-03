#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CRAG 检索器 - 应用层封装
=========================

基于 supports.crag.CRAGEngine 的应用层封装，
提供面向 Agent 的便捷接口。

同时支持:
- 远程 Embedding API 的语义向量检索 (可选)
- 本地 TF-IDF 的轻量检索 (默认)
- CRAG 纠正流程 (Retrieve → Grade → Correct)

用法:
    retriever = CRAGRetriever()
    retriever.load_knowledge_dir("knowledge/")
    context = retriever.retrieve("查询文本", top_k=5)
"""

import os
import math
from pathlib import Path
from typing import List, Optional

import requests
from dotenv import load_dotenv

from supports.crag import CRAGEngine, CRAGResult
from .utils import log_title, log_section

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ---------------------------------------------------------------------------
# 远程 Embedding 客户端 (可选增强)
# ---------------------------------------------------------------------------

class RemoteEmbedder:
    """远程 Embedding API 客户端，用于语义向量增强"""

    def __init__(
        self,
        model: str = "BAAI/bge-large-zh-v1.5",
        api_key: str = "",
        base_url: str = "",
    ):
        self.model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._base_url = (base_url or os.getenv("OPENAI_BASE_URL",
                          "https://api.siliconflow.cn/v1")).rstrip("/")

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    async def embed(self, text: str) -> Optional[List[float]]:
        """调用远程 API 生成向量"""
        if not self.available:
            return None
        try:
            resp = requests.post(
                f"{self._base_url}/embeddings",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                json={"model": self.model, "input": text, "encoding_format": "float"},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
        except Exception:
            return None


# ---------------------------------------------------------------------------
# CRAGRetriever 主类
# ---------------------------------------------------------------------------

class CRAGRetriever:
    """
    CRAG 检索器

    整合 CRAG 引擎 + 可选远程 Embedding，提供统一的检索接口。

    两种检索模式:
    1. 纯本地 CRAG: 使用 TF-IDF + 相关性评估 + 纠正流程
    2. 混合 CRAG: 远程 Embedding 语义检索 + CRAG 纠正流程
    """

    def __init__(
        self,
        use_remote: bool = False,
        embedding_model: str = "BAAI/bge-large-zh-v1.5",
        api_key: str = "",
        base_url: str = "",
    ):
        self._engine = CRAGEngine()
        self._embedder = RemoteEmbedder(embedding_model, api_key, base_url)
        self._use_remote = use_remote and self._embedder.available
        self._remote_vectors: List[List[float]] = []
        self._remote_docs: List[str] = []

    # ------------------------------------------------------------------
    # 文档管理
    # ------------------------------------------------------------------

    def add_document(self, document: str) -> int:
        """添加单个文档到索引"""
        return self._engine.index_document(document)

    def add_documents(self, documents: List[str]) -> List[int]:
        """批量添加文档"""
        return self._engine.index_documents(documents)

    def load_knowledge_dir(self, directory: str, pattern: str = "*.md") -> int:
        """
        从目录加载知识文档

        Args:
            directory: 知识库目录路径
            pattern: 文件通配符

        Returns:
            加载的文档数量
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            return 0

        count = 0
        for f in sorted(dir_path.glob(pattern)):
            try:
                content = f.read_text(encoding="utf-8")
                if content.strip():
                    self._engine.index_document(content)
                    count += 1
            except Exception:
                continue
        return count

    async def add_document_remote(self, document: str) -> None:
        """使用远程 Embedding 索引文档（用于混合模式）"""
        if not self._embedder.available:
            return
        vec = await self._embedder.embed(document)
        if vec:
            self._remote_vectors.append(vec)
            self._remote_docs.append(document)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        enable_correction: bool = True,
    ) -> CRAGResult:
        """
        执行 CRAG 检索

        Args:
            query: 查询文本
            top_k: 初始检索文档数
            enable_correction: 是否启用纠正（False 则为简单 RAG）

        Returns:
            CRAGResult: 包含相关文档和操作日志
        """
        log_title("CRAG RETRIEVAL")
        result = self._engine.process(
            query, top_k=top_k, enable_correction=enable_correction
        )

        # 打印操作日志
        for action in result.action_log:
            print(f"  {action}")

        log_section("RESULT")
        print(f"  最终上下文: {len(result.docs)} 个文档片段")

        return result

    def get_context(
        self,
        query: str,
        top_k: int = 5,
        enable_correction: bool = True,
    ) -> str:
        """
        便捷方法: 直接返回上下文字符串

        Args:
            query: 查询文本
            top_k: 检索文档数
            enable_correction: 是否启用 CRAG 纠正

        Returns:
            拼接后的上下文字符串
        """
        result = self.retrieve(query, top_k, enable_correction)
        if result.corrected_context:
            return result.corrected_context
        # 回退: 直接拼接
        return "\n\n---\n\n".join(d.content for d in result.docs)

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    @property
    def doc_count(self) -> int:
        return self._engine.get_stats()["total_docs"]

    def get_stats(self) -> dict:
        return self._engine.get_stats()

    def clear(self) -> None:
        self._engine.clear()
        self._remote_vectors.clear()
        self._remote_docs.clear()
