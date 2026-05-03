#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
向量检索模块 (RAG - Retrieval Augmented Generation)
提供基于 API 的文本向量化和语义检索功能，包括：
- 文档向量化存储
- 查询向量化
- 余弦相似度检索

使用硅基流动 (SiliconFlow) 的 Embedding API，
默认模型为 BAAI/bge-large-zh-v1.5（中文优化，高性价比）。
"""

import os
import math
import json
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass, field

import requests
from dotenv import load_dotenv

from .utils import log_title, log_section

# 加载环境变量
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ---------------------------------------------------------------------------
# 默认配置常量
# ---------------------------------------------------------------------------

SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"


def _resolve_api_key() -> str:
    """安全地从环境变量获取 API Key"""
    return os.getenv("OPENAI_API_KEY", os.getenv("SILICONFLOW_API_KEY", ""))


def _resolve_api_base() -> str:
    """安全地从环境变量获取 API Base URL"""
    return os.getenv("OPENAI_BASE_URL", os.getenv("SILICONFLOW_BASE_URL", SILICONFLOW_BASE_URL))


# ---------------------------------------------------------------------------
# 向量存储
# ---------------------------------------------------------------------------

class VectorStore:
    """
    简单的内存向量存储

    存储文档及其对应的向量表示，支持基于余弦相似度的语义检索。
    适用于小规模文档集合（< 10K 文档），大规模场景可替换为 FAISS / ChromaDB。
    """

    def __init__(self):
        self._vectors: List[List[float]] = []
        self._documents: List[str] = []

    def add(self, vector: List[float], document: str) -> None:
        """
        添加文档向量

        Args:
            vector: 文档向量
            document: 原始文档内容
        """
        self._vectors.append(vector)
        self._documents.append(document)

    def search(self, query_vector: List[float], top_k: int = 3) -> List[str]:
        """
        基于余弦相似度检索最相关的 top_k 个文档

        Args:
            query_vector: 查询向量
            top_k: 返回结果数

        Returns:
            最相关文档内容列表
        """
        if not self._vectors:
            return []

        similarities = [
            self._cosine_similarity(query_vector, doc_vec)
            for doc_vec in self._vectors
        ]

        # 按相似度降序取 top_k
        top_indices = sorted(
            range(len(similarities)),
            key=lambda i: similarities[i],
            reverse=True,
        )[:top_k]

        return [self._documents[i] for i in top_indices]

    @property
    def size(self) -> int:
        return len(self._documents)

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """计算两个向量的余弦相似度"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# EmbeddingRetriever 主类
# ---------------------------------------------------------------------------

class EmbeddingRetriever:
    """
    RAG 检索器，负责文档向量化和语义检索

    使用远程 Embedding API 生成向量，本地使用 VectorStore 存储和检索。

    Usage:
        retriever = EmbeddingRetriever()
        await retriever.embed_document("这是一段知识文本...")
        results = await retriever.retrieve("查询文本", top_k=3)
    """

    def __init__(
        self,
        model: str = SILICONFLOW_EMBEDDING_MODEL,
        api_key: str = "",
        base_url: str = "",
    ):
        """
        初始化检索器

        Args:
            model: Embedding 模型名称
            api_key: API 密钥（为空则从环境变量读取）
            base_url: API 地址（为空则从环境变量读取）
        """
        self.model = model
        self._api_key = api_key or _resolve_api_key()
        self._base_url = (base_url or _resolve_api_base()).rstrip("/")
        self._vector_store = VectorStore()

        if not self._api_key:
            raise ValueError("未设置 API Key，请设置 SILICONFLOW_API_KEY 环境变量")

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    async def embed_document(self, document: str) -> List[float]:
        """
        将文档向量化并存储到向量库中

        Args:
            document: 文档内容

        Returns:
            文档向量
        """
        log_section("EMBEDDING DOCUMENT")
        vector = await self._get_embedding(document)
        self._vector_store.add(vector, document)
        return vector

    async def embed_query(self, query: str) -> List[float]:
        """
        将查询文本向量化

        Args:
            query: 查询文本

        Returns:
            查询向量
        """
        log_section("EMBEDDING QUERY")
        return await self._get_embedding(query)

    async def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        检索与查询最相关的 top_k 个文档

        Args:
            query: 查询文本
            top_k: 返回文档数

        Returns:
            最相关文档列表
        """
        query_vector = await self.embed_query(query)
        return self._vector_store.search(query_vector, top_k)

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    async def _get_embedding(self, text: str) -> List[float]:
        """
        调用远程 Embedding API 生成文本向量

        Args:
            text: 输入文本

        Returns:
            浮点数向量列表
        """
        url = f"{self._base_url}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        payload = {
            "model": self.model,
            "input": text,
            "encoding_format": "float",
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Embedding API 调用失败: {e}") from e
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Embedding API 返回格式异常: {e}") from e
