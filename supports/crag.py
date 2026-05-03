#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CRAG (Corrective Retrieval-Augmented Generation) 引擎
======================================================

纯 Python 实现的 CRAG 算法，无需依赖 LangChain/LlamaIndex 等框架。

CRAG 算法流程:
  1. Retrieve   - 从知识库检索相关文档
  2. Grade      - 评估每个文档与查询的相关性
  3. Correct    - 纠正/补充检索结果
     - 高相关文档 → 直接使用 (Correct)
     - 模糊相关文档 → 补充知识搜索 (Ambiguous → Web Search)
     - 不相关文档 → 替换为知识搜索结果 (Incorrect → Web Search)
  4. Generate   - 基于纠正后的上下文生成回答

参考论文: https://arxiv.org/pdf/2401.15884
"""

import json
import math
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

class RelevanceLevel(str, Enum):
    """文档相关性等级"""
    CORRECT = "correct"         # 高度相关，直接使用
    AMBIGUOUS = "ambiguous"     # 模糊相关，需要补充搜索
    INCORRECT = "incorrect"     # 不相关，需替换


@dataclass
class RetrievalDoc:
    """检索到的文档"""
    content: str
    source: str = ""
    score: float = 0.0
    relevance: RelevanceLevel = RelevanceLevel.AMBIGUOUS


@dataclass
class CRAGResult:
    """CRAG 处理结果"""
    docs: List[RetrievalDoc] = field(default_factory=list)
    corrected_context: str = ""
    knowledge_search_results: List[str] = field(default_factory=list)
    action_log: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 向量存储 (轻量级，无外部依赖)
# ---------------------------------------------------------------------------

class LightweightVectorStore:
    """
    轻量级向量存储，纯 Python 实现

    使用 TF-IDF 风格的稀疏向量进行初步检索，
    结合关键词匹配进行相关性评估。
    适用于中小规模文档集合 (< 10K 文档)。
    """

    def __init__(self):
        self._documents: List[str] = []
        self._term_index: Dict[str, Dict[int, float]] = {}  # term -> {doc_id: tf}
        self._doc_freq: Dict[str, int] = {}  # term -> document frequency
        self._doc_count = 0

    def add(self, document: str) -> int:
        """添加文档并返回文档 ID"""
        doc_id = self._doc_count
        self._documents.append(document)
        self._doc_count += 1
        self._index_document(doc_id, document)
        return doc_id

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.01,
    ) -> List[Tuple[str, float]]:
        """
        使用 TF-IDF 余弦相似度检索

        Returns:
            List of (document_content, score) sorted by relevance
        """
        if not self._documents:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        # 计算查询向量
        query_vec: Dict[str, float] = {}
        for term in query_terms:
            query_vec[term] = query_vec.get(term, 0.0) + 1.0

        # IDF 权重
        for term in query_vec:
            df = self._doc_freq.get(term, 1)
            query_vec[term] *= math.log((self._doc_count + 1) / (df + 1)) + 1

        # 计算每个文档的相似度
        scores: List[Tuple[int, float]] = []
        for doc_id in range(self._doc_count):
            score = self._cosine_similarity(query_vec, doc_id)
            if score >= min_score:
                scores.append((doc_id, score))

        # 按分数降序排列
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self._documents[doc_id], score) for doc_id, score in scores[:top_k]]

    def _index_document(self, doc_id: int, text: str) -> None:
        """索引文档的术语"""
        terms = self._tokenize(text)
        term_freq: Dict[str, float] = {}
        for term in terms:
            term_freq[term] = term_freq.get(term, 0.0) + 1.0

        for term, tf in term_freq.items():
            if term not in self._term_index:
                self._term_index[term] = {}
            self._term_index[term][doc_id] = tf
            self._doc_freq[term] = self._doc_freq.get(term, 0) + 1

    def _cosine_similarity(self, query_vec: Dict[str, float], doc_id: int) -> float:
        """计算查询向量与文档的余弦相似度"""
        dot = 0.0
        query_norm = 0.0
        doc_norm = 0.0

        for term, q_weight in query_vec.items():
            query_norm += q_weight * q_weight
            if term in self._term_index and doc_id in self._term_index[term]:
                tf = self._term_index[term][doc_id]
                df = self._doc_freq.get(term, 1)
                idf = math.log((self._doc_count + 1) / (df + 1)) + 1
                d_weight = tf * idf
                dot += q_weight * d_weight
                doc_norm += d_weight * d_weight

        if query_norm == 0 or doc_norm == 0:
            return 0.0
        return dot / (math.sqrt(query_norm) * math.sqrt(doc_norm))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        多语言分词（中英文混合）

        英文: 按空格和标点分词，转小写
        中文: 按 2-gram 字符组分词
        """
        # 分离中英文
        tokens = []

        # 英文单词（含数字）
        eng_words = re.findall(r'[a-zA-Z0-9]+', text.lower())
        tokens.extend(eng_words)

        # 中文字符 2-gram
        chinese_chars = re.findall(r'[一-鿿]', text)
        for i in range(len(chinese_chars)):
            if i < len(chinese_chars) - 1:
                tokens.append(chinese_chars[i] + chinese_chars[i + 1])
            else:
                tokens.append(chinese_chars[i])

        return tokens

    @property
    def size(self) -> int:
        return self._doc_count

    def clear(self) -> None:
        self._documents.clear()
        self._term_index.clear()
        self._doc_freq.clear()
        self._doc_count = 0


# ---------------------------------------------------------------------------
# 相关性评估器
# ---------------------------------------------------------------------------

class RelevanceGrader:
    """
    文档相关性评估器

    使用关键词密度和语义相似度评估文档与查询的相关性。

    评估策略:
    1. 关键词覆盖率: query 中的关键词在文档中出现的比例
    2. 语义相关性: 使用词袋向量余弦相似度
    3. 综合评分: 加权平均
    """

    # 相关性阈值
    CORRECT_THRESHOLD = 0.50    # >= 此值: 高相关
    AMBIGUOUS_THRESHOLD = 0.15  # >= 此值且 < CORRECT_THRESHOLD: 模糊相关
    # < AMBIGUOUS_THRESHOLD: 不相关

    def grade(
        self,
        query: str,
        docs: List[Tuple[str, float]],
    ) -> List[RetrievalDoc]:
        """
        评估文档相关性

        Args:
            query: 用户查询
            docs: (doc_content, retrieval_score) 列表

        Returns:
            带相关性标注的文档列表
        """
        results = []
        for content, retrieval_score in docs:
            relevance_score = self._compute_relevance(query, content)
            combined_score = 0.4 * retrieval_score + 0.6 * relevance_score

            if combined_score >= self.CORRECT_THRESHOLD:
                level = RelevanceLevel.CORRECT
            elif combined_score >= self.AMBIGUOUS_THRESHOLD:
                level = RelevanceLevel.AMBIGUOUS
            else:
                level = RelevanceLevel.INCORRECT

            results.append(RetrievalDoc(
                content=content,
                score=combined_score,
                relevance=level,
            ))

        return results

    def _compute_relevance(self, query: str, document: str) -> float:
        """计算查询与文档的关键词相关度"""
        if not query or not document:
            return 0.0

        query_tokens = set(LightweightVectorStore._tokenize(query))
        doc_tokens = LightweightVectorStore._tokenize(document.lower())
        doc_lower = document.lower()

        if not query_tokens:
            return 0.0

        # 关键词覆盖率
        matched = sum(1 for t in query_tokens if t in doc_lower)
        coverage = matched / len(query_tokens)

        # 关键词频率密度
        density = sum(doc_tokens.count(t) for t in query_tokens if t in doc_tokens) / max(len(doc_tokens), 1)

        # 组合评分
        return 0.6 * coverage + 0.4 * min(density * 10, 1.0)


# ---------------------------------------------------------------------------
# 知识搜索器
# ---------------------------------------------------------------------------

class KnowledgeSearch:
    """
    外部知识搜索

    当检索文档不相关或模糊时，通过以下方式补充:
    1. 从已有知识库中扩展搜索
    2. 可选: 调用 Web Search API
    """

    def __init__(self, vector_store: Optional[LightweightVectorStore] = None):
        self._vector_store = vector_store
        self._search_log: List[str] = []

    def search(
        self,
        query: str,
        existing_docs: List[str],
        max_results: int = 3,
    ) -> List[str]:
        """
        执行知识搜索以补充或替换不相关文档

        Args:
            query: 用户查询
            existing_docs: 已有文档（已检索的）
            max_results: 最大返回结果数

        Returns:
            搜索结果列表
        """
        results = []

        # 1. 从本地向量库扩展搜索 (使用查询扩展)
        expanded_queries = self._expand_query(query)
        if self._vector_store and self._vector_store.size > 0:
            for eq in expanded_queries:
                docs = self._vector_store.search(eq, top_k=2, min_score=0.005)
                for content, score in docs:
                    if content not in existing_docs and content not in results:
                        results.append(content)
                        self._search_log.append(f"本地扩展: '{eq}' → {content[:80]}...")
                        if len(results) >= max_results:
                            return results[:max_results]

        return results

    @property
    def search_log(self) -> List[str]:
        return self._search_log

    @staticmethod
    def _expand_query(query: str) -> List[str]:
        """查询扩展: 生成多个查询变体"""
        # 简单实现: 提取关键词组合
        tokens = LightweightVectorStore._tokenize(query)
        expansions = [query]

        if len(tokens) >= 3:
            # 前一半关键词
            expansions.append(" ".join(tokens[:len(tokens)//2]))
            # 后一半关键词
            expansions.append(" ".join(tokens[len(tokens)//2:]))

        return expansions


# ---------------------------------------------------------------------------
# CRAG 主引擎
# ---------------------------------------------------------------------------

class CRAGEngine:
    """
    Corrective RAG 引擎

    完整实现 CRAG 算法:
    Retrieve → Grade → Correct → Generate

    Usage:
        engine = CRAGEngine()
        engine.index_documents(["doc1", "doc2", ...])
        result = engine.process("用户查询")
        print(result.corrected_context)
    """

    def __init__(self):
        self._vector_store = LightweightVectorStore()
        self._grader = RelevanceGrader()
        self._knowledge_search = KnowledgeSearch(self._vector_store)
        self._action_log: List[str] = []

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def index_document(self, document: str) -> int:
        """索引单个文档"""
        return self._vector_store.add(document)

    def index_documents(self, documents: List[str]) -> List[int]:
        """批量索引文档"""
        return [self._vector_store.add(doc) for doc in documents]

    def process(
        self,
        query: str,
        top_k: int = 5,
        enable_correction: bool = True,
    ) -> CRAGResult:
        """
        执行完整的 CRAG 流程

        Args:
            query: 用户查询
            top_k: 初始检索文档数
            enable_correction: 是否启用纠正步骤

        Returns:
            CRAGResult: 包含处理后的上下文和操作日志
        """
        self._action_log.clear()
        result = CRAGResult()

        # Step 1: Retrieve
        self._log(f"[Retrieve] 检索 top_k={top_k} 文档")
        raw_docs = self._vector_store.search(query, top_k=top_k)
        if not raw_docs:
            self._log("[Retrieve] 未检索到文档，尝试知识搜索")
            if enable_correction:
                raw_docs = [(r, 0.1) for r in self._knowledge_search.search(query, [])]
            if not raw_docs:
                return result

        # Step 2: Grade
        self._log(f"[Grade] 评估 {len(raw_docs)} 个文档的相关性")
        graded_docs = self._grader.grade(query, raw_docs)
        result.docs = graded_docs

        correct_docs = []
        ambiguous_docs = []
        incorrect_docs = []

        for doc in graded_docs:
            if doc.relevance == RelevanceLevel.CORRECT:
                correct_docs.append(doc)
            elif doc.relevance == RelevanceLevel.AMBIGUOUS:
                ambiguous_docs.append(doc)
            else:
                incorrect_docs.append(doc)

        self._log(
            f"[Grade] 结果: {len(correct_docs)} 高相关, "
            f"{len(ambiguous_docs)} 模糊, "
            f"{len(incorrect_docs)} 不相关"
        )

        if not enable_correction:
            result.corrected_context = self._docs_to_context(graded_docs)
            return result

        # Step 3: Correct
        final_docs = list(correct_docs)

        # 对模糊文档进行补充搜索
        if ambiguous_docs:
            self._log(f"[Correct] 对 {len(ambiguous_docs)} 个模糊文档进行知识补充")
            existing = [d.content for d in final_docs + ambiguous_docs + incorrect_docs]
            supplement = self._knowledge_search.search(query, existing, max_results=3)
            result.knowledge_search_results = supplement
            for s in supplement:
                # 补充结果标记为模糊相关
                final_docs.append(RetrievalDoc(
                    content=s,
                    source="knowledge_search",
                    score=0.3,
                    relevance=RelevanceLevel.AMBIGUOUS,
                ))
            # 保留模糊文档
            final_docs.extend(ambiguous_docs)

        # 对不相关文档进行替换搜索
        if incorrect_docs:
            self._log(f"[Correct] {len(incorrect_docs)} 个文档不相关，替换为知识搜索结果")
            existing = [d.content for d in final_docs]
            replacement = self._knowledge_search.search(query, existing, max_results=3)
            for r in replacement:
                if r not in [d.content for d in final_docs]:
                    result.knowledge_search_results.append(r)
                    final_docs.append(RetrievalDoc(
                        content=r,
                        source="knowledge_search",
                        score=0.2,
                        relevance=RelevanceLevel.AMBIGUOUS,
                    ))

        # Step 4: 构建纠正后的上下文
        result.corrected_context = self._docs_to_context(final_docs)
        result.action_log = list(self._action_log)

        self._log(f"[Done] 最终上下文: {len(final_docs)} 个文档片段")
        return result

    def retrieve_only(self, query: str, top_k: int = 5) -> List[str]:
        """仅执行检索（不进行纠正），用于简单 RAG 场景"""
        raw_docs = self._vector_store.search(query, top_k=top_k)
        return [doc for doc, _ in raw_docs]

    def get_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        return {
            "total_docs": self._vector_store.size,
            "action_log": self._action_log,
        }

    def clear(self) -> None:
        """清空所有索引和日志"""
        self._vector_store.clear()
        self._action_log.clear()

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        self._action_log.append(msg)

    @staticmethod
    def _docs_to_context(docs: List[RetrievalDoc]) -> str:
        """将文档列表拼接为上下文字符串"""
        if not docs:
            return ""
        parts = []
        for i, doc in enumerate(docs):
            prefix = f"[文档 {i+1}]" + (f" ({doc.source})" if doc.source else "")
            parts.append(f"{prefix}\n{doc.content}")
        return "\n\n---\n\n".join(parts)
