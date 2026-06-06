# =============================================================================
# bm25_retriever.py — BM25 关键词检索器
# 基于 rank_bm25 库实现经典的 BM25 检索算法，配合 jieba 中文分词。
# 核心设计：中文分词 + BM25 评分 + 结果结构化封装
# BM25 优势：精确关键词匹配、对术语和专有名词敏感、无需 Embedding 模型
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import jieba
from rank_bm25 import BM25Okapi

from .logger import logger


@dataclass
class BM25Result:
    """BM25 检索结果的数据类。"""
    text: str                   # 文档片段的文本内容
    score: float                # BM25 相关性分数
    source: str                 # 来源文件名
    filepath: str               # 文件路径
    metadata: Dict[str, Any]    # 完整元数据
    doc_index: int              # 文档在语料库中的索引（用于调试）


class BM25Retriever:
    """基于 BM25 算法的关键词检索器。

    使用 jieba 进行中文分词，BM25Okapi 进行文档评分。
    BM25 特别擅长：
    - 精确术语匹配（如 "FastAPI"、"Docker"）
    - 专有名词检索
    - 关键词密度相关的排序
    """

    def __init__(self):
        # BM25 索引实例（构建后赋值）
        self._bm25: Optional[BM25Okapi] = None
        # 语料库——分词后的文档片段列表
        self._tokenized_corpus: List[List[str]] = []
        # 原始文档片段——保留原文用于返回结果
        self._documents: List[Dict[str, Any]] = []
        # 是否已构建索引
        self._is_built: bool = False

    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """从文档片段列表构建 BM25 索引。

        Args:
            documents: 文档片段列表，每个片段是包含 text/metadata 的字典
        """
        if not documents:
            logger.warning("BM25 索引构建失败：文档列表为空")
            return

        self._documents = documents
        self._tokenized_corpus = []

        for doc in documents:
            text = doc.get("text", "")
            # jieba 分词：将中文文本切分为词列表
            # cut_for_search 模式更适合搜索场景（更细粒度的分词）
            tokens = list(jieba.cut_for_search(text))
            # 过滤掉空白 token 和单字符标点
            tokens = [t.strip() for t in tokens if t.strip() and len(t.strip()) > 1]
            self._tokenized_corpus.append(tokens)

        # 构建 BM25 索引
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        self._is_built = True

        logger.info("BM25 索引构建完成，文档片段数: %s", len(documents))

    def retrieve(self, query: str, top_k: int = 8) -> List[BM25Result]:
        """使用 BM25 算法检索最相关的文档片段。

        Args:
            query: 用户查询
            top_k: 返回的最大结果数

        Returns:
            BM25Result 列表，按相关性分数降序排列
        """
        if not self._is_built or self._bm25 is None:
            logger.warning("BM25 索引未构建，无法执行检索")
            return []

        # 对查询进行分词
        query_tokens = list(jieba.cut_for_search(query))
        query_tokens = [t.strip() for t in query_tokens if t.strip() and len(t.strip()) > 1]

        if not query_tokens:
            logger.warning("BM25 查询分词结果为空: %s", query)
            return []

        # BM25 评分
        scores = self._bm25.get_scores(query_tokens)

        # 获取 top_k 个最高分的索引
        # argsort 返回从小到大的索引，取最后 top_k 个再反转
        top_indices = scores.argsort()[-top_k:][::-1]

        results: List[BM25Result] = []
        for idx in top_indices:
            score = float(scores[idx])
            # 跳过分数为 0 的结果（完全不匹配）
            if score <= 0.0:
                continue

            doc = self._documents[idx]
            metadata = doc.get("metadata", {})

            results.append(BM25Result(
                text=doc.get("text", ""),
                score=score,
                source=str(metadata.get("source") or metadata.get("file_name") or "unknown"),
                filepath=str(metadata.get("filepath") or metadata.get("file_path") or "unknown"),
                metadata=metadata,
                doc_index=int(idx),
            ))

        logger.info(
            "BM25 检索完成: 查询词=%s, 候选数=%s, 返回数=%s",
            query_tokens, len(self._documents), len(results),
        )
        return results

    @property
    def is_built(self) -> bool:
        """索引是否已构建。"""
        return self._is_built

    @property
    def corpus_size(self) -> int:
        """语料库中的文档片段总数。"""
        return len(self._documents)
