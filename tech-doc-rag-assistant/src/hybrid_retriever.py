# =============================================================================
# hybrid_retriever.py — 混合检索器（BM25 + 向量检索 + RRF 融合）
# 将 BM25 关键词检索和向量语义检索的结果通过倒数排名融合（RRF）合并。
# 核心设计：双通道检索 + RRF 排名融合 + 去重 + 阈值过滤
#
# 为什么需要混合检索？
# - 向量检索：擅长语义理解（"Python 数据类型" 能匹配到 "列表、字典、元组"）
# - BM25 检索：擅长精确匹配（"FastAPI" 精确匹配到包含 "FastAPI" 的文档）
# - 两者互补，RRF 融合无需额外训练，是业界最常用的混合检索策略
# =============================================================================

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .bm25_retriever import BM25Retriever, BM25Result
from .logger import logger
from .retriever import RetrievedChunk, TechDocRetriever


class HybridRetriever:
    """融合 BM25 和向量检索的混合检索器。

    使用倒数排名融合（Reciprocal Rank Fusion, RRF）算法合并两路结果。
    RRF 公式：score(d) = Σ 1 / (k + rank_i(d))
    其中 k 是平滑常数（默认 60），rank_i 是文档在第 i 路检索中的排名。
    """

    def __init__(
        self,
        vector_retriever: TechDocRetriever,    # 向量检索器
        bm25_retriever: BM25Retriever,          # BM25 检索器
        top_k: int = 6,                         # 最终返回的结果数
        rrf_k: int = 60,                        # RRF 平滑常数
        similarity_threshold: float = 0.0,      # 融合后的最低分数阈值
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.similarity_threshold = similarity_threshold

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """执行混合检索：双通道检索 → RRF 融合 → 去重 → 排序 → 截断。"""

        # ---- 通道 1：向量语义检索 ----
        vector_chunks = self.vector_retriever.retrieve(query)
        logger.info("向量检索返回: %s 个结果", len(vector_chunks))

        # ---- 通道 2：BM25 关键词检索 ----
        bm25_top_k = max(self.top_k * 2, 10)  # BM25 多取一些，增加融合覆盖面
        bm25_results = self.bm25_retriever.retrieve(query, top_k=bm25_top_k)
        logger.info("BM25 检索返回: %s 个结果", len(bm25_results))

        # ---- RRF 融合 ----
        fused = self._rrf_fusion(vector_chunks, bm25_results)

        # ---- 阈值过滤 ----
        if self.similarity_threshold > 0:
            fused = [
                (chunk, score) for chunk, score in fused
                if score >= self.similarity_threshold
            ]

        # ---- 截断到 top_k ----
        fused = fused[:self.top_k]

        # ---- 构建最终结果 ----
        result_chunks: List[RetrievedChunk] = []
        for chunk, rrf_score in fused:
            # 用 RRF 融合分数替换原始分数
            chunk.score = rrf_score
            result_chunks.append(chunk)

        logger.info(
            "混合检索完成: 向量=%s, BM25=%s, 融合后=%s, 最终返回=%s",
            len(vector_chunks), len(bm25_results),
            len(fused), len(result_chunks),
        )
        return result_chunks

    def _rrf_fusion(
        self,
        vector_chunks: List[RetrievedChunk],
        bm25_results: List[BM25Result],
    ) -> List[Tuple[RetrievedChunk, float]]:
        """执行倒数排名融合（RRF）。

        RRF 核心思想：不直接比较分数（向量余弦相似度 vs BM25 分数无法直接比较），
        而是用排名来融合——排名越靠前，RRF 分数越高。

        公式：RRF_score(d) = Σ 1 / (k + rank_i(d))
        """

        # 用文本内容作为去重键（相同内容的文档片段只保留一个）
        # 存储：text → (RetrievedChunk, rrf_score)
        score_map: Dict[str, Tuple[RetrievedChunk, float]] = {}

        # ---- 处理向量检索结果 ----
        for rank, chunk in enumerate(vector_chunks):
            text_key = chunk.text.strip()
            rrf_score = 1.0 / (self.rrf_k + rank + 1)

            if text_key in score_map:
                # 已存在——累加 RRF 分数
                existing_chunk, existing_score = score_map[text_key]
                score_map[text_key] = (existing_chunk, existing_score + rrf_score)
            else:
                score_map[text_key] = (chunk, rrf_score)

        # ---- 处理 BM25 检索结果 ----
        for rank, bm25_result in enumerate(bm25_results):
            text_key = bm25_result.text.strip()
            rrf_score = 1.0 / (self.rrf_k + rank + 1)

            if text_key in score_map:
                # 已存在——累加 RRF 分数
                existing_chunk, existing_score = score_map[text_key]
                score_map[text_key] = (existing_chunk, existing_score + rrf_score)
            else:
                # BM25 独有的结果——转换为 RetrievedChunk
                new_chunk = RetrievedChunk(
                    text=bm25_result.text,
                    score=0.0,  # 临时分数，后面会被 RRF 分数覆盖
                    source=bm25_result.source,
                    filepath=bm25_result.filepath,
                    metadata=bm25_result.metadata,
                )
                score_map[text_key] = (new_chunk, rrf_score)

        # ---- 按 RRF 分数降序排序 ----
        sorted_results = sorted(
            score_map.values(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_results
