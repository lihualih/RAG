# =============================================================================
# retriever.py — 检索过滤器
# 封装 LlamaIndex 的检索入口，增加相似度阈值过滤逻辑。
# 核心设计：Top-K 检索 + 相关性二次过滤 + 结构化结果封装
# =============================================================================

# 从 __future__ 导入 annotations，让所有类型注解使用延迟求值
from __future__ import annotations

# dataclass：数据类装饰器，自动生成 __init__ / __repr__ / __eq__
from dataclasses import dataclass
# typing 的类型标注工具
from typing import Any, Dict, List

# VectorStoreIndex：LlamaIndex 的向量索引类，提供检索入口
from llama_index.core import VectorStoreIndex

# 导入项目统一的日志器
from .logger import logger


# 检索结果的数据类 —— 用 @dataclass 保证类型安全
@dataclass
class RetrievedChunk:
    """检索到的文档片段，包含文本内容和溯源信息。"""
    text: str                   # 文档片段的文本内容
    score: float                # 相似度分数（0~1，Cosine 距离）
    source: str                 # 来源文件名（如 "fastapi_intro.md"）
    filepath: str               # 文件的绝对路径
    metadata: Dict[str, Any]    # 完整的元数据字典（包含 source/filepath/file_type 等）


class TechDocRetriever:
    """封装 RAG 检索逻辑与相关性过滤。"""

    def __init__(
        self,
        index: VectorStoreIndex,        # LlamaIndex 的向量索引实例
        top_k: int,                     # 检索时返回的最大文档片段数
        similarity_threshold: float,    # 相似度阈值——低于此值的片段被丢弃
    ):
        # 从索引创建检索器
        # index.as_retriever(similarity_top_k=top_k)：创建一个检索器实例
        # similarity_top_k 指定了每次检索返回的最大结果数
        self.retriever = index.as_retriever(similarity_top_k=top_k)
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """检索并按阈值过滤弱相关结果。
        核心流程：原始检索 → 遍历结果 → 阈值过滤 → 构建结构化对象。"""

        # 调用 LlamaIndex 检索器执行向量检索
        # 返回的是 List[NodeWithScore]——每个结果包含文档节点和相似度分数
        raw_nodes = self.retriever.retrieve(query)
        # 存储过滤后的结果
        filtered: List[RetrievedChunk] = []

        # 遍历原始检索结果
        for node_with_score in raw_nodes:
            # 提取分数——如果 score 是 None，默认为 0.0（这种节点无法通过阈值）
            score = float(node_with_score.score) if node_with_score.score is not None else 0.0
            # 相似度过滤：分数低于阈值的节点直接丢弃
            # 这是关键逻辑——向量检索总是会返回"距离最近"的结果，
            # 但"距离最近"不等于"内容相关"，需要用阈值做质量把关
            if score < self.similarity_threshold:
                continue

            # 提取元数据——如果为 None 用空字典兜底
            metadata = dict(node_with_score.node.metadata or {})
            # 提取来源文件名——容错：先用 source 字段，没有就用 file_name，都没有用 "unknown"
            source = str(metadata.get("source") or metadata.get("file_name") or "unknown")
            # 提取文件路径——同样的容错逻辑
            filepath = str(metadata.get("filepath") or metadata.get("file_path") or "unknown")

            # 构建 RetrievedChunk 对象并加入过滤后的结果列表
            filtered.append(
                RetrievedChunk(
                    text=node_with_score.node.get_content(),  # 获取节点的文本内容
                    score=score,
                    source=source,
                    filepath=filepath,
                    metadata=metadata,
                )
            )

        # 记录检索统计：原始命中数 vs 过滤后数量 vs 阈值
        # 这个日志对调试和优化检索参数非常有价值
        logger.info(
            "检索完成: 原始命中=%s, 阈值过滤后=%s, threshold=%s",
            len(raw_nodes),     # 原始检索结果数（= top_k）
            len(filtered),      # 过滤后剩余的数量
            self.similarity_threshold,  # 使用的过滤阈值
        )
        return filtered
