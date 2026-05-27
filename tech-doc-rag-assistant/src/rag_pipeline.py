# =============================================================================
# rag_pipeline.py — 流程编排器（门面模式 Facade Pattern）
# 将检索（Retriever）和生成（Generator）串联为一个统一的问答接口。
# 核心设计：单一入口 ask() 方法 + 结构化返回 + 引用来源构建
# =============================================================================

# 从 __future__ 导入 annotations，让所有类型注解使用延迟求值
from __future__ import annotations

# typing 的类型标注工具：Any（任意类型）、Dict（字典泛型）、List（列表泛型）
from typing import Any, Dict, List

# Settings：项目的配置类（只用于类型标注）
from .config import Settings
# AnswerGenerator：答案生成器
from .generator import AnswerGenerator
# TechDocRetriever：检索过滤器
from .retriever import TechDocRetriever
# truncate_text：文本压缩截断工具函数
from .utils import truncate_text


# 预设消息：当检索不到相关内容时返回的统一提示
NO_CONTEXT_MSG = "未在知识库中检索到足够相关内容，请尝试换一个问法或补充相关文档。"


class RAGPipeline:
    """统一编排检索与生成流程，供前端直接调用。
    这是整个系统对外暴露的唯一接口——前端 app.py 只需要调用 ask() 方法。"""

    def __init__(self, retriever: TechDocRetriever, settings: Settings):
        # 持有检索器实例
        self.retriever = retriever
        # 持有答案生成器实例
        # 注意：AnswerGenerator 只需要 settings（用来创建 OpenAI 客户端），不需要 retriever
        self.generator = AnswerGenerator(settings)

    def ask(self, question: str) -> Dict[str, Any]:
        """用户提问的核心方法。
        输入：用户问题字符串
        输出：包含 answer（答案）、citations（引用）、retrieved_chunks（原文片段）的字典"""

        # ---- Step 1：问题预处理 ----
        # strip() 去除首尾空白
        clean_question = question.strip()
        # 空问题——直接返回提示
        if not clean_question:
            return {
                "answer": "请输入有效问题。",
                "citations": [],
                "retrieved_chunks": [],
            }

        # ---- Step 2：语义检索 ----
        # 调用检索器从知识库中找到相关文档片段
        chunks = self.retriever.retrieve(clean_question)

        # ---- Step 3：空结果检查 ----
        # 检索不到任何结果（可能全部被阈值过滤了）——返回预设提示
        if not chunks:
            return {
                "answer": NO_CONTEXT_MSG,
                "citations": [],
                "retrieved_chunks": [],
            }

        # ---- Step 4：LLM 答案生成 ----
        # 将问题和检索到的上下文一起交给 LLM
        answer = self.generator.generate(clean_question, chunks)

        # ---- Step 5：构建 citations 列表（供前端展示引用来源） ----
        citations: List[Dict[str, Any]] = []
        retrieved_chunks: List[Dict[str, Any]] = []
        for chunk in chunks:
            # citations：精简版——只包含前端展示需要的字段
            citations.append(
                {
                    "source": chunk.source,                          # 来源文件名
                    "filepath": chunk.filepath,                      # 文件路径
                    # snippet：截断后的文本摘要（最长 180 字符，美化展示）
                    "snippet": truncate_text(chunk.text, max_length=180),
                    # score：四舍五入到 4 位小数，便于前端显示百分比
                    "score": round(chunk.score, 4),
                }
            )
            # retrieved_chunks：完整版——保留全部信息供调试使用
            retrieved_chunks.append(
                {
                    "text": chunk.text,        # 完整的文本内容
                    "score": chunk.score,      # 原始的相似度分数
                    "source": chunk.source,    # 来源文件名
                    "filepath": chunk.filepath, # 文件路径
                    "metadata": chunk.metadata, # 完整的元数据字典
                }
            )

        # ---- Step 6：返回结构化结果 ----
        return {
            "answer": answer,              # LLM 生成的答案文本
            "citations": citations,        # 引用来源（精简版，前端展示用）
            "retrieved_chunks": retrieved_chunks,  # 检索原文片段（完整版，调试用）
        }
