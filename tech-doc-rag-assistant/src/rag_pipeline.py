# =============================================================================
# rag_pipeline.py — 流程编排器（门面模式 Facade Pattern）
# 将检索（Retriever）和生成（Generator）串联为一个统一的问答接口。
# 核心设计：单一入口 ask() 方法 + 结构化返回 + 引用来源构建
#
# 包含两个 Pipeline：
# 1. RAGPipeline — 原始的简单 RAG 流程（检索 → 生成）
# 2. AgentRAGPipeline — 增强版 Agent 协同流程（路由 → 改写 → 混合检索 → 生成）
# =============================================================================

# 从 __future__ 导入 annotations，让所有类型注解使用延迟求值
from __future__ import annotations

# typing 的类型标注工具：Any（任意类型）、Dict（字典泛型）、List（列表泛型）
from typing import Any, Dict, List

# httpx：HTTP 客户端，用于创建 OpenAI 客户端
import httpx
# OpenAI：OpenAI 官方 SDK
from openai import OpenAI

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


# =============================================================================
# AgentRAGPipeline — 增强版 Agent 协同流程
# =============================================================================

class AgentRAGPipeline:
    """基于多 Agent 协同的增强版 RAG 流程。

    协同流程：
    1. RouterAgent → 判断问题类型（rag / general / direct）
    2. QueryRewriteAgent → 改写查询（仅 RAG 类问题）
    3. HybridRetriever → BM25 + 向量混合检索
    4. RetrievalAgent → ReAct 循环 + 工具调用 → 生成答案

    对外暴露的接口与 RAGPipeline 一致：ask(question) → dict
    """

    def __init__(
        self,
        hybrid_retriever: Any,   # HybridRetriever 实例
        settings: Settings,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.settings = settings

        # 创建 OpenAI 客户端（供 Agent 使用）
        client_kwargs = {"api_key": settings.api_key}
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url
        self.client = OpenAI(
            **client_kwargs,
            http_client=httpx.Client(timeout=120.0),
        )

        # 创建三个 Agent
        from .agent_core import RouterAgent, QueryRewriteAgent, RetrievalAgent
        self.router = RouterAgent(self.client, settings.llm_model)
        self.rewriter = QueryRewriteAgent(self.client, settings.llm_model)
        self.retrieval_agent = RetrievalAgent(
            self.client, settings.llm_model, settings.agent_max_iterations,
        )

        # 初始化工具的全局引用
        from .agent_tools import init_tools
        init_tools(hybrid_retriever, settings.docs_dir)

    def ask(self, question: str) -> Dict[str, Any]:
        """Agent 协同问答入口。

        流程：路由决策 → 查询改写 → 混合检索 → Agent 生成答案
        """
        from .agent_core import AgentStep

        # ---- Step 0：问题预处理 ----
        clean_question = question.strip()
        if not clean_question:
            return {
                "answer": "请输入有效问题。",
                "citations": [],
                "retrieved_chunks": [],
                "agent_steps": [],
                "route": "direct",
            }

        agent_steps: List[AgentStep] = []

        # ---- Step 1：路由决策 ----
        route_result = self.router.route(clean_question)
        route = route_result["route"]
        reason = route_result["reason"]

        agent_steps.append(AgentStep(
            agent_name="Router",
            step_type="think",
            content=f"问题类型: {route} | 理由: {reason}",
        ))

        # ---- Step 2：直接回答（问候/闲聊） ----
        if route == "direct":
            answer = self._handle_direct(clean_question)
            return {
                "answer": answer,
                "citations": [],
                "retrieved_chunks": [],
                "agent_steps": agent_steps,
                "route": route,
            }

        # ---- Step 3：通用知识回答 ----
        if route == "general":
            answer = self._handle_general(clean_question)
            return {
                "answer": answer,
                "citations": [],
                "retrieved_chunks": [],
                "agent_steps": agent_steps,
                "route": route,
            }

        # ---- Step 4：RAG 通道——查询改写 ----
        rewrite_result = self.rewriter.rewrite(clean_question)
        queries = rewrite_result["queries"]
        explanation = rewrite_result["explanation"]

        agent_steps.append(AgentStep(
            agent_name="QueryRewrite",
            step_type="think",
            content=f"改写查询: {queries} | 理由: {explanation}",
        ))

        # ---- Step 5：混合检索（使用所有改写后的查询） ----
        all_chunks = []
        seen_texts = set()
        for q in queries:
            chunks = self.hybrid_retriever.retrieve(q)
            for chunk in chunks:
                if chunk.text not in seen_texts:
                    seen_texts.add(chunk.text)
                    all_chunks.append(chunk)

        agent_steps.append(AgentStep(
            agent_name="Retrieval",
            step_type="observation",
            content=f"混合检索完成，获取 {len(all_chunks)} 个相关文档片段",
        ))

        # ---- Step 6：构建上下文 ----
        if not chunks:
            return {
                "answer": NO_CONTEXT_MSG,
                "citations": [],
                "retrieved_chunks": [],
                "agent_steps": agent_steps,
                "route": route,
                "rewritten_query": queries[0] if queries else None,
            }

        context_blocks = []
        for idx, chunk in enumerate(all_chunks, start=1):
            block = (
                f"[{idx}] 来源文件: {chunk.source}\n"
                f"路径: {chunk.filepath}\n"
                f"内容: {chunk.text}"
            )
            context_blocks.append(block)
        context_text = "\n\n".join(context_blocks)

        # ---- Step 7：RetrievalAgent 生成答案 ----
        tools_desc = (
            "1. search_knowledge(query, source_file, file_type, max_results) - 【核心】混合检索知识库\n"
            "2. expand_context(query, context_size) - 扩展检索上下文，获取更多相关片段\n"
            "3. extract_code(query, language) - 提取相关代码示例\n"
            "4. compare_sources(query) - 对比不同来源对同一问题的说法\n"
            "5. entity_lookup(term) - 查找技术术语的定义和用法\n"
            "6. document_outline(filename) - 查看文档目录结构\n"
            "7. knowledge_stats() - 查看知识库统计信息"
        )

        agent_result = self.retrieval_agent.run(
            query=clean_question,
            context=context_text,
            tools_description=tools_desc,
        )
        answer = agent_result["answer"]
        agent_steps.extend(agent_result.get("steps", []))

        # ---- Step 8：构建 citations ----
        citations: List[Dict[str, Any]] = []
        retrieved_chunks: List[Dict[str, Any]] = []
        for chunk in all_chunks:
            citations.append({
                "source": chunk.source,
                "filepath": chunk.filepath,
                "snippet": truncate_text(chunk.text, max_length=180),
                "score": round(chunk.score, 4),
            })
            retrieved_chunks.append({
                "text": chunk.text,
                "score": chunk.score,
                "source": chunk.source,
                "filepath": chunk.filepath,
                "metadata": chunk.metadata,
            })

        return {
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": retrieved_chunks,
            "agent_steps": agent_steps,
            "route": route,
            "rewritten_query": queries[0] if queries else None,
        }

    def _handle_direct(self, question: str) -> str:
        """处理直接回答（问候/闲聊）。"""
        try:
            response = self.client.chat.completions.create(
                model=self.settings.llm_model,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": "你是一个友好的技术文档助手。简洁回复用户的问候。"},
                    {"role": "user", "content": question},
                ],
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            return "你好！我是技术文档助手，有什么技术问题可以帮你解答？"

    def _handle_general(self, question: str) -> str:
        """处理通用技术知识问题。"""
        try:
            response = self.client.chat.completions.create(
                model=self.settings.llm_model,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": (
                        "你是一个技术文档助手。请基于你的知识回答技术问题。"
                        "回答简洁、专业、面向工程实践。"
                        "如果不确定，坦诚告知。"
                    )},
                    {"role": "user", "content": question},
                ],
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.exception("通用回答失败: %s", exc)
            return "回答时出现错误，请稍后重试。"
