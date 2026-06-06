# =============================================================================
# agent_core.py — Agent 核心引擎
# 实现多 Agent 协同架构：Router → QueryRewrite → Retrieval（ReAct 循环）
# 核心设计：
# 1. RouterAgent：路由决策——判断问题类型，选择处理通道
# 2. QueryRewriteAgent：查询改写——优化检索查询，提升召回率
# 3. RetrievalAgent：ReAct 循环——思考→调用工具→观察→最终回答
#
# 技术实现：
# - 使用 LlamaIndex OpenAIAgent 实现 ReAct + 工具调用
# - Router 和 QueryRewrite 使用简单的 LLM 调用（无需工具）
# - 所有 Agent 共享同一个 LLM 实例（由 LlamaIndex 全局配置）
# =============================================================================

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from openai import OpenAI

from .config import Settings
from .logger import logger


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class AgentStep:
    """Agent 推理过程的单步记录，用于前端展示思考链。"""
    agent_name: str         # Agent 名称（如 "Router"、"Retrieval"）
    step_type: str          # 步骤类型：think / action / observation / answer
    content: str            # 步骤内容


@dataclass
class AgentResult:
    """Agent 协同的最终结果。"""
    answer: str                                 # 最终答案
    route: str                                  # 路由决策：rag / general / direct
    agent_steps: List[AgentStep] = field(default_factory=list)  # 思考过程
    rewritten_query: Optional[str] = None       # 改写后的查询


# =============================================================================
# Router Agent（路由决策）
# =============================================================================

class RouterAgent:
    """路由 Agent：分析用户问题类型，决定处理通道。

    路由策略：
    - rag：需要从知识库检索的技术问题（如 "FastAPI 怎么用？"）
    - general：通用知识问题（如 "什么是 REST？"）
    - direct：简单问候或闲聊（如 "你好"）
    """

    ROUTE_PROMPT = """你是一个问题分类器。根据用户问题，判断应该走哪个处理通道。

通道类型：
1. rag — 需要从技术文档知识库中检索答案的问题（如具体技术的用法、配置、API 等）
2. general — 通用技术知识问题（如概念解释、最佳实践、技术比较）
3. direct — 简单问候、闲聊、或与技术无关的问题

请只返回一个 JSON 对象，格式如下：
{"route": "rag" 或 "general" 或 "direct", "reason": "简短理由"}

注意：
- 如果问题涉及知识库中可能有的具体技术（FastAPI、Docker、Python 等），优先选 rag
- 如果是通用概念或知识库不太可能覆盖的内容，选 general
- 如果是打招呼、感谢等非技术内容，选 direct"""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def route(self, query: str, history: Sequence[Dict[str, str]] = ()) -> Dict[str, str]:
        """分析问题并返回路由决策。

        Args:
            query: 当前用户问题
            history: 对话历史，格式 [{"role": "user", "content": "..."}, ...]
        Returns:
            {"route": "rag"|"general"|"direct", "reason": "理由"}
        """
        try:
            messages = [{"role": "system", "content": self.ROUTE_PROMPT}]
            # 加入对话历史，帮助判断意图（如"还有呢"、"它怎么用"需要上下文）
            for msg in history:
                messages.append(msg)
            messages.append({"role": "user", "content": query})

            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                messages=messages,
            )
            content = (response.choices[0].message.content or "").strip()

            # 尝试解析 JSON
            # 处理可能的 markdown 代码块包裹
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            result = json.loads(content)
            route = result.get("route", "rag")
            reason = result.get("reason", "")

            # 校验路由值
            if route not in ("rag", "general", "direct"):
                route = "rag"

            return {"route": route, "reason": reason}

        except Exception as exc:
            logger.warning("路由决策失败，默认走 RAG 通道: %s", exc)
            return {"route": "rag", "reason": "路由解析失败，默认走 RAG"}


# =============================================================================
# Query Rewrite Agent（查询改写）
# =============================================================================

class QueryRewriteAgent:
    """查询改写 Agent：优化用户的原始查询，提升检索召回率。

    改写策略：
    - 扩展同义词（"怎么用" → "使用方法 教程 示例"）
    - 补充技术关键词（"部署" → "Docker 部署 容器化"）
    - 拆解复杂问题为多个子查询
    """

    REWRITE_PROMPT = """你是一个查询优化专家。你的任务是改写用户的搜索查询，使其更适合在技术文档知识库中检索。

改写规则：
1. 保留原始查询的核心意图
2. 添加相关的同义词和技术术语
3. 如果问题包含多个子问题，拆分为独立的查询
4. 保持查询简洁，每个查询不超过 30 字

请返回 JSON 对象，格式如下：
{"queries": ["改写后的查询1", "改写后的查询2"], "explanation": "改写理由"}

注意：
- 如果原始查询已经很清晰，可以只返回一个改写后的查询
- 不要改变用户问题的核心含义
- 优先使用技术文档中常见的术语表达"""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def rewrite(self, query: str, history: Sequence[Dict[str, str]] = ()) -> Dict[str, Any]:
        """改写查询，支持对话历史上下文（解决指代消解）。

        Args:
            query: 当前用户问题
            history: 对话历史
        Returns:
            {"queries": ["查询1", "查询2"], "explanation": "理由"}
        """
        try:
            # 如果有历史，构造带上下文的改写请求
            if history:
                history_text = "\n".join(
                    f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
                    for m in history
                )
                user_content = (
                    f"对话历史：\n{history_text}\n\n"
                    f"当前用户问题：{query}\n\n"
                    "请基于对话历史理解当前问题的完整含义后改写。"
                    "特别是如果当前问题中存在代词（如'它'、'这个'、'那个'），"
                    "请用对话历史中对应的具体术语替换。"
                )
            else:
                user_content = query

            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": self.REWRITE_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            content = (response.choices[0].message.content or "").strip()

            # 处理可能的 markdown 代码块包裹
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            result = json.loads(content)
            queries = result.get("queries", [query])
            explanation = result.get("explanation", "")

            # 确保 queries 是列表且非空
            if not isinstance(queries, list) or not queries:
                queries = [query]

            return {"queries": queries, "explanation": explanation}

        except Exception as exc:
            logger.warning("查询改写失败，使用原始查询: %s", exc)
            return {"queries": [query], "explanation": "改写失败，使用原始查询"}


# =============================================================================
# Retrieval Agent（ReAct 循环 + 工具调用）
# =============================================================================

class RetrievalAgent:
    """检索 Agent：基于 ReAct 模式执行工具调用和答案生成。

    使用 LlamaIndex 的 OpenAIAgent 实现 ReAct 循环：
    Thought → Action → Observation → ... → Final Answer

    如果 LlamaIndex Agent 不可用（如 DashScope 不支持 function calling），
    回退到简单的 LLM 调用模式。
    """

    def __init__(self, client: OpenAI, model: str, max_iterations: int = 5):
        self.client = client
        self.model = model
        self.max_iterations = max_iterations

    def run(
        self,
        query: str,
        context: str,
        tools_description: str,
        history: Sequence[Dict[str, str]] = (),
    ) -> Dict[str, Any]:
        """执行 ReAct 检索循环。

        Args:
            query: 用户问题（或改写后的查询）
            context: 检索到的上下文（来自混合检索）
            tools_description: 可用工具的描述
            history: 对话历史

        Returns:
            {"answer": "答案", "steps": [AgentStep, ...]}
        """
        steps: List[AgentStep] = []

        # 尝试使用 LlamaIndex OpenAIAgent
        try:
            return self._run_with_llama_agent(query, context, tools_description, steps, history)
        except Exception as exc:
            logger.info("LlamaIndex Agent 不可用，回退到直接模式: %s", exc)
            return self._run_direct(query, context, steps, history)

    def _run_with_llama_agent(
        self, query: str, context: str, tools_description: str,
        steps: List[AgentStep], history: Sequence[Dict[str, str]] = (),
    ) -> Dict[str, Any]:
        """使用 LlamaIndex OpenAIAgent 执行 ReAct 循环。"""
        from llama_index.agent.openai import OpenAIAgent
        from .agent_tools import create_tools

        tools = create_tools()

        # 构建系统提示
        system_prompt = (
            "你是一个技术文档问答助手。你可以使用工具来搜索知识库、提取代码、"
            "对比不同来源的说法、查找术语定义、浏览文档结构等。\n\n"
            "使用工具的策略：\n"
            "1. 遇到技术问题 → 先用 search_knowledge 检索\n"
            "2. 检索结果不够 → 用 expand_context 获取更多上下文\n"
            "3. 需要看代码 → 用 extract_code 提取代码示例\n"
            "4. 对比不同说法 → 用 compare_sources 查看异同\n"
            "5. 解释术语 → 用 entity_lookup 查找定义\n"
            "6. 了解文档结构 → 用 document_outline 查看目录\n"
            "7. 了解知识库范围 → 用 knowledge_stats 查看统计\n\n"
            f"可用工具：\n{tools_description}\n\n"
            "回答规则：\n"
            "1. 严格基于检索到的内容回答，不要编造\n"
            "2. 如果知识库中没有相关信息，坦诚告知\n"
            "3. 如果检索结果之间有矛盾，应该指出并分析\n"
            "4. 代码相关问题优先展示代码示例\n"
            "5. 回答保持简洁、专业、面向工程实践"
        )

        # 如果有对话历史，追加到系统提示中
        if history:
            history_text = "\n".join(
                f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
                for m in history
            )
            system_prompt += f"\n\n以下是之前的对话记录，供你理解上下文：\n{history_text}"

        # 创建 LlamaIndex Agent
        agent = OpenAIAgent.from_tools(
            tools=tools,
            system_prompt=system_prompt,
            verbose=True,
            max_iterations=self.max_iterations,
        )

        # 记录思考步骤
        steps.append(AgentStep(
            agent_name="Retrieval",
            step_type="think",
            content=f"开始执行 ReAct 检索循环，用户问题: {query}",
        ))

        # 执行 Agent
        response = agent.chat(query)
        answer = str(response)

        # 提取 Agent 的思考过程
        if hasattr(response, 'sources') and response.sources:
            for source in response.sources:
                steps.append(AgentStep(
                    agent_name="Retrieval",
                    step_type="action",
                    content=f"调用工具: {getattr(source, 'tool_name', 'unknown')}",
                ))

        steps.append(AgentStep(
            agent_name="Retrieval",
            step_type="answer",
            content=answer[:200] + "..." if len(answer) > 200 else answer,
        ))

        return {"answer": answer, "steps": steps}

    def _run_direct(
        self, query: str, context: str, steps: List[AgentStep],
        history: Sequence[Dict[str, str]] = (),
    ) -> Dict[str, Any]:
        """回退模式：直接使用 LLM 基于上下文回答。"""
        steps.append(AgentStep(
            agent_name="Retrieval",
            step_type="think",
            content="使用直接模式（无工具调用），基于检索上下文生成答案",
        ))

        system_prompt = (
            "你是一名技术文档问答助手。"
            "你必须严格基于给定上下文回答，不允许编造。"
            "若上下文不足或没有明确依据，直接回答：'知识库中未检索到足够相关信息。'"
            "回答风格保持简洁、专业、面向工程实践。"
        )

        user_prompt = (
            f"用户问题：\n{query}\n\n"
            f"检索到的上下文：\n{context}\n\n"
            "请基于上述上下文给出答案。"
        )

        try:
            messages = [{"role": "system", "content": system_prompt}]
            # 插入对话历史
            for msg in history:
                messages.append(msg)
            messages.append({"role": "user", "content": user_prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                messages=messages,
            )
            answer = (response.choices[0].message.content or "").strip()
            if not answer:
                answer = "知识库中未检索到足够相关信息。"
        except Exception as exc:
            logger.exception("LLM 调用失败: %s", exc)
            answer = "生成回答时出现错误，请稍后重试。"

        steps.append(AgentStep(
            agent_name="Retrieval",
            step_type="answer",
            content=answer[:200] + "..." if len(answer) > 200 else answer,
        ))

        return {"answer": answer, "steps": steps}
