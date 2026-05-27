# =============================================================================
# generator.py — 答案生成器
# 调用大语言模型（LLM）基于检索到的文档上下文生成最终答案。
# 核心设计：结构化上下文组装 + 严格 System Prompt + 低温度稳定性
# =============================================================================

# 从 __future__ 导入 annotations，让所有类型注解使用延迟求值
from __future__ import annotations

# List：typing 的列表泛型，用于类型标注
from typing import List

# httpx：现代化的 HTTP 客户端，显式管理以避免 SDK 版本兼容问题
import httpx
# OpenAI：OpenAI 官方 SDK 的同步客户端
from openai import OpenAI

# Settings：项目的配置类（只用于类型标注和构造函数的参数）
from .config import Settings
# 导入项目统一的日志器
from .logger import logger
# RetrievedChunk：检索结果的 dataclass（从 retriever 模块导入）
from .retriever import RetrievedChunk


# 预设消息：当检索不到相关内容时返回给用户的友好提示
NO_CONTEXT_MSG = "未在知识库中检索到足够相关内容，请尝试换一个问法或补充相关文档。"


class AnswerGenerator:
    """根据检索上下文调用大模型生成回答。"""

    def __init__(self, settings: Settings):
        # 构建 OpenAI 客户端的关键字参数
        client_kwargs = {"api_key": settings.api_key}
        # 如果配置了自定义 base_url（如 DashScope 的兼容接口地址）
        # 传给 OpenAI 客户端——这是支持多提供方的关键
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url

        # 显式传入 httpx 客户端，避免 openai/httpx 版本组合导致 proxies 参数冲突
        # timeout=120.0：LLM 生成回答可能需要较长时间（10-30 秒），设置 2 分钟超时
        # 如果不显式传 httpx.Client，SDK 内部可能创建带 proxies 参数的客户端导致报错
        self.client = OpenAI(
            **client_kwargs,
            http_client=httpx.Client(timeout=120.0),
        )
        # 保存模型名，用于后续 API 调用
        self.model = settings.llm_model

    def generate(self, question: str, chunks: List[RetrievedChunk]) -> str:
        """严格基于检索结果回答，禁止编造。
        这是 RAG 系统最核心的方法——如何让 LLM 真的使用检索结果而不是自己编。"""

        # 如果没有检索到任何相关片段，直接返回预设提示消息
        # 不调用 LLM——没有上下文的情况下 LLM 一定会编造
        if not chunks:
            return NO_CONTEXT_MSG

        # --- 第一步：组装结构化上下文 ---
        # 将每个 RetrieveChunk 格式化为带编号的文本块
        context_blocks = []
        for idx, chunk in enumerate(chunks, start=1):  # start=1 → 序号从 1 开始
            # 每个块包含三个信息：来源文件名、文件路径、内容正文
            block = (
                f"[{idx}] 来源文件: {chunk.source}\n"   # [1] 来源文件: fastapi_intro.md
                f"路径: {chunk.filepath}\n"               # 路径: /path/to/docs/sample/...
                f"内容: {chunk.text}"                     # 内容: FastAPI 是一个基于 Python...
            )
            context_blocks.append(block)

        # 用双换行分隔每个块——让 LLM 能清晰区分不同来源的文档片段
        context_text = "\n\n".join(context_blocks)

        # --- 第二步：设计 System Prompt（RAG 的灵魂） ---
        # 三条硬约束，每一条都直指 RAG 的核心风险：
        # 1. "必须严格基于给定上下文回答" — 防止 LLM 无视检索结果，凭借参数化知识回答
        # 2. "不允许编造" — 直接下令禁止幻觉（指令越强硬，LLM 遵循率越高）
        # 3. "上下文不足时直接告知" — 让 LLM 诚实地承认无知，而非糊弄
        system_prompt = (
            "你是一名技术文档问答助手。"
            "你必须严格基于给定上下文回答，不允许编造。"
            "若上下文不足或没有明确依据，直接回答："
            "'知识库中未检索到足够相关信息。'"
            "回答风格保持简洁、专业、面向工程实践。"
        )

        # --- 第三步：组装 User Prompt ---
        # 用户问题 + 检索到的完整上下文
        user_prompt = (
            f"用户问题：\n{question}\n\n"
            f"检索到的上下文：\n{context_text}\n\n"
            "请基于上述上下文给出答案。"
        )

        # --- 第四步：调用 LLM ---
        try:
            # 调用 OpenAI Chat Completion API
            # model：模型名称（qwen-plus / gpt-4o-mini）
            # temperature=0.1：极低的随机性——技术问答需要确定性和一致性
            #   0.1 意味着模型几乎只选择概率最高的 token
            # messages：系统提示 + 用户问题 + 上下文
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},  # 系统角色约束
                    {"role": "user", "content": user_prompt},       # 用户问题+上下文
                ],
            )
            # 提取答案文本——response.choices[0].message.content
            # content 可能为 None（极端情况），用 or "" 兜底
            answer = (response.choices[0].message.content or "").strip()
            # 如果答案为空字符串，返回预设提示
            if not answer:
                return "知识库中未检索到足够相关信息。"
            return answer
        except Exception as exc:  # noqa: BLE001
            # API 调用失败——记录完整的异常堆栈日志
            # logger.exception 会自动包含 traceback 信息
            logger.exception("调用模型失败: %s", exc)
            # 返回友好提示而不是让程序崩溃
            return "生成回答时出现错误，请检查模型配置或稍后重试。"
