# =============================================================================
# config.py — 配置管理中心
# 负责从 .env 文件加载所有运行参数，提供类型安全的 Settings 不可变配置对象。
# 核心设计：不可变对象 + 自动校验 + 全局单例缓存
# =============================================================================

# 从 __future__ 导入 annotations，让所有类型注解使用延迟求值（PEP 563）
from __future__ import annotations

# os：操作系统接口，用于读取环境变量
import os
# dataclass：数据类装饰器，自动生成 __init__ / __repr__ / __eq__ 等方法
from dataclasses import dataclass
# lru_cache：LRU（最近最少使用）缓存装饰器，用于缓存函数返回值
from functools import lru_cache
# Path：pathlib 的路径操作类
from pathlib import Path

# load_dotenv：python-dotenv 库的函数，将 .env 文件中的变量加载到操作系统环境变量中
from dotenv import load_dotenv


# frozen=True：创建不可变（immutable）的数据类
# 一旦实例化后，任何修改字段值的操作都会抛出 FrozenInstanceError
# 这确保了 Settings 对象在整个生命周期中保持一致
@dataclass(frozen=True)
class Settings:
    """项目运行所需配置。"""

    # ---- 字段定义 ----
    # api_provider：API 提供方标识，如 "dashscope" 或 "openai"
    api_provider: str
    # api_key：API 密钥，用于认证
    api_key: str
    # openai_base_url：API 端点地址，为 None 时使用 SDK 默认地址
    # str | None 表示这个字段可以是字符串或 None（Python 3.10+ 联合类型）
    openai_base_url: str | None
    # llm_model：大语言模型名称，如 "qwen-plus" 或 "gpt-4o-mini"
    llm_model: str
    # embed_model：Embedding 模型名称，如 "text-embedding-v3"
    embed_model: str
    # docs_dir：知识库文档目录路径
    docs_dir: Path
    # chroma_dir：ChromaDB 向量数据库持久化目录路径
    chroma_dir: Path
    # top_k：检索时返回的最相关文档片段数量
    top_k: int
    # chunk_size：文本切分的每个块大小（tokens）
    chunk_size: int
    # chunk_overlap：相邻文本块之间的重叠量（tokens）
    chunk_overlap: int
    # similarity_threshold：相似度阈值，低于此分数的检索结果会被丢弃
    similarity_threshold: float
    # bm25_top_k：BM25 检索返回的最大文档片段数
    bm25_top_k: int
    # rrf_k：倒数排名融合（RRF）算法的平滑常数
    rrf_k: int
    # agent_max_iterations：Agent 最大推理循环次数（防止无限循环）
    agent_max_iterations: int

    # ---- 类方法：从环境变量构建 Settings ----
    @classmethod
    def from_env(cls) -> "Settings":
        """从环境变量读取配置并做基础校验。"""

        # 从环境变量中获取 OpenAI API Key，不存在或为空则返回空字符串
        # os.getenv(key, default)：获取环境变量，不存在时返回 default
        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        # 从环境变量中获取 DashScope API Key（阿里云）
        dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()

        # --- 智能识别 API 提供方 ---
        # 优先使用 DASHSCOPE_API_KEY —— 因为阿里云 DashScope 在国内更常用且成本更低
        # 只有当 DASHSCOPE_API_KEY 不存在时才使用 OPENAI_API_KEY
        if dashscope_api_key:
            # 存在 DashScope Key → 使用阿里云 DashScope
            provider = "dashscope"
            api_key = dashscope_api_key
        elif openai_api_key:
            # 只有 OpenAI Key → 使用 OpenAI
            # API_PROVIDER 环境变量允许用户显式指定提供方
            provider = os.getenv("API_PROVIDER", "openai").strip().lower() or "openai"
            api_key = openai_api_key
        else:
            # 两个 Key 都没有 → 抛出异常，提示用户至少配置一个
            raise ValueError(
                "缺少关键配置，请至少设置 OPENAI_API_KEY 或 DASHSCOPE_API_KEY。"
            )

        # --- 配置 API 端点地址 ---
        # 读取 OPENAI_BASE_URL 环境变量，如果为空则设为 None（使用 SDK 默认地址）
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        # 如果是 DashScope 且用户没有自定义 base_url
        # 自动设置为阿里云 DashScope 的 OpenAI 兼容接口地址
        if provider == "dashscope" and not base_url:
            # DashScope 提供 OpenAI 兼容接口
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        # --- 选择默认模型 ---
        # 根据提供方自动选择最合适的默认模型
        # DashScope → qwen-plus（通义千问增强版）
        # OpenAI → gpt-4o-mini（性价比最高的 GPT-4 系列模型）
        default_llm = "qwen-plus" if provider == "dashscope" else "gpt-4o-mini"
        # Embedding 模型同理
        default_embed = "text-embedding-v3" if provider == "dashscope" else "text-embedding-3-small"

        # --- 构建并返回 Settings 实例 ---
        # LLM_MODEL 和 EMBED_MODEL 从环境变量读取（如果用户设置了自定义值）
        return cls(
            api_provider=provider,
            api_key=api_key,
            openai_base_url=base_url,
            llm_model=os.getenv("LLM_MODEL", default_llm).strip(),
            embed_model=os.getenv("EMBED_MODEL", default_embed).strip(),
            # docs_dir：转为绝对路径（resolve()），确保后续文件操作不受当前工作目录影响
            docs_dir=Path(os.getenv("DOCS_DIR", "docs")).resolve(),
            # chroma_dir：同理转为绝对路径
            chroma_dir=Path(os.getenv("CHROMA_DIR", "storage/chroma_db")).resolve(),
            # 以下参数通过自定义的校验函数进行类型转换和合法性检查
            top_k=_parse_positive_int("TOP_K", 4),
            chunk_size=_parse_positive_int("CHUNK_SIZE", 512),
            chunk_overlap=_parse_non_negative_int("CHUNK_OVERLAP", 80),
            similarity_threshold=_parse_float("SIMILARITY_THRESHOLD", 0.45),
            bm25_top_k=_parse_positive_int("BM25_TOP_K", 8),
            rrf_k=_parse_positive_int("RRF_K", 60),
            agent_max_iterations=_parse_positive_int("AGENT_MAX_ITERATIONS", 5),
        )




# ---- 配置校验函数 ----

def _parse_positive_int(key: str, default: int) -> int:
    """解析环境变量为「正整数」—— 值必须 > 0。"""
    raw = os.getenv(key, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {key} 必须是整数，当前值: {raw}") from exc
    if value <= 0:
        raise ValueError(f"环境变量 {key} 必须大于 0，当前值: {value}")
    return value


def _parse_non_negative_int(key: str, default: int) -> int:
    """解析环境变量为「非负整数」—— 值必须 >= 0。"""
    raw = os.getenv(key, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {key} 必须是整数，当前值: {raw}") from exc
    if value < 0:
        raise ValueError(f"环境变量 {key} 不能为负数，当前值: {value}")
    return value


def _parse_float(key: str, default: float) -> float:
    """解析环境变量为「浮点数」—— 值必须 >= 0。"""
    raw = os.getenv(key, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {key} 必须是浮点数，当前值: {raw}") from exc
    if value < 0:
        raise ValueError(f"环境变量 {key} 不能为负数，当前值: {value}")
    return value


# ---- 全局配置单例 ----

# lru_cache(maxsize=1)：只缓存最近 1 次调用的结果
# 效果：get_settings() 只会执行一次（读取 .env + 构建 Settings），
# 后续所有调用直接返回缓存的 Settings 对象，不会重复加载
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """读取并缓存配置，避免频繁重复加载。"""

    # load_dotenv：将 .env 文件中的键值对加载为环境变量
    # override=False：如果环境变量已经存在（如系统级），优先使用已有值
    load_dotenv(override=False)
    # 从环境变量构建 Settings 实例
    return Settings.from_env()
