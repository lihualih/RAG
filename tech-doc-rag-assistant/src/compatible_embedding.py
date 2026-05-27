# =============================================================================
# compatible_embedding.py — Embedding 适配器（项目的核心技术创新点）
# 继承 LlamaIndex 的 BaseEmbedding，实现了一个通用的 OpenAI 兼容接口适配器。
# 核心设计：策略模式 + 依赖倒置 + 同步异步双接口 + 显式 httpx 管理
# =============================================================================

# 从 __future__ 导入 annotations，让所有类型注解使用延迟求值
from __future__ import annotations

# typing 的类型标注工具
from typing import Any, List, Optional

# httpx：现代化的 HTTP 客户端库，支持同步和异步两种模式
# 我们显式管理 httpx 客户端，避免 openai SDK 和 httpx 版本组合的兼容性问题
import httpx
# BaseEmbedding：LlamaIndex 的 Embedding 抽象基类
# 继承它就能无缝接入 LlamaIndex 的索引构建和检索流程
from llama_index.core.base.embeddings.base import BaseEmbedding
# AsyncOpenAI、OpenAI：OpenAI 官方 SDK 的异步和同步客户端
from openai import AsyncOpenAI, OpenAI
# pydantic 的字段声明工具
# Field：声明公开配置字段（会参与序列化/校验）
# PrivateAttr：声明私有属性（不参与序列化，用于持有内部状态）
from pydantic import Field, PrivateAttr


class OpenAICompatibleEmbedding(BaseEmbedding):
    """
    通用 OpenAI 兼容 Embedding 适配器。

    作用：
    1. 兼容任意 OpenAI 风格 embedding 模型名（如 text-embedding-v3）
    2. 支持自定义 base_url，适配 DashScope 等兼容网关
    3. 同时提供同步和异步向量化接口
    """

    # ---- 公开配置字段（pydantic Field） ----
    # Field(description=...)：pydantic 的字段声明方式，会自动生成 schema 和校验逻辑
    model_name: str = Field(description="Embedding 模型名")
    api_key: str = Field(description="API Key")
    # Optional[str] = Field(default=None)：base_url 是可选的，不传就用 SDK 默认地址
    base_url: Optional[str] = Field(default=None, description="OpenAI 兼容接口地址")
    # dimensions 也是可选的——不是所有 Embedding 模型都支持自定义维度
    dimensions: Optional[int] = Field(default=None, description="可选向量维度")

    # ---- 私有属性（PrivateAttr，不参与 pydantic 的序列化和校验） ----
    # _client：同步的 OpenAI 客户端，用于构建索引和同步检索
    _client: OpenAI = PrivateAttr()
    # _aclient：异步的 OpenAI 客户端，预留用于高并发检索场景
    _aclient: AsyncOpenAI = PrivateAttr()

    # ---- 构造函数 ----
    # **data：接收所有 pydantic Field 声明的参数
    def __init__(self, **data: Any) -> None:
        # 先调用父类 BaseEmbedding 的 __init__，让 pydantic 完成 Field 的赋值和校验
        super().__init__(**data)

        # 构建 OpenAI SDK 客户端的公共参数
        client_kwargs = {"api_key": self.api_key}
        # 如果配置了 base_url，传给 SDK——这是支持 DashScope 等兼容网关的关键
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        # 显式传入 httpx 客户端，避免 openai/httpx 版本组合导致 proxies 参数冲突
        # httpx.Client(timeout=60.0)：同步客户端，60 秒超时
        # 如果不显式传入，openai SDK 内部可能会尝试创建带有 proxies 参数的 httpx 客户端
        # 在某些环境下 proxies 参数会报错
        self._client = OpenAI(
            **client_kwargs,
            http_client=httpx.Client(timeout=60.0),
        )
        # httpx.AsyncClient(timeout=60.0)：异步客户端，同样 60 秒超时
        self._aclient = AsyncOpenAI(
            **client_kwargs,
            http_client=httpx.AsyncClient(timeout=60.0),
        )

    # ---- 类方法：返回类名 ----
    # LlamaIndex 内部用它来标识 Embedding 模型的类型
    @classmethod
    def class_name(cls) -> str:
        return "OpenAICompatibleEmbedding"

    # ---- 构建 API 调用参数 ----
    def _build_embedding_kwargs(self, text: str) -> dict:
        """构建 OpenAI Embedding API 调用的参数字典。"""
        # 基础参数：模型名 + 输入文本
        kwargs = {
            "model": self.model_name,
            "input": text,
        }
        # 如果配置了 dimensions（向量维度），加入参数
        # 不是所有模型都支持 dimensions 参数，所以是可选的
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        return kwargs

    # ---- 四个核心方法（必须实现，LlamaIndex 的接口契约） ----

    def _get_query_embedding(self, query: str) -> List[float]:
        """同步方法：将用户查询转为向量。
        LlamaIndex 在检索阶段调用这个方法向量化用户问题。"""
        # 调用 OpenAI Embedding API（同步客户端）
        response = self._client.embeddings.create(**self._build_embedding_kwargs(query))
        # 返回第一个（也是唯一一个）Embedding 结果的数据
        # embeddings.create 返回一个列表，每个输入文本对应一个结果
        return response.data[0].embedding

    def _get_text_embedding(self, text: str) -> List[float]:
        """同步方法：将文档文本转为向量。
        LlamaIndex 在索引构建阶段调用这个方法向量化每个文档块。"""
        # 和查询向量化的逻辑完全一样，只是输入是文档文本
        response = self._client.embeddings.create(**self._build_embedding_kwargs(text))
        return response.data[0].embedding

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """异步方法：查询向量化的异步版本。
        预留用于高并发检索场景——异步不会阻塞事件循环。"""
        # 使用异步客户端调用
        response = await self._aclient.embeddings.create(**self._build_embedding_kwargs(query))
        return response.data[0].embedding

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """异步方法：文本向量化的异步版本。
        预留用于批量文档处理的异步场景。"""
        response = await self._aclient.embeddings.create(**self._build_embedding_kwargs(text))
        return response.data[0].embedding
