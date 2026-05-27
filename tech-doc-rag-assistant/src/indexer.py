# =============================================================================
# indexer.py — 索引管理器
# 负责 ChromaDB 向量库与 LlamaIndex 索引的完整生命周期管理。
# 核心设计：全局配置 + 索引复用 + 生命周期管理 + 增量和重建接口
# =============================================================================

# 从 __future__ 导入 annotations，让所有类型注解使用延迟求值
from __future__ import annotations

# typing 的类型标注工具：Any（任意类型）、Sequence（序列泛型）
from typing import Any, Sequence

# chromadb：ChromaDB 向量数据库的 Python SDK
import chromadb
# ChromaSettings：ChromaDB 的配置类，这里用来关闭遥测
from chromadb.config import Settings as ChromaSettings
# LlamaSettings：LlamaIndex 的全局配置类，统一管理 LLM、Embedding、NodeParser
from llama_index.core import Settings as LlamaSettings
# StorageContext：LlamaIndex 的存储上下文，指定向量库等存储后端
# VectorStoreIndex：LlamaIndex 的向量索引类
from llama_index.core import StorageContext, VectorStoreIndex
# SentenceSplitter：LlamaIndex 的句子级文本切分器
from llama_index.core.node_parser import SentenceSplitter
# Document：LlamaIndex 的文档对象类型（只在类型标注中使用）
from llama_index.core.schema import Document
# OpenAI：LlamaIndex 封装的 OpenAI LLM 接口
from llama_index.llms.openai import OpenAI
# ChromaVectorStore：LlamaIndex 的 ChromaDB 向量库适配器
# 它让 LlamaIndex 的索引操作能直接对接 ChromaDB
from llama_index.vector_stores.chroma import ChromaVectorStore

# 导入项目的自定义 Embedding 适配器
from .compatible_embedding import OpenAICompatibleEmbedding
# 导入项目的配置类（只用于类型标注）
from .config import Settings
# 导入项目的统一日志器
from .logger import logger
# 导入工具函数：确保目录存在
from .utils import ensure_dir


class IndexManager:
    """负责 Chroma 向量库与索引生命周期管理。"""

    # ChromaDB 的 Collection 名称——所有文档的向量都存在这个 Collection 里
    COLLECTION_NAME = "tech_docs"

    def __init__(self, settings: Settings):
        # 保存配置引用
        self.settings = settings
        # 确保 ChromaDB 的持久化目录存在（不存在则创建）
        ensure_dir(self.settings.chroma_dir)
        # 创建 ChromaDB 的 PersistentClient（本地持久化客户端）
        # path：向量数据库文件的存储路径
        # anonymized_telemetry=False：关闭匿名遥测，避免无关报错噪音
        self.client = chromadb.PersistentClient(
            path=str(self.settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # 配置 LlamaIndex 的全局参数
        self._configure_llama_index()

    def _configure_llama_index(self) -> None:
        """统一配置 LLM、Embedding 与文本切分器。
        这个方法只执行一次，设置了三个全局组件，后续所有索引操作都使用这些配置。"""

        # --- 配置 LLM ---
        # 构建 LLM 初始化参数
        llm_kwargs = {
            "model": self.settings.llm_model,      # 模型名称（如 qwen-plus）
            "api_key": self.settings.api_key,       # API 密钥
            "temperature": 0.1,                      # 低温度保证答案一致性
        }
        # 如果配置了自定义 base_url（如 DashScope 的地址），传入 api_base 参数
        # LlamaIndex 的 OpenAI 类使用 api_base 而不是 base_url（两个命名不同）
        if self.settings.openai_base_url:
            # LlamaIndex OpenAI 兼容接口参数
            llm_kwargs["api_base"] = self.settings.openai_base_url

        # 设置 LlamaIndex 的全局 LLM——所有检索和生成操作默认使用这个 LLM
        LlamaSettings.llm = OpenAI(**llm_kwargs)

        # --- 配置 Embedding ---
        # 使用我们自定义的 OpenAICompatibleEmbedding 适配器
        # 它支持任意 OpenAI 兼容的 Embedding 服务（DashScope/OpenAI/Ollama 等）
        LlamaSettings.embed_model = OpenAICompatibleEmbedding(
            model_name=self.settings.embed_model,
            api_key=self.settings.api_key,
            base_url=self.settings.openai_base_url,
        )

        # --- 配置文本切分器 ---
        # SentenceSplitter：按句子边界智能切分，不会在句子中间截断
        # chunk_size：每个文本块的目标大小（tokens）
        # chunk_overlap：相邻块之间的重叠量（tokens）——保证跨块语义连续
        LlamaSettings.node_parser = SentenceSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

    def _get_collection(self) -> Any:
        """获取或创建 ChromaDB Collection。
        如果 Collection 不存在则创建；如果已存在则直接返回。"""
        return self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            # metadata 指定了向量空间的配置
            # "hnsw:space": "cosine" → 使用 HNSW 索引 + Cosine 余弦距离度量
            metadata={"hnsw:space": "cosine"},
        )

    def _build_index(self, documents: Sequence[Document]) -> VectorStoreIndex:
        """从文档列表构建全新的向量索引。
        这是一个完整重建的过程——文档 → 切分 → 向量化 → 存入 ChromaDB。"""

        # 获取或创建 ChromaDB Collection
        collection = self._get_collection()
        # 将 ChromaDB Collection 包装为 LlamaIndex 的 ChromaVectorStore 适配器
        vector_store = ChromaVectorStore(chroma_collection=collection)
        # 创建 StorageContext（存储上下文），指定向量存储后端为我们的 ChromaDB
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 记录构建开始
        logger.info("开始构建向量索引，文档数: %s", len(documents))
        # VectorStoreIndex.from_documents() 会自动完成：
        # 1. 用 SentenceSplitter 切分文档
        # 2. 用 Embedding 模型将每个 chunk 向量化
        # 3. 将向量存入 ChromaDB（通过 storage_context 指定的后端）
        # documents=list(documents)：确保是 list 类型
        # show_progress=True：显示索引进度条（在终端中）
        
        index = VectorStoreIndex.from_documents(
            documents=list(documents),       # 传入所有 Document 对象
            storage_context=storage_context,  # 指定存到 ChromaDB
            show_progress=True,              # 显示进度条
        )

        # 记录构建完成和向量总数
        logger.info("索引构建完成，当前向量数: %s", collection.count())
        return index

    def _load_index_from_store(self) -> VectorStoreIndex:
        """从已有的 ChromaDB Collection 加载索引。
        这个方法不会触发任何 Embedding 计算——只是加载已存在的向量数据。"""

        collection = self._get_collection()
        vector_store = ChromaVectorStore(chroma_collection=collection)
        # 记录加载情况和向量数
        logger.info("加载已有索引，当前向量数: %s", collection.count())
        # from_vector_store：从已有向量库加载，不执行任何文档处理
        return VectorStoreIndex.from_vector_store(vector_store=vector_store)

    def get_or_create_index(self, documents: Sequence[Document]) -> VectorStoreIndex:
        """优先复用已有索引，不存在时再构建。
        这是最常用的方法——首次启动时构建索引，后续重启直接加载。"""

        collection = self._get_collection()
        # 核心判断：如果 Collection 中已有向量（count > 0），说明之前已经构建过索引
        if collection.count() > 0:
            # 直接加载已有索引——跳过文档加载和 Embedding 计算
            return self._load_index_from_store()

        # Collection 为空，需要从文档构建新索引
        if not documents:
            # 如果连文档都没有，抛出明确的错误信息
            raise ValueError("docs 目录中没有可用文档，无法构建索引。")

        return self._build_index(documents)

    def rebuild_index(self, documents: Sequence[Document]) -> VectorStoreIndex:
        """重建索引：删除旧集合后全量导入。
        适用于：文档大量更新、切换 Embedding 模型、索引损坏修复。"""

        if not documents:
            raise ValueError("docs 目录中没有可用文档，无法重建索引。")

        try:
            # 删除整个 Collection——清空所有旧的向量数据
            self.client.delete_collection(self.COLLECTION_NAME)
            logger.info("已删除旧 collection: %s", self.COLLECTION_NAME)
        except Exception:  # noqa: BLE001
            # Collection 不存在也算正常——首次调用 rebuild 时旧 Collection 可能还没创建
            logger.info("旧 collection 不存在，直接创建新索引。")

        # 全量构建新索引
        return self._build_index(documents)

    def incremental_update(self, all_documents: Sequence[Document]) -> VectorStoreIndex:
        """
        增量导入接口（v1 占位实现）。

        当前版本默认回退到全量重建，保留统一调用入口，后续可替换为真正增量 upsert。
        这样设计的好处：调用方（app.py）不需要关心底层是增量还是全量，
        以后升级实现时只需要改这个方法，调用方代码零改动。
        """

        logger.info("incremental_update 当前使用全量重建策略。")
        # 当前回退到全量重建
        return self.rebuild_index(all_documents)

    def get_index_status(self) -> dict:
        """获取索引状态信息，供前端侧边栏展示。"""
        collection = self._get_collection()
        return {
            "collection_name": self.COLLECTION_NAME,  # Collection 名称
            "vector_count": collection.count(),        # 向量条目总数
            "chroma_dir": str(self.settings.chroma_dir),  # 持久化目录路径
        }
