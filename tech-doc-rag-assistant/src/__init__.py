# =============================================================================
# __init__.py — 包入口，统一导出公共 API
# 外部模块只需 from src import XXX 即可使用所有核心组件。
# 设计模式：外观模式（Facade）——隐藏内部模块结构，对外提供简洁的统一接口。
# =============================================================================

# 从 config 模块导出 Settings 类和 get_settings 函数
from .config import Settings, get_settings
# 从 indexer 模块导出 IndexManager 类
from .indexer import IndexManager
# 从 loaders 模块导出 DocumentLoader 类
from .loaders import DocumentLoader
# 从 rag_pipeline 模块导出 RAGPipeline 和 AgentRAGPipeline 类
from .rag_pipeline import AgentRAGPipeline, RAGPipeline
# 从 retriever 模块导出 RetrievedChunk 数据类和 TechDocRetriever 类
from .retriever import RetrievedChunk, TechDocRetriever
# 从 bm25_retriever 模块导出 BM25Retriever 类
from .bm25_retriever import BM25Retriever
# 从 hybrid_retriever 模块导出 HybridRetriever 类
from .hybrid_retriever import HybridRetriever
# 从 agent_core 模块导出 Agent 相关类
from .agent_core import AgentResult, AgentStep, QueryRewriteAgent, RetrievalAgent, RouterAgent
# 从 agent_tools 模块导出工具初始化函数
from .agent_tools import create_tools, init_tools

# __all__ 变量：定义了 from src import * 时会导入哪些名称
# 这也是一种文档——明确告诉使用者哪些是公共 API
__all__ = [
    "Settings",              # 配置对象（frozen dataclass）
    "get_settings",          # 配置加载函数（@lru_cache 单例）
    "IndexManager",          # ChromaDB 索引管理器
    "DocumentLoader",        # 多格式文档加载器
    "RAGPipeline",           # 原始 RAG 流程编排器（简单模式）
    "AgentRAGPipeline",      # Agent 协同 RAG 流程编排器（增强模式）
    "RetrievedChunk",        # 检索结果数据类
    "TechDocRetriever",      # 向量检索过滤器
    "BM25Retriever",         # BM25 关键词检索器
    "HybridRetriever",       # 混合检索器（BM25 + 向量 + RRF）
    "RouterAgent",           # 路由决策 Agent
    "QueryRewriteAgent",     # 查询改写 Agent
    "RetrievalAgent",        # 检索+生成 Agent
    "AgentResult",           # Agent 结果数据类
    "AgentStep",             # Agent 步骤数据类
    "create_tools",          # 创建 Agent 工具列表
    "init_tools",            # 初始化工具全局引用
]
