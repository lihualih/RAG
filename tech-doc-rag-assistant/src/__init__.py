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
# 从 rag_pipeline 模块导出 RAGPipeline 类（前端直接调用它）
from .rag_pipeline import RAGPipeline
# 从 retriever 模块导出 RetrievedChunk 数据类和 TechDocRetriever 类
from .retriever import RetrievedChunk, TechDocRetriever

# __all__ 变量：定义了 from src import * 时会导入哪些名称
# 这也是一种文档——明确告诉使用者哪些是公共 API
__all__ = [
    "Settings",          # 配置对象（frozen dataclass）
    "get_settings",      # 配置加载函数（@lru_cache 单例）
    "IndexManager",      # ChromaDB 索引管理器
    "DocumentLoader",    # 多格式文档加载器
    "RAGPipeline",       # RAG 流程编排器（核心入口）
    "RetrievedChunk",    # 检索结果数据类
    "TechDocRetriever",  # 检索过滤器
]
