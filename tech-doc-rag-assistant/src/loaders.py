# =============================================================================
# loaders.py — 文档加载器
# 负责扫描知识库目录、加载和解析多格式技术文档。
# 核心设计：格式支持 + 元数据注入 + 异常隔离
# =============================================================================

# 从 __future__ 导入 annotations，让所有类型注解使用延迟求值
from __future__ import annotations

# Path：pathlib 的路径操作类
from pathlib import Path
# List：typing 的列表泛型，用于类型标注
from typing import List

# LlamaIndex 的 SimpleDirectoryReader：一个智能的多格式文档加载器
# 它能自动检测文件类型（PDF / Markdown / HTML 等）并调用对应的解析器
from llama_index.core import SimpleDirectoryReader
# Document：LlamaIndex 的文档对象，包含文本内容和元数据
from llama_index.core.schema import Document

# 导入项目统一的日志器
from .logger import logger


class DocumentLoader:
    """负责扫描并加载本地技术文档。"""

    # 类属性：支持的文件扩展名集合
    # 使用 set 集合——成员检查 O(1) 比 list 的 O(n) 快
    SUPPORTED_EXTENSIONS = {".md", ".txt", ".html", ".pdf"}

    def __init__(self, docs_dir: str | Path):
        # 统一转为 Path 对象，兼容字符串和 Path 两种传参方式
        self.docs_dir = Path(docs_dir)

    def scan_files(self) -> List[Path]:
        """递归扫描 docs 目录下支持的文件。"""

        # 检查文档目录是否存在，不存在时记录警告并返回空列表
        # 不抛异常——目录不存在不应该导致程序崩溃
        if not self.docs_dir.exists():
            logger.warning("文档目录不存在: %s", self.docs_dir)
            return []

        # 列表推导式：遍历 docs_dir 下的所有文件和子目录
        # rglob("*")：递归匹配所有文件和目录（相当于 shell 的 **/*）
        # is_file()：只保留文件，排除目录
        # suffix.lower() in SUPPORTED_EXTENSIONS：按后缀过滤（lower() 做大小写容错）
        files = [
            file_path
            for file_path in self.docs_dir.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]
        # 排序返回——保证每次加载的文件顺序一致，便于调试
        return sorted(files)

    def load_documents(self) -> List[Document]:
        """逐文件加载，异常时记录日志并跳过。"""

        # 先扫描文件列表
        files = self.scan_files()
        # 没有扫描到文件：记录警告，返回空列表
        if not files:
            logger.warning("未扫描到可用文档，请检查目录: %s", self.docs_dir)
            return []

        # 累积所有文档对象
        all_documents: List[Document] = []
        # 记录成功加载的文件数（用于统计日志）
        success_files = 0

        # 逐个文件加载
        for file_path in files:
            try:
                # 创建 SimpleDirectoryReader 实例
                # input_files=[str(file_path)]：只加载当前这一个文件
                # filename_as_id=True：用文件名作为文档的唯一标识 ID
                reader = SimpleDirectoryReader(
                    input_files=[str(file_path)],
                    filename_as_id=True,
                )
                # 执行加载——reader.load_data() 返回该文件的 Document 对象列表
                # 一个文件可能包含多个 Document（如 PDF 的每一页）
                docs = reader.load_data()

                # 为每个 Document 注入元数据
                for doc in docs:
                    # 获取已有元数据（doc.metadata 可能为 None，用 or {} 兜底）
                    metadata = dict(doc.metadata or {})
                    # 注入 source：文件名（如 "fastapi_intro.md"）
                    metadata["source"] = file_path.name
                    # 注入 filepath：文件的绝对路径
                    metadata["filepath"] = str(file_path.resolve())
                    # 注入 file_type：文件后缀名（如 ".md"）
                    metadata["file_type"] = file_path.suffix.lower()
                    # 更新文档的元数据
                    doc.metadata = metadata

                # 将当前文件的所有 Document 对象追加到总列表
                all_documents.extend(docs)
                success_files += 1
            except Exception as exc:  # noqa: BLE001
                # 异常隔离：单个文件加载失败只记录警告日志
                # 不会因为一个文件坏了就让整个加载流程崩溃
                logger.warning("文件加载失败，已跳过: %s | 错误: %s", file_path, exc)

        # 加载完成：记录统计信息
        logger.info(
            "文档加载完成: 成功文件=%s / 总文件=%s, 文档对象=%s",
            success_files,  # 成功加载的文件数
            len(files),     # 扫描到的总文件数
            len(all_documents),  # 生成的 Document 对象数（可能多于文件数）
        )
        return all_documents
