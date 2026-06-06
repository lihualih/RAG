# =============================================================================
# agent_tools.py — Agent 工具定义（企业级）
# 从真实业务场景出发设计工具，覆盖检索增强、上下文扩展、证据对比、
# 实体查询、文档导航、知识运营等维度。
#
# 工具清单：
# ├── Tier 1 核心检索增强 ─────────────────────────────
# │   1. search_knowledge — 带元数据过滤的混合检索（主力工具）
# │   2. expand_context  — 获取指定内容的前后相邻段落
# │
# ├── Tier 2 证据对比与实体查询 ───────────────────────
# │   3. extract_code     — 从知识库中提取相关代码示例
# │   4. compare_sources  — 对比不同来源对同一问题的说法
# │   5. entity_lookup    — 查找技术术语在知识库中的定义
# │
# └── Tier 3 知识运营与管理 ──────────────────────────
# │   6. document_outline — 获取文档的目录结构/标题层级
# │   7. knowledge_stats  — 知识库统计信息
# =============================================================================

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from llama_index.core.tools import FunctionTool

from .logger import logger

# 类型检查时导入（避免运行时循环引用）
if TYPE_CHECKING:
    from .hybrid_retriever import HybridRetriever


# ---- 全局引用：由 AgentRAGPipeline 在初始化时注入 ----
_hybrid_retriever: Optional["HybridRetriever"] = None
_docs_dir: Optional[Path] = None

# 存储最近一次检索结果，供 expand_context 等关联工具使用
_last_search_results: List[Dict[str, Any]] = []


def init_tools(hybrid_retriever: "HybridRetriever", docs_dir: Path) -> None:
    """初始化工具的全局引用（由 AgentRAGPipeline 调用）。"""
    global _hybrid_retriever, _docs_dir
    _hybrid_retriever = hybrid_retriever
    _docs_dir = docs_dir


# =============================================================================
# 工具 1（核心）：带元数据过滤的混合检索
# =============================================================================

def search_knowledge(
    query: str,
    source_file: str = "",
    file_type: str = "",
    max_results: int = 5,
) -> str:
    """【核心工具】在技术文档知识库中执行混合检索（BM25关键词 + 向量语义），支持元数据过滤。

    这是回答技术问题的主力检索工具。当你需要查找某个技术概念、API用法、
    配置方法、故障排查步骤时，优先使用此工具。

    支持的过滤条件：
    - source_file: 限定在某个文档内搜索（如 "fastapi_intro.md"）
    - file_type: 限定文件类型（如 ".md", ".pdf", ".txt"）
    - max_results: 返回的最大结果数（默认5，范围1-10）

    返回结果包含：内容片段、来源文件、相关度分数。

    Args:
        query: 搜索查询，自然语言或关键词均可
        source_file: 可选，限定在指定文档内搜索。不填则搜索全部文档
        file_type: 可选，限定文件类型。如 ".md" / ".pdf" / ".txt"
        max_results: 返回的最大结果数，默认5，建议不超过8
    Returns:
        格式化的检索结果，包含序号、来源、分数、内容
    """
    global _last_search_results

    if _hybrid_retriever is None:
        return "错误：检索器未初始化，请检查系统配置。"

    # 规范化 max_results
    max_results = max(1, min(max_results, 10))

    # 执行混合检索
    chunks = _hybrid_retriever.retrieve(query)

    # 保存原始结果用于上下文扩展
    _last_search_results = []

    if not chunks:
        return (
            "未找到相关文档片段。建议：\n"
            "1. 尝试使用更通用或更简洁的关键词\n"
            "2. 使用 knowledge_stats 工具查看知识库覆盖范围\n"
            "3. 使用 document_outline 工具浏览可用的文档结构"
        )

    # 元数据过滤
    filtered = []
    for chunk in chunks:
        # source_file 过滤
        if source_file and chunk.source != source_file:
            continue
        # file_type 过滤
        if file_type:
            ext = Path(chunk.filepath).suffix.lower()
            if ext != file_type.lower().lstrip("."):
                continue
        filtered.append(chunk)

    if not filtered:
        return (
            f"按过滤条件（source_file={source_file or '不限'}, "
            f"file_type={file_type or '不限'}）未找到匹配结果。"
            "请放宽过滤条件后再试。"
        )

    # 截断到 max_results
    filtered = filtered[:max_results]

    # 保存到全局状态
    for i, chunk in enumerate(filtered):
        _last_search_results.append({
            "index": i + 1,
            "text": chunk.text,
            "source": chunk.source,
            "filepath": chunk.filepath,
            "score": chunk.score,
        })

    # 格式化结果
    results = [f"检索到 {len(filtered)}/{len(chunks)} 条相关结果（已过滤）：\n"]
    for i, chunk in enumerate(filtered, 1):
        score_pct = int(chunk.score * 100)
        relevance = "  高相关" if score_pct >= 8 else "  相关" if score_pct >= 5 else ""
        results.append(
            f"── 结果 [{i}] {relevance} ──\n"
            f"  来源: {chunk.source}\n"
            f"  分数: {score_pct}%\n"
            f"  内容: {chunk.text.strip()}"
        )

    return "\n".join(results)


# =============================================================================
# 工具 2：上下文扩展
# =============================================================================

def expand_context(query: str, context_size: int = 3) -> str:
    """获取指定查询的更广泛上下文，解决单个chunk信息不完整的问题。

    当检索到的代码片段缺少开头、配置段落被截断、或者需要理解某个概念的
    前后文关系时使用。此工具会扩大搜索范围获取更多相关内容。

    Args:
        query: 与上次 search_knowledge 相关的补充查询或关键词
        context_size: 额外获取的上下文片段数，默认3
    Returns:
        扩展后的上下文内容
    """
    if _hybrid_retriever is None:
        return "错误：检索器未初始化"

    if not _last_search_results:
        return "提示：请先使用 search_knowledge 工具进行检索，然后再使用 expand_context 扩展上下文。"

    # 用更大的 top_k 重新检索
    chunks = _hybrid_retriever.retrieve(query)

    if not chunks:
        return "未找到额外的上下文内容。"

    # 过滤掉已在上次结果中的内容
    existing_texts = {r["text"] for r in _last_search_results}
    new_chunks = [c for c in chunks if c.text not in existing_texts]

    if not new_chunks:
        return "未找到新的上下文内容，当前检索结果已是最完整的。"

    new_chunks = new_chunks[:context_size]

    results = [f"扩展上下文（+{len(new_chunks)} 个新片段）：\n"]
    for i, chunk in enumerate(new_chunks, 1):
        results.append(
            f"── 扩展 [{i}] ──\n"
            f"  来源: {chunk.source}\n"
            f"  内容: {chunk.text.strip()}"
        )

    return "\n".join(results)


# =============================================================================
# 工具 3：代码提取
# =============================================================================

def extract_code(query: str, language: str = "") -> str:
    """从知识库中检索并提取相关的代码示例。

    当用户询问"怎么实现"、"有代码示例吗"、"给我写一段"时使用。
    会在检索结果中优先提取代码块（``` 包裹的内容），直接返回可用的代码。

    Args:
        query: 搜索查询，描述需要什么代码（如 "FastAPI 路由定义"）
        language: 可选，限定编程语言（如 "python", "bash", "yaml"）
    Returns:
        提取到的代码示例，带来源标注
    """
    if _hybrid_retriever is None:
        return "错误：检索器未初始化"

    chunks = _hybrid_retriever.retrieve(query)
    if not chunks:
        return "未找到相关代码示例。请尝试更换搜索关键词。"

    code_blocks = []
    code_pattern = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)

    for chunk in chunks:
        matches = code_pattern.findall(chunk.text)
        for lang, code in matches:
            lang = lang.strip().lower()
            # 语言过滤
            if language and language.lower() != lang:
                continue
            # 过滤太短的片段（可能是内联代码）
            if len(code.strip()) < 20:
                continue
            code_blocks.append({
                "language": lang or "text",
                "code": code.strip(),
                "source": chunk.source,
            })

    if not code_blocks:
        return (
            "检索到的内容中没有找到完整的代码块。"
            "建议使用 search_knowledge 工具查看原始文档内容。"
        )

    # 去重
    seen = set()
    unique_blocks = []
    for block in code_blocks:
        key = block["code"][:100]
        if key not in seen:
            seen.add(key)
            unique_blocks.append(block)

    results = [f"从知识库中提取到 {len(unique_blocks)} 个代码示例：\n"]
    for i, block in enumerate(unique_blocks[:5], 1):
        results.append(
            f"── 代码示例 [{i}] ({block['language']}) ──\n"
            f"  来源: {block['source']}\n"
            f"```{block['language']}\n{block['code']}\n```"
        )

    return "\n".join(results)


# =============================================================================
# 工具 4：来源对比
# =============================================================================

def compare_sources(query: str) -> str:
    """对比知识库中不同文档对同一问题的描述，找出异同点。

    当用户问"A 和 B 有什么区别"、"不同文档说的是否一致"、或者需要排查
    矛盾信息时使用。会返回多个来源的观点摘要，帮助形成全面的理解。

    Args:
        query: 需要对比的技术问题
    Returns:
        各来源的观点对比
    """
    if _hybrid_retriever is None:
        return "错误：检索器未初始化"

    # 多取一些结果以便对比
    chunks = _hybrid_retriever.retrieve(query)

    if len(chunks) < 2:
        return "检索结果不足，无法进行来源对比。请尝试更宽泛的查询。"

    # 按来源文件分组
    groups: Dict[str, List[str]] = {}
    for chunk in chunks[:10]:
        if chunk.source not in groups:
            groups[chunk.source] = []
        groups[chunk.source].append(chunk.text)

    results = [f"来源对比分析（{len(groups)} 个文档涉及此问题）：\n"]

    for source, texts in groups.items():
        combined = " | ".join(texts)
        # 截断摘要
        summary = combined[:300] + "..." if len(combined) > 300 else combined
        results.append(
            f"── 来源 [{source}] ──\n"
            f"  相关内容摘要: {summary}"
        )

    # 如果只有单一来源，提示
    if len(groups) == 1:
        results.append(
            "\n⚠ 注意：只有一个文档涉及此问题，知识库中可能缺少多角度的信息。"
        )

    return "\n".join(results)


# =============================================================================
# 工具 5：实体查询
# =============================================================================

def entity_lookup(term: str, context_lines: int = 2) -> str:
    """在知识库中查找某个技术术语/实体的所有定义和用法。

    当用户问"什么是 X"、"X 是什么意思"、"X 在项目中怎么用"时使用。
    会返回该术语在知识库中的所有出现位置和上下文。

    与 search_knowledge 的区别：entity_lookup 侧重于精确术语匹配，
    会找出术语的所有出现位置；search_knowledge 侧重于语义匹配。

    Args:
        term: 技术术语，如 "Docker Compose"、"asyncio"、"JWT"
        context_lines: 每个匹配位置展示的上下文行数，默认2行
    Returns:
        术语的定义和使用位置汇总
    """
    if _hybrid_retriever is None:
        return "错误：检索器未初始化"

    # 混合检索，侧重于关键词
    chunks = _hybrid_retriever.retrieve(term)

    if not chunks:
        return (
            f"知识库中未找到关于 '{term}' 的信息。\n"
            "建议：\n"
            "1. 检查拼写是否正确\n"
            "2. 尝试更通用的术语（如用 'docker' 代替 'docker compose'）\n"
            "3. 使用 search_knowledge 工具进行语义搜索"
        )

    # 在结果中高亮术语出现
    matches = []
    for chunk in chunks[:8]:
        text = chunk.text
        # 不区分大小写查找
        positions = [
            m.start() for m in re.finditer(re.escape(term), text, re.IGNORECASE)
        ]

        for pos in positions[:3]:  # 每个chunk最多展示3处
            start = max(0, pos - 80)
            end = min(len(text), pos + len(term) + 80)
            context = text[start:end]
            if start > 0:
                context = "..." + context
            if end < len(text):
                context = context + "..."

            matches.append({
                "source": chunk.source,
                "context": context.strip(),
            })

    if not matches:
        return f"在检索结果中没有精确匹配到 '{term}' 的文本，但可能有语义相关的描述。建议使用 search_knowledge 工具查看。"

    # 去重
    seen = set()
    unique_matches = []
    for m in matches:
        key = m["context"]
        if key not in seen:
            seen.add(key)
            unique_matches.append(m)

    results = [f"术语 '{term}' 在知识库中的出现（{len(unique_matches)} 处）：\n"]
    for i, m in enumerate(unique_matches[:10], 1):
        results.append(
            f"── 位置 [{i}] / {m['source']} ──\n"
            f"  {m['context']}"
        )

    return "\n".join(results)


# =============================================================================
# 工具 6：文档结构导航
# =============================================================================

def document_outline(filename: str) -> str:
    """获取指定文档的目录/标题结构，帮助快速了解文档覆盖范围。

    当用户问"文档里有什么内容"、"从哪里可以找到 XXX"时使用。
    先获取文档结构，再决定需要深入阅读哪些章节。

    Args:
        filename: 文档文件名，如 "fastapi_intro.md"
    Returns:
        文档的标题层级结构
    """
    if _docs_dir is None:
        return "错误：文档目录未配置"

    matches = list(_docs_dir.rglob(filename))
    if not matches:
        return f"未找到文件 '{filename}'。使用 knowledge_stats 工具查看可用文档列表。"

    file_path = matches[0]

    if file_path.suffix.lower() == ".pdf":
        # PDF 文件用检索结果推断主题
        chunks = _hybrid_retriever.retrieve(filename.replace(".pdf", "")) if _hybrid_retriever else []
        if chunks:
            sources = list({c.source for c in chunks[:10]})
            return f"[PDF 文档] {file_path.name}\n相关主题: {', '.join(sources[:5])}"

        return (
            f"[PDF 文档] {file_path.name}\n"
            f"文件大小: {file_path.stat().st_size / 1024:.1f} KB\n"
            "PDF 文件无法直接提取标题结构，请使用 search_knowledge 工具搜索文档内容。"
        )

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"读取文件失败: {exc}"

    # 提取 Markdown 标题
    heading_pattern = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)
    headings = heading_pattern.findall(content)

    if not headings:
        return f"[{file_path.name}] 文档无标题结构（{len(content)} 字符）"

    result_lines = [
        f"── {file_path.name} 文档结构 ──",
        f"共 {len(headings)} 个标题节点\n",
    ]

    for level_marker, title in headings:
        level = len(level_marker)
        indent = "  " * (level - 1)
        prefix = ["", "", "├─ ", "│  ├─ ", "│  │  ├─ "][level - 1]
        result_lines.append(f"{indent}{prefix}{title.strip()}")

    return "\n".join(result_lines)


# =============================================================================
# 工具 7：知识库运营统计
# =============================================================================

def knowledge_stats() -> str:
    """获取知识库的整体统计信息和覆盖范围概况。

    当用户问"知识库有多大"、"有哪些类型的文档"、"什么时候更新的"时使用。
    也可以用作检索前的摸底——先了解覆盖范围，再决定搜索策略。

    Returns:
        知识库统计报告
    """
    if _docs_dir is None or not _docs_dir.exists():
        return "知识库未配置或目录不存在。"

    extensions = {".md", ".txt", ".html", ".pdf"}
    files = sorted([
        f for f in _docs_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in extensions
    ])

    if not files:
        return "知识库中暂无文档。"

    # 统计文件类型分布
    type_counts = Counter()
    total_size = 0
    latest_time = None
    for f in files:
        type_counts[f.suffix.lower()] += 1
        total_size += f.stat().st_size
        mtime = f.stat().st_mtime
        if latest_time is None or mtime > latest_time:
            latest_time = mtime

    latest_str = datetime.fromtimestamp(latest_time).strftime("%Y-%m-%d %H:%M") if latest_time else "未知"

    # 获取向量库统计
    vector_count = 0
    if _hybrid_retriever:
        try:
            test_chunks = _hybrid_retriever.retrieve("test")
            # 通过检索器间接了解规模
            _ = test_chunks  # 不实际使用，只是触发日志
        except Exception:
            pass

    lines = [
        "── 知识库运营报告 ──\n",
        f"  文档总数: {len(files)}",
        f"  总大小: {total_size / 1024:.1f} KB",
        f"  最新更新: {latest_str}",
        f"  文件类型分布:",
    ]

    for ext, count in type_counts.most_common():
        labels = {".md": "Markdown", ".txt": "纯文本", ".html": "HTML", ".pdf": "PDF"}
        lines.append(f"    {labels.get(ext, ext)} ({ext}): {count} 个文件")

    lines.append(f"\n  文档清单:")
    for f in files:
        lines.append(f"    - {f.name} ({f.suffix}, {f.stat().st_size / 1024:.1f}KB)")

    return "\n".join(lines)


# =============================================================================
# 工具注册
# =============================================================================

def create_tools() -> List[FunctionTool]:
    """创建并返回所有 Agent 可用的企业级工具列表。

    使用 LlamaIndex 的 FunctionTool.from_defaults() 封装普通函数。
    LlamaIndex 会自动从函数签名和 docstring 提取函数的名称、描述和参数 schema。

    LLM 会根据工具的 description 自动选择何时调用哪个工具。

    Returns:
        FunctionTool 列表
    """
    tools = [
        # Tier 1：核心检索增强
        FunctionTool.from_defaults(
            fn=search_knowledge,
            name="search_knowledge",
            description=(
                "【核心工具】在技术文档知识库中执行 BM25+向量混合检索。"
                "支持按文档名(source_file)和文件类型(file_type)过滤。"
                "当用户询问任何技术问题、API用法、配置方法、故障排查时，"
                "这是你的首选工具。参数: query(必填,搜索关键词), "
                "source_file(可选,限定文档), file_type(可选,如.md/.pdf), "
                "max_results(可选,默认5)。"
            ),
        ),
        FunctionTool.from_defaults(
            fn=expand_context,
            name="expand_context",
            description=(
                "扩展上一次检索结果的上下文。当检索到的片段被截断、信息不完整、"
                "或者需要理解前后文关系时使用。必须先调用 search_knowledge 后再使用此工具。"
                "参数: query(补充查询), context_size(扩展片段数,默认3)。"
            ),
        ),

        # Tier 2：证据对比与实体查询
        FunctionTool.from_defaults(
            fn=extract_code,
            name="extract_code",
            description=(
                "从知识库中提取与查询相关的代码示例。从文档中的代码块(```)提取。"
                "当用户问'怎么实现'、'有代码吗'、'写个示例'时使用。"
                "参数: query(搜索查询), language(可选,限定语言如python/bash/yaml)。"
            ),
        ),
        FunctionTool.from_defaults(
            fn=compare_sources,
            name="compare_sources",
            description=(
                "对比不同文档对同一问题的描述，找出异同。"
                "当用户问'有什么区别'、'不同文档说的是否一致'、或排查矛盾信息时使用。"
                "参数: query(需要对比的技术问题)。"
            ),
        ),
        FunctionTool.from_defaults(
            fn=entity_lookup,
            name="entity_lookup",
            description=(
                "在知识库中查找技术术语的所有定义和用法。精确匹配术语名称。"
                "当用户问'什么是 X'、'X 是什么意思'时使用。"
                "与 search_knowledge 的区别：entity_lookup 侧重精确术语匹配，search_knowledge 侧重语义搜索。"
                "参数: term(技术术语), context_lines(每个匹配的上下文行数)。"
            ),
        ),

        # Tier 3：知识运营与管理
        FunctionTool.from_defaults(
            fn=document_outline,
            name="document_outline",
            description=(
                "获取文档的目录/标题结构。帮助了解文档覆盖范围，定位信息在哪个章节。"
                "当用户问'文档里有什么'、'从哪里可以找到'时使用。"
                "参数: filename(文档文件名,如 'fastapi_intro.md')。"
            ),
        ),
        FunctionTool.from_defaults(
            fn=knowledge_stats,
            name="knowledge_stats",
            description=(
                "获取知识库的整体统计信息：文档数量、类型分布、最后更新时间。"
                "当用户问'知识库有多大'、'有哪些文档'、'覆盖什么内容'时使用。"
                "也可以作为检索前的摸底——先了解知识库范围，再决定搜索策略。"
            ),
        ),
    ]

    logger.info("Agent 企业级工具注册完成，工具数: %s", len(tools))
    return tools
