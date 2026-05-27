# =============================================================================
# app.py — Streamlit 前端入口
# 基于 Streamlit 构建的交互式 Chat UI，是整个 RAG 系统的用户界面。
# 核心设计：侧边栏状态面板 + Chat 对话区 + Session State 管理 + 引用来源展示
# =============================================================================

# 从 __future__ 导入 annotations，让所有类型注解使用延迟求值
from __future__ import annotations

# os：操作系统接口，这里用于设置环境变量
import os
# warnings：Python 的警告控制模块，用于过滤不必要的警告信息
import warnings

# streamlit：Streamlit Web 框架，别名 st 是官方惯例
import streamlit as st

# ---- 环境初始化（在 Streamlit 渲染页面之前执行） ----

# 关闭 ChromaDB 的匿名遥测——避免每次启动都向 Chroma 官方发送统计数据
# 也避免了遥测相关的报错噪音干扰
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# 尝试过滤 pydantic 内部模块产生的 UnsupportedFieldAttributeWarning 警告
# 这个警告在某些 pydantic 版本下是正常的，不影响功能但会污染日志输出
try:
    from pydantic._internal._generate_schema import UnsupportedFieldAttributeWarning
    # 忽略这个特定类型的警告
    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
except Exception:
    # pydantic 内部路径可能随版本变化（不同版本的模块路径不同）
    # 导入失败时不影响主流程——只是不能过滤这个警告而已
    pass

# ---- 导入项目核心模块 ----
# 从 src 包导入所有需要的公共 API（由 __init__.py 统一导出）
from src import DocumentLoader, IndexManager, RAGPipeline, TechDocRetriever, get_settings
# 导入项目的统一日志器
from src.logger import logger

# ---- 页面全局配置（必须是第一个 Streamlit 命令） ----
st.set_page_config(
    page_title="Tech Doc RAG Assistant",  # 浏览器标签页标题
    page_icon="",                          # 浏览器标签页图标（emoji 图书）
    layout="wide",                          # 使用宽屏布局（而非默认的窄屏居中）
    initial_sidebar_state="expanded",       # 侧边栏默认展开
)

# ---- 全局自定义 CSS 样式 ----
st.markdown("""
<style>
    /* 隐藏 Streamlit 默认的页脚和右上角菜单按钮 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* 主页面背景：浅灰蓝色渐变，比纯白更有层次感 */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
    }

    /* 侧边栏：深色主题，模拟专业产品的控制台风格 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f36 0%, #2d3458 100%);
        border-right: none;
    }
    [data-testid="stSidebar"] * {
        color: #e0e4f0 !important;  /* 侧边栏所有文字改为浅色 */
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffffff !important;  /* 标题纯白 */
    }
    [data-testid="stSidebar"] button {
        background: rgba(255,255,255,0.1) !important;     /* 半透明白色背景 */
        border: 1px solid rgba(255,255,255,0.2) !important; /* 半透明白色边框 */
        color: #e0e4f0 !important;
        border-radius: 8px !important;                      /* 圆角按钮 */
        transition: all 0.2s;                               /* 悬停过渡动画 */
    }
    [data-testid="stSidebar"] button:hover {
        background: rgba(255,255,255,0.2) !important;      /* 悬停时加深背景 */
        border-color: rgba(255,255,255,0.4) !important;    /* 悬停时边框更亮 */
    }

    /* 聊天消息气泡：白色卡片 + 圆角 + 微弱阴影——现代 UI 风格 */
    [data-testid="stChatMessage"] {
        background: #ffffff;
        border-radius: 14px;
        padding: 20px 24px !important;
        margin: 8px 0 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        border: 1px solid #eef0f4;
    }

    /* 输入框：圆角 + focus 蓝色边框过渡 */
    [data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
        border: 2px solid #e2e6ed !important;
        padding: 14px 18px !important;
        font-size: 15px !important;
        transition: border-color 0.2s;  /* 边框颜色过渡动画 */
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #4f6ef6 !important;                   /* focus 时变蓝 */
        box-shadow: 0 0 0 3px rgba(79,110,246,0.12) !important; /* focus 时蓝色光晕 */
    }

    /* 引用来源卡片：浅灰背景 + hover 效果 */
    .citation-box {
        background: #f8f9fc;
        border: 1px solid #e8ecf2;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 13px;
        transition: all 0.2s;
    }
    .citation-box:hover {
        border-color: #c8d4e8;
        background: #f0f3f9;
    }
    /* 相关度百分比徽章：小圆角标签 */
    .citation-score {
        display: inline-block;
        background: #eef2ff;
        color: #4f6ef6;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }

    /* 侧边栏指标卡片：暗色背景上的亮色数字 */
    .metric-card {
        background: rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 10px 14px;
        margin: 4px 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .metric-value {
        font-size: 18px;
        font-weight: 700;
        color: #8bb4ff !important;  /* 亮蓝色数字 */
    }
    .metric-label {
        font-size: 11px;
        color: #8899bb !important;  /* 灰色标签 */
        margin-top: 2px;
    }

    /* 自定义滚动条样式：细条 + 圆角 */
    ::-webkit-scrollbar {width: 6px;}
    ::-webkit-scrollbar-track {background: transparent;}
    ::-webkit-scrollbar-thumb {background: #c8d4e8; border-radius: 3px;}
</style>
""", unsafe_allow_html=True)  # unsafe_allow_html=True：允许嵌入原始 HTML/CSS


# ---- 系统初始化函数 ----

def build_pipeline(force_rebuild: bool = False) -> dict:
    """加载文档并初始化 RAG 流程。
    这是整个系统的启动链路——配置→加载→索引→检索器→Pipeline"""

    # Step 1：加载配置（@lru_cache 保证只加载一次 .env）
    settings = get_settings()

    # Step 2：创建文档加载器 + 加载所有文档
    loader = DocumentLoader(settings.docs_dir)
    documents = loader.load_documents()

    # Step 3：创建索引管理器
    index_manager = IndexManager(settings)
    # 根据 force_rebuild 参数选择加载方式
    if force_rebuild:
        # 强制重建：删除旧索引 → 全量重新 Embedding → 构建新索引
        index = index_manager.rebuild_index(documents)
    else:
        # 默认模式：已有索引则复用，没有则构建（最常用的逻辑）
        index = index_manager.get_or_create_index(documents)

    # Step 4：创建检索器（配置检索参数）
    retriever = TechDocRetriever(
        index=index,
        top_k=settings.top_k,                         # 返回 Top-4 个最相关片段
        similarity_threshold=settings.similarity_threshold,  # 相似度阈值 0.45
    )

    # Step 5：创建 RAGPipeline（流程编排器）
    pipeline = RAGPipeline(retriever=retriever, settings=settings)

    # 返回初始化结果字典——前端用这些信息展示系统状态
    return {
        "settings": settings,           # 完整配置对象
        "pipeline": pipeline,           # RAG Pipeline 实例（核心入口）
        "doc_count": len(documents),    # 成功加载的文档数量
        "index_status": index_manager.get_index_status(),  # 索引状态信息
    }


def init_state(force_rebuild: bool = False) -> None:
    """将初始化结果写入 st.session_state。
    Streamlit 的 session_state 是会话级的状态存储——页面刷新后数据仍然保留。"""

    # st.spinner：显示加载动画和提示文字，异步操作完成后自动消失
    with st.spinner("正在加载知识库..."):
        state = build_pipeline(force_rebuild=force_rebuild)

    # 将初始化结果写入 session_state——后续所有操作从这里读取
    st.session_state.settings = state["settings"]
    st.session_state.pipeline = state["pipeline"]
    st.session_state.doc_count = state["doc_count"]
    st.session_state.index_status = state["index_status"]


# ---- Session State 初始化 ----

# 检查 messages 是否已存在于 session_state
# 如果不存在（首次打开页面或用户清除了缓存），初始化为空列表
if "messages" not in st.session_state:
    st.session_state.messages = []

# 检查 pipeline 是否已初始化
# 如果不存在（首次打开页面），执行初始化流程
if "pipeline" not in st.session_state:
    try:
        init_state(force_rebuild=False)  # 默认使用复用模式
    except Exception as exc:  # noqa: BLE001
        # 初始化失败——记录完整日志 + 在前端展示错误信息
        logger.exception("初始化失败: %s", exc)
        st.error(f"系统初始化失败，请检查配置文件。")
        st.stop()  # 停止后续渲染，防止在未初始化状态下执行操作


# ---- 侧边栏：知识库控制台 ----

with st.sidebar:
    # 顶部：系统标识 + 状态灯
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:10px 0 20px 0;">
        <div style="font-size:32px;"></div>
        <div>
            <div style="font-size:18px;font-weight:700;color:#fff;">知识库控制台</div>
            <div style="font-size:11px;color:#8899bb;">Knowledge Base Console</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 系统状态指示灯
    # 根据向量条目数判断状态：> 0 表示就绪，否则等待索引
    status_emoji = "" if st.session_state.index_status["vector_count"] > 0 else ""
    status_text = "就绪" if st.session_state.index_status["vector_count"] > 0 else "等待索引"
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px;
        background:rgba(255,255,255,0.06);border-radius:10px;padding:12px 16px;">
        <div style="font-size:24px;">{status_emoji}</div>
        <div>
            <div style="font-weight:600;color:#fff;">系统状态：{status_text}</div>
            <div style="font-size:11px;color:#8899bb;">模型提供方：{st.session_state.settings.api_provider}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 四个关键指标卡片（2x2 网格布局）
    col1, col2 = st.columns(2)  # 第一行两列
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.doc_count}</div>
            <div class="metric-label">已加载文档</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.index_status['vector_count']}</div>
            <div class="metric-label">向量条目</div>
        </div>
        """, unsafe_allow_html=True)

    col3, col4 = st.columns(2)  # 第二行两列
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">4</div>
            <div class="metric-label">Top-K</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">512</div>
            <div class="metric-label">Chunk Size</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")  # 分隔线

    # 操作按钮区
    c1, c2 = st.columns(2)  # 两个按钮并排
    with c1:
        # 重建索引按钮：删除旧索引 + 全量重建
        if st.button(" 重建索引", use_container_width=True):
            try:
                init_state(force_rebuild=True)  # 传入 force_rebuild=True
                st.rerun()  # 刷新页面——触发侧边栏和状态显示更新
            except Exception as exc:
                logger.exception("重建索引失败: %s", exc)
                st.error("重建失败")
    with c2:
        # 清空对话按钮：只清除 messages，不动索引
        if st.button(" 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # 底部：支持的文件格式提示
    st.markdown("---")
    st.markdown("""
    <div style="font-size:10px;color:#667;line-height:1.6;">
        <div style="color:#8899bb;font-weight:600;margin-bottom:4px;">支持格式</div>
        <div>.md &nbsp; .txt &nbsp; .html &nbsp; .pdf</div>
    </div>
    """, unsafe_allow_html=True)


# ---- 主页面：Chat UI ----

# 顶部标题区（产品品牌展示）
st.markdown("""
<div style="padding:8px 0 0 0;">
    <div style="display:flex;align-items:center;gap:14px;">
        <div style="font-size:38px;"></div>
        <div>
            <div style="font-size:26px;font-weight:800;color:#1a1f36;letter-spacing:-0.5px;">
                Tech Doc Assistant
            </div>
            <div style="font-size:13px;color:#8899bb;margin-top:2px;">
                智能技术文档问答  ·  让你的知识库开口说话
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 首次打开时的欢迎引导（只在没有对话历史时显示）
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center;padding:40px 20px 20px 20px;">
        <div style="font-size:56px;margin-bottom:12px;"></div>
        <div style="font-size:17px;color:#555;margin-bottom:8px;">
            向知识库提问，获取精准答案
        </div>
        <div style="font-size:13px;color:#999;">
            支持自然语言检索  ·  答案追溯源文档  ·  完全本地化运行
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---- 展示历史消息 ----
# 遍历 session_state.messages 列表，每条消息渲染一个聊天气泡
for message in st.session_state.messages:
    # st.chat_message(role)：自动根据 role 显示对应的头像和样式
    # role 可以是 "user" 或 "assistant"
    with st.chat_message(message["role"]):
        # 显示消息的文本内容（支持 Markdown 格式）
        st.markdown(message["content"])

        # 如果是助手消息且有引用来源，展示引用区域
        if message["role"] == "assistant" and message.get("citations"):
            # st.expander：折叠面板——默认收起，用户点击展开查看引用详情
            with st.expander(f"查看引用来源 ({len(message['citations'])})"):
                # 遍历每条引用
                for idx, item in enumerate(message["citations"], 1):
                    # 将相似度分数（0~1）转为百分比（0~100）
                    score_pct = int(item["score"] * 100)
                    # 根据百分比分档显示不同颜色：
                    # >= 70% → 绿色（高相关度）
                    # >= 50% → 橙色（中等相关度）
                    # < 50%  → 灰色（低相关度）
                    score_color = "#10b981" if score_pct >= 70 else "#f59e0b" if score_pct >= 50 else "#94a3b8"
                    # 渲染引用卡片
                    st.markdown(f"""
                    <div class="citation-box">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                            <strong style="color:#1a1f36;">{idx}. {item['source']}</strong>
                            <span class="citation-score" style="background:{score_color}18;color:{score_color};">
                                相关度 {score_pct}%
                            </span>
                        </div>
                        <div style="color:#64748b;font-size:12px;line-height:1.6;">
                            {item['snippet']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# ---- 聊天输入框 ----
# st.chat_input：固定在页面底部的输入框，placeholder 是灰色提示文字
user_query = st.chat_input("输入你的技术问题，例如：python有哪些数据类型？")

# 用户输入了内容
if user_query:
    # 将用户消息追加到对话历史
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 渲染用户消息气泡
    with st.chat_message("user"):
        st.markdown(user_query)

    # 渲染助手消息区域
    with st.chat_message("assistant"):
        # st.spinner：在等待检索和生成时显示加载动画
        with st.spinner(""):
            # 调用核心 Pipeline——ask() 方法串联了检索+生成全流程
            result = st.session_state.pipeline.ask(user_query)

        # 显示 LLM 生成的答案
        st.markdown(result["answer"])

        # 如果有引用来源，展示折叠面板
        if result["citations"]:
            with st.expander(f"引用来源 ({len(result['citations'])})"):
                for idx, item in enumerate(result["citations"], 1):
                    score_pct = int(item["score"] * 100)
                    score_color = "#10b981" if score_pct >= 70 else "#f59e0b" if score_pct >= 50 else "#94a3b8"
                    st.markdown(f"""
                    <div class="citation-box">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                            <strong style="color:#1a1f36;">{item['source']}</strong>
                            <span class="citation-score" style="background:{score_color}18;color:{score_color};">
                                相关度 {score_pct}%
                            </span>
                        </div>
                        <div style="color:#64748b;font-size:12px;line-height:1.6;">
                            {item['snippet']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # 将助手消息追加到对话历史（包括 citations 供后续展示）
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "citations": result["citations"],
    })
