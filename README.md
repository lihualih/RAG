# Tech Doc RAG Assistant — 企业级技术文档智能问答系统

## 1. 项目简介

`tech-doc-rag-assistant` 是一个基于 RAG（检索增强生成）技术栈的本地技术文档智能问答系统。项目在传统 RAG 基础上进行了三项核心优化：**BM25+向量混合检索**、**多 Agent 协同**、**企业级工具体系**。

系统实现「文档导入 → 文本切分 → Embedding 向量化 → BM25关键词索引 → RRF混合检索 → Agent路由/改写/推理 → 答案生成 → 引用展示 → 思考过程可视化」的完整闭环，适合应届生简历展示与面试演示。

## 2. 核心亮点（面试重点）

### 2.1 BM25 + 向量混合检索（RRF 融合）

| 检索方式 | 擅长 | 短板 |
|---------|------|------|
| 向量语义检索 | 语义理解（"部署" → "容器化发布"） | 精确术语匹配弱 |
| BM25 关键词检索 | 精确匹配（"FastAPI" → 含"FastAPI"的文档） | 不理解语义 |

通过 **倒数排名融合（RRF）** 合并两路结果：`score(d) = Σ 1/(k + rank_i(d))`，k=60。无需归一化，天然兼容不同量纲的分数，且有交叉验证效应——两路排名都靠前的文档获得最高分。

### 2.2 多 Agent 协同架构

```
用户问题
  │
  ▼
RouterAgent（路由决策） → direct / general / rag
  │
  ▼
QueryRewriteAgent（查询改写） → 同义词扩展 + 子问题拆解
  │
  ▼
HybridRetriever（混合检索） → 对每个改写查询执行 BM25+向量检索
  │
  ▼
RetrievalAgent（ReAct循环） → 思考→调用工具→观察→最终回答
```

### 2.3 企业级工具体系（7 个分层工具）

| 层级 | 工具 | 用途 |
|------|------|------|
| Tier1 核心检索 | `search_knowledge` | 混合检索 + 元数据过滤 |
| Tier1 核心检索 | `expand_context` | 解决 chunk 截断，获取完整上下文 |
| Tier2 证据实体 | `extract_code` | 从文档中提取代码示例 |
| Tier2 证据实体 | `compare_sources` | 多来源观点对比 |
| Tier2 证据实体 | `entity_lookup` | 术语精确匹配 |
| Tier3 知识运营 | `document_outline` | 文档目录结构导航 |
| Tier3 知识运营 | `knowledge_stats` | 知识库覆盖范围统计 |

## 3. 功能列表

- 支持 `docs/` 目录递归加载文档（`.md` / `.txt` / `.html` / `.pdf`）
- 异常文件隔离：单个文件损坏不影响整体流程
- 自研 Embedding 适配器：兼容任何 OpenAI 兼容接口（DashScope / OpenAI / Ollama）
- ChromaDB 本地持久化向量库，启动自动复用已有索引
- BM25 关键词索引：jieba 中文分词 + rank_bm25 评分
- 混合检索：BM25 + 向量双通道 + RRF 倒数排名融合
- 相似度阈值过滤 + 上下文扩展，防止弱相关结果
- 查询改写：LLM 自动扩展同义词、拆解复合问题
- Agent ReAct 循环：自主决策工具调用，max_iterations 防无限循环
- 路由决策：自动区分 RAG 检索 / 通用知识 / 直接回答
- 结构化 Prompt：防止幻觉、禁止编造、上下文不足时诚实告知
- 回答附带引用来源（文件名、路径、片段摘要、相关度分数）
- 前端展示 Agent 思考过程、路由标签、改写查询
- 一键重建索引（`force_rebuild`）
- 基于 session_state 的简单多轮对话

## 4. 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| 前端 | Streamlit 1.39 | 交互式 Chat UI |
| Agent 框架 | LlamaIndex Agent (OpenAI) 0.3 | ReAct 循环 + 工具调度 |
| LLM | OpenAI SDK 1.51（兼容 DashScope） | 问答生成 + Agent 推理 |
| Embedding | 自研 OpenAICompatibleEmbedding | 文本向量化 |
| 向量库 | ChromaDB 0.5 | 余弦相似度检索 |
| 关键词检索 | rank_bm25 0.2 | BM25 算法 |
| 中文分词 | jieba 0.42 | 中文文本分词 |
| 文档加载 | LlamaIndex SimpleDirectoryReader | 多格式解析 |
| 文本切分 | LlamaIndex SentenceSplitter | 句子级智能切分 |
| 配置管理 | python-dotenv + frozen dataclass | 环境变量管理 |

## 5. 项目结构

```text
tech-doc-rag-assistant/
├── app.py                        # Streamlit 前端入口（Chat UI + 状态面板）
├── requirements.txt              # 依赖清单（锁版本）
├── .env.example                  # 环境变量模板
├── .gitignore
├── docs/
│   └── sample/                   # 示例技术文档
│       ├── fastapi_intro.md
│       ├── python_notes.txt
│       ├── docker_basics.md
│       └── jls25.pdf
├── storage/
│   └── chroma_db/                # ChromaDB 持久化向量数据
│       └── .gitkeep
├── logs/                         # 运行日志
└── src/                          # 核心代码包
    ├── __init__.py               # 公共 API 统一导出（Facade 模式）
    ├── config.py                 # 配置管理（frozen dataclass + 校验）
    ├── logger.py                 # 统一日志（console + file）
    ├── loaders.py                # 多格式文档加载 + 元数据注入
    ├── indexer.py                # ChromaDB 索引生命周期 + BM25 语料提取
    ├── compatible_embedding.py   # 自研 Embedding 适配器
    ├── retriever.py              # 向量检索器 + 阈值过滤
    ├── bm25_retriever.py         # BM25 关键词检索器（jieba + rank_bm25）
    ├── hybrid_retriever.py       # 混合检索器（BM25 + 向量 + RRF 融合）
    ├── generator.py              # LLM 答案生成 + 防幻觉 Prompt
    ├── rag_pipeline.py           # 流程编排（RAGPipeline + AgentRAGPipeline）
    ├── agent_core.py             # 多 Agent 引擎（Router / Rewrite / Retrieval）
    ├── agent_tools.py            # 企业级工具体系（7 个分层工具）
    └── utils.py                  # 工具函数（ensure_dir / truncate_text）
```

## 6. 安装步骤

```bash
# 1) 进入项目目录
cd tech-doc-rag-assistant

# 2) 创建虚拟环境（强烈建议）
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

# 3) 升级构建工具
python -m pip install -U pip setuptools wheel

# 4) 安装依赖
pip install --prefer-binary -r requirements.txt
```

## 7. 环境变量配置

```bash
# 复制配置模板
cp .env.example .env
```

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DASHSCOPE_API_KEY` | — | 阿里云 DashScope API Key（国内推荐） |
| `OPENAI_API_KEY` | — | OpenAI API Key |
| `OPENAI_BASE_URL` | 自动识别 | 自定义 API 端点 |
| `LLM_MODEL` | `qwen-plus` / `gpt-4o-mini` | 大语言模型 |
| `EMBED_MODEL` | `text-embedding-v3` / `text-embedding-3-small` | Embedding 模型 |
| `TOP_K` | 4 | 最终返回结果数 |
| `BM25_TOP_K` | 8 | BM25 检索返回数 |
| `CHUNK_SIZE` | 512 | 文本切分块大小 |
| `CHUNK_OVERLAP` | 80 | 相邻块重叠量 |
| `SIMILARITY_THRESHOLD` | 0.45 | 向量检索相似度阈值 |
| `RRF_K` | 60 | RRF 融合平滑常数 |
| `AGENT_MAX_ITERATIONS` | 5 | Agent 最大推理循环次数 |

## 8. 准备文档 & 启动

```bash
# 把技术文档放到 docs/ 目录即可（支持任意子目录）
# 示例文档已内置在 docs/sample/

streamlit run app.py
```

浏览器打开 `http://localhost:8501`，输入技术问题开始问答。

## 9. 一次完整问答流程（以 "FastAPI 怎么配置路由？" 为例）

```
用户: "FastAPI 怎么配置路由？"
  │
  ├─ RouterAgent: {"route": "rag", "reason": "具体技术问题，需知识库检索"}
  │
  ├─ QueryRewriteAgent:
  │   改写为 ["FastAPI 路由配置", "FastAPI app.get 装饰器", "FastAPI Router 定义"]
  │
  ├─ HybridRetriever（对每个改写查询并行执行）:
  │   ├─ 向量通道: embedding → ChromaDB cosine top-4
  │   ├─ BM25通道: jieba分词 → BM25Okapi 评分 top-8
  │   └─ RRF融合: 合并去重排序 → top-6
  │
  ├─ RetrievalAgent:
  │   Thought: 已有足够上下文，可直接回答
  │   Answer: "FastAPI 使用装饰器定义路由，主要有以下几种方式..."
  │
  └─ 返回: {answer, citations, agent_steps, route, rewritten_query}
```

## 10. RAG 优化演进路线

```
v1.0 — 基础 RAG
  └── 文档加载 → 向量索引 → 语义检索 → LLM 生成

v2.0 — 混合检索（当前）
  ├── BM25 + 向量双通道
  ├── RRF 倒数排名融合
  └── 元数据过滤检索

v3.0 — Agent 协同（当前）
  ├── RouterAgent 路由决策
  ├── QueryRewriteAgent 查询改写
  ├── RetrievalAgent ReAct 循环
  └── 7 个企业级分层工具

v4.0 — 未来规划
  ├── 用户反馈闭环（thumbs up/down → 检索权重调整）
  ├── LLM 重排序（提升 top-1 准确率）
  ├── 增量索引更新（无需全量重建）
  ├── 多轮对话记忆（上下文关联）
  └── 多模态文档（图片 OCR、表格提取）
```

## 11. 常见问题

**Q: 启动时报 chromadb 连接错误？**  
A: 删除 `storage/chroma_db/` 目录后重新启动，系统会自动重建索引。

**Q: DashScope 模型名报错？**  
A: 项目使用自研的 `OpenAICompatibleEmbedding` 适配器，兼容任意 OpenAI 兼容接口，不需要模型名在官方枚举中。

**Q: 检索结果不准怎么办？**  
A: 调整 `.env` 中的 `TOP_K`、`SIMILARITY_THRESHOLD`、`CHUNK_SIZE` 参数。也可以使用 `expand_context` 工具获取更多上下文，或开启 Agent 模式让 QueryRewriteAgent 改写查询。
