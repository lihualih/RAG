# tech-doc-rag-assistant

## 1. 项目简介
`tech-doc-rag-assistant` 是一个基于本地公开技术文档的 RAG（检索增强生成）问答系统。项目使用 Python + LlamaIndex + Chroma + Streamlit，实现“文档导入 -> 向量检索 -> 生成回答 -> 引用展示”的完整闭环，适合应届生用于简历项目展示与面试演示。

## 2. 功能列表
- 支持从 `docs/` 目录递归加载文档（`md/txt/html/pdf`）
- 支持异常文件跳过与日志记录，不因单个文件损坏而崩溃
- 使用 LlamaIndex 进行切分与索引构建
- 使用 Chroma 本地持久化向量库（`storage/chroma_db`）
- 支持一键重建索引
- 提供检索阈值过滤，避免弱相关内容误答
- 基于 OpenAI 兼容接口生成答案（兼容 OpenAI / DashScope）
- 回答附带引用来源（文件名、路径、片段摘要）
- 当上下文不足时返回安全提示，不胡乱生成
- 支持 Streamlit 简单多轮对话（基于 session state）

## 3. 技术栈
- Python 3.10+
- Streamlit
- LlamaIndex
- Chroma (PersistentClient)
- OpenAI Compatible API（OpenAI / DashScope）
- python-dotenv

## 4. 项目结构
```text
tech-doc-rag-assistant/
├─ app.py
├─ requirements.txt
├─ README.md
├─ .env.example
├─ .gitignore
├─ docs/
│  └─ sample/
│     ├─ fastapi_intro.md
│     ├─ python_notes.txt
│     └─ docker_basics.md
├─ storage/
│  └─ chroma_db/
│     └─ .gitkeep
└─ src/
   ├─ __init__.py
   ├─ config.py
   ├─ logger.py
   ├─ loaders.py
   ├─ indexer.py
   ├─ retriever.py
   ├─ generator.py
   ├─ rag_pipeline.py
   └─ utils.py
```

## 5. 安装步骤（重点：避免 resolution-too-deep）
```bash
# 1) 进入项目目录
cd tech-doc-rag-assistant

# 2) 创建全新虚拟环境（强烈建议）
python -m venv .venv
.\.venv\Scripts\activate

# 3) 先升级基础构建工具
python -m pip install -U pip setuptools wheel

# 4) 再安装锁版本依赖
pip install --prefer-binary -r requirements.txt
```

如果你之前失败过，建议先清理旧环境再装，不要在同一个脏环境里反复重试。

## 6. 环境变量配置
1. 复制配置模板：
```bash
# Windows PowerShell
Copy-Item .env.example .env
```

2. 二选一配置 API Key：
- 方案 A（OpenAI）：填写 `OPENAI_API_KEY`
- 方案 B（DashScope）：填写 `DASHSCOPE_API_KEY`

3. 可选配置：
- `OPENAI_BASE_URL`：不填时，DashScope 会自动使用 `https://dashscope.aliyuncs.com/compatible-mode/v1`
- `LLM_MODEL`：如 OpenAI `gpt-4o-mini`，DashScope `qwen-plus`
- `EMBED_MODEL`：如 OpenAI `text-embedding-3-small`，DashScope `text-embedding-v3`
- `TOP_K`：检索条数
- `SIMILARITY_THRESHOLD`：相似度阈值

## 7. 如何准备 docs 文档
- 把公开技术文档放到 `docs/` 下任意子目录即可
- 支持文件类型：`.md`, `.txt`, `.html`, `.pdf`
- 示例文档已内置在 `docs/sample/`，可直接测试
- 若有文件损坏或格式异常，系统会记录日志并跳过

## 8. 如何启动项目
```bash
streamlit run app.py
```
启动后在浏览器打开终端提示的本地地址（默认 `http://localhost:8501`）。

使用流程：
1. 首次启动自动扫描 `docs/` 并构建/加载索引
2. 在页面输入技术问题进行问答
3. 查看回答下方引用来源
4. 若文档更新，可点击侧边栏“重建索引”

## 9. 常见问题
### Q1：启动时报错“找不到 chromadb / llama_index 模块”
通常是依赖没安装成功或 Python 解释器不一致。

请确认：
1. 你已激活 `.venv`
2. `python -m pip install --prefer-binary -r requirements.txt` 成功执行
3. IDE 选择的是 `.venv` 对应解释器

### Q2：我只用 DashScope，可以吗？
可以。只需要在 `.env` 填 `DASHSCOPE_API_KEY`，`OPENAI_API_KEY` 留空即可。

### Q3：为什么回答提示“未在知识库中检索到足够相关内容”？
常见原因：
- 问题和文档主题不匹配
- `SIMILARITY_THRESHOLD` 设得过高
- `docs/` 文档数量不足

可尝试降低阈值、补充文档或换一种提问方式。

### Q4：修改了 docs 文档但结果没更新
点击侧边栏“重建索引”即可全量更新。当前版本保留了增量导入扩展位，默认走全量重建以保证稳定。

## 10. 简历可写亮点
- 独立完成从 0 到 1 的 RAG 问答系统落地，支持本地知识库导入、向量检索与大模型回答
- 使用 LlamaIndex + Chroma 构建可持久化索引，支持一键重建并预留增量导入接口
- 设计检索阈值与无结果保护策略，显著降低大模型“幻觉回答”风险
- 在前端实现引用来源可视化（文件名/路径/摘要），增强答案可追溯性
- 兼容 OpenAI 与 DashScope 两种 API 供应商，具备良好工程可迁移性


### Q5：报错 `TypeError: __init__() got an unexpected keyword argument 'proxies'`
项目代码已通过显式 `http_client` 规避该兼容问题，先确认已拉取最新代码并重启进程。

若你需要重装依赖，建议执行：

```bash
pip install --upgrade openai==1.51.2 "httpx>=0.27,<0.29"
```
