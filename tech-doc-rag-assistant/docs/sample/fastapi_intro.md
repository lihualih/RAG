# FastAPI 简介

FastAPI 是一个基于 Python 类型提示构建的高性能 Web 框架，适合快速开发 API 服务。

## 核心特点
1. 自动生成 OpenAPI 文档（Swagger UI）
2. 基于 Pydantic 做请求参数校验
3. 支持异步编程（`async def`）
4. 开发效率高，适合微服务和 AI 应用后端

## 简单示例
```python
from fastapi import FastAPI

app = FastAPI()

@app.get('/ping')
def ping():
    return {"message": "pong"}
```

## 启动命令
通常使用 Uvicorn 启动：
`uvicorn main:app --reload --port 8000`

启动后访问 `/docs` 即可查看交互式接口文档。
