# Docker 基础概念

Docker 是一种容器化技术，可以把应用及其依赖打包成镜像，在不同环境中一致运行。

## 常见概念
- Image（镜像）：运行环境模板
- Container（容器）：镜像启动后的实例
- Dockerfile：构建镜像的脚本
- Registry：镜像仓库，例如 Docker Hub

## 常见命令
1. 构建镜像：`docker build -t my-app:latest .`
2. 运行容器：`docker run -p 8000:8000 my-app:latest`
3. 查看容器：`docker ps`
4. 停止容器：`docker stop <container_id>`

## 使用建议
开发阶段可以通过容器统一依赖，减少“在我电脑上能跑”的环境问题。
