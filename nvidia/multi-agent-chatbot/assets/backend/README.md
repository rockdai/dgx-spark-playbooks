# 后端

FastAPI Python 应用程序充当聊天机器人演示的 API 后端。

## 概述

后端处理：
- multimodalLLM整合（本地模式）
- RAG 的文档摄取和向量存储
- 用于实时聊天流的 WebSocket 连接
- 图像处理与分析
- 聊天记录管理
- 模型控制协议 (MCP) 集成

## 主要特点

- **多模型支持**：集成各种LLM提供商和本地模型
- **RAG 管道**：文档处理、嵌入生成和检索
- **流式响应**：通过 WebSocket 进行实时令牌流式传输
- **图像分析**：用于图像理解的多模态功能
- **矢量数据库**：文档检索的高效相似性搜索
- **会话管理**：聊天历史记录和上下文持久化

## 建筑学

具有异步支持的 FastAPI 应用程序，与用于 RAG 功能的矢量数据库和用于实时通信的 WebSocket 端点集成。

## Docker 故障排除

### 容器问题
- **端口冲突**：确保端口 8000 未被使用
- **内存问题**：后端需要大量 RAM 来加载模型
- **启动失败**：检查是否设置了所需的环境变量

### 模型加载问题
```bash
# Check model download status
docker logs backend | grep -i "model"

# Verify model files exist
docker exec -it cbackend ls -la /app/models/

# Check available disk space
docker exec -it backend df -h
```

### 常用命令
```bash
# View backend logs
docker logs -f backend

# Restart backend container
docker restart backend

# Rebuild backend
docker-compose up --build -d backend

# Access container shell
docker exec -it backend /bin/bash

# Check API health
curl http://localhost:8000/health
```

### 性能问题
- **响应缓慢**：检查 GPU 可用性和模型大小
- **内存错误**：增加 Docker 内存限制或使用较小的模型
- **连接超时**：验证 WebSocket 连接和防火墙设置
