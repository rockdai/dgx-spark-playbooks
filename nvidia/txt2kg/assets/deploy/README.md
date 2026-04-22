# 部署配置

该目录包含 txt2kg 项目的所有与部署相关的配置。

## 结构

- **compose/**：Docker Compose 配置
  - `docker-compose.yml`：ArangoDB + Ollama（默认）
  - `docker-compose.vllm.yml`：Neo4j + vLLM（GPU 加速）

- **app/**：前端应用程序Docker配置
  - Next.js 应用程序的 Dockerfile

- **服务/**：容器化服务
  - **ollama/**：Ollama LLM 推理服务（默认）
  - **vllm/**：具有 GPU 支持的 vLLM 推理服务（通过 `--vllm` 标志）
  - **sentence-transformers/**：用于嵌入的句子转换器服务（通过 `--vector-search` 标志）
  - **gpu-viz/**：GPU加速的图形可视化服务（单独运行）
  - **gnn_model/**：图神经网络模型服务（实验）

## 用法

**推荐：使用启动脚本**

```bash
# Default: ArangoDB + Ollama
./start.sh

# Use Neo4j + vLLM (GPU-accelerated, for DGX Spark/GB300)
./start.sh --vllm

# Enable vector search (Qdrant + Sentence Transformers)
./start.sh --vector-search

# Combine options
./start.sh --vllm --vector-search

# Development mode (run frontend without Docker)
./start.sh --dev-frontend
```

**手动 Docker Compose 命令：**

```bash
# Default: ArangoDB + Ollama
docker compose -f deploy/compose/docker-compose.yml up -d

# Neo4j + vLLM
docker compose -f deploy/compose/docker-compose.vllm.yml up -d

# With vector search services (add --profile vector-search)
docker compose -f deploy/compose/docker-compose.yml --profile vector-search up -d
docker compose -f deploy/compose/docker-compose.vllm.yml --profile vector-search up -d
```

## 包含的服务

### 默认堆栈（ArangoDB + Ollama）
- **Next.js 应用**：端口 3001 上的 Web UI
- **ArangoDB**：端口 8529 上的图形数据库
- **Ollama**：端口 11434 上的本地 LLM 推理

### vLLM 堆栈（`--vllm` 标志）- Neo4j + vLLM
- **Next.js 应用**：端口 3001 上的 Web UI
- **Neo4j**：端口 7474 (HTTP) 和 7687 (Bolt) 上的图形数据库
- **vLLM**：端口 8001 上的 GPU 加速 LLM 推理

### 矢量搜索（`--vector-search` 配置文件）
- **Qdrant**：端口 6333 上的矢量数据库
- **句子转换器**：在端口 8000 上嵌入生成

### 可选服务（单独运行）
- **GPU-Viz 服务**：有关 GPU 加速可视化，请参阅 `services/gpu-viz/README.md`
- **GNN 模型服务**：有关基于 GNN 的实验性 RAG，请参阅 `services/gnn_model/README.md`

## 建筑学

```
┌─────────────────────────────────────────────────────────────────┐
│  Default Stack (./start.sh)          │  vLLM Stack (--vllm)     │
├──────────────────────────────────────┼──────────────────────────┤
│                                      │                          │
│  ┌─────────────┐                     │  ┌─────────────┐         │
│  │   Next.js   │ port 3001           │  │   Next.js   │ 3001    │
│  └──────┬──────┘                     │  └──────┬──────┘         │
│         │                            │         │                │
│  ┌──────┴──────┐  ┌─────────────┐    │  ┌──────┴──────┐  ┌─────┐│
│  │  ArangoDB   │  │   Ollama    │    │  │   Neo4j     │  │vLLM ││
│  │  port 8529  │  │ port 11434  │    │  │  port 7474  │  │8001 ││
│  └─────────────┘  └─────────────┘    │  └─────────────┘  └─────┘│
│                                      │                          │
└──────────────────────────────────────┴──────────────────────────┘

Optional (--vector-search): Qdrant (6333) + Sentence Transformers (8000)
```
