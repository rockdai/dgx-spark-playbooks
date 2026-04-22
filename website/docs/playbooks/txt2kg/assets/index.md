---
id: txt2kg-assets
title: assets
sidebar_label: assets
---

# 英伟达txt2kg

## 概述

本手册可作为知识图谱提取和检索增强生成 (RAG) 查询的参考解决方案。这个 txt2kg 剧本从文本中提取知识三元组，并构建用于可视化和查询的知识图，与传统的 RAG 方法相比，创建了更加结构化的信息检索形式。通过利用图形数据库和实体关系，本手册提供了上下文更丰富的答案，可以更好地表示数据中的复杂关系。

<details>
<summary>📋 目录</summary>


- [Overview](#overview)
- [Key Features](#key-features)
- [Software Components](#software-components)
- [Technical Diagram](#technical-diagram)
- [Minimum System Requirements](#minimum-system-requirements)
- [Deployment Guide](#deployment-guide)
- [Available Customizations](#available-customizations)
- [License](#license)

</details>

默认情况下，本剧本利用 **Ollama** 进行本地 LLM 推理，提供完全独立的解决方案，完全在您自己的硬件上运行。您可以选择使用 **vLLM** 在 DGX Spark/GB300 上进行 GPU 加速推理，或者使用 [NVIDIA API Catalog](https://build.nvidia.com) 中提供的 NVIDIA 托管模型来实现高级功能。

## 主要特点

![Screenshot](./frontend/public/txt2kg.png)

- 从文本文档中提取知识三元组
- 知识图谱构建与可视化
- **本地优先架构**与 Ollama 进行 LLM 推理
- 图数据库与 ArangoDB 集成
- 使用 Three.js WebGPU 进行交互式知识图可视化
- 使用 Ollama 进行 GPU 加速的 LLM 推理
- 使用 Docker Compose 进行完全容器化部署
- 适用于基于云的模型的可选 NVIDIA API 集成
- 使用 Qdrant 进行可选向量搜索以获得语义相似性
- 用于上下文答案的可选基于图形的 RAG

## 软件组件

### 核心组件（默认）

* **法学硕士推理**
  * **Ollama**：具有 GPU 加速的本地 LLM 推理
    * 默认型号：`llama3.1:8b`
    * 支持任何 Ollama 兼容型号
* **知识图数据库**
  * **ArangoDB**：用于存储知识三元组（实体和关系）的图形数据库
    * 端口 8529 上的 Web 界面
    * 无需身份验证（可配置）
* **图形可视化**
  * **Three.js WebGPU**：客户端 GPU 加速图形渲染
* **前端和API**
  * **Next.js**：具有 API 路由的现代 React 框架

### 可选组件

* **vLLM 堆栈**（带有 `--vllm` 标志）
  * **vLLM**：针对 DGX Spark/GB300 优化的 GPU 加速 LLM 推理
    * 默认型号：`nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8`
  * **Neo4j**：替代图数据库
* **矢量数据库和嵌入**（带有 `--vector-search` 标志）
  * **SentenceTransformer**：本地嵌入生成（模型：`all-MiniLM-L6-v2`）
  * **Qdrant**：自托管向量存储和相似性搜索
* **云模型**（单独配置）
  * **NVIDIA API**：通过 NVIDIA API Catalog 的基于云的模型

## 技术图解

### 默认架构（最小设置）

知识图谱构建和可视化的核心工作流程：
1. 用户通过 txt2kg Web UI 上传文档
2. 对文档进行处理和分块以供分析
3. **Ollama** 使用本地 LLM 推理从文本中提取知识三元组（主语-谓语-宾语）
4. 三元组存储在 **ArangoDB** 图形数据库中
5. 知识图通过浏览器中的 **Three.js WebGPU** 渲染进行可视化
6. 用户可以使用 Ollama 查询图表并生成见解

### 未来的增强功能

可以添加额外的功能：
- **矢量搜索**：使用 Qdrant 和 SentenceTransformer 嵌入添加语义相似性搜索
- **S3 存储**：用于可扩展文档存储的 MinIO
- **基于 GNN 的 GraphRAG**：用于增强检索的图神经网络

## GPU 加速的 LLM 推理

本手册包括使用 Ollama 进行 **GPU 加速的 LLM 推理**：

### Ollama 功能（默认）
- **完全本地推理**：无需云依赖项或 API 密钥
- **GPU 加速**：NVIDIA GPU 的自动 CUDA 支持
- **多型号支持**：使用任何 Ollama 兼容型号
- **优化推理**：Flash 注意力、KV 缓存优化和量化
- **轻松的模型管理**：使用简单的命令拉动和切换模型
- **隐私第一**：所有数据处理都发生在您的硬件上

### vLLM 替代方案（通过 `--vllm` 标志）
- **高性能推理**：针对 DGX Spark/GB300 统一内存进行优化
- **FP8 量化**：高效内存使用，质量损失最小
- **大上下文支持**：高达 32K 令牌上下文长度
- **连续批处理**：多个请求的高吞吐量

### 默认 Olama 配置
- 型号：`llama3.1:8b`
- GPU 内存比例：0.9（可用 VRAM 的 90%）
- 启用闪光注意
- Q8_0 KV 缓存可提高内存效率

## 软件要求

- CUDA 12.0+
- Docker 与 NVIDIA 容器工具包

## 部署指南

### 环境变量

**默认部署不需要 API 密钥！** 所有服务都在本地运行。

默认配置使用：
- 本地 Ollama（无需 API 密钥）
- 本地ArangoDB（默认情况下不进行身份验证）

用于自定义的可选环境变量：
```bash
# Ollama configuration (optional - defaults are set)
OLLAMA_BASE_URL=http://ollama:11434/v1
OLLAMA_MODEL=llama3.1:8b

# NVIDIA API (optional - for cloud models)
NVIDIA_API_KEY=your-nvidia-api-key
```

### 快速入门

1. **克隆存储库：**
```bash
git clone <repository-url>
cd txt2kg
```

2. **启动应用程序：**
```bash
./start.sh
```

就是这样！无需配置。该脚本将：
- 使用 Docker Compose 启动所有必需的服务
- 设置 ArangoDB 数据库
- 启动具有 GPU 加速功能的 Ollama
- 启动 Next.js 前端

3. **拉一个 Ollama 模型（仅限第一次）：**
```bash
docker exec ollama-compose ollama pull llama3.1:8b
```

4. **访问应用程序：**
- **网络用户界面**：http://localhost:3001
- **ArangoDB**：`http://localhost:8529`（无需身份验证）
- **Ollama API**：http://localhost:11434

### 替代方案：使用 vLLM（适用于 DGX Spark/GB300）

对于使用 vLLM 的 GPU 加速推理：

```bash
./start.sh --vllm
```

然后等待 vLLM 加载模型：
```bash
docker logs vllm-service -f
```

服务：
- **网络用户界面**：http://localhost:3001
- **Neo4j 浏览器**：`http://localhost:7474`（用户：`neo4j`，密码：`password123`）
- **vLLM API**：http://localhost:8001

### 添加矢量搜索

启用语义相似性搜索：
```bash
./start.sh --vector-search
```

这增加了：
- **Qdrant**：http://localhost:6333
- **句子转换器**：http://localhost:8000

## 可用的定制

- **切换 LLM 后端**：对 vLLM 使用 `--vllm` 标志，对 Ollama 使用默认标志
- **添加矢量搜索**：使用 `--vector-search` 标志进行 Qdrant + 嵌入
- **切换 Ollama 模型**：使用 Ollama 库中的任何模型（Llama、Mistral、Qwen 等）
- **修改提取提示**：自定义从文本中提取三元组的方式
- **添加特定领域的知识源**：集成外部本体或分类法
- **使用 NVIDIA API**：连接到特定用例的云模型

## 执照

[MIT](LICENSE)

该项目将下载并安装其他第三方开源软件项目和容器。
