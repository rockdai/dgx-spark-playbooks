---
id: txt2kg
title: txt2kg
sidebar_label: txt2kg
---

# DGX Spark 上的文本到知识图

> 通过 LLM 推理和图形可视化将非结构化文本转换为交互式知识图


## 目录

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思路

本手册演示了如何构建和部署全面的知识图谱生成和可视化解决方案，为知识图谱提取提供参考。
统一内存架构支持运行更大、更准确的模型，生成更高质量的知识图并提供卓越的下游 GraphRAG 性能。

这个 txt2kg 剧本使用以下方法将非结构化文本文档转换为结构化知识图：
- **知识三重提取**：使用 Ollama 配合 GPU 加速进行本地 LLM 推理，提取主谓宾关系
- **图数据库存储**：ArangoDB用于存储和查询具有关系遍历的知识三元组
- **GPU 加速可视化**：用于交互式 2D/3D 图形探索的 Three.js WebGPU 渲染

> **未来增强**：矢量嵌入和 GraphRAG 功能是计划中的增强功能。

## 你将完成什么

您将拥有一个功能齐全的系统，能够处理文档、生成和编辑知识图并提供查询，并可通过交互式 Web 界面进行访问。
设置包括：
- **本地 LLM 推理**：Ollama 用于 GPU 加速的 LLM 推理，无需 API 密钥
- **图数据库**：ArangoDB，用于通过关系遍历来存储和查询三元组
- **交互式可视化**：使用 Three.js WebGPU 进行 GPU 加速图形渲染
- **现代 Web 界面**：带有文档管理和查询界面的 Next.js 前端
- **完全容器化**：通过 Docker Compose 和 GPU 支持进行可重复部署

## 先决条件

-  带有最新 NVIDIA 驱动程序的 DGX Spark
-  使用 NVIDIA Container Toolkit 安装和配置 Docker
-  Docker 组合


## 时间与风险

- **期间**：
  - 初始设置和容器部署需要 2-3 分钟
  - Ollama 模型下载需要 5-10 分钟（取决于模型大小）
  - 即时文档处理和知识图生成

- **风险**：
  - GPU 内存要求取决于所选的 Ollama 模型大小
  - 文档处理时间与文档大小和复杂性相关

- **回滚**：停止并删除 Docker 容器，根据需要删除下载的模型
- **最后更新**：2025 年 1 月 8 日
  - 从 Pinecone 迁移到 Qdrant 以实现 ARM64 兼容性
  - 添加了 Neo4j 的 vLLM 支持
  - 添加了 Palette UI 组件并改进了辅助功能
  - 添加了仅 CPU 开发模式 (`./start.sh --cpu`)
  - 具有确定性键和 BM25 搜索的优化 ArangoDB
  - 添加了用于知识图训练的 GNN 预处理脚本

## 指示

## 步骤 1. 克隆存储库

在终端中，克隆 txt2kg 存储库并导航到项目目录。

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/txt2kg/assets
```

## 步骤2.启动txt2kg服务

使用提供的启动脚本启动所有必需的服务。这将设置 Ollama、ArangoDB 和 Next.js 前端：

```bash
./start.sh
```

该脚本将自动：
- 检查 GPU 可用性
- 启动 Docker Compose 服务
- 设置 ArangoDB 数据库
- 启动网络界面

## 步骤 3. 拉取 Ollama 模型（可选）

下载用于知识提取的语言模型。默认加载的模型是 Llama 3.1 8B：

```bash
docker exec ollama-compose ollama pull <model-name>
```

浏览 [https://ollama.com/search](https://ollama.com/search) 上的可用型号

> [!NOTE]
> 统一的内存架构支持运行更大的模型，例如 70B 参数，从而产生更加准确的知识三元组。

## 步骤 4. 访问 Web 界面

打开浏览器并导航至：

```
http://localhost:3001
```

您还可以访问个人服务：
- **ArangoDB Web 界面**：http://localhost:8529 
- **Ollama API**：http://localhost:11434

## 步骤5.上传文档并构建知识图谱

#### 5.1.文件上传
- 使用Web界面上传文本文档（支持markdown、text、CSV）
- 文档自动分块并处理以进行三重提取

#### 5.2.知识图谱生成
- 系统使用 Ollama 提取主谓宾三元组
- 三元组存储在ArangoDB中用于关系查询

#### 5.3.交互式可视化
- 使用 GPU 加速渲染以 2D 或 3D 形式查看知识图
- 以交互方式探索节点和关系

#### 5.4.基于图的查询
- 使用查询界面询问有关您的文档的问题
- 图遍历通过 ArangoDB 中的实体关系增强上下文
- LLM 使用丰富的图形上下文生成响应

> **未来增强**：计划使用基于向量的 KNN 搜索进行实体检索的 GraphRAG 功能。

## 步骤 6. 清理和回滚

停止所有服务并可选择删除容器：

```bash
## Stop services
docker compose down

## Remove containers and volumes (optional)
docker compose down -v

## Remove downloaded models (optional)
docker exec ollama-compose ollama rm llama3.1:8b
```

## 步骤 7. 后续步骤

- 使用不同的 Ollama 模型进行不同提取质量的实验
- 自定义特定领域知识的三重提取提示
- 探索高级图形查询和可视化功能

## 故障排除

| 症状 | 原因 | 使固定 |
|---------|--------|-----|
| 奥拉马性能问题 | DGX Spark 的次优设置 | 设置环境变量：`OLLAMA_FLASH_ATTENTION=1`、`OLLAMA_KEEP_ALIVE=30m`、`OLLAMA_MAX_LOADED_MODELS=1`、`OLLAMA_KV_CACHE_TYPE=q8_0` |
| VRAM 耗尽或内存压力（例如在 Ollama 模型之间切换时） | Linux 缓冲区高速缓存消耗 GPU 内存 | 刷新缓冲区高速缓存：`sudo sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'` |
| 慢三重萃取 | 大模型或大上下文窗口 | 减少文档块大小或使用更快的模型 |
| ArangoDB 连接被拒绝 | 服务未完全启动 | start.sh后等待30秒，用`docker ps`验证 |

> [!NOTE]
> DGX Spark 使用统一内存架构 (UMA)，可实现 GPU 和 CPU 之间的动态内存共享。 
> 由于许多应用程序仍在更新以利用 UMA，因此即使在 
> DGX Spark 的内存容量。如果发生这种情况，请使用以下命令手动刷新缓冲区缓存：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
