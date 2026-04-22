---
id: txt2kg-assets-frontend
title: frontend
sidebar_label: frontend
---

# 前端应用程序

此目录包含 txt2kg 项目的 Next.js 前端应用程序。

## 结构

- **app/**：Next.js 15 应用程序目录，包含页面和 API 路由
  - LLM 提供商的 API 路线（Ollama、vLLM、NVIDIA API）
  - 三重提取和图形查询端点
  - 设置和健康检查端点
- **组件/**：React 19 个组件
  - 图形可视化（Three.js WebGPU）
  - PyGraphistry 集成用于 GPU 加速渲染
  - RAG查询接口
  - 文件上传及处理
- **contexts/**：用于状态管理的 React 上下文提供程序
- **hooks/**：自定义 React 钩子
- **lib/**：实用函数和共享逻辑
  - LLM 服务（Ollama、vLLM、NVIDIA API 集成）
  - 图数据库服务（ArangoDB、Neo4j）
  - Qdrant载体数据库集成
  - 用于知识图谱查询的RAG服务
- **public/**：静态资产
- **types/**：图形数据结构的 TypeScript 类型定义

## 技术栈

- **Next.js 15**：带有 App Router 的 React 框架
- **React 19**：具有改进的并发功能的最新 React
- **TypeScript**：类型安全开发
- **Tailwind CSS**：实用优先的样式
- **Three.js**：WebGL/WebGPU 3D 图形可视化
- **D3.js**：数据驱动的可视化
- **LangChain**：法学硕士编排和链接

## 发展

启动开发服务器：

```bash
cd frontend
npm install
npm run dev
```

或者使用项目根目录的启动脚本：

```bash
./start.sh --dev-frontend
```

开发服务器将在 http://localhost:3000 上运行

## 生产建筑

```bash
cd frontend
npm run build
npm start
```

或者使用 Docker（推荐）：

```bash
# From project root
./start.sh
```

生产应用程序将在 http://localhost:3001 上运行

## 环境变量

所需的环境变量在 docker-compose 文件中配置：

- `ARANGODB_URL`：ArangoDB 连接 URL
- `OLLAMA_BASE_URL`：Ollama API 端点
- `VLLM_BASE_URL`：vLLM API 端点（可选）
- `NVIDIA_API_KEY`：NVIDIA API 密钥（可选）
- `QDRANT_URL`：Qdrant 向量数据库 URL（可选）
- `SENTENCE_TRANSFORMER_URL`：嵌入服务 URL（可选）

## 特征

- **知识图谱提取**：使用法学硕士从文本中提取三元组
- **图形可视化**：使用 Three.js WebGPU 进行交互式 3D 可视化
- **RAG 查询**：通过检索增强生成来查询知识图
- **多个 LLM 提供商**：支持 Ollama、vLLM 和 NVIDIA API
- **GPU 加速渲染**：针对大型图形的可选 PyGraphistry 集成
- **矢量搜索**：用于语义搜索的 Qdrant 集成 
