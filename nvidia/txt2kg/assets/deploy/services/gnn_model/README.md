# GNN 模型服务（实验性）

**状态**：这是一项实验性服务，用于为经过增强 RAG 检索训练的图神经网络模型提供服务。

**注意**：此服务**不包含**在默认的 docker-compose 配置中，必须单独部署。

## 概述

该服务提供了一个 REST API，用于提供来自图神经网络 (GNN) 模型的预测，从而增强知识图检索：

- 加载预训练的GNN模型（GAT架构）
- 使用图结构知识处理查询
- 将 GNN 嵌入与 LLM 生成相结合
- 比较基于 GNN 的检索与传统 RAG

## 入门

### 先决条件

- Python 3.8+
- PyTorch 和 PyTorch 几何
- 经过训练的模型文件（使用 `scripts/gnn/` 中的 `train_export.py` 创建）
- 码头工人（可选）

### 训练模型

在使用该服务之前，您必须使用训练管道训练 GNN 模型：

```bash
# See scripts/gnn/README.md for full instructions

# 1. Preprocess data from ArangoDB
python scripts/gnn/preprocess_data.py --use_arango --output_dir ./output

# 2. Train the model
python scripts/gnn/train_test_gnn.py --output_dir ./output

# 3. Export model for serving
python deploy/services/gnn_model/train_export.py --output_dir models
```

这将创建服务所需的 `tech-qa-model.pt` 文件。

### 运行服务

#### 选项A：直接Python

```bash
cd deploy/services/gnn_model
pip install -r requirements.txt
python app.py
```

服务运行于：http://localhost:5000

#### 选项B：Docker

```bash
cd deploy/services/gnn_model
docker build -t gnn-model-service .
docker run -p 5000:5000 -v $(pwd)/models:/app/models gnn-model-service
```


## API端点

### 健康检查

```
GET /health
```

返回服务的健康状态。

### 预言

```
POST /predict
```

请求正文：
```json
{
  "question": "Your question here",
  "context": "Retrieved context information"
}
```

回复：
```json
{
  "question": "Your question here",
  "answer": "The generated answer"
}
```

## 使用客户端示例

提供了一个简单的客户端脚本来测试服务：

```bash
python deploy/services/gnn_model/client_example.py --question "What is the capital of France?" --context "France is a country in Western Europe. Its capital is Paris, which is known for the Eiffel Tower."
```

该脚本还包含一个占位符，用于将基于 GNN 的方法与传统的 RAG 方法进行比较。

## 建筑学

GNN 模型服务使用：
- 用于处理图结构化数据的图注意力网络（GAT）
- 用于生成答案的语言模型 (LLM)
- 利用两个组件的组合架构 (GRetriever)

## 与 txt2kg 集成

要将此服务与主 txt2kg 应用程序集成：

1. 使用 GNN 训练管道训练模型
2. 在单独的端口上部署 GNN 服务
3. 更新前端以调用 GNN 服务端点
4. 比较 GNN 增强检索与标准 RAG

## 目前状态

这是一个实验性功能。服务代码存在，但需要：
- 经过训练的 GNN 模型
- 与前端查询管道集成
- 从txt2kg知识图构建图
- 性能基准测试与传统 RAG

## 未来的增强功能

- Docker Compose 集成，更轻松部署
- 从 txt2kg 图自动训练模型
- 随着图表的增长实时模型更新
- 前端UI对比 
