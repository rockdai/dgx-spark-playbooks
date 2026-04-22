# GNN 训练管道（实验）

**状态**：这是一项实验性功能，用于训练图神经网络模型以增强 RAG 检索。

该管道提供了一个用于训练基于 GNN 的知识图检索模型的两阶段过程：

1. **数据预处理** (`preprocess_data.py`)：从 ArangoDB 中提取知识图三元组并准备训练数据集。
2. **模型训练和测试** (`train_test_gnn.py`)：使用 PyTorch Geometric 训练和评估基于 GNN 的检索器模型。

## 先决条件

- Python 3.8+
- ArangoDB 正在运行（可以使用提供的 docker-compose.yml 进行设置）
- 安装了 PyTorch 和 PyTorch Geometric
- requirements.txt 中列出的所有依赖项

## 安装

1. 安装所需的依赖项：

```bash
pip install -r scripts/requirements.txt
```

2. 确保 ArangoDB 正在运行。您可以使用主启动脚本：

```bash
# From project root
./start.sh
```

## 用法

### 第一阶段：数据预处理

运行预处理脚本来准备数据集：

```bash
python scripts/preprocess_data.py --use_arango --output_dir ./output
```

#### 从 ArangoDB 加载数据

使用 `--use_arango` 标志从 ArangoDB 加载三元组，而不是使用 TXT2KG 生成它们：

```bash
python scripts/preprocess_data.py --use_arango
```

该脚本将使用 docker-compose.yml 中的默认设置连接到 ArangoDB：
- 网址：http://localhost:8529
- 数据库：txt2kg
- 无身份验证（用户名和密码为空）

#### 自定义 ArangoDB 连接

您可以指定自定义 ArangoDB 连接参数：

```bash
python scripts/preprocess_data.py --use_arango --arango_url "http://localhost:8529" --arango_db "your_db" --arango_user "username" --arango_password "password"
```

#### 使用直接三重提取

如果您不传递 `--use_arango` 标志，脚本将使用配置的 LLM 提供程序直接提取三元组。

### 第二阶段：模型训练和测试

数据预处理后，训练和测试模型：

```bash
python scripts/train_test_gnn.py --output_dir ./output
```

#### 培训选项

您可以使用选项自定义培训：

```bash
python scripts/train_test_gnn.py --output_dir ./output --gnn_hidden_channels 2048 --num_gnn_layers 6 --epochs 5 --batch_size 2
```

#### 仅评估

要评估先前训练的模型而不需要重新训练：

```bash
python scripts/train_test_gnn.py --output_dir ./output --eval_only
```

## ArangoDB 中的预期数据格式

该脚本期望 ArangoDB 具有：

1. 名为 `entities` 的文档集合，其中包含具有 `name` 属性的节点
2. 名为 `relationships` 的边集合，其中：
   - 边具有 `type` 属性（谓词/关系类型）
   - 边连接 `entities` 集合中的实体和连接到 `entities` 集合中的实体

## 它是如何运作的

### 数据预处理 (`preprocess_data.py`)
1. 连接到 ArangoDB 并以“主谓谓宾”格式查询所有三元组（或使用 TXT2KG 生成它们）
2. 从这些三元组创建知识图
3. 准备包含训练、验证和测试拆分的数据集

### 模型训练和测试 (`train_test_gnn.py`)
1. 加载预处理的数据集
2. 初始化一个 GNN 模型（GAT 架构）和一个用于生成的 LLM
3. 在训练集上训练模型，在验证集上进行验证
4. 使用 LLMJudge 评分来评估测试集上的训练模型

## 局限性

- 该脚本假设您的 ArangoDB 实例包含上述格式的数据
- 您需要提供问答对和语料库文档
- 确保您的 ArangoDB 包含与您的语料库相关的知识图三元组
- 大型 LLM 模型需要大量 GPU 内存进行训练 
