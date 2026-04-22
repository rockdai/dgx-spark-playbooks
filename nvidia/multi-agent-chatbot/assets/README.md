# 聊天机器人 Spark：DGX Spark 的本地多代理系统 

## 项目概况

Chatbot Spark 是一个基于 DGX Spark 构建的完全本地多代理系统。凭借 128GB 统一内存，DGX Spark 可以并行运行多个 LLM 和 VLM，从而实现跨代理的交互。 

其核心是由 GPT-OSS-120B 提供支持的主管代理，协调专门的下游代理进行编码、检索增强生成 (RAG) 和图像理解。得益于 DGX Spark 对流行 AI 框架和库的开箱即用支持，开发和原型设计变得快速且顺畅。这些组件共同展示了如何在本地高性能硬件上高效执行复杂的multimodal工作流程。

> **注意**：默认情况下，此演示使用 DGX Spark 128GB 内存中的约 120 内存，因此请确保使用 `nvidia-smi` 的 Spark 上没有运行其他工作负载，或切换到较小的管理程序模型（如 gpt-oss-20B）。

该项目是为了可定制而构建的，作为开发人员可以定制的框架。 

## 主要特点
  - **MCP 服务器集成**：Chatbot Spark 还展示了通过简单且可自定义的多服务器客户端连接到自定义 MCP 服务器的能力

  - **工具调用**：该项目使用代理即工具框架，并展示了创建作为工具连接的其他代理的能力。还可以添加通用工具。

  - **轻松交换模型**：模型使用 Llama CPP 和 Ollama 加载和提供服务，并通过 OpenAI API 提供服务。任何与 OpenAI 兼容的模型都可以集成到该项目中。

  - **矢量索引和检索**：GPU 加速的 Milvus 用于高性能文档检索。

  - **实时 LLM 流**：我们提供定制的 LLM 流基础设施，使开发人员可以轻松地从任何 OpenAI 兼容模型流式传输主管响应。 

  - **gpt-oss集成**：默认的聊天/工具调用模型是gpt-oss:120b，提供与OpenAI最新开源工具调用模型的无缝集成。


## 系统概览
<img src="assets/system-diagram.png" alt="System Diagram" style="max-width:600px;border-radius:5px;justify-content:center">

## 默认型号
| 模型                        | 量化 | 型号类型 | 显存        |
|------------------------------|--------------|------------|-------------|
| GPT-OSS：120B                 | MXFP4        | 聊天       | ~ 63.5 GB   |
| Deepseek-Coder:6.7B-指令 | Q8           | 编码     | ~ 9.5 GB   |
| Qwen2.5-VL:7B-指令       | BF16         | 图像      | ~ 35.4 GB   |
| Qwen3-Embedding-4B           | Q8           | 嵌入  | ~ 5.39 GB   |

**所需的总 VRAM：** ~114 GB

> **警告**：
> 由于默认模型使用大部分可用 VRAM，因此请确保您没有使用 `nvidia-smi` 在 DGX Spark 上运行任何内容。如果这样做，请切换到 [this guide](#using-different-models) 之后的 `gpt-oss-20b`。

---

## 快速入门
#### 1. 克隆存储库并将目录更改为多代理聊天机器人目录。

#### 2.配置docker权限
```bash
sudo usermod -aG docker $USER
newgrp docker
```

> **警告**：运行 usermod 后，您可能需要使用 `sudo reboot` 重新启动以启动新的
> 具有更新的组权限的会话。

#### 3.运行模型下载脚本
安装脚本将负责从 HuggingFace 中提取模型 GGUF 文件。正在提取的模型文件包括 gpt-oss-120B (~63GB)、Deepseek-Coder:6.7B-Instruct (~7GB) 和 Qwen3-Embedding-4B (~4GB)。这可能需要 30 分钟到 2 小时，具体取决于网络速度。
```bash
chmod +x model_download.sh
./model_download.sh
```

#### 4. 启动应用程序的docker容器
此步骤构建基础 llama cpp 服务器映像，并启动为模型、后端 API 服务器以及前端 UI 提供服务所需的所有 docker 服务。此步骤可能需要 10 到 20 分钟，具体取决于网络速度。
```bash
docker compose -f docker-compose.yml -f docker-compose-models.yml up -d --build
```
> 注意：Qwen2.5 VL模型容器启动时可能会报unhealthy，可以忽略。

等待所有容器准备就绪并正常运行。 
```bash
watch 'docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"'
```

>**注意**：如果任何模型下载失败，请将目录更改为 `models/` 目录并删除有问题的文件，然后重新从步骤 3 开始。
```bash
cd models/
rm -rf <model_file>
./model_download.sh
```

#### 5. 访问前端UI

打开浏览器并转到：[http://localhost:3000](http://localhost:3000)

> 注意：如果您通过 ssh 连接在远程 GPU 上运行此程序，则在新的终端窗口中，您需要运行才能访问 localhost:3000 处的 UI，并让 UI 能够与 localhost:8000 处的后端进行通信：
>````bash
> ssh -L 3000:localhost:3000 -L 8000:localhost:8000 用户名@IP 地址
>````

您应该在浏览器中看到以下 UI：
<img src="assets/multi-agent-chatbot.png" alt="Frontend UI" style="max-width:600px;border-radius:5px;justify-content:center">

### 6. 尝试示例提示
单击前端的任何图块即可试用主管和其他代理。

#### 拉格代理：
在尝试 RAG 代理的示例提示之前，请通过以下方式上传示例 PDF 文档 [NVIDIA Blackwell Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf) 作为上下文：转至链接，将 PDF 下载到本地文件系统，单击左侧边栏中“上下文”下的绿色“上传文档”按钮，然后确保选中“选择源”部分中的框。

<img src="assets/document-ingestion.png" alt="Ingest Documents" style="max-width:300px;border-radius:5px;justify-content:center">

> **注意**：您可以上传您选择的任何 PDF，并提出相应的疑问。默认提示需要 NVIDIA Blackwell 白皮书。

#### 图像理解代理：

**提示示例：**

描述此图像：https://en.wikipedia.org/wiki/London_Bridge#/media/File:London_Bridge_from_St_Olaf_Stairs.jpg


## 清理

请按照以下步骤完全删除容器并释放资源。

从 multi-agent-chatbot 项目的根目录中，运行以下命令：

```bash
docker compose -f docker-compose.yml -f docker-compose-models.yml down

docker volume rm "$(basename "$PWD")_postgres_data"
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
您可以选择运行 `docker volume prune` 以在演示结束时删除所有未使用的卷。
> **注意**：如果容器不执行这些命令，将会继续运行并占用内存。

## 定制

### 使用不同的模型

您可以使用交换主管代理正在使用的模型，例如 gpt-oss-20b。

1. 在 `setup.sh` 中，取消注释行以下载 gpt-oss-20b。
> 注意：如果您已经下载了模型文件，则可以跳至步骤 2。
2. 在 `docker-compose-models.yml` 中，取消注释 gpt-oss-20b 的块。 
> 注意：由于默认模型使用所有现有 VRAM，因此您需要注释掉 `docker-compose-models.yml` 中的 gpt-oss-120b 块。
3. 在 `docker-compose.yml` 中，将 `gpt-oss-20b` 添加到 `MODELS` 环境变量（第 40 行）。
> 注意：此名称应与您在 `docker-compose-models.yml` 中为此模型设置的容器名称匹配。

### 添加 MCP 服务器和工具

1. 您可以按照现有示例在 [backend/tools/mcp_servers](backend/tools/mcp_servers/) 下添加更多 MCP 服务器和工具。

2. 如果您添加了 MCP 服务器，请记住将其添加到 [backend/client.py](backend/client.py) 中的服务器配置中
