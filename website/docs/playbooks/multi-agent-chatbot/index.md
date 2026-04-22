---
id: multi-agent-chatbot
title: multi-agent-chatbot
sidebar_label: multi-agent-chatbot
---

# 构建和部署多代理聊天机器人

> 部署多代理聊天机器人系统并与 Spark 上的代理聊天

## 目录

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思路

本手册向您展示了如何使用 DGX Spark 进行原型设计、构建和部署完全本地的多代理系统。 
凭借 128GB 统一内存，DGX Spark 可以并行运行多个 LLM 和 VLM，从而实现跨代理的交互。

其核心是由 gpt-oss-120B 提供支持的主管代理，协调专门的下游代理以进行编码、检索增强生成 (RAG) 和图像理解。 
得益于 DGX Spark 对流行 AI 框架和库的开箱即用支持，开发和原型设计变得快速且顺畅。 
这些组件共同展示了如何在本地高性能硬件上高效执行复杂的multimodal工作流程。

## 你将完成什么

您将在 DGX Spark 上运行一个全栈多代理聊天机器人系统，可通过
您本地的网络浏览器。 
设置包括：
- 使用 llama.cpp 服务器和 TensorRT LLM 服务器提供 LLM 和 VLM 模型服务
- 用于模型推理和文档检索的 GPU 加速
- 使用由 gpt-oss-120B 提供支持的主管代理进行多代理系统编排
- MCP（模型上下文协议）服务器作为主管代理的工具

## 先决条件

-  DGX Spark 设备已设置并可访问
-  DGX Spark GPU 上没有运行其他进程
-  有足够的磁盘空间用于模型下载

> [！笔记]
> 默认情况下，此演示使用 DGX Spark 128GB 内存中的约 120 内存。 
> 请确保使用 `nvidia-smi` 的 Spark 上没有运行其他工作负载，或者切换到较小的管理程序模型，例如 gpt-oss-20B。


## 时间与风险

* **预计时间**：30 分钟到一小时
* **风险**：
  * Docker 权限问题可能需要更改用户组并重新启动会话
  * 安装包括下载 gpt-oss-120B (~63GB)、Deepseek-Coder:6.7B-Instruct (~7GB) 和 Qwen3-Embedding-4B (~4GB) 的模型文件，这可能需要 30 分钟到 2 小时，具体取决于网络速度
* **回滚**：使用提供的清理命令停止并删除 Docker 容器。
* **最后更新**：2025 年 11 月 20 日
  * 修复了在 DGX Spark 上运行 llama.cpp 的中断命令

## 指示

## 步骤1.配置Docker权限

要在不使用 sudo 的情况下轻松管理容器，您必须位于 `docker` 组中。如果您选择跳过此步骤，则需要使用 sudo 运行 Docker 命令。

打开新终端并测试 Docker 访问。在终端中，运行：

```bash
docker ps
```

如果您看到权限被拒绝错误（例如尝试连接到 Docker 守护进程套接字时权限被拒绝），请将您的用户添加到 docker 组，这样您就不需要使用 sudo 运行命令。

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## 步骤 2. 克隆存储库

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/multi-agent-chatbot/assets
```

## 步骤 3. 运行模型下载脚本

```bash
chmod +x model_download.sh
./model_download.sh
```

安装脚本将负责从 HuggingFace 中提取模型 GGUF 文件。 
正在提取的模型文件包括 gpt-oss-120B (~63GB)、Deepseek-Coder:6.7B-Instruct (~7GB) 和 Qwen3-Embedding-4B (~4GB)。 
这可能需要 30 分钟到 2 小时，具体取决于网络速度。


## 步骤 4. 启动应用程序的 docker 容器

```bash
  docker compose -f docker-compose.yml -f docker-compose-models.yml up -d --build
```
此步骤构建基础 llama.cpp 服务器映像并启动为模型、后端 API 服务器以及前端 UI 提供服务所需的所有 docker 服务。 
此步骤可能需要 10 到 20 分钟，具体取决于网络速度。
等待所有容器准备就绪并正常运行。

```bash
watch 'docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"'
```

## 步骤 5. 访问前端 UI

打开浏览器并转到：http://localhost:3000

> [！笔记]
> 如果您通过 SSH 连接在远程 GPU 上运行此程序，则在新的终端窗口中，您需要运行以下命令才能访问 localhost:3000 处的 UI，并让 UI 能够与 localhost:8000 处的后端进行通信。

>``ssh -L 3000:localhost:3000 -L 8000:localhost:8000  username@IP-address```

## 步骤 6. 尝试示例提示

单击前端的任何图块即可试用主管和其他代理。

**RAG 代理**：
在尝试 RAG 代理的示例提示之前，请上传示例 PDF 文档 [NVIDIA Blackwell Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf) 
作为上下文，转到链接，将 PDF 下载到本地文件系统，单击左侧边栏中“上下文”下的绿色“上传文档”按钮，然后确保选中“选择源”部分中的框。

## 步骤 8. 清理和回滚

完全删除容器并释放资源的步骤。

从 multi-agent-chatbot 项目的根目录中，运行以下命令：

```bash
docker compose -f docker-compose.yml -f docker-compose-models.yml down

docker volume rm "$(basename "$PWD")_postgres_data"
```

## 步骤 9. 后续步骤

- 使用多代理聊天机器人系统尝试不同的提示。
- 按照存储库中的说明尝试不同的模型。
- 尝试添加新的 MCP（模型上下文协议）服务器作为主管代理的工具。

## 故障排除

| 症状 | 原因 | 使固定 |
|---------|--------|-----|
| 无法访问 URL 的门禁存储库 | 某些 HuggingFace 模型的访问受到限制 | 重新生成你的 [HuggingFace token](https://huggingface.co/docs/hub/en/security-tokens);并请求在您的网络浏览器上访问 [gated model](https://huggingface.co/docs/hub/en/models-gated#customize-requested-information) |

> [！笔记]
> DGX Spark 使用统一内存架构 (UMA)，可实现 GPU 和 CPU 之间的动态内存共享。 
> 由于许多应用程序仍在更新以利用 UMA，因此即使在 
> DGX Spark 的内存容量。如果发生这种情况，请使用以下命令手动刷新缓冲区缓存：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
