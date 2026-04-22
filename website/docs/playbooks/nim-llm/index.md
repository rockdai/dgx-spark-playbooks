---
id: nim-llm
title: nim-llm
sidebar_label: nim-llm
---

# Spark 上的 NIM

> 在 Spark 上部署 NIM

## 目录

- [Overview](#overview)
  - [Basic idea](#basic-idea)
  - [What you'll accomplish](#what-youll-accomplish)
  - [What to know before starting](#what-to-know-before-starting)
  - [Prerequisites](#prerequisites)
  - [Time & risk](#time-risk)
- [Instructions](#instructions)
  - [Step 2. Configure NGC authentication](#step-2-configure-ngc-authentication)
- [Troubleshooting](#troubleshooting)

---

## 概述

### 基本思路

NVIDIA NIM 是容器化软件，可在 NVIDIA GPU 上提供快速、可靠的 AI 模型服务和推理。本手册演示了如何在 DGX Spark 设备上为 LLM 运行 NIM 微服务，通过简单的 Docker 工作流程实现本地 GPU 推理。您将使用 NVIDIA 的注册表进行身份验证，启动 NIM 推理微服务，并执行基本推理测试以验证功能。

### 你将完成什么

您将在 DGX Spark 设备上启动 NIM 容器，以公开用于文本完成的 GPU 加速的 HTTP 端点。虽然这些指令适用于 Llama 3.1 8B NIM，但包括 [Qwen3-32 NIM](https://catalog.ngc.nvidia.com/orgs/nim/teams/qwen/containers/qwen3-32b-dgx-spark) 在内的其他 NIM 可用于 DGX Spark（请参阅 [here](https://docs.nvidia.com/nim/large-language-models/1.14.0/release-notes.html#new-language-models%20)）。

### 开始之前需要了解什么

- 在终端环境中工作
- 使用 Docker 命令和支持 GPU 的容器
- 基本熟悉 REST API 和curl 命令
- 了解 NVIDIA GPU 环境和 CUDA

### 先决条件

- 安装了 NVIDIA 驱动程序的 DGX Spark 设备
  ```bash
  nvidia-smi
  ```
- 配置了 NVIDIA Container Toolkit 的 Docker，指令 [here](https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html)
  ```bash
  docker run -it --gpus=all nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 nvidia-smi
  ```
- 具有来自 [here](https://ngc.nvidia.com/setup/api-key) 的 API 密钥的 NGC 帐户
  ```bash
  echo $NGC_API_KEY | grep -E '^[a-zA-Z0-9]{86}=='
  ```
- 足够的磁盘空间用于模型缓存（因模型而异，通常为 10-50GB）
  ```bash
  df -h ~
  ```


### 时间与风险

* **预计时间：** 15-30 分钟用于设置和验证
* **风险：**
  * 大型模型下载可能需要大量时间，具体取决于网络速度
  * GPU 内存要求因模型大小而异
  * 容器启动时间取决于模型加载
* **回滚：** 使用 `docker stop CONTAINER_NAME && docker rm CONTAINER_NAME` 停止并删除容器。如果需要恢复磁盘空间，请从 `~/.cache/nim` 中删除缓存的模型。
* **最后更新：** 2025 年 12 月 22 日
  * 将 docker 容器版本更新为 cuda:13.0.1-devel-ubuntu24.04
  * 添加docker容器权限设置说明

## 指示

## 步骤 1. 验证环境先决条件

检查您的系统是否满足运行支持 GPU 的容器的基本要求。

```bash
nvidia-smi
docker --version
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 nvidia-smi
```

如果您看到权限被拒绝错误（例如尝试连接到 Docker 守护进程套接字时权限被拒绝），请将您的用户添加到 docker 组，这样您就不需要使用 sudo 运行命令。

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### 步骤 2. 配置 NGC 身份验证

使用 NGC API 密钥设置对 NVIDIA 容器注册表的访问。

```bash
export NGC_API_KEY="`<YOUR_NGC_API_KEY>`"
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

## 步骤 3. 选择并配置 NIM 容器

从 NGC 选择特定的 LLM NIM 并为模型资产设置本地缓存。

```bash
export CONTAINER_NAME="nim-llm-demo"
export IMG_NAME="nvcr.io/nim/meta/llama-3.1-8b-instruct-dgx-spark:latest"
export LOCAL_NIM_CACHE=~/.cache/nim
export LOCAL_NIM_WORKSPACE=~/.local/share/nim/workspace
mkdir -p "$LOCAL_NIM_WORKSPACE"
chmod -R a+w "$LOCAL_NIM_WORKSPACE"
mkdir -p "$LOCAL_NIM_CACHE"
chmod -R a+w "$LOCAL_NIM_CACHE"
```

## 步骤 4. 启动 NIM 容器

通过 GPU 加速和适当的资源分配启动容器化 LLM 服务。

```bash
docker run -it --rm --name=$CONTAINER_NAME \
  --gpus all \
  --shm-size=16GB \
  -e NGC_API_KEY=$NGC_API_KEY \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -v "$LOCAL_NIM_WORKSPACE:/opt/nim/workspace" \
  -p 8000:8000 \
  $IMG_NAME
```

容器将在首次运行时下载模型，可能需要几分钟才能启动。寻找
指示服务已准备就绪的启动消息。

## 步骤 5. 验证推理端点

使用基本完成请求测试已部署的服务以验证功能。在新终端中运行以下curl命令。


```bash
curl -X 'POST' \
    'http://0.0.0.0:8000/v1/chat/completions' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "meta/llama-3.1-8b-instruct",
      "messages": [
        {
          "role":"system",
          "content":"detailed thinking on"
        },
        {
          "role":"user",
          "content":"Can you write me a song?"
        }
      ],
      "top_p": 1,
      "n": 1,
      "max_tokens": 15,
      "frequency_penalty": 1.0,
      "stop": ["hello"]

    }'
    
```

预期输出应该是一个 JSON 响应，其中包含带有生成文本的完成字段。

## 步骤 6. 清理和回滚

删除正在运行的容器并可选择清理缓存的模型文件。

> [！警告]
> 删除缓存的模型将需要在下次运行时重新下载。

```bash
docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME
```

要删除缓存的模型并释放磁盘空间：
```bash
rm -rf "$LOCAL_NIM_CACHE"
```

## 步骤 7. 后续步骤

通过有效的 NIM 部署，您可以：

- 使用 OpenAI 兼容接口将 API 端点集成到您的应用程序中
- 尝试 NGC 目录中提供的不同模型
- 使用容器编排工具扩展部署
- 监控资源使用情况并优化容器资源分配

测试与您首选的 HTTP 客户端或 SDK 的集成，以开始构建应用程序。

## 故障排除

| 症状 | 原因 | 使固定 |
|---------|--------|-----|
| 容器因 GPU 错误而无法启动 | 未配置 NVIDIA 容器工具包 | 安装 nvidia-container-toolkit 并重新启动 Docker |
| docker 登录期间“凭据无效” | NGC API 密钥格式不正确 | 从 NGC 门户验证 API 密钥，确保没有多余的空格 |
| 模型下载挂起或失败 | 网络连接或磁盘空间不足 | 检查互联网连接和缓存目录中的可用磁盘空间 |
| API返回404或连接被拒绝 | 容器未完全启动或端口错误 | 等待容器启动完成，验证端口8000是否可访问 |
| 未找到运行时 | NVIDIA 容器工具包未正确配置 | 运行 `sudo nvidia-ctk runtime configure --runtime=docker` 并重新启动 Docker |

> [！笔记]
> DGX Spark 使用统一内存架构 (UMA)，可实现 GPU 和 CPU 之间的动态内存共享。 
> 由于许多应用程序仍在更新以利用 UMA，因此即使在 
> DGX Spark 的内存容量。如果发生这种情况，请使用以下命令手动刷新缓冲区缓存：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
