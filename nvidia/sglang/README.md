# 用于推理的 SGLang

> 在 DGX Spark 上安装和使用 SGLang

## 目录

- [Overview](#overview)
  - [Time & risk](#time-risk)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思想

SGLang 是一个用于大型语言模型和视觉语言模型的快速服务框架，使得
通过共同设计后端运行时，您与模型的交互更快、更可控
前端语言。此设置在单个 NVIDIA 上使用优化的 NVIDIA SGLang NGC 容器
采用 Blackwell 架构的 Spark 设备，提供具有所有依赖项的 GPU 加速推理
预安装。

## 你将完成什么

您将在 NVIDIA Spark 设备上以服务器和离线推理模式部署 SGLang，
通过支持文本生成、聊天完成和
使用 DeepSeek-V2-Lite 等模型执行视觉语言任务。

## 开始之前需要了解什么

- 在 Linux 系统的终端环境中工作
- 对 Docker 容器和容器管理有基本了解
- 熟悉 NVIDIA GPU 驱动程序和 CUDA 工具包概念
- 拥有 HTTP API 端点和 JSON 请求/响应处理的经验

## 先决条件

- 采用 Blackwell 架构的 NVIDIA Spark 设备
- Docker 引擎已安装并正在运行：`docker --version`
- 安装的 NVIDIA GPU 驱动程序：`nvidia-smi`
- 配置的 NVIDIA 容器工具包：`docker run --rm --gpus all nvcr.io/nvidia/sglang:26.02-py3 nvidia-smi`
- 足够的磁盘空间（>20GB可用）：`df -h`
- 用于拉取 NGC 容器的网络连接：`ping nvcr.io`

## 附属文件

- 离线推理Python脚本[found here on GitHub](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/sglang/assets/offline-inference.py)

## 模型支持矩阵

Spark 上的 SGLang 支持以下模型。所有列出的型号均可供使用：

| 模型 | 量化 | 支持状态 | 高频手柄 |
|-------|-------------|----------------|-----------|
| **GPT-OSS-20B** | MXFP4 | ✅ | `openai/gpt-oss-20b` |
| **GPT-OSS-120B** | MXFP4 | ✅ | `openai/gpt-oss-120b` |
| **Llama-3.1-8B-Instruct** | FP8 | ✅ | `nvidia/Llama-3.1-8B-Instruct-FP8` |
| **Llama-3.1-8B-Instruct** | NVFP4 | ✅ | `nvidia/Llama-3.1-8B-Instruct-FP4` |
| **Llama-3.3-70B-Instruct** | NVFP4 | ✅ | `nvidia/Llama-3.3-70B-Instruct-FP4` |
| **Qwen3-8B** | FP8 | ✅ | `nvidia/Qwen3-8B-FP8` |
| **Qwen3-8B** | NVFP4 | ✅ | `nvidia/Qwen3-8B-FP4` |
| **Qwen3-14B** | FP8 | ✅ | `nvidia/Qwen3-14B-FP8` |
| **Qwen3-14B** | NVFP4 | ✅ | `nvidia/Qwen3-14B-FP4` |
| **Qwen3-32B** | NVFP4 | ✅ | `nvidia/Qwen3-32B-FP4` |
| **Phi-4-multimodalInstruct** | FP8 | ✅ | `nvidia/Phi-4-multimodal-instruct-FP8` |
| **Phi-4-multimodalInstruct** | NVFP4 | ✅ | `nvidia/Phi-4-multimodal-instruct-FP4` |
| **Phi-4-reasoning+** | FP8 | ✅ | `nvidia/Phi-4-reasoning-plus-FP8` |
| **Phi-4-reasoning+** | NVFP4 | ✅ | `nvidia/Phi-4-reasoning-plus-FP4` |

注意：对于 NVFP4 型号，请添加 `--quantization modelopt_fp4` 标志。

### 时间与风险

* **预计时间：** 初始设置和验证需要 30 分钟
* **风险级别：** 低 - 使用预构建、经过验证的 SGLang 容器，配置最少
* **回滚：** 使用 `docker stop` 和 `docker rm` 命令停止并删除容器
* **最后更新：** 2026 年 3 月 15 日
    * 使用最新的 NGC SGLang 容器：nvcr.io/nvidia/sglang:26.02-py3

## 指示

## 步骤 1. 验证系统先决条件

在继续之前，请检查您的 NVIDIA Spark 设备是否满足所有要求。此步骤运行于
您的主机系统，并确保正确配置 Docker、GPU 驱动程序和容器工具包。 

> 注意：如果您在拉取容器映像时遇到超时或“连接被拒绝”错误，您可能需要使用 VPN 或代理，因为某些注册表可能受到您的本地网络或 ISP 的限制。

```bash
## Verify Docker installation
docker --version

## Check NVIDIA GPU drivers
nvidia-smi

## Verify Docker GPU support
docker run --rm --gpus all nvcr.io/nvidia/sglang:26.02-py3 nvidia-smi

## Check available disk space
df -h /
```

如果您看到权限被拒绝错误（例如尝试连接到 Docker 守护进程套接字时权限被拒绝），请将您的用户添加到 docker 组，这样您就不需要使用 sudo 运行命令。

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## 步骤 2. 拉取 SGLang 容器

下载最新的 SGLang 容器。此步骤在主机上运行，​​可能需要
几分钟，具体取决于您的网络连接。


```bash
## Pull the SGLang container
docker pull nvcr.io/nvidia/sglang:26.02-py3

## Verify the image was downloaded
docker images | grep sglang
```

## 步骤 3. 启动服务器模式的 SGLang 容器

在服务器模式下启动 SGLang 容器以启用 HTTP API 访问。这运行推理
服务器位于容器内，将其暴露在端口 30000 上以供客户端连接。

```bash
## Launch container with GPU support and port mapping
docker run --gpus all -it --rm \
  -p 30000:30000 \
  -v /tmp:/tmp \
  nvcr.io/nvidia/sglang:26.02-py3 \
  bash
```

## 步骤 4. 启动 SGLang 推理服务器

在容器内，使用受支持的模型启动 HTTP 推理服务器。这一步运行
在 Docker 容器内并启动 SGLang 服务器守护进程。

```bash
## Start the inference server with DeepSeek-V2-Lite model
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V2-Lite \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --tp 1 \
  --attention-backend flashinfer \
  --mem-fraction-static 0.75 &

## Wait for server to initialize
sleep 30

## Check server status
curl http://localhost:30000/health
```

## 步骤 5. 测试客户端-服务器推理

从主机系统上的新终端测试 SGLang 服务器 API 以确保其正常工作
正确。这验证服务器正在接受请求并生成响应。

```bash
## Test with curl
curl -X POST http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
      "text": "What does NVIDIA love?",
      "sampling_params": {
          "temperature": 0.7,
          "max_new_tokens": 100
      }
  }'
```

## 步骤 6. 测试 Python 客户端 API

创建一个简单的 Python 脚本来测试对 SGLang 服务器的编程访问。这运行于
主机系统并演示如何将 SGLang 集成到应用程序中。

```python
import requests

## Send prompt to server
response = requests.post('http://localhost:30000/generate', json={
  'text': 'What does NVIDIA love?',
  'sampling_params': {
      'temperature': 0.7,
      'max_new_tokens': 100,
  },
})

print(f"Response: {response.json()['text']}")
```

## 步骤 7. 验证安装

确认服务器和离线模式均正常工作。此步骤验证
完整的 SGLang 设置并确保可靠运行。

```bash
## Check server mode (from host)
curl http://localhost:30000/health
curl -X POST http://localhost:30000/generate -H "Content-Type: application/json" \
  -d '{"text": "Hello", "sampling_params": {"max_new_tokens": 10}}'

## Check container logs
docker ps
docker logs <CONTAINER_ID>
```

## 步骤 8. 清理和回滚

停止并删除容器以清理资源。此步骤将您的系统恢复到原来的状态
原始状态。

> [!WARNING]
> 这将停止所有 SGLang 容器并删除临时数据。

```bash
## Stop all SGLang containers
docker ps | grep sglang | awk '{print $1}' | xargs docker stop

## Remove stopped containers
docker container prune -f

## Remove SGLang images (optional)
docker rmi nvcr.io/nvidia/sglang:26.02-py3
```

## 步骤 9. 后续步骤

成功部署 SGLang 后，您现在可以：

- 使用 `/generate` 端点将 HTTP API 集成到您的应用程序中
- 通过更改 `--model-path` 参数来试验不同的模型
- 通过调整 `--tp`（张量并行）设置来使用多个 GPU 进行扩展
- 使用您选择的容器编排平台部署生产工作负载

## 故障排除

常见问题及其解决方案：

| 症状 | 原因 | 使固定 |
|---------|-------|-----|
| 容器因 GPU 错误而无法启动 | 缺少 NVIDIA 驱动程序/工具包 | 安装nvidia-container-toolkit，重启Docker |
| 服务器响应 404 或连接被拒绝 | 服务器未完全初始化 | 等待60秒，检查容器日志 |
| 模型加载期间出现内存不足错误 | GPU显存不足 | 使用较小的模型或增加--tp参数 |
| 模型下载失败 | 网络连接问题 | 检查互联网连接，重试下载 |
| 访问 /tmp 的权限被拒绝 | 卷安装问题 | 使用完整路径：-v /tmp:/tmp 或创建专用目录 |

> [!NOTE]
> DGX Spark 使用统一内存架构 (UMA)，可实现 GPU 和 CPU 之间的动态内存共享。 
> 由于许多应用程序仍在更新以利用 UMA，因此即使在 
> DGX Spark 的内存容量。如果发生这种情况，请使用以下命令手动刷新缓冲区缓存：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
