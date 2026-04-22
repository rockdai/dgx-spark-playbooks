---
id: vllm
title: vllm
sidebar_label: vllm
---

# vLLM 推理

> 在 DGX Spark 上安装和使用 vLLM

## 目录

- [Overview](#overview)
- [Instructions](#instructions)
- [Run on two Sparks](#run-on-two-sparks)
  - [Step 11. (Optional) Launch 405B inference server](#step-11-optional-launch-405b-inference-server)
- [Run on multiple Sparks through a switch](#run-on-multiple-sparks-through-a-switch)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思路

vLLM 是一种推理引擎，旨在高效运行大型语言模型。关键思想是在为 LLM 提供服务时**最大化吞吐量并最小化内存浪费**。

- 它使用名为 **PagedAttention** 的内存高效注意力算法来处理长序列，而不会耗尽 GPU 内存。
- 可以通过**连续批处理**将新请求添加到已处理的批次中，以保持 GPU 得到充分利用。
- 它具有 **OpenAI 兼容 API**，因此为 OpenAI API 构建的应用程序可以切换到 vLLM 后端，只需很少的修改或无需修改。

## 你将完成什么

您将使用 Blackwell 架构在 DGX Spark 上设置 vLLM 高吞吐量 LLM 服务，
使用预构建的 Docker 容器或使用自定义 LLVM/Triton 从源代码构建
支持ARM64。

## 开始之前需要了解什么

- 体验使用 Docker 构建和配置容器
- 熟悉CUDA工具包安装和版本管理
- 了解Python虚拟环境和包管理
- 了解使用 CMake 和 Ninja 从源代码构建软件
- 有 Git 版本控制和补丁管理经验

## 先决条件

- 具有 ARM64 处理器和 Blackwell GPU 架构的 DGX Spark 设备
- 安装的 CUDA 13.0 工具包：`nvcc --version` 显示 CUDA 工具包版本。
- Docker 安装并配置：`docker --version` 成功
- 已安装 NVIDIA 容器工具包
- Python 3.12 可用：`python3.12 --version` 成功
- Git 安装：`git --version` 成功
- 网络访问下载包和容器镜像


## 模型支持矩阵

Spark 上的 vLLM 支持以下模型。所有列出的型号均可供使用：

| 模型 | 量化 | 支持状态 | 高频手柄 |
|-------|-------------|----------------|-----------|
| **Gemma 4 31B IT** | BF16 | ✅ | [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) |
| **Gemma 4 31B IT** | NVFP4 | ✅ | [`nvidia/Gemma-4-31B-IT-NVFP4`](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4) |
| **Gemma 4 26B A4B IT** | BF16 | ✅ | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| **Gemma 4 E4B IT** | BF16 | ✅ | [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) |
| **Gemma 4 E2B IT** | BF16 | ✅ | [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) |
| **Nemotron-3-Super-120B** | NVFP4 | ✅ | [`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4) |
| **GPT-OSS-20B** | MXFP4 | ✅ | [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) |
| **GPT-OSS-120B** | MXFP4 | ✅ | [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) |
| **Llama-3.1-8B-Instruct** | FP8 | ✅ | [`nvidia/Llama-3.1-8B-Instruct-FP8`](https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8) |
| **Llama-3.1-8B-Instruct** | NVFP4 | ✅ | [`nvidia/Llama-3.1-8B-Instruct-NVFP4`](https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-NVFP4) |
| **Llama-3.3-70B-Instruct** | NVFP4 | ✅ | [`nvidia/Llama-3.3-70B-Instruct-NVFP4`](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-NVFP4) |
| **Qwen3-8B** | FP8 | ✅ | [`nvidia/Qwen3-8B-FP8`](https://huggingface.co/nvidia/Qwen3-8B-FP8) |
| **Qwen3-8B** | NVFP4 | ✅ | [`nvidia/Qwen3-8B-NVFP4`](https://huggingface.co/nvidia/Qwen3-8B-NVFP4) |
| **Qwen3-14B** | FP8 | ✅ | [`nvidia/Qwen3-14B-FP8`](https://huggingface.co/nvidia/Qwen3-14B-FP8) |
| **Qwen3-14B** | NVFP4 | ✅ | [`nvidia/Qwen3-14B-NVFP4`](https://huggingface.co/nvidia/Qwen3-14B-NVFP4) |
| **Qwen3-32B** | NVFP4 | ✅ | [`nvidia/Qwen3-32B-NVFP4`](https://huggingface.co/nvidia/Qwen3-32B-NVFP4) |
| **Qwen2.5-VL-7B-Instruct** | NVFP4 | ✅ | [`nvidia/Qwen2.5-VL-7B-Instruct-NVFP4`](https://huggingface.co/nvidia/Qwen2.5-VL-7B-Instruct-NVFP4) |
| **Qwen3-VL-Reranker-2B** | BF16 | ✅ | [`Qwen/Qwen3-VL-Reranker-2B`](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) |
| **Qwen3-VL-Reranker-8B** | BF16 | ✅ | [`Qwen/Qwen3-VL-Reranker-8B`](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B) |
| **Qwen3-VL-Embedding-2B** | BF16 | ✅ | [`Qwen/Qwen3-VL-Embedding-2B`](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) |
| **Phi-4-multimodal-instruct** | FP8 | ✅ | [`nvidia/Phi-4-multimodal-instruct-FP8`](https://huggingface.co/nvidia/Phi-4-multimodal-instruct-FP8) |
| **Phi-4-multimodal-instruct** | NVFP4 | ✅ | [`nvidia/Phi-4-multimodal-instruct-NVFP4`](https://huggingface.co/nvidia/Phi-4-multimodal-instruct-NVFP4) |
| **Phi-4-reasoning-plus** | FP8 | ✅ | [`nvidia/Phi-4-reasoning-plus-FP8`](https://huggingface.co/nvidia/Phi-4-reasoning-plus-FP8) |
| **Phi-4-reasoning-plus** | NVFP4 | ✅ | [`nvidia/Phi-4-reasoning-plus-NVFP4`](https://huggingface.co/nvidia/Phi-4-reasoning-plus-NVFP4) |
| **Nemotron-3-Nano** | BF16 | ✅ | [`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) |
| **Nemotron-3-Nano** | FP8 | ✅ | [`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8) |

> [!NOTE]
> Phi-4-multimodal-instruct 模型在启动 vLLM 时需要 `--trust-remote-code`。

> [!NOTE]
> 您可以使用 NVFP4 量化文档为您喜欢的模型生成您自己的 NVFP4 量化检查点。这使您能够利用 NVFP4 量化的性能和内存优势，即使对于 NVIDIA 尚未发布的模型也是如此。

提醒：并非所有模型架构都支持 NVFP4 量化。

## 时间与风险

* **持续时间：** Docker 方法需要 30 分钟
* **风险：** 容器注册表访问需要内部凭据
* **回滚：**容器方法是非破坏性的。
* **最后更新：** 2026 年 4 月 2 日
  * 添加对 Gemma 4 模型系列的支持

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

## 步骤 2. 拉取 vLLM 容器映像

从 https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm 查找最新的容器版本

```bash
export LATEST_VLLM_VERSION=`<latest_container_version>`
## example
## export LATEST_VLLM_VERSION=26.02-py3

export HF_MODEL_HANDLE=`<HF_HANDLE>`
## example
## export HF_MODEL_HANDLE=openai/gpt-oss-20b

docker pull nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION}
```

对于 Gemma 4 模型系列，使用 vLLM 自定义容器：
```bash
docker pull vllm/vllm-openai:gemma4-cu130
```

## 步骤 3. 在容器中测试 vLLM

启动容器并使用测试模型启动 vLLM 服务器以验证基本功能。

```bash
docker run -it --gpus all -p 8000:8000 \
nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION} \
vllm serve ${HF_MODEL_HANDLE}
```

要运行 Gemma 4 模型系列中的模型（例如 `google/gemma-4-31B-it`）：
```bash
docker run -it --gpus all -p 8000:8000 \
vllm/vllm-openai:gemma4-cu130 ${HF_MODEL_HANDLE}
```

预期输出应包括：
- 模型加载确认
- 服务器在端口 8000 上启动
- GPU内存分配详细信息

在另一个终端中，测试服务器：

```bash
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "'"${HF_MODEL_HANDLE}"'",
    "messages": [{"role": "user", "content": "12*17"}],
    "max_tokens": 500
}'
```

预期响应应包含 `"content": "204"` 或类似的数学计算。

## 步骤 4. 清理和回滚

对于容器方法（非破坏性）：

```bash
docker rm $(docker ps -aq --filter ancestor=nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION})
docker rmi nvcr.io/nvidia/vllm
```

## 步骤 5. 后续步骤

- **生产部署：** 根据您的特定模型要求配置 vLLM
- **性能调整：** 根据您的工作负载调整批量大小和内存设置
- **监控：** 设置日志记录和指标收集以供生产使用
- **模型管理：**探索其他模型格式和量化选项

## 在两个 Spark 上运行

## 步骤 1. 配置网络连接

按照 [Connect two Sparks](https://build.nvidia.com/spark/connect-two-sparks) 手册中的网络设置说明在 DGX Spark 节点之间建立连接。

这包括：
- 物理 QSFP 电缆连接
- 网络接口配置（自动或手动IP分配）
- 无密码 SSH 设置
- 网络连接验证

## 步骤2.下载集群部署脚本

获取两个节点上的vLLM集群部署脚本。该脚本协调分布式推理所需的 Ray 集群设置。

```bash
## Download on both nodes
wget https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/examples/online_serving/run_cluster.sh
chmod +x run_cluster.sh
```

## 步骤 3. 从 NGC 提取 NVIDIA vLLM 映像

首先，您需要配置 docker 以从 NGC 拉取
如果这是您第一次使用 docker run：
```bash
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

之后，您应该能够在不使用 `sudo` 的情况下运行 docker 命令。


```bash
docker pull nvcr.io/nvidia/vllm:25.11-py3
export VLLM_IMAGE=nvcr.io/nvidia/vllm:25.11-py3
```


## 步骤4.启动Ray头节点

在节点 1 上启动 Ray 集群头节点。该节点协调分布式推理并为 API 端点提供服务。

```bash
## On Node 1, start head node

## Get the IP address of the high-speed interface
## Use the interface that shows "(Up)" from ibdev2netdev (enp1s0f0np0 or enp1s0f1np1)
export MN_IF_NAME=enp1s0f1np1
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')

echo "Using interface $MN_IF_NAME with IP $VLLM_HOST_IP"

bash run_cluster.sh $VLLM_IMAGE $VLLM_HOST_IP --head ~/.cache/huggingface \
  -e VLLM_HOST_IP=$VLLM_HOST_IP \
  -e UCX_NET_DEVICES=$MN_IF_NAME \
  -e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
  -e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME \
  -e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
  -e TP_SOCKET_IFNAME=$MN_IF_NAME \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR=$VLLM_HOST_IP
```


## 步骤5.启动Ray工作节点

将节点 2 作为工作节点连接到 Ray 集群。这为张量并行性提供了额外的 GPU 资源。

```bash
## On Node 2, join as worker

## Set the interface name (same as Node 1)
export MN_IF_NAME=enp1s0f1np1

## Get Node 2's own IP address
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')

## IMPORTANT: Set HEAD_NODE_IP to Node 1's IP address
## You must get this value from Node 1 (run: echo $VLLM_HOST_IP on Node 1)
export HEAD_NODE_IP=`<NODE_1_IP_ADDRESS>`

echo "Worker IP: $VLLM_HOST_IP, connecting to head node at: $HEAD_NODE_IP"

bash run_cluster.sh $VLLM_IMAGE $HEAD_NODE_IP --worker ~/.cache/huggingface \
  -e VLLM_HOST_IP=$VLLM_HOST_IP \
  -e UCX_NET_DEVICES=$MN_IF_NAME \
  -e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
  -e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME \
  -e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
  -e TP_SOCKET_IFNAME=$MN_IF_NAME \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR=$HEAD_NODE_IP
```
> **注意：** 将 `<NODE_1_IP_ADDRESS>` 替换为节点 1 中的实际 IP 地址，特别是 [Connect two Sparks](https://build.nvidia.com/spark/connect-two-sparks) 剧本中配置的 QSFP 接口 nep1s0f1np1。

## 步骤 6. 验证集群状态

确认两个节点在 Ray 集群中均已识别且可用。

```bash
## On Node 1 (head node)
## Find the vLLM container name (it will be node-<random_number>)
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
echo "Found container: $VLLM_CONTAINER"

docker exec $VLLM_CONTAINER ray status
```

预期输出显示 2 个具有可用 GPU 资源的节点。

## 步骤7.下载Llama 3.3 70B模型

使用 Hugging Face 进行身份验证并下载推荐的生产就绪模型。

```bash
## From within the same container where `ray status` ran, run the following
hf auth login
hf download meta-llama/Llama-3.3-70B-Instruct
```

## 步骤 8. 启动 Llama 3.3 70B 的推理服务器

启动 vLLM 推理服务器，并在两个节点上实现张量并行。

```bash
## On Node 1, enter container and start server
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
docker exec -it $VLLM_CONTAINER /bin/bash -c '
  vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 2 --max_model_len 2048'
```

## 步骤9.测试70B模型推理

使用示例推理请求验证部署。

```bash
## Test from Node 1 or external client
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.3-70B-Instruct",
    "prompt": "Write a haiku about a GPU",
    "max_tokens": 32,
    "temperature": 0.7
  }'
```

预期输出包括生成的俳句响应。

## 步骤10.（可选）部署Llama 3.1 405B模型

> [!WARNING]
> 405B 型号的内存空间不足以供生产使用。

下载量化 405B 模型仅用于测试目的。

```bash
## On Node 1, download quantized model
huggingface-cli download hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4
```

### 步骤 11.（可选）启动 405B 推理服务器

使用大型模型的内存受限参数启动服务器。

```bash
## On Node 1, launch with restricted parameters
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
docker exec -it $VLLM_CONTAINER /bin/bash -c '
  vllm serve hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4 \
    --tensor-parallel-size 2 --max-model-len 64 --gpu-memory-utilization 0.9 \
    --max-num-seqs 1 --max_num_batched_tokens 64'
```

## 步骤 12.（可选）测试 405B 模型推理

使用受限参数验证 405B 部署。

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4",
    "prompt": "Write a haiku about a GPU",
    "max_tokens": 32,
    "temperature": 0.7
  }'
```

## 步骤 13. 验证部署

对分布式推理系统进行全面验证。

```bash
## Check Ray cluster health
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
docker exec $VLLM_CONTAINER ray status

## Verify server health endpoint
curl http://192.168.100.10:8000/health

## Monitor GPU utilization on both nodes
nvidia-smi
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
docker exec $VLLM_CONTAINER nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## 步骤 14. 后续步骤

访问 Ray 仪表板进行集群监控并探索其他功能：

```bash
## Ray dashboard available at:
http://<head-node-ip>:8265

## Consider implementing for production:
## - Health checks and automatic restarts
## - Log rotation for long-running services
## - Persistent model caching across restarts
## - Alternative quantization methods (FP8, INT4)
```

## 通过交换机在多个 Spark 上运行

## 步骤 1. 配置网络连接

按照 [Multi Sparks through switch](https://build.nvidia.com/spark/multi-sparks-through-switch) 手册中的网络设置说明在 DGX Spark 节点之间建立连接。

这包括：
- Sparks 和 Switch 之间的物理 QSFP 电缆连接
- 网络接口配置（自动或手动IP分配）
- 无密码 SSH 设置
- 网络连接验证
- NCCL 带宽测试

## 步骤2.下载集群部署脚本

在所有节点上下载 vLLM 集群部署脚本。该脚本协调分布式推理所需的 Ray 集群设置。

```bash
## Download on all nodes
wget https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/examples/online_serving/run_cluster.sh
chmod +x run_cluster.sh
```

## 步骤 3. 从 NGC 提取 NVIDIA vLLM 映像

在所有节点上执行此步骤。

首先，您需要配置 docker 以从 NGC 拉取
如果这是您第一次使用 docker run：
```bash
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

之后，您应该能够在不使用 `sudo` 的情况下运行 docker 命令。

从 https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm 查找最新的容器版本

```bash
docker pull nvcr.io/nvidia/vllm:26.02-py3
export VLLM_IMAGE=nvcr.io/nvidia/vllm:26.02-py3
```

## 步骤4.启动Ray头节点

在节点 1 上启动 Ray 集群头节点。该节点协调分布式推理并为 API 端点提供服务。

```bash
## On Node 1, start head node

## Get the IP address of the high-speed interface
## Use the interface that shows "(Up)" from ibdev2netdev (enp1s0f0np0 or enp1s0f1np1)
export MN_IF_NAME=enp1s0f1np1
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')

echo "Using interface $MN_IF_NAME with IP $VLLM_HOST_IP"

bash run_cluster.sh $VLLM_IMAGE $VLLM_HOST_IP --head ~/.cache/huggingface \
  -e VLLM_HOST_IP=$VLLM_HOST_IP \
  -e UCX_NET_DEVICES=$MN_IF_NAME \
  -e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
  -e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME \
  -e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
  -e TP_SOCKET_IFNAME=$MN_IF_NAME \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR=$VLLM_HOST_IP
```

## 步骤 5. 启动 Ray 工作节点

将其余节点作为工作节点连接到 Ray 集群。这为张量并行性提供了额外的 GPU 资源。

```bash
## On other Nodes, join as workers

## Set the interface name (same as Node 1)
export MN_IF_NAME=enp1s0f1np1

## Get Node's own IP address
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')

## IMPORTANT: Set HEAD_NODE_IP to Node 1's IP address
## You must get this value from Node 1 (run: echo $VLLM_HOST_IP on Node 1)
export HEAD_NODE_IP=`<NODE_1_IP_ADDRESS>`

echo "Worker IP: $VLLM_HOST_IP, connecting to head node at: $HEAD_NODE_IP"

bash run_cluster.sh $VLLM_IMAGE $HEAD_NODE_IP --worker ~/.cache/huggingface \
  -e VLLM_HOST_IP=$VLLM_HOST_IP \
  -e UCX_NET_DEVICES=$MN_IF_NAME \
  -e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
  -e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME \
  -e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
  -e TP_SOCKET_IFNAME=$MN_IF_NAME \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR=$HEAD_NODE_IP
```
> **注意：** 将 `<NODE_1_IP_ADDRESS>` 替换为节点 1 中的实际 IP 地址，特别是 [Multi Sparks through switch](https://build.nvidia.com/spark/multi-sparks-through-switch) 剧本中配置的 QSFP 接口 enp1s0f1np1。

## 步骤 6. 验证集群状态

确认 Ray 集群中的所有节点均已识别且可用。

```bash
## On Node 1 (head node)
## Find the vLLM container name (it will be node-<random_number>)
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
echo "Found container: $VLLM_CONTAINER"

docker exec $VLLM_CONTAINER ray status
```

预期输出显示具有可用 GPU 资源的所有节点。

## 步骤 7. 下载 MiniMax M2.5 模型

如果您正在运行四个或更多火花，则可以轻松地使用张量并行性运行此模型。使用 Hugging Face 进行身份验证并下载模型。

```bash
## On all nodes, from within the docker containers created in previous steps, run the following
hf auth login
hf download MiniMaxAI/MiniMax-M2.5
```

## 步骤 8. 启动 MiniMax M2.5 推理服务器

启动 vLLM 推理服务器，并在所有节点上实现张量并行。

```bash
## On Node 1, enter container and start server
## Assuming that you run on a 4 node cluster, set --tensor-parallel-size as 4
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
docker exec -it $VLLM_CONTAINER /bin/bash -c '
  vllm serve MiniMaxAI/MiniMax-M2.5 \
    --tensor-parallel-size 4 --max-model-len 129000 --max-num-seqs 4 --trust-remote-code'
```

## 步骤9.测试MiniMax M2.5模型推理

使用示例推理请求验证部署。

```bash
## Test from Node 1 or external client.
## If testing with external client change localhost to the Node 1 Mgmt IP address.
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMaxAI/MiniMax-M2.5",
    "prompt": "Write a haiku about a GPU",
    "max_tokens": 32,
    "temperature": 0.7
  }'
```

## 步骤 10. 验证部署

对分布式推理系统进行全面验证。

```bash
## Check Ray cluster health
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
docker exec $VLLM_CONTAINER ray status

## Verify server health endpoint on Node 1
curl http://localhost:8000/health

## Monitor GPU utilization on all nodes
nvidia-smi
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
docker exec $VLLM_CONTAINER nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## 步骤 11. 后续步骤

访问 Ray 仪表板进行集群监控并探索其他功能：

```bash
## Ray dashboard available at:
http://<head-node-ip>:8265

## Consider implementing for production:
## - Health checks and automatic restarts
## - Log rotation for long-running services
## - Persistent model caching across restarts
## - Other models which can fit on the cluster with different quantization methods (FP8, NVFP4)
```

## 故障排除

## 在单个 Spark 上运行的常见问题

| 症状 | 原因 | 使固定 |
|---------|--------|-----|
| CUDA版本不匹配错误 | CUDA工具包版本错误 | 使用精确安装程序重新安装 CUDA 12.9 |
| 容器注册表身份验证失败 | GitLab 令牌无效或过期 | 生成新的身份验证令牌 |
| SM_121a 架构无法识别 | 缺少 LLVM 补丁 | 验证应用于 LLVM 源的 SM_121a 补丁 |

## 在两个 Spark 上运行的常见问题
| 症状 | 原因 | 使固定 |
|---------|--------|-----|
| 节点 2 在 Ray 集群中不可见 | 网络连接问题 | 验证 QSFP 电缆连接，检查 IP 配置 |
| 无法访问 URL 的门禁存储库 | 某些 HuggingFace 模型的访问受到限制 | 重新生成你的 [HuggingFace token](https://huggingface.co/docs/hub/en/security-tokens);并请求在您的网络浏览器上访问 [gated model](https://huggingface.co/docs/hub/en/models-gated#customize-requested-information) |
| 模型下载失败 | 身份验证或网络问题 | 重新运行 `huggingface-cli login`，检查互联网访问情况 |
| 无法访问 URL 的门禁存储库 | 某些 HuggingFace 模型的访问受到限制 | 重新生成您的 HuggingFace 令牌；并请求在您的网络浏览器上访问门控模型 |
| CUDA 内存不足，405B | GPU显存不足 | 使用70B模型或减少max_model_len参数 |
| 容器启动失败 | 缺少 ARM64 映像 | 按照 ARM64 指令重建 vLLM 映像 |

> [!NOTE]
> DGX Spark 使用统一内存架构 (UMA)，可实现 GPU 和 CPU 之间的动态内存共享。
> 由于许多应用程序仍在更新以利用 UMA，因此即使在
> DGX Spark 的内存容量。如果发生这种情况，请使用以下命令手动刷新缓冲区缓存：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
