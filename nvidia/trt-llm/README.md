# 使用 TensorRT-LLM 进行推理

> 在 DGX Spark 上安装和使用 TensorRT-LLM

## 目录

- [概述](#overview)
- [单台 Spark](#single-spark)
- [在两台 Spark 上运行](#run-on-two-sparks)
  - [步骤 1. 配置网络连接](#step-1-configure-network-connectivity)
  - [步骤2.配置Docker权限](#step-2-configure-docker-permissions)
  - [步骤 3. 创建 OpenMPI 主机文件](#step-3-create-openmpi-hostfile)
  - [步骤 4. 在两个节点上启动容器](#step-4-start-containers-on-both-nodes)
  - [步骤 5. 验证容器正在运行](#step-5-verify-containers-are-running)
  - [步骤 6. 将主机文件复制到主容器](#step-6-copy-hostfile-to-primary-container)
  - [步骤 7. 保存容器引用](#step-7-save-container-reference)
  - [步骤8.生成配置文件](#step-8-generate-configuration-file)
  - [步骤9.下载模型](#step-9-download-model)
  - [步骤 10. 为模型提供服务](#step-10-serve-the-model)
  - [步骤 11. 验证 API 服务器](#step-11-validate-api-server)
  - [步骤 12. 清理和回滚](#step-12-cleanup-and-rollback)
  - [步骤 13. 后续步骤](#step-13-next-steps)
- [TensorRT-LLM 的 Open WebUI](#open-webui-for-TensorRT-LLM)
  - [步骤 1. 设置使用 Open WebUI 与 TRT-LLM 的先决条件](#step-1-set-up-the-prerequisites-to-use-open-webui-with-trt-llm)
  - [步骤 2. 启动 Open WebUI 容器](#step-2-launch-open-webui-container)
  - [步骤3.访问Open WebUI 界面](#step-3-access-the-open-webui-interface)
  - [步骤 4. 清理和回滚](#step-4-cleanup-and-rollback)
- [故障排查](#troubleshooting)

---

<a id="overview"></a>
## 概述

## 基本思路

**NVIDIA TensorRT-LLM (TRT-LLM)** 是一个开源库，用于优化和加速 NVIDIA GPU 上的大语言模型 (LLM) 推理。

它提供高效的内核、内存管理和并行策略（例如张量、管道和序列并行），因此开发人员可以以更低的延迟和更高的吞吐量为 LLM 提供服务。

TRT-LLM 与 Hugging Face 和 PyTorch 等框架集成，使得大规模部署最先进的模型变得更加容易。


## 你将完成什么

您将设置 TensorRT-LLM 以在 DGX Spark 上优化和部署大型语言模型，从而实现比标准 PyTorch 显着更高的吞吐量和更低的延迟
通过内核级优化、高效内存布局和高级量化进行推理。

## 开始之前需要了解什么

- 对 PyTorch 或类似 ML 框架的 Python 熟练程度和经验
- 运行 CLI 工具和 Docker 容器的命令行舒适性
- 对 GPU 概念的基本了解，包括 VRAM、批处理和量化 (FP16/INT8)
- 熟悉 NVIDIA 软件堆栈（CUDA 工具包、驱动程序）
- 具有推理服务器和容器化环境的经验

## 先决条件

- DGX Spark 设备
- 与 CUDA 12.x 兼容的 NVIDIA 驱动程序：`nvidia-smi`
- 安装了 Docker 并配置了 GPU 支持：`docker run --rm --gpus all nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13 nvidia-smi`
- 使用 Hugging Face 模型访问令牌：`echo $HF_TOKEN`
- 足够的 GPU VRAM（70B 模型建议使用 40GB+）
- 用于下载模型和容器镜像的互联网连接
- 网络：在主机上打开 TCP 端口 8355 (LLM) 和 8356 (VLM)，用于 OpenAI 兼容服务

## 附属文件

所有必需的资产都可以在 [GitHub](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main) 中找到

- [**trtllm-mn-entrypoint.sh**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/trt-llm/assets/trtllm-mn-entrypoint.sh) — 用于多节点设置的容器入口点脚本

## 模型支持矩阵

Spark 上的 TensorRT-LLM 支持以下模型。所有列出的模型均可供使用：

| 模型 | 量化 | 支持状态 | 模型标识 |
|-------|-------------|----------------|-----------|
| **Nemotron-3-Nano-Omni-30B-A3B-Reasoning** | BF16 | ✅ | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` |
| **Nemotron-3-Nano-Omni-30B-A3B-Reasoning** | FP8 | ✅ | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8` |
| **Nemotron-3-Nano-Omni-30B-A3B-Reasoning** | NVFP4 | ✅ | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` |
| **Nemotron-3-Super-120B** | NVFP4 | ✅ | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` |
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
| **Phi-4-multimodal-instruct** | FP8 | ✅ | `nvidia/Phi-4-multimodal-instruct-FP8` |
| **Phi-4-multimodal-instruct** | NVFP4 | ✅ | `nvidia/Phi-4-multimodal-instruct-FP4` |
| **Phi-4-reasoning+** | FP8 | ✅ | `nvidia/Phi-4-reasoning-plus-FP8` |
| **Phi-4-reasoning+** | NVFP4 | ✅ | `nvidia/Phi-4-reasoning-plus-FP4` |
| **Qwen3-30B-A3B** | NVFP4 | ✅ | `nvidia/Qwen3-30B-A3B-FP4` |
| **Llama-4-Scout-17B-16E-指示** | NVFP4 | ✅ | `nvidia/Llama-4-Scout-17B-16E-Instruct-FP4` |
| **Qwen3-235B-A22B（仅两个 Spark）** | NVFP4 | ✅ | `nvidia/Qwen3-235B-A22B-FP4` |

> [!NOTE]
> 您可以使用 NVFP4 量化文档为您喜欢的模型生成您自己的 NVFP4 量化检查点。这使您能够利用 NVFP4 量化的性能和内存优势，即使对于 NVIDIA 尚未发布的模型也是如此。

提醒：并非所有模型架构都支持 NVFP4 量化。

## 时间与风险

* **预计时间**：设置和 API 服务器部署需要 45-60 分钟
* **风险级别**：中 - 容器拉取和模型下载可能会因网络问题而失败
* **回滚**：停止推理服务器并删除下载的模型以释放资源。
* **最后更新：** 2026 年 4 月 28 日
  * Docker 镜像升级到 1.3.0rc13；模型矩阵新增 Nemotron Omni reasoning BF16、FP8、NVFP4

<a id="single-spark"></a>
## 单台 Spark

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

## 步骤 2. 验证环境先决条件

确认您的 Spark 设备具有下载所需的 GPU 访问权限和网络连接
模型和容器。

```bash
## Check GPU visibility and driver
nvidia-smi

## Verify Docker GPU support
docker run --rm --gpus all nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13 nvidia-smi

```

## 步骤3.设置环境变量

```bash
## Set `HF_TOKEN` for model access.
export HF_TOKEN=<your-huggingface-token>

export DOCKER_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13"
```

## 步骤 4. 验证 TensorRT-LLM 安装

确认GPU访问后，验证TensorRT-LLM是否可以导入到容器内。

```bash
docker run --rm -it --gpus all \
  $DOCKER_IMAGE \
  python -c "import tensorrt_llm; print(f'TensorRT-LLM version: {tensorrt_llm.__version__}')"
```

预期输出：
```
[TensorRT-LLM] TensorRT-LLM version: 1.3.0rc13
TensorRT-LLM version: 1.3.0rc13
```

## 步骤5.创建缓存目录

设置本地缓存以避免在后续运行时重新下载模型。

```bash
## Create Hugging Face cache directory
mkdir -p $HOME/.cache/huggingface/
```

## 步骤 6. 使用 faststart_advanced 验证设置

本快速入门通过测试模型加载、推理引擎初始化和真实文本生成的 GPU 执行来端到端验证您的 TensorRT-LLM 设置。这是在启动推理 API 服务器之前确认一切正常的最快方法。

**LLM 快速入门示例**

#### 骆驼 3.1 8B Instruct
```bash
export MODEL_HANDLE="nvidia/Llama-3.1-8B-Instruct-FP4"

docker run \
  -e MODEL_HANDLE=$MODEL_HANDLE \
  -e HF_TOKEN=$HF_TOKEN \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host --network host \
  $DOCKER_IMAGE \
  bash -c '
    hf download $MODEL_HANDLE && \
    python examples/llm-api/quickstart_advanced.py \
      --model_dir $MODEL_HANDLE \
      --prompt "Paris is great because" \
      --max_tokens 64
    '
```

#### GPT-OSS 20B
```bash
export MODEL_HANDLE="openai/gpt-oss-20b"

docker run \
  -e MODEL_HANDLE=$MODEL_HANDLE \
  -e HF_TOKEN=$HF_TOKEN \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host --network host \
  $DOCKER_IMAGE \
  bash -c '
    export TIKTOKEN_ENCODINGS_BASE="/tmp/harmony-reqs" && \
    mkdir -p $TIKTOKEN_ENCODINGS_BASE && \
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken && \
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken && \
    hf download $MODEL_HANDLE && \
    python examples/llm-api/quickstart_advanced.py \
      --model_dir $MODEL_HANDLE \
      --prompt "Paris is great because" \
      --max_tokens 64
    '
```

#### GPT-OSS 120B
```bash
export MODEL_HANDLE="openai/gpt-oss-120b"

docker run \
  -e MODEL_HANDLE=$MODEL_HANDLE \
  -e HF_TOKEN=$HF_TOKEN \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host --network host \
  $DOCKER_IMAGE \
  bash -c '
    export TIKTOKEN_ENCODINGS_BASE="/tmp/harmony-reqs" && \
    mkdir -p $TIKTOKEN_ENCODINGS_BASE && \
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken && \
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken && \
    hf download $MODEL_HANDLE && \
    python examples/llm-api/quickstart_advanced.py \
      --model_dir $MODEL_HANDLE \
      --prompt "Paris is great because" \
      --max_tokens 64
    '
```

## 步骤 7. 使用 Quickstart_multimodal 验证设置

**VLM 快速入门示例**

这通过运行图像理解推理来展示视觉语言模型的功能。该示例使用多模态输入来验证文本和视觉处理管道。

#### Phi-4-multimodal-instruct

该模型需要 LoRA（低秩适应）配置，因为它使用参数高效的微调。 `--load_lora` 标志允许加载 LoRA 权重，以适应多模态指令跟踪的基本模型。
```bash
export MODEL_HANDLE="nvidia/Phi-4-multimodal-instruct-FP4"

docker run \
  -e MODEL_HANDLE=$MODEL_HANDLE \
  -e HF_TOKEN=$HF_TOKEN \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host --network host \
  $DOCKER_IMAGE \
  bash -c '
  python3 examples/llm-api/quickstart_multimodal.py \
    --model_type phi4mm \
    --model_dir $MODEL_HANDLE \
    --modality image \
    --media "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png" \
    --prompt "What is happening in this image?" \
    --load_lora \
    --auto_model_name Phi4MMForCausalLM
  '
```


> [!NOTE]
> 如果您在下载或首次运行期间遇到主机 OOM，请释放主机上的操作系统页面缓存（容器外部）并重试：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

## 步骤 8. 使用兼容 OpenAI 的 API 为 LLM 提供服务

通过 trtllm-serve 使用兼容 OpenAI 的 API 提供服务：

#### Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16

此示例会写入 **`nano_v3.yaml`** 来配置 KV 缓存、MoE 与 CUDA graph，然后在端口 **8000** 启动 **`trtllm-serve`**，并启用 Nemotron Omni reasoning 解析器。

```bash
export MODEL_HANDLE="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"

docker run --name trtllm_llm_server --rm -it --gpus all --ipc host --network host \
  -e HF_TOKEN=$HF_TOKEN \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \
  $DOCKER_IMAGE \
  bash -c '
    hf download $MODEL_HANDLE && \
    cat > nano_v3.yaml <<EOF
kv_cache_config:
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.80
  mamba_ssm_cache_dtype: float32
moe_config:
  backend: CUTLASS
cuda_graph_config:
  enable_padding: true
  max_batch_size: 1
max_batch_size: 1
EOF
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    trtllm-serve serve "$MODEL_HANDLE" \
      --host 0.0.0.0 \
      --port 8355 \
      --trust_remote_code \
      --reasoning_parser nano-v3 \
      --tool_parser qwen3_coder \
      --extra_llm_api_options nano_v3.yaml
  '
```

#### 骆驼 3.1 8B Instruct
```bash
export MODEL_HANDLE="nvidia/Llama-3.1-8B-Instruct-FP4"

docker run --name trtllm_llm_server --rm -it --gpus all --ipc host --network host \
  -e HF_TOKEN=$HF_TOKEN \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \
  $DOCKER_IMAGE \
  bash -c '
    hf download $MODEL_HANDLE && \
    cat > /tmp/extra-llm-api-config.yml <<EOF
print_iter_log: false
kv_cache_config:
  dtype: "auto"
  free_gpu_memory_fraction: 0.9
cuda_graph_config:
  enable_padding: true
disable_overlap_scheduler: true
EOF
    trtllm-serve "$MODEL_HANDLE" \
      --max_batch_size 64 \
      --trust_remote_code \
      --port 8355 \
      --extra_llm_api_options /tmp/extra-llm-api-config.yml
  '
```

#### GPT-OSS 20B
```bash
export MODEL_HANDLE="openai/gpt-oss-20b"

docker run --name trtllm_llm_server --rm -it --gpus all --ipc host --network host \
  -e HF_TOKEN=$HF_TOKEN \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \
  $DOCKER_IMAGE \
  bash -c '
    export TIKTOKEN_ENCODINGS_BASE="/tmp/harmony-reqs" && \
    mkdir -p $TIKTOKEN_ENCODINGS_BASE && \
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken && \
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken && \
    hf download $MODEL_HANDLE && \
    cat > /tmp/extra-llm-api-config.yml <<EOF
print_iter_log: false
kv_cache_config:
  dtype: "auto"
  free_gpu_memory_fraction: 0.9
cuda_graph_config:
  enable_padding: true
disable_overlap_scheduler: true
EOF
    trtllm-serve "$MODEL_HANDLE" \
      --max_batch_size 64 \
      --trust_remote_code \
      --port 8355 \
      --extra_llm_api_options /tmp/extra-llm-api-config.yml
  '
```

最小的 OpenAI 风格的聊天请求。从单独的终端运行它。

```bash
curl -s http://localhost:8355/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL_HANDLE"'",
    "messages": [{"role": "user", "content": "Paris is great because"}],
    "max_tokens": 64
  }'
```

## 步骤 9. 清理和回滚

测试完成后，删除下载的模型和容器以释放空间。

> [!WARNING]
> 这将删除所有缓存的模型，并且可能需要重新下载以供将来运行。

```bash
## Remove Hugging Face cache
sudo chown -R "$USER:$USER" "$HOME/.cache/huggingface"
rm -rf $HOME/.cache/huggingface/

## Clean up Docker images
docker image prune -f
docker rmi $DOCKER_IMAGE
```

<a id="run-on-two-sparks"></a>
## 在两台 Spark 上运行

<a id="step-1-configure-network-connectivity"></a>
### 步骤 1. 配置网络连接

按照 [Connect two Sparks](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks) 手册中的网络设置说明在 DGX Spark 节点之间建立连接。

这包括：
- 物理 QSFP 电缆连接
- 网络接口配置（自动或手动IP分配）
- 无密码 SSH 设置
- 网络连接验证

<a id="step-2-configure-docker-permissions"></a>
### 步骤2.配置Docker权限

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

在两个节点上重复此步骤。

<a id="step-3-create-openmpi-hostfile"></a>
### 步骤 3. 创建 OpenMPI 主机文件

使用两个节点的 IP 地址创建主机文件以进行 MPI 操作。在每个节点上，获取网络接口的 IP 地址：

```bash
ip a show enp1s0f0np0
```

或者，如果您使用第二个界面：

```bash
ip a show enp1s0f1np1
```

查找 `inet` 行以查找 IP 地址（例如 `192.168.1.10/24`）。

在主节点上，使用收集的 IP 创建主机文件 `~/openmpi-hostfile`：

```bash
cat > ~/openmpi-hostfile <<EOF
192.168.1.10
192.168.1.11
EOF
```

将 IP 地址替换为您的实际节点 IP。

<a id="step-4-start-containers-on-both-nodes"></a>
### 步骤 4. 在两个节点上启动容器

在**每个节点**（主节点和工作节点）上，运行以下命令来启动 TRT-LLM 容器：

```bash
  docker run -d --rm \
  --name trtllm-multinode \
  --gpus '"device=all"' \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --device /dev/infiniband:/dev/infiniband \
  -e UCX_NET_DEVICES="enp1s0f0np0,enp1s0f1np1" \
  -e NCCL_SOCKET_IFNAME="enp1s0f0np0,enp1s0f1np1" \
  -e OMPI_MCA_btl_tcp_if_include="enp1s0f0np0,enp1s0f1np1" \
  -e OMPI_MCA_orte_default_hostfile="/etc/openmpi-hostfile" \
  -e OMPI_MCA_rmaps_ppr_n_pernode="1" \
  -e OMPI_ALLOW_RUN_AS_ROOT="1" \
  -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1" \
  -e CPATH=/usr/local/cuda/include \
  -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
  -v ~/.cache/huggingface/:/root/.cache/huggingface/ \
  -v ~/.ssh:/tmp/.ssh:ro \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc5 \
  sh -c "curl https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/trt-llm/assets/trtllm-mn-entrypoint.sh | sh"
```

> [!NOTE]
> 确保在主节点和工作节点上都运行此命令。

<a id="step-5-verify-containers-are-running"></a>
### 步骤 5. 验证容器正在运行

在每个节点上，验证容器正在运行：

```bash
docker ps
```

您应该看到类似于以下内容的输出：

```
CONTAINER ID   IMAGE                                                 COMMAND                  CREATED          STATUS          PORTS     NAMES
abc123def456   nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc5         "sh -c 'curl https:…"    10 seconds ago   Up 8 seconds              trtllm-multinode
```

<a id="step-6-copy-hostfile-to-primary-container"></a>
### 步骤 6. 将主机文件复制到主容器

在主节点上，将 OpenMPI 主机文件复制到容器中：

```bash
docker cp ~/openmpi-hostfile trtllm-multinode:/etc/openmpi-hostfile
```

<a id="step-7-save-container-reference"></a>
### 步骤 7. 保存容器引用

在主节点上，为了方便起见，将容器名称保存在变量中：

```bash
export TRTLLM_MN_CONTAINER=trtllm-multinode
```

<a id="step-8-generate-configuration-file"></a>
### 步骤8.生成配置文件

在主节点上，在容器内生成配置文件：

```bash
docker exec $TRTLLM_MN_CONTAINER bash -c 'cat <<EOF > /tmp/extra-llm-api-config.yml
print_iter_log: false
kv_cache_config:
  dtype: "auto"
  free_gpu_memory_fraction: 0.9
cuda_graph_config:
  enable_padding: true
EOF'
```

<a id="step-9-download-model"></a>
### 步骤9.下载模型

我们可以使用以下命令下载模型。您可以将 `nvidia/Qwen3-235B-A22B-FP4` 替换为您选择的模型。

```bash
## Need to specify huggingface token for model download.
export HF_TOKEN=<your-huggingface-token>

docker exec \
  -e MODEL="nvidia/Qwen3-235B-A22B-FP4" \
  -e HF_TOKEN=$HF_TOKEN \
  -it $TRTLLM_MN_CONTAINER bash -c 'mpirun -x HF_TOKEN bash -c "hf download $MODEL"'
```

<a id="step-10-serve-the-model"></a>
### 步骤 10. 为模型提供服务

在主节点上，启动 TensorRT-LLM 服务器：

```bash
docker exec \
  -e MODEL="nvidia/Qwen3-235B-A22B-FP4" \
  -e HF_TOKEN=$HF_TOKEN \
  -it $TRTLLM_MN_CONTAINER bash -c '
    mpirun -x HF_TOKEN trtllm-llmapi-launch trtllm-serve $MODEL \
      --tp_size 2 \
      --backend pytorch \
      --max_num_tokens 32768 \
      --max_batch_size 4 \
      --extra_llm_api_options /tmp/extra-llm-api-config.yml \
      --port 8355'
```

这将在端口 8355 上启动 TensorRT-LLM 服务器。然后您可以使用 OpenAI 兼容的 API 格式向 `http://localhost:8355` 发出推理请求。

> [!NOTE]
> 您可能会看到诸如 `UCX  WARN  network device 'enp1s0f0np0' is not available, please use one or more of` 之类的警告。如果推理成功，您可以忽略此警告，因为它与仅使用两个 CX-7 端口之一有关，而另一个未使用。

**预期输出：** 服务器启动日志和就绪消息。

<a id="step-11-validate-api-server"></a>
### 步骤 11. 验证 API 服务器

服务器运行后，您可以使用 CURL 请求对其进行测试。在主节点上运行：

```bash
curl -s http://localhost:8355/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Qwen3-235B-A22B-FP4",
    "messages": [{"role": "user", "content": "Paris is great because"}],
    "max_tokens": 64
  }'
```

**预期输出：** JSON 响应以及生成的文本完成。

<a id="step-12-cleanup-and-rollback"></a>
### 步骤 12. 清理和回滚

停止并删除**每个节点**上的容器。通过 SSH 连接到每个节点并运行：

```bash
docker stop trtllm-multinode
```

> [!WARNING]
> 这将删除所有推理数据和性能报告。如果需要，请在清理之前复制任何必要的文件。

删除下载的模型以释放每个节点上的磁盘空间：

```bash
rm -rf $HOME/.cache/huggingface/hub/models--nvidia--Qwen3*
```

<a id="step-13-next-steps"></a>
### 步骤 13. 后续步骤

您现在可以在 DGX Spark 集群上部署其他模型。

<a id="open-webui-for-TensorRT-LLM"></a>
## TensorRT-LLM 的 Open WebUI

<a id="step-1-set-up-the-prerequisites-to-use-open-webui-with-trt-llm"></a>
### 步骤 1. 设置使用 Open WebUI 与 TRT-LLM 的先决条件

在单节点或多节点配置中设置 TensorRT-LLM 推理服务器后，
您可以部署 Open WebUI 以通过 Open WebUI 与您的模型进行交互。要进行设置，只需确保以下内容
已就绪：

- TensorRT-LLM 推理服务器在 http://localhost:8355 运行并可访问
- Docker 安装并配置（请参阅前面的步骤）
- DGX Spark 上可用的端口 3000

<a id="step-2-launch-open-webui-container"></a>
### 步骤 2. 启动 Open WebUI 容器

在运行 TensorRT-LLM 推理服务器的 DGX Spark 节点上运行以下命令。
对于多节点设置，这将是主节点。

> [!NOTE]
> 如果您为兼容 OpenAI 的 API 服务器使用了不同的端口，请调整 `OPENAI_API_BASE_URL="http://localhost:8355/v1"` 以匹配 TensorRT-LLM 推理服务器的 IP 和端口。

```bash
docker run \
  -d \
  -e OPENAI_API_BASE_URL="http://localhost:8355/v1" \
  -v open-webui:/app/backend/data \
  --network host \
  --add-host=host.docker.internal:host-gateway \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

这个命令：
- 连接到位于 http://localhost:8355 的 TensorRT-LLM 的 OpenAI 兼容 API 服务器
- 提供对位于 http://localhost:8080 的 Open WebUI 界面的访问
- 将聊天数据保留在 Docker 卷中
- 启用容器自动重启
- 使用最新的 Open WebUI 镜像

<a id="step-3-access-the-open-webui-interface"></a>
### 步骤3.访问Open WebUI 界面

打开您的网络浏览器并导航至：

```
http://localhost:8080
```

您应该在 http://localhost:8080 处看到 Open WebUI 界面，您可以在其中：
- 与您部署的模型聊天
- 调整模型参数
- 查看聊天记录
- 管理模型配置

您可以从左上角的下拉菜单中选择您的模型。这就是开始在部署的模型中使用 Open WebUI 所需要做的全部工作。

> [!NOTE]
> 如果从远程计算机访问，请将 localhost 替换为 DGX Spark 的 IP 地址。

<a id="step-4-cleanup-and-rollback"></a>
### 步骤 4. 清理和回滚
> [!WARNING]
> 这会删除所有聊天数据，并且可能需要重新上传以供将来运行。

使用以下命令删除容器：
```bash
docker stop open-webui
docker rm open-webui
docker volume rm open-webui
docker rmi ghcr.io/open-webui/open-webui:main
```

<a id="troubleshooting"></a>
## 故障排查
## 在单个 Spark 上运行的常见问题

| 症状 | 原因 | 使固定 |
|---------|-------|-----|
| 无法访问 URL 的门禁仓库 | 某些 Hugging Face 模型的访问受到限制 | 重新生成你的 [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens);并请求在您的网络浏览器上访问 [gated model](https://huggingface.co/docs/hub/en/models-gated#customize-requested-information) |
| 重量加载期间出现 OOM（例如 [Nemotron Super 49B](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5)） | 并行权重加载内存压力 | `export TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL=1` |
| “CUDA内存不足” | GPU VRAM 不足以支持模型 | 减少 `free_gpu_memory_fraction: 0.9` 或批量大小或使用较小的模型 |
| “找不到模型”错误 | HF_TOKEN 无效或模型无法访问 | 验证令牌和模型权限 |
| 容器拉取超时 | 网络连接问题 | 重试拉取或使用本地镜像 |
| 导入tensorrt_llm失败 | 容器运行时问题 | 重新启动 Docker 守护进程并重试 |

## 在两台 Spark 上运行的常见问题

| 症状 | 原因 | 使固定 |
|---------|-------|-----|
| MPI 主机名测试返回单个主机名 | 网络连接问题 | 验证两个节点的 IP 地址均可访问 |
| Hugging Face 下载时显示“权限被拒绝” | HF_TOKEN 无效或缺失 | 设置有效令牌：`export HF_TOKEN=<TOKEN>` |
| 无法访问 URL 的门禁仓库 | 某些 Hugging Face 模型的访问受到限制 | 重新生成你的 [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens);并请求在您的网络浏览器上访问 [gated model](https://huggingface.co/docs/hub/en/models-gated#customize-requested-information) |
| “CUDA 内存不足”错误 | GPU显存不足 | 减少 `--max_batch_size` 或 `--max_num_tokens` |
| 容器立即退出 | 缺少入口点脚本 | 确保 `trtllm-mn-entrypoint.sh` 下载成功并具有可执行权限，还要确保您没有在节点上运行容器。如果端口 2233 已被使用，则入口点脚本将不会启动。 |
| 来自守护程序的错误响应：验证根 CA 证书时出错 | 系统时钟不同步或证书过期 | 更新系统时间以与 NTP 服务器 `sudo timedatectl set-ntp true` 同步|
| “类型‘绑定’的安装配置无效” | 入口点脚本丢失或不可执行 | 运行 `docker inspect <container_id>` 查看完整的错误消息。验证 `trtllm-mn-entrypoint.sh` 存在于主目录 (`ls -la $HOME/trtllm-mn-entrypoint.sh`) 的两个节点上并且具有可执行权限 (`chmod +x $HOME/trtllm-mn-entrypoint.sh`) |
| “任务：非零退出（255）” | 容器退出，错误代码为 255 | 使用 `docker ps -a --filter "name=trtllm-multinode"` 检查容器日志以获取容器 ID，然后使用 `docker logs <container_id>` 查看详细的错误消息 |
| Docker 状态停留在“待处理”状态，并显示“没有合适的节点（不足...）” | Docker 守护进程未正确配置 GPU 访问 | 验证步骤 2-4 是否已成功完成，并检查 `/etc/docker/daemon.json` 是否包含正确的 GPU 配置 |
| 服务模型失败 `ptxas fatal` 错误 | 模型需要运行时 triton 内核编译 | 在步骤 10 中，将 `-x TRITON_PTXAS_PATH` 添加到 `mpirun` 命令中 |

> [!NOTE]
> DGX Spark 使用统一内存架构 (UMA)，可实现 GPU 和 CPU 之间的动态内存共享。
> 由于许多应用程序仍在更新以利用 UMA，因此即使在
> DGX Spark 的内存容量。如果发生这种情况，请使用以下命令手动刷新缓冲区缓存：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
