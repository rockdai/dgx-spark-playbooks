# NVFP4 量化

> 使用 TensorRT 模型优化器将模型量化为 NVFP4 以在 Spark 上运行

## 目录

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思路

NVFP4 是 NVIDIA Blackwell GPU 引入的 4 位浮点格式，用于保持模型准确性，同时降低推理工作负载的内存带宽和存储要求。 
与统一的 INT4 量化不同，NVFP4 保留了具有共享指数和紧凑尾数的浮点语义，从而允许更高的动态范围和更稳定的收敛。
NVIDIA Blackwell Tensor Core 本身支持 FP16、FP8 和 FP4 的混合精度执行，使模型能够使用 FP4 进行权重和激活，同时以更高的精度（通常为 FP16）进行累积。 
该设计最大限度地减少了矩阵乘法期间的量化误差，并支持 TensorRT-LLM 中的高效转换管道，以实现逐层微调的量化。

直接的好处是：
  - 与 FP16 相比，内存使用量减少约 3.5 倍，与 FP8 相比减少约 1.8 倍
  - 保持接近 FP8 的精度（通常损失 <1%）
  - 提高推理速度和能源效率


## 你将完成什么

您将使用 NVIDIA 的 TensorRT 模型优化器量化 DeepSeek-R1-Distill-Llama-8B 模型
在 TensorRT-LLM 容器内，生成 NVFP4 量化模型以部署在 NVIDIA DGX Spark 上。

这些示例使用 NVIDIA FP4 量化模型，通过降低模型层的精度，有助于将模型大小减少约 2 倍。
这种量化方法旨在保持准确性，同时显着提高吞吐量。但是，请务必注意，量化可能会影响模型的准确性 - 我们建议运行评估来验证量化模型是否为您的用例保持可接受的性能。

## 开始之前需要了解什么

- 使用 Docker 容器和 GPU 加速的工作负载
- 了解模型量化概念及其对推理性能的影响
- 拥有 NVIDIA TensorRT 和 CUDA 工具包环境的经验
- 熟悉 Hugging Face 模型存储库和身份验证

## 先决条件

- 配备 Blackwell 架构 GPU 的 NVIDIA Spark 设备
- Docker 安装并支持 GPU
- 已配置 NVIDIA 容器工具包
- 模型文件和输出的可用存储
- 可以访问目标模型的 Hugging Face 帐户

验证您的设置：
```bash
## Check Docker GPU access
docker run --rm --gpus all nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev nvidia-smi

## Verify sufficient disk space
df -h .
```

## 时间与风险

* **预计持续时间**：45-90 分钟，具体取决于网络速度和模型大小
* **风险**：
  * 由于网络问题或Hugging Face认证问题，模型下载可能会失败
  * 量化过程需要占用大量内存，在 GPU 内存不足的系统上可能会失败
  * 输出文件很大（几GB）并且需要足够的存储空间
* **回滚**：删除输出目录和任何拉取的 Docker 映像以恢复原始状态。
* **最后更新**：2025 年 12 月 15 日
  * 修复第 8 步中损坏的客户端 CURL 请求
  * 更新 ModelOptimizer 项目名称

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

## 步骤2.准备环境

创建一个本地输出目录，用于存储量化模型文件。该目录将被挂载到容器中，以在容器退出后保留结果。

```bash
mkdir -p ./output_models
chmod 755 ./output_models
```

## 步骤3.抱脸认证

通过设置 Hugging Face 身份验证令牌，确保您有权访问 DeepSeek 模型。

```bash
## Export your Hugging Face token as an environment variable
## Get your token from: https://huggingface.co/settings/tokens
export HF_TOKEN="your_token_here"
```

容器将自动使用该令牌进行模型下载。

## 步骤 4. 运行 TensorRT 模型优化器容器

启动具有 GPU 访问、针对多 GPU 工作负载优化的 IPC 设置以及用于模型缓存和输出持久性的卷安装的 TensorRT-LLM 容器。

```bash
docker run --rm -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "./output_models:/workspace/output_models" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN=$HF_TOKEN \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  bash -c "
    git clone -b 0.35.0 --single-branch https://github.com/NVIDIA/Model-Optimizer.git /app/TensorRT-Model-Optimizer && \
    cd /app/TensorRT-Model-Optimizer && pip install -e '.[dev]' && \
    export ROOT_SAVE_PATH='/workspace/output_models' && \
    /app/TensorRT-Model-Optimizer/examples/llm_ptq/scripts/huggingface_example.sh \
    --model 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B' \
    --quant nvfp4 \
    --tp 1 \
    --export_fmt hf
  "
```

注意：您可能会遇到这个`pynvml.NVMLError_NotSupported: Not Supported`。这在某些环境中是预期的，不会影响结果，并将在即将发布的版本中修复。
注意：请注意，如果您的模型太大，您可能会遇到内存不足错误。您可以尝试量化较小的模型。

这个命令：
- 使用完整的 GPU 访问权限和优化的共享内存设置来运行容器
- 挂载输出目录以保留量化模型文件
- 安装您的 Hugging Face 缓存以避免重新下载模型
- 从源克隆并安装 TensorRT 模型优化器
- 使用 NVFP4 量化参数执行量化脚本

## 步骤 5. 监控量化过程

量化过程将显示进度信息，包括：
- Hugging Face 的模型下载进度
- 量化校准步骤
- 模型导出和验证阶段

## 步骤 6. 验证量化模型

容器完成后，验证量化模型文件是否已成功创建。

```bash
## Check output directory contents
ls -la ./output_models/

## Verify model files are present
find ./output_models/ -name "*.bin" -o -name "*.safetensors" -o -name "config.json"
```

您应该在输出目录中看到模型权重文件、配置文件和分词器文件。

## 步骤 7. 测试模型加载

首先，设置量化模型的路径：

```bash
## Set path to quantized model directory
export MODEL_PATH="./output_models/saved_models_DeepSeek-R1-Distill-Llama-8B_nvfp4_hf/"
```

现在使用简单的测试验证量化模型是否可以正确加载：

```bash
docker run \
  -e HF_TOKEN=$HF_TOKEN \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \
  -v "$MODEL_PATH:/workspace/model" \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host --network host \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  bash -c '
    python examples/llm-api/quickstart_advanced.py \
      --model_dir /workspace/model/ \
      --prompt "Paris is great because" \
      --max_tokens 64
    '
```

## 步骤 8. 使用兼容 OpenAI 的 API 为模型提供服务
使用量化模型启动 TensorRT-LLM OpenAI 兼容 API 服务器。
首先，设置量化模型的路径：

```bash
## Set path to quantized model directory
export MODEL_PATH="./output_models/saved_models_DeepSeek-R1-Distill-Llama-8B_nvfp4_hf/"

docker run \
  -e HF_TOKEN=$HF_TOKEN \
  -v "$MODEL_PATH:/workspace/model" \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host --network host \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  trtllm-serve /workspace/model \
    --backend pytorch \
    --max_batch_size 4 \
    --port 8000
```

运行以下命令以使用客户端 CURL 请求测试服务器：

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [{"role": "user", "content": "What is artificial intelligence?"}],
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": false
  }'
```

## 步骤 10. 清理和回滚

要清理环境并删除生成的文件：

> [！警告]
> 这将永久删除所有量化模型文件和缓存数据。

```bash
## Remove output directory and all quantized models
rm -rf ./output_models

## Remove Hugging Face cache (optional)
rm -rf ~/.cache/huggingface

## Remove Docker image (optional)
docker rmi nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev
```

## 步骤 11. 后续步骤

量化模型现已准备好部署。常见的后续步骤包括：
- 与原始模型相比，对推理性能进行基准测试。
- 将量化模型集成到您的推理管道中。
- 部署到 NVIDIA Triton 推理服务器以提供生产服务。
- 对您的特定用例运行额外的验证测试。

## 故障排除

| 症状 | 原因 | 使固定 |
|---------|--------|-----|
| 访问拥抱脸时“权限被拒绝” | HF 令牌缺失或无效 | 使用有效令牌运行 `huggingface-cli login` |
| 容器因 CUDA 内存不足而退出 | GPU显存不足 | 减少批量大小或使用具有更多 GPU 内存的机器 |
| 在输出目录中找不到模型文件 | 卷安装失败或路径错误 | 验证 `$(pwd)/output_models` 正确解析 |
| Git 克隆在容器内失败 | 网络连接问题 | 检查互联网连接并重试 |
| 量化过程挂起 | 容器资源限制 | 增加 Docker 内存限制或使用 `--ulimit` 标志 |
| 无法访问 URL 的门禁存储库 | 某些 HuggingFace 模型的访问受到限制 | 重新生成你的 [HuggingFace token](https://huggingface.co/docs/hub/en/security-tokens);并请求在您的网络浏览器上访问 [gated model](https://huggingface.co/docs/hub/en/models-gated#customize-requested-information) |

> [！笔记]
> DGX Spark 使用统一内存架构 (UMA)，可实现 GPU 和 CPU 之间的动态内存共享。 
> 由于许多应用程序仍在更新以利用 UMA，因此即使在 
> DGX Spark 的内存容量。如果发生这种情况，请使用以下命令手动刷新缓冲区缓存：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
