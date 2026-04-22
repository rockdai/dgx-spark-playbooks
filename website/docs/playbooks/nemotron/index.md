---
id: nemotron
title: nemotron
sidebar_label: nemotron
---

# Nemotron-3-Nano 与 llama.cpp

> 在 DGX Spark 上使用 llama.cpp 运行 Nemotron-3-Nano-30B 模型

## 目录

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思路

Nemotron-3-Nano-30B-A3B 是 NVIDIA 强大的语言模型，具有 300 亿个参数的专家混合 (MoE) 架构，且仅有 30 亿个活动参数。这种高效的设计能够以较低的计算要求实现高质量的推理，使其成为 DGX Spark 的 GB10 GPU 的理想选择。

本手册演示了如何使用 llama.cpp 运行 Nemotron-3-Nano，它会在构建时专门针对您的 GPU 架构编译 CUDA 内核。该模型包括内置推理（思维模式）和通过聊天模板调用工具的支持。

## 你将完成什么

您将拥有在 DGX Spark 上运行的功能齐全的 Nemotron-3-Nano-30B-A3B 推理服务器，可通过 OpenAI 兼容的 API 进行访问。此设置可以：

- 本地法学硕士推理
- 兼容 OpenAI 的 API 端点，可轻松与现有工具集成
- 内置推理和工具调用功能

## 开始之前需要了解什么

- 基本熟悉 Linux 命令行和终端命令
- 了解 git 并使用分支
- 使用 CMake 从源代码构建软件的经验
- 用于测试的 REST API 和 cURL 的基本知识
- 熟悉 Hugging Face Hub 进行模型下载

## 先决条件

**硬件要求：**
- 配备 GB10 GPU 的 NVIDIA DGX Spark
- 至少 40GB 可用 GPU 内存（模型使用 ~38GB VRAM）
- 至少 50GB 可用存储空间用于模型下载和构建工件

**软件要求：**
- NVIDIA DGX 操作系统
- git：`git --version`
- CMake（3.14+）：`cmake --version`
- CUDA 工具包：`nvcc --version`
- 网络访问 GitHub 和 Hugging Face

## 时间与风险

* **预计时间：** 30 分钟（包括约 38GB 的​​模型下载）
* **风险级别：**低
  * 构建过程从源代码编译但不修改系统文件
  * 如果模型下载中断，可以恢复
* **回滚：**删除克隆的`llama.cpp`目录和下载的模型文件以完全删除安装
* **最后更新：** 2025 年 12 月 17 日
  * 首次出版

## 指示

## 步骤 1. 验证先决条件

在继续之前，请确保您已在 DGX Spark 上安装了所需的工具。

```bash
git --version
cmake --version
nvcc --version
```

所有命令都应返回版本信息。如果缺少任何内容，请在继续之前安装它们。

安装 Hugging Face CLI：

```bash
python3 -m venv nemotron-venv
source nemotron-venv/bin/activate
pip install -U "huggingface_hub[cli]"
```

验证安装：

```bash
hf version
```

## 步骤 2. 克隆 llama.cpp 存储库

克隆 llama.cpp 存储库，它提供了用于运行 Nemotron 模型的推理框架。

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

## 步骤 3. 使用 CUDA 支持构建 llama.cpp

启用 CUDA 并针对 GB10 的 sm_121 计算架构构建 llama.cpp。这会编译专门针对您的 DGX Spark GPU 优化的 CUDA 内核。

```bash
mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="121" -DLLAMA_CURL=OFF
make -j8
```

构建过程大约需要 5-10 分钟。您应该看到编译进度并最终看到成功的构建消息。

## 步骤 4. 下载 Nemotron GGUF 模型

从 Hugging Face 下载 Q8 量化 GGUF 模型。该型号提供卓越的品质，同时适合 GB10 的内存容量。

```bash
hf download unsloth/Nemotron-3-Nano-30B-A3B-GGUF \
  Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf \
  --local-dir ~/models/nemotron3-gguf
```

下载大约 38GB。如果中断，可以继续下载。

## 步骤5.启动llama.cpp服务器

使用 Nemotron 模型启动推理服务器。服务器提供与 OpenAI 兼容的 API 端点。

```bash
./bin/llama-server \
  --model ~/models/nemotron3-gguf/Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf \
  --host 0.0.0.0 \
  --port 30000 \
  --n-gpu-layers 99 \
  --ctx-size 8192 \
  --threads 8
```

**参数说明：**
- `--host 0.0.0.0`：监听所有网络接口
- `--port 30000`：API服务器端口
- `--n-gpu-layers 99`：将所有层卸载到 GPU
- `--ctx-size 8192`：上下文窗口大小（最多可增加1M）
- `--threads 8`：用于非 GPU 操作的 CPU 线程

您应该看到服务器启动消息，指示模型已加载并准备就绪：
```
llama_new_context_with_model: n_ctx = 8192
...
main: server is listening on 0.0.0.0:30000
```

## 步骤 6. 测试 API

打开新终端并使用 OpenAI 兼容的聊天完成端点测试推理服务器。

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemotron",
    "messages": [{"role": "user", "content": "New York is a great city because..."}],
    "max_tokens": 100
  }'
```

预期响应格式：
```json
{
  "choices": [
    {
      "finish_reason": "length",
      "index": 0,
      "message": {
        "role": "assistant",
        "reasoning_content": "We need to respond to user statement: \"New York is a great city because...\". Probably they want continuation, maybe a discussion. It's a simple open-ended prompt. Provide reasons why New York is great. No policy issues. Just respond creatively.",
        "content": "New York is a great city because it's a living, breathing collage of cultures, ideas, and possibilities—all stacked into one vibrant, never‑sleeping metropolis. Here are just a few reasons that many people ("
      }
    }
  ],
  "created": 1765916539,
  "model": "Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 100,
    "prompt_tokens": 25,
    "total_tokens": 125
  },
  "id": "chatcmpl-...",
  "timings": {
    ...
  }
}
```

## 步骤7.测试推理能力

Nemotron-3-Nano 包含内置推理功能。使用更复杂的提示进行测试：

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemotron",
    "messages": [{"role": "user", "content": "Solve this step by step: If a train travels 120 miles in 2 hours, what is its average speed?"}],
    "max_tokens": 500
  }'
```

模型在给出最终答案之前会提供详细的推理链。

## 步骤 8. 清理

要停止服务器，请在运行服务器的终端中按 `Ctrl+C`。

要完全删除安装：

```bash
## Remove llama.cpp build
rm -rf ~/llama.cpp

## Remove downloaded models
rm -rf ~/models/nemotron3-gguf
```

## 步骤 9. 后续步骤

1. **增加上下文大小**：对于较长的对话，请将 `--ctx-size` 增加到 1048576（1M 令牌），但这会使用更多内存
3. **与应用程序集成**：将 OpenAI 兼容的 API 与 Open WebUI、Continue.dev 或自定义应用程序等工具结合使用

该服务器支持完整的 OpenAI API 规范，包括流响应、函数调用和多轮对话。

## 故障排除

| 症状 | 原因 | 使固定 |
|---------|-------|-----|
| `cmake` 失败并显示“未找到 CUDA” | CUDA 工具包不在 PATH 中 | 运行 `export PATH=/usr/local/cuda/bin:$PATH` 并重试 |
| 模型下载失败或中断 | 网络问题 | 重新运行 `hf download` 命令 - 它将从停止的地方恢复 |
| 启动服务器时出现“CUDA 内存不足” | GPU显存不足 | 将 `--ctx-size` 减少到 4096 或使用较小的量化 (Q4_K_M) |
| 服务器启动但推理速度慢 | 模型未完全加载到 GPU | 验证 `--n-gpu-layers 99` 已设置并检查 `nvidia-smi` 显示 GPU 使用情况 |
| 端口 30000 上“连接被拒绝” | 服务器未运行或端口错误 | 验证服务器正在运行并检查 `--port` 参数 |
| API 响应中“未找到模型” | 模型路径错误 | 验证 `--model` 参数中的模型路径与下载的文件位置匹配 |


> [！笔记] 
> DGX Spark 使用统一内存架构 (UMA)，可实现 GPU 和 CPU 之间的动态内存共享。 
> 由于许多应用程序仍在更新以利用 UMA，因此即使在 
> DGX Spark 的内存容量。如果发生这种情况，请使用以下命令手动刷新缓冲区缓存：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

有关最新的已知问题，请查看 [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)。
