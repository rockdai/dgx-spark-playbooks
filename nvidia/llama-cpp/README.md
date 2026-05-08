# 在 DGX Spark 上使用 llama.cpp 运行模型

> 使用 CUDA 构建 llama.cpp 并通过 OpenAI 兼容的 API 提供模型（以 Nemotron 3 Nano Omni 为例）


## 目录

- [概述](#overview)
- [操作步骤](#instructions)
- [故障排查](#troubleshooting)

---

<a id="overview"></a>
## 概述

## 基本思路

[llama.cpp](https://github.com/ggml-org/llama.cpp) 是用于大型语言模型的轻量级 C/C++ 推理堆栈。您可以使用 CUDA 构建它，以便张量工作在 DGX Spark GB10 GPU 上运行，然后加载 GGUF 权重并通过 `llama-server` 的 OpenAI 兼容 HTTP API 公开聊天。

本剧本以 **Nemotron 3 Nano Omni**（NVIDIA 的 MoE 系列，能在 Spark 上以量化 GGUF 良好运行）作为实战示例，从头到尾地遍历该堆栈。所有受支持模型的检查点选择和路径都汇总在下面的矩阵中；命令位于操作步骤中。

## 你将完成什么

您将使用 GB10 的 CUDA 构建 llama.cpp，下载 **Nemotron 3 Nano Omni** 示例检查点，并使用 GPU 卸载运行 **`llama-server`**。你得到：

- 通过 llama.cpp 进行本地推理（无需单独的 Python 推理框架）
- 用于工具和应用程序的 OpenAI 兼容 `/v1/chat/completions` 端点
- **Nemotron 3 Nano Omni** 示例在 DGX Spark 的该堆栈上运行的具体验证

## 开始之前需要了解什么

- 基本熟悉 Linux 命令行和终端命令
- 了解 git 并使用 CMake 从源代码构建
- 用于测试的 REST API 和 cURL 的基本知识
- 熟悉使用 Hugging Face Hub 下载 GGUF 文件

## 先决条件

**硬件要求**

- 配备 GB10 GPU 的 NVIDIA DGX Spark
- 为示例 **Q8_0** 检查点提供足够的统一内存（权重约为 **~35GB**，加上 KV 缓存和运行时开销——如选择更大的量化或更长上下文则需扩容）
- 至少 **~40GB** 可用磁盘用于示例下载和构建工件（如果保留多个 GGUF 则需要更多）

**软件要求**

- NVIDIA DGX 操作系统
- git：`git --version`
- CMake（3.14+）：`cmake --version`
- CUDA 工具包：`nvcc --version`
- 网络访问 GitHub 和 Hugging Face

## 模型支持矩阵

Spark 上的 llama.cpp 支持以下模型。说明默认使用 **Nemotron 3 Nano Omni** 示例行。

| 模型 | 支持状态 | 模型标识 |
|-------|----------------|-----------|
| **Nemotron 3 Nano Omni**（示例演练） | ✅ | `ggml-org/NVIDIA-Nemotron-3-Nano-Omni` |
| **Qwen3.6-35B-A3B** | ✅ | `unsloth/Qwen3.6-35B-A3B-GGUF` |
| **Qwen3.6-27B** | ✅ | `unsloth/Qwen3.6-27B-GGUF` |
| **Gemma 4 31B IT** | ✅ | `ggml-org/gemma-4-31B-it-GGUF` |
| **Gemma 4 26B A4B IT** | ✅ | `ggml-org/gemma-4-26B-A4B-it-GGUF` |
| **Gemma 4 E4B IT** | ✅ | `ggml-org/gemma-4-E4B-it-GGUF` |
| **Gemma 4 E2B IT** | ✅ | `ggml-org/gemma-4-E2B-it-GGUF` |
| **Nemotron-3-Nano** | ✅ | `unsloth/Nemotron-3-Nano-30B-A3B-GGUF` |

## 时间与风险

* **预计时间：** 大约 30 分钟，加上下载示例 GGUF（默认量化约 ~35GB 量级）
* **风险级别：** 低 — 构建是您的克隆本地的；以下步骤无需进行系统范围内的安装
* **回滚：**删除`llama.cpp`克隆以及`~/models/`下的模型目录以回收磁盘空间
* **最后更新：** 2026 年 4 月 28 日
  * 演练改用 Nemotron Omni；其他模型行仍可用

<a id="instructions"></a>
## 操作步骤
## 步骤 1. 验证先决条件

**示例**检查点为 Hugging Face 仓库 **`ggml-org/NVIDIA-Nemotron-3-Nano-Omni`** 中的 **`nemotron-3-nano-omni-ga_v1.0-Q8_0.gguf`**（完整标识：`ggml-org/NVIDIA-Nemotron-3-Nano-Omni/nemotron-3-nano-omni-ga_v1.0-Q8_0.gguf`）。其他受支持的 GGUF——包括 Qwen3.6、Gemma 以及其他 Nemotron Omni 构建——使用相同的构建和服务器步骤；只需更改 `hf download` 与 `--model` 路径（见上方模型矩阵）。

确保安装了所需的工具：

```bash
git --version
cmake --version
nvcc --version
```

所有命令都应返回版本信息。如果缺少任何内容，请在继续之前安装它们。

安装 Hugging Face CLI：

```bash
python3 -m venv llama-cpp-venv
source llama-cpp-venv/bin/activate
pip install -U "huggingface_hub[cli]"
```

验证安装：

```bash
hf version
```

## 步骤 2. 克隆 llama.cpp 仓库

克隆上游 llama.cpp — 您正在构建的框架：

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

## 步骤 3. 使用 CUDA 构建 llama.cpp

使用 CUDA 和 GB10 的 **sm_121** 架构配置 CMake，以便 GGML 的 CUDA 后端与您的 GPU 匹配：

```bash
mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="121" -DLLAMA_CURL=OFF
make -j8
```

构建通常需要 5-10 分钟左右。完成后，`llama-server` 等二进制文件将出现在 `build/bin/` 下。

## 步骤 4. 下载示例 Nemotron 3 Nano Omni GGUF

llama.cpp 以 **GGUF** 格式加载模型。本剧本使用来自 `ggml-org/NVIDIA-Nemotron-3-Nano-Omni` 的 **Q8_0** 检查点，可在 DGX Spark GB10 的统一内存上平衡质量和内存。

```bash
hf download ggml-org/NVIDIA-Nemotron-3-Nano-Omni \
  nemotron-3-nano-omni-ga_v1.0-Q8_0.gguf \
  --local-dir ~/models/NVIDIA-Nemotron-3-Nano-Omni
```

文件量级约为 **~35GB**（具体大小可能不同）。如果中断，可以继续下载。

## 步骤 5. 使用 Nemotron 3 Nano Omni 启动 llama-server

从 `llama.cpp/build` 目录中，启动具有 GPU 卸载功能的 OpenAI 兼容服务器：

```bash
./bin/llama-server \
  --model ~/models/NVIDIA-Nemotron-3-Nano-Omni/nemotron-3-nano-omni-ga_v1.0-Q8_0.gguf \
  --host 0.0.0.0 \
  --port 30000 \
  --n-gpu-layers 99 \
  --ctx-size 8192 \
  --threads 8
```

**参数（简短）：**

- `--host` / `--port`：HTTP API 的绑定地址和端口
- `--n-gpu-layers 99`：将层卸载到 GPU（如果使用不同的模型，请进行调整）
- `--ctx-size`：上下文长度（可以增加到模型/服务器限制；使用更多内存）
- `--threads`：用于非 GPU 工作的 CPU 线程

您应该看到类似于以下内容的日志行：

```
llama_new_context_with_model: n_ctx = 8192
...
main: server is listening on 0.0.0.0:30000
```

**测试时保持此终端打开**。大型 GGUF 可能需要一分钟以上才能加载；在您看到 `server is listening` 之前，端口 30000 上没有任何内容接受连接（请参阅排除 `curl` 报告连接被拒绝的情况）。

## 步骤 6. 测试 API

使用运行 `llama-server` 的同一台计算机上的第二个终端（例如 DGX Spark 的另一个 SSH 会话）。如果您在笔记本电脑上运行 `curl`，而服务器仅在 Spark 上运行，请使用 Spark 主机名或 IP，而不是 `localhost`。

```bash
curl -X POST http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemotron",
    "messages": [{"role": "user", "content": "New York is a great city because..."}],
    "max_tokens": 100
  }'
```

如果您看到 `curl: (7) Failed to connect`，则服务器仍在加载，进程已退出（检查服务器日志中是否有 OOM 或路径错误），或者您没有卷曲运行 `llama-server` 的主机。

响应的示例形状（字段因 llama.cpp 版本而异；`message` 可能包含额外的键）：

```json
{
  "choices": [
    {
      "finish_reason": "length",
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "New York is a great city because it's a living, breathing collage of cultures, ideas, and possibilities—all stacked into one vibrant, never‑sleeping metropolis. Here are just a few reasons that many people ("
      }
    }
  ],
  "created": 1765916539,
  "model": "nemotron-3-nano-omni-ga_v1.0-Q8_0.gguf",
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

## 步骤 7. 更长的完成时间（使用 Nemotron 3 Nano Omni）

尝试使用稍长的提示来确认 **Nemotron 3 Nano Omni** 的稳定生成：

```bash
curl -X POST http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemotron",
    "messages": [{"role": "user", "content": "Solve this step by step: If a train travels 120 miles in 2 hours, what is its average speed?"}],
    "max_tokens": 500
  }'
```

## 步骤 8. 清理

在运行服务器的终端中使用 `Ctrl+C` 停止服务器。

要删除本教程的工件：

```bash
rm -rf ~/llama.cpp
rm -rf ~/models/NVIDIA-Nemotron-3-Nano-Omni
```

如果不再需要 `hf`，请停用 Python venv：

```bash
deactivate
```

## 步骤 9. 后续步骤

1. **上下文长度：** 增加 `--ctx-size` 以获得更长的聊天时间（监视内存；仅当构建、模型和硬件允许时才可以使用 1M 令牌类上下文）。
2. **其他模型：** 将 `--model` 指向任何兼容的 GGUF； llama.cpp 服务器 API 保持不变。
3. **集成：** 使用 OpenAI 客户端模式在 `http://<spark-host>:30000/v1` 点 Open WebUI、Continue.dev 或自定义客户端。

服务器实现了 llama.cpp 构建启用的常见 OpenAI 风格聊天功能（包括支持的流和工具相关流程）。

<a id="troubleshooting"></a>
## 故障排查
| 症状 | 原因 | 使固定 |
|---------|-------|-----|
| `cmake` 失败并显示“未找到 CUDA” | CUDA 工具包不在 PATH 中 | 运行 `export PATH=/usr/local/cuda/bin:$PATH` 并从干净的构建目录重新运行 CMake |
| 构建错误提到错误的 GPU 架构 | CMake `CMAKE_CUDA_ARCHITECTURES` 与 GB10 不匹配 | 按照说明对 DGX Spark GB10 使用 `-DCMAKE_CUDA_ARCHITECTURES="121"` |
| GGUF 下载失败或停止 | 网络或Hugging Face可用性 | 重新运行`hf download`；它恢复部分文件 |
| 启动 `llama-server` 时出现“CUDA 内存不足” | 模型对于当前上下文或 VRAM 来说太大 | 降低 `--ctx-size` （例如 4096）或使用同一仓库中较小的量化 |
| 服务器运行但延迟很高 | 不在 GPU 上的层 | 确认 `--n-gpu-layers` 对于您的模型来说足够高；在请求期间检查 `nvidia-smi` |
| 端口 30000 上的 `curl: (7) Failed to connect` | 还没有侦听器、主机错误或崩溃 | 等待`server is listening`；在与 `llama-server`（或 Spark 的 IP）相同的主机上运行 `curl`；运行 `ss -tln` 并确认 `:30000`；读取服务器 stderr 是否存在 OOM 或错误的 `--model` 路径 |
| 聊天 API 错误或空回复 | `--model` 路径错误或 GGUF 不兼容 | 验证 `.gguf` 文件的路径；如果 GGUF 需要更新的格式，请更新 llama.cpp |

> [！笔记]
> DGX Spark 使用统一内存架构（UMA），允许 GPU 和 CPU 内存之间灵活共享。一些软件仍在追赶 UMA 行为。如果意外遇到内存压力，可以尝试刷新页面缓存（在共享系统上请小心使用）：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

有关最新的平台问题，请参阅 [DGX Spark known issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html) 文档。
