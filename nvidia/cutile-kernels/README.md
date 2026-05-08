# cuTile Kernels

> 在 DGX Spark 上使用 cuTile 高性能 GPU kernel 与 TileGym 基准套件

## 目录

- [概述](#overview)
- [操作步骤](#instructions)
  - [Kernel Benchmarks 工作流](#kernel-benchmarks)
  - [End-to-End Inference 工作流](#e2e-inference)
  - [FMHA Implementation Guide](#fmha)
- [性能对比](#performance-comparison)
- [故障排查](#troubleshooting)

---

<a id="overview"></a>
## 概述

## 基本思路

[TileGym](https://github.com/NVIDIA/TileGym) 是 NVIDIA 为 cuTile kernel 提供的基准套件与集成框架——这些是使用 cuTile Python DSL 编写的高性能 GPU kernel。cuTile 编译为 Tile IR，使开发者无需进行底层 CUDA 编程即可写出高效的 kernel。

本 playbook 涵盖三种工作流：
1. **[Kernel Benchmarks](#kernel-benchmarks)** —— 运行独立的 cuTile kernel 基准测试（FMHA、MatMul、RMSNorm 等）
2. **[End-to-End Inference](#e2e-inference)** —— 通过 monkey-patching 的方式，使用 cuTile 优化的 kernel 运行 LLM 推理
3. **[FMHA Implementation](#fmha)** —— 一步步从伪代码到优化后的 cuTile，构建一个 Flash Multi-Head Attention kernel 的实战教程，并附带可运行和基准测试的脚本

同一份 cuTile 代码可以同时在 DGX Spark（sm_121）和 B300（sm_103）上运行——cuTile 会在 JIT 编译时自动适配对应的 GPU 架构。

## 你将完成什么

- 在 DGX Spark 上运行 TileGym 基准套件
- 使用 cuTile 优化的 kernel 运行 Qwen2-7B 或 DeepSeek-V2-Lite 推理
- 观察 DGX Spark 与 B300 之间的性能扩展表现
- 一步步从伪代码构建一个优化的 FMHA kernel 实现

## 开始之前需要了解什么

- 对 Docker 与命令行工具有基本的熟悉
- 了解 GPU 计算相关概念（TFLOPS、内存带宽）
- 不需要 CUDA 编程经验
- 拥有带访问令牌的 HuggingFace 账号（仅 LLM 推理需要）

## 先决条件

**硬件要求：**
- 运行 Ubuntu 24.04 的 DGX Spark，或 B300 云实例
- 至少 16GB GPU 显存以运行 LLM 推理
- 至少 50GB 可用存储空间用于下载模型

**软件要求：**
- 已安装并配置好 Docker：`docker ps`
- CUDA Toolkit 13.x，并支持 Tile IR
- 用于访问模型的 HuggingFace token（仅 LLM 推理需要）
- 能访问网络以拉取容器和下载模型

验证 Docker 是否可用：
```bash
docker ps
```

如果遇到权限错误：
```bash
sudo usermod -aG docker $USER
newgrp docker
```

## Kernel 支持矩阵

| Kernel | 类别 | 数据类型 | 描述 |
|--------|----------|------------|-------------|
| **FMHA** | Attention | float16, float8 | Flash Multi-Head Attention |
| **MLA** | Attention | bfloat16, float8 | Multi-head Latent Attention |
| **MLA Decoding** | Attention | float16, float8 | 用于 decode 阶段的 MLA |
| **MatMul** | Matrix Ops | float16, float8 | 矩阵乘法 |
| **BMM** | Matrix Ops | float16 | 批量矩阵乘法 |
| **Group GEMM** | Matrix Ops | float16, float8 | 用于 MoE 的分组 GEMM |
| **RMSNorm** | Normalization | float16, bfloat16 | Root mean square normalization |
| **RoPE** | Positional | float16 | Rotary position embedding |
| **SiLU** | Activation | float16, float32 | 带乘法的 SiLU 激活 |
| **SwiGLU** | Activation | float16, float32 | 融合的 SwiGLU 操作 |
| **Softmax** | Activation | float16 | Softmax 归一化 |
| **Dropout** | Regularization | float16, float32 | Dropout 前向 |

## LLM 推理的模型支持

| 模型 | 支持的 Kernel | Batch Size | 输出 Token 数 | 备注 |
|-------|-------------------|------------|---------------|-------|
| **Qwen2-7B** | RoPE, RMSNorm, SwiGLU, FMHA | 16 | 50 | 标准 transformer |
| **DeepSeek-V2-Lite** | RoPE, RMSNorm, SiLU, MLA, MoE | 1 | 100 | MLA attention，MoE 层 |

## 附属文件

所需的全部资源均可在 [TileGym 仓库](https://github.com/NVIDIA/TileGym) 中找到。

- `tests/benchmark/run_all.sh` —— 运行所有 kernel 基准测试
- `modeling/transformers/bench_qwen.sh` —— Qwen2-7B 基准测试脚本
- `modeling/transformers/bench_deepseek.sh` —— DeepSeek-V2-Lite 基准测试脚本
- `modeling/transformers/infer.py` —— 集成了 TileGym 的主推理脚本
- [`assets/fmha_optimization_tutorial.py`](assets/fmha_optimization_tutorial.py) —— FMHA 分步优化教程
- [`assets/fmha_scaling_analysis.py`](assets/fmha_scaling_analysis.py) —— 跨序列长度的 FMHA 扩展性分析

## 时间与风险

* **预计时间：** 30-45 分钟（包含 LLM 推理所需的模型下载时间）
* **风险等级：** 低
  * 大文件下载可能因网络问题失败
  * 首次运行包含 JIT 编译开销
* **回滚：** 移除 Docker 容器即可撤销所有更改
* **最后更新：** 2026 年 2 月
  * 首次发布

<a id="instructions"></a>
## 操作步骤

<a id="kernel-benchmarks"></a>
### Kernel Benchmarks 工作流

## 步骤 1. 拉取带 CTK 13.x 的 CUDA NGC 容器

```bash
docker pull nvcr.io/nvidia/cuda:13.1-devel-ubuntu24.04
```

启动一个具备 GPU 访问权限的交互式会话：

```bash
docker run --gpus all -it --rm \
  -v ~/TileGym:/workspace/TileGym \
  nvcr.io/nvidia/cuda:13.1-devel-ubuntu24.04 \
  /bin/bash
```

> [!NOTE]
> `-v` 参数将本地目录挂载到容器内，从而持久化保存 TileGym 仓库。`--rm` 参数会在退出时自动移除容器；如果你希望之后还能继续使用该容器，可以省略此参数。

如果不在容器中运行，也可以直接安装 Tile IR：

```bash
# 需要 root 权限——使用 sudo 或以 root 身份运行
sudo apt-get install cuda-tile-ir-13-1 cuda-compiler-13-1
```

## 步骤 2. 克隆 TileGym 仓库

```bash
git clone https://github.com/NVIDIA/TileGym
cd TileGym
pip install .
```

## 步骤 3. 运行基准套件

```bash
cd tests/benchmark/
bash run_all.sh
```

> [!NOTE]
> 基准测试按顺序运行，以确保计时结果准确。完成所有 kernel 可能需要 10-15 分钟。

## 步骤 4. 查看结果

结果会展示每种 kernel 在不同序列长度下的 cuTile 性能。

预期输出大致如下：

```text
==========================================
Running bench_fused_attention.py...
==========================================
fused-attention-batch4-head32-d128-fwd-causal=True-float16-TFLOPS:
     N_CTX     CuTile
0   1024.0  58.188262
1   2048.0  80.906892
2   4096.0  86.189532
3   8192.0  88.891086
4  16384.0  89.491869
✓ PASSED: bench_fused_attention.py
```

## 步骤 5. 运行单独的基准测试

如需运行某个 kernel 的基准测试：

```bash
# Flash Multi-Head Attention
python bench_fused_attention.py

# 矩阵乘法
python bench_matrix_multiplication.py

# RMSNorm
python bench_rmsnorm.py

# RoPE
python bench_rope.py

# SwiGLU
python bench_swiglu.py
```

## 步骤 6. 清理

退出容器：

```bash
exit
```

移除当前工作流的容器（如果运行时未加 `--rm`）：

```bash
# 推荐：仅移除来自该工作流镜像的容器
docker rm $(docker ps -a --filter ancestor=nvcr.io/nvidia/cuda:13.1-devel-ubuntu24.04 --format '{{.ID}}')

# 备选：清理所有已停止的容器（会提示确认）
# docker container prune
```

移除镜像（可选）：

```bash
docker rmi nvcr.io/nvidia/cuda:13.1-devel-ubuntu24.04
```

## 步骤 7. 在 B300 上重复

在 B300 硬件上重复步骤 1-6，以观察性能扩展。预期的扩展结果请见 **性能对比** 章节。

<a id="e2e-inference"></a>
### End-to-End Inference 工作流

## 步骤 1. 配置环境

如果你尚未拉取 CUDA 容器并克隆 TileGym，请先完成（详见 **Kernel Benchmarks** 章节）。

首先在主机上克隆 TileGym：

```bash
mkdir -p ~/TileGym
git clone https://github.com/NVIDIA/TileGym ~/TileGym
```

然后启动容器并挂载该仓库：

```bash
docker run --gpus all -it --rm \
  -v ~/TileGym:/workspace/TileGym \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/cuda:13.1-devel-ubuntu24.04 \
  /bin/bash
```

> [!NOTE]
> `-v ~/.cache/huggingface:/root/.cache/huggingface` 用于挂载 HuggingFace 缓存，避免重复下载模型。

在容器内安装 TileGym：

```bash
cd /workspace/TileGym
pip install .
```

设置 HuggingFace token 以访问受限模型：

```bash
export HF_TOKEN=<your_huggingface_token>
```

> [!WARNING]
> 你需要一个 HuggingFace 账号和访问令牌。可在 https://huggingface.co/settings/tokens 获取。

## 步骤 2. 运行推理基准测试

进入 transformers 基准测试目录：

```bash
cd modeling/transformers
```

**选项 A：运行 Qwen2-7B 基准测试**

```bash
./bench_qwen.sh
```

配置：模型 `Qwen/Qwen2-7B`，Batch size 16，输出长度 50 tokens。

**选项 B：运行 DeepSeek-V2-Lite 基准测试**

```bash
./bench_deepseek.sh
```

配置：模型 `deepseek-ai/DeepSeek-V2-Lite-Chat`，Batch size 1，输出长度 100 tokens。

两个脚本都会运行两种配置：
1. **PyTorch 基线** —— 标准的 HuggingFace 推理
2. **TileGym cuTile** —— 使用 cuTile kernel 替换后的版本

## 步骤 3. 查看结果

**Qwen2-7B 在 DGX Spark（GB10）上的样例结果：**

```text
========================================
  Benchmark Results
========================================
Qwen2-7B_naive_bfloat16    |  15.66 tokens/s |  51.10s |  51151.0ms CUDA
Qwen2-7B_cutile_attn       |  18.52 tokens/s |  43.20s |  43079.7ms CUDA
========================================
```

**cuTile Kernel 拆解（DGX Spark - Qwen2）：**

| Kernel | CUDA 时间 (ms) | 调用次数 |
|--------|----------------|-------|
| `fmha_kernel` | 4185.9 | 28 |
| `swiglu_forward_kernel` | 2459.8 | 1400 |
| `attention_decode_kernel_grouped` | 2271.8 | 1372 |
| `rms_norm_kernel_static_persistent` | 634.7 | 57 |
| `rope_kernel` | 355.6 | 1400 |

## 步骤 4. TileGym monkey-patching 的工作方式

TileGym 会用 cuTile kernel 替换 PyTorch 模型中的算子。下面这段代码片段取自 TileGym 的 [`src/tilegym/transformers/monkey_patch.py`](https://github.com/NVIDIA/TileGym/blob/main/src/tilegym/transformers/monkey_patch.py)，由 [`modeling/transformers/infer.py`](https://github.com/NVIDIA/TileGym/blob/main/modeling/transformers/infer.py) 调用：

```python
from tilegym.transformers import apply_tilegym_kernel_to_qwen2

apply_tilegym_kernel_to_qwen2(
    rope=True,      # Replace RoPE with cuTile kernel
    rms_norm=True,  # Replace RMSNorm with cuTile kernel  
    swiglu=True,    # Replace SwiGLU with cuTile kernel
    attn=True,      # Replace attention with cuTile FMHA
    use_cutile=True # Use cuTile backend (vs Triton)
)
```

**Qwen2 被替换的 Kernel：**

| Kernel | PyTorch 操作 | cuTile 替换 |
|--------|-------------------|-------------------|
| `rms_norm_kernel_static_persistent` | `nn.RMSNorm` | Persistent RMSNorm |
| `rope_kernel` | Rotary position embedding | Fused RoPE |
| `fmha_kernel` | `F.scaled_dot_product_attention` | Flash Attention |
| `swiglu_forward_kernel` | SiLU + Mul | Fused SwiGLU |
| `attention_decode_kernel_grouped` | Decode attention | Grouped decode |

**DeepSeek-V2 被替换的 Kernel：**（参见 [`src/tilegym/transformers/monkey_patch.py`](https://github.com/NVIDIA/TileGym/blob/main/src/tilegym/transformers/monkey_patch.py)）

```python
from tilegym.transformers import apply_tilegym_kernel_to_deepseek_v2

apply_tilegym_kernel_to_deepseek_v2(
    rope=True,      # Replace RoPE with cuTile kernel
    rms_norm=True,  # Replace RMSNorm with cuTile kernel  
    swiglu=True,    # Replace SiLU+Mul with cuTile kernel
    attn=True,      # Replace MLA attention with cuTile
    moe=True,       # Replace MoE routing with cuTile
    use_cutile=True
)
```

| Kernel | PyTorch 操作 | cuTile 替换 |
|--------|-------------------|-------------------|
| `prefill_mla` | MLA prefill attention | Multi-head Latent Attention |
| `_mla_decoding_split_kv` | MLA decode attention | Split-KV decoding |
| `fused_moe_kernel` | MoE expert routing | Fused MoE |
| `group_gemm_kernel` | Expert FFN | Grouped GEMM |

## 步骤 5. 平台特定调优（进阶）

cuTile 提供了两种互补的性能调优机制：

- **[`ct.ByTarget`](https://docs.nvidia.com/cuda/cutile-python/performance.html)** —— 为不同 GPU 架构（`sm_<major><minor>`）选择不同的 kernel 启动参数。编译器在 JIT 时会选择与当前目标匹配的取值；如果没有匹配项，则使用 `default` 值。详见 [Performance Tuning](https://docs.nvidia.com/cuda/cutile-python/performance.html) 与 [Execution Model](https://docs.nvidia.com/cuda/cutile-python/execution.html) 文档。
- **`num_ctas`** —— 每次 kernel 调用启动的 Cooperative Thread Array（thread block）数量。可根据目标 GPU 的 SM 数量进行调优。
- **`occupancy`** —— 编译器为每个 SM 目标的并发 CTA 数提示。更高的 occupancy 有助于隐藏内存延迟，但会增加寄存器/共享内存压力。详见 [Execution Model](https://docs.nvidia.com/cuda/cutile-python/execution.html) 文档。
- **[`ct.autotune`](https://docs.nvidia.com/cuda/cutile-python/performance.html)** —— 在运行时搜索一组候选取值，并选出最快的配置。结果通过 [`cuda.tile.tune.TuningResult`](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.tune.TuningResult.html) / [`Measurement`](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.tune.Measurement.html) 报告。

```python
import cuda.tile as ct

@ct.kernel(
    # num_ctas: how many thread blocks to launch.
    # Use ByTarget to pick an arch-specific value at JIT time.
    num_ctas=ct.ByTarget({
        "sm_103": 8,   # B300 - more SMs, launch more CTAs
        "sm_121": 4,   # DGX Spark - fewer SMs (48), use fewer CTAs
        "default": 1,  # Fallback for any other GPU architecture
    }),
    # occupancy: hint for concurrent CTAs per SM (latency hiding vs. register pressure).
    occupancy=ct.ByTarget({
        "sm_103": 16,  # B300 - high occupancy, plenty of registers/SMEM
        "sm_121": 12,  # DGX Spark - moderate occupancy
        "default": 8,  # Conservative fallback
    }),
    opt_level=3       # Maximum compiler optimization level
)
def optimized_kernel(A, B, C):
    # Same kernel code works on all platforms;
    # ByTarget swaps in the arch-specific launch params automatically.
    ...
```

如需自动调优，可使用 [`ct.autotune`](https://docs.nvidia.com/cuda/cutile-python/performance.html) 在候选取值中搜索，并在运行时选出最快的配置：

```python
@ct.kernel(
    # autotune: benchmark each value and pick the fastest.
    num_ctas=ct.autotune([1, 2, 4, 8, 16]),
    occupancy=ct.autotune([8, 12, 16, 24]),
    opt_level=3
)
def autotuned_kernel(A, B, C):
    ...
```

## 步骤 6. 在 B300 上重复

在 B300 硬件上重复步骤 1-3。**同一份代码无需修改即可运行** —— cuTile 会自动针对 sm_103 进行 JIT 编译。

详细的扩展结果请见 **性能对比** 章节。

<a id="fmha"></a>
### FMHA Implementation Guide

> [!NOTE]
> 这是一份理解 cuTile 中 FMHA 实现的指南，并非完整参考。完整文档请参阅 [cuTile Python Documentation](https://docs.nvidia.com/cuda/cutile-python/)。

## Attention 基础

Attention 让神经网络能够聚焦于输入中相关的部分。在 transformer（GPT、LLaMA、Qwen）中，每个位置都会通过三个向量计算它对其他每个位置的关注程度：

- **Query (Q)**：「我在找什么？」
- **Key (K)**：「我包含什么？」
- **Value (V)**：「这是我的内容」

```text
Attention(Q, K, V) = softmax(Q × K^T / √d) × V

Shapes:
  Q, K, V = [batch, heads, seq_len, head_dim]
  Q × K^T = [batch, heads, seq_len, seq_len]  # Attention scores
  Output  = [batch, heads, seq_len, head_dim]
```

对于自回归模型，**causal masking** 确保每个 token 只关注先前的 token —— 在 softmax 之前将未来位置的分数置为 -infinity。

## Flash Attention 算法

标准 attention 会显式构建一个 [seq_len × seq_len] 矩阵（例如 seq_len=32768 时为 2 GB）。Flash Attention 通过分块（tiles）配合 **online softmax** 来避免这一开销：

```text
m = -infinity    # Running maximum
l = 0            # Running sum of exp(x - m)
acc = 0          # Running weighted sum of values

FOR each K,V tile:
    scores = Q_tile @ K_tile.T * scale
    m_new = max(m, max(scores))
    correction = exp(m - m_new)
    l = l * correction + sum(exp(scores - m_new))
    acc = acc * correction + exp(scores - m_new) @ V_tile
    m = m_new

output = acc / l
```

## cuTile 伪代码 → 实际映射

| 概念 | 伪代码 | cuTile |
|---|---|---|
| 定义 kernel | `KERNEL fmha(...)` | `@ct.kernel()` |
| 获取 block ID | `block_x = BLOCK_ID_X` | `bid_x = ct.bid(0)` |
| 创建索引 | `range(0, N)` | `ct.arange(N, dtype=ct.int32)` |
| 创建常量 tile | `tile = zeros(M, N)` | `ct.full((M, N), 0.0, dtype)` |
| 从内存加载 | `tile = LOAD(ptr, shape)` | `ct.load(tensor, index, shape)` |
| 写回内存 | `STORE(ptr, tile)` | `ct.store(tensor, index, tile)` |
| 矩阵乘法 | `C = A @ B + C` | `ct.mma(A, B, C)` |
| 归约 | `max_val = MAX(tile, axis)` | `ct.max(tile, axis, keepdims)` |

## Kernel 伪代码

```text
KERNEL fmha(Q, K, V, Out, scale, TILE_M, TILE_N):
    tile_row = BLOCK_ID_X
    batch_head = BLOCK_ID_Y
    batch = batch_head // num_heads
    head = batch_head % num_heads

    m_i = full(TILE_M, -infinity)
    l_i = full(TILE_M, 0)
    acc = zeros(TILE_M, head_dim)

    q = LOAD(Q[batch, head, tile_row*TILE_M : (tile_row+1)*TILE_M, :])

    FOR j = 0 to num_k_tiles:
        k = LOAD(K[batch, head, j*TILE_N : (j+1)*TILE_N, :])
        v = LOAD(V[batch, head, j*TILE_N : (j+1)*TILE_N, :])
        scores = MMA(q, transpose(k)) * scale
        IF causal AND in_mask_region:
            scores = WHERE(valid_mask, scores, -infinity)
        m_new = max(m_i, row_max(scores))
        correction = exp(m_i - m_new)
        p = exp(scores - m_new)
        l_i = l_i * correction + row_sum(p)
        acc = acc * correction + MMA(p, v)
        m_i = m_new

    out = acc / l_i
    STORE(Out[batch, head, tile_row*TILE_M :, :], out)
```

## cuTile 实现

```python
import cuda.tile as ct
import math
ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

@ct.kernel()
def fmha_kernel(Q, K, V, Out, qk_scale: float, TILE_D: ConstInt, H: ConstInt,
                TILE_M: ConstInt, TILE_N: ConstInt, CAUSAL: ConstBool):
    bid_x, bid_y = ct.bid(0), ct.bid(1)
    batch_idx, head_idx = bid_y // H, bid_y % H

    offs_m = (bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32))[:, None]
    offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)[None, :]

    m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0),
                shape=(1, 1, TILE_M, TILE_D)).reshape((TILE_M, TILE_D))

    k_seqlen = K.shape[2]
    if CAUSAL:
        Tc = ct.cdiv(min((bid_x + 1) * TILE_M, k_seqlen), TILE_N)
        mask_start = (bid_x * TILE_M) // TILE_N
    else:
        Tc = ct.cdiv(k_seqlen, TILE_N)
        mask_start = k_seqlen // TILE_N

    for j in range(0, Tc):
        k_tile = ct.load(K, index=(batch_idx, head_idx, j, 0),
                        shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
        k_t = ct.permute(k_tile, (1, 0))

        qk = ct.mma(q, k_t, ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32))
        qk = qk * qk_scale

        if CAUSAL and j >= mask_start:
            offs_n = j * TILE_N + offs_n_tile
            qk = ct.where(offs_m >= offs_n, qk,
                         ct.full((TILE_M, TILE_N), -math.inf, dtype=ct.float32))

        m_ij = ct.maximum(m_i, ct.max(qk, axis=-1, keepdims=True))
        qk = qk - m_ij
        p = ct.exp(qk)
        alpha = ct.exp(m_i - m_ij)
        l_i = l_i * alpha + ct.sum(p, axis=-1, keepdims=True)
        acc = acc * alpha

        v_tile = ct.load(V, index=(batch_idx, head_idx, j, 0),
                        shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
        acc = ct.mma(p.astype(Q.dtype), v_tile, acc)
        m_i = m_ij

    acc = (acc / l_i).reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)
```

## 启动 Kernel

```python
def run_fmha(q, k, v, sm_scale, is_causal=True):
    import torch
    TILE_M, TILE_N = 64, 64  # Platform-specific (see below)
    batch, num_heads, seq_len, head_dim = q.shape
    out = torch.empty_like(q)
    grid = (math.ceil(seq_len / TILE_M), batch * num_heads, 1)
    ct.launch(
        torch.cuda.current_stream(), grid, fmha_kernel,
        (q, k, v, out, sm_scale, head_dim, num_heads, TILE_M, TILE_N, is_causal)
    )
    return out
```

## 优化技巧

### exp2 + flush_to_zero

`exp2(x) = 2^x` 在 GPU 上比 `exp(x)` 更快。需要把 scale 调整为乘以 `1/log(2)`。

```python
# Convert natural-exp scale to base-2 so we can use the faster ct.exp2 intrinsic.
# exp(x) == exp2(x / log(2)) == exp2(x * INV_LOG_2).
INV_LOG_2 = 1.0 / math.log(2)  # ≈ 1.4427
qk_scale_log2 = qk_scale * INV_LOG_2  # Pre-multiply the softmax scale once

# ... in loop:
# Fuse the running-max update with the scale multiplication.
m_ij = ct.max(qk, axis=-1, keepdims=True) * qk_scale_log2
# Subtract the running max for numerical stability (online softmax).
qk = qk * qk_scale_log2 - m_ij
# flush_to_zero=True: flush denormals to 0 -> avoids slow denormal handling on GPU.
p = ct.exp2(qk, flush_to_zero=True)
alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)  # Correction factor for previous acc/l_i
```

### Load Order Transpose（加载时转置）

通过 `order` 参数在加载 K 时直接得到转置结果，避免显式的 permute。

```python
# order=(0,1,3,2) swaps the last two axes during the load,
# producing K^T directly in registers -- no extra ct.permute() needed.
# shape is expressed in the transposed layout: (1, 1, TILE_D, TILE_N).
k_t = ct.load(K, index=(..., 0, j), shape=(1,1,TILE_D,TILE_N),
              order=(0,1,3,2)).reshape((TILE_D, TILE_N))
```

### Latency Hints（延迟提示）

预取数据，让内存加载与计算相互重叠。完整的 load/store 提示列表（如 `allow_tma`、`latency`）请参阅 [Performance Tuning 文档](https://docs.nvidia.com/cuda/cutile-python/performance.html)。

```python
# latency=N tells the compiler to issue this load N loop iterations in
# advance of its use, so the memory transfer overlaps with the MMA work
# from earlier iterations. Larger latency = deeper software pipeline but
# more register pressure.
k_t = ct.load(K, ..., latency=2)    # Prefetch K 2 iterations ahead
v_tile = ct.load(V, ..., latency=4) # Prefetch V 4 iterations ahead (used later in the loop)
```

### Occupancy

允许每个 SM 上同时驻留多个 thread block，以隐藏内存延迟。`occupancy` 与寄存器、共享内存的相互关系详见 [Execution Model 文档](https://docs.nvidia.com/cuda/cutile-python/execution.html)。

```python
# occupancy=N is a hint to the compiler to target N concurrent CTAs per SM.
# Higher occupancy -> more warps available to hide memory latency,
# but constrains the per-CTA register/SMEM budget.
@ct.kernel(occupancy=2)  # 2 thread blocks (CTAs) co-resident per SM
def fmha_optimized(...):
```

### Approximate Division（近似除法）

在最终归一化阶段使用快速近似除法。

```python
from cuda.tile import RoundingMode as RMd
# RMd.APPROX -> hardware approximate reciprocal/divide (MUFU), much faster
# than IEEE-compliant division. Safe here because it's the final softmax
# normalization step where a small ULP error is acceptable.
# flush_to_zero=True flushes denormals to 0 to avoid the slow path.
acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
```

## 平台配置

同一份 kernel 代码可以在所有平台上运行，仅需修改配置参数。可使用 [`ct.ByTarget`](https://docs.nvidia.com/cuda/cutile-python/performance.html) 为不同架构选择取值，或使用 [`ct.autotune`](https://docs.nvidia.com/cuda/cutile-python/performance.html) 自动搜索候选取值。

| 平台 | TILE_M | TILE_N | Occupancy | 原因 |
|---|---|---|---|---|
| DGX Spark (sm_121) | 64 | 64 | 2 | 较小的 tile，配合 48 个 SM 实现较高 occupancy |
| B300 (sm_103) | 256 | 128 | 1 | 大 tile 最大化 HBM3e 吞吐 |
| B300 备选 | 128 | 128 | 2 | 更高的 occupancy，平衡并行度 |

```python
import cuda.tile as ct

@ct.kernel(
    # TILE_M / TILE_N: rows/cols of the Q and K/V tiles processed per CTA.
    # Larger tiles -> more arithmetic intensity; smaller tiles -> higher occupancy.
    # occupancy: target concurrent CTAs per SM (latency hiding vs. register pressure).
    occupancy=ct.ByTarget({
        "sm_121": 2,   # DGX Spark (48 SMs): 2 CTAs/SM for latency hiding
        "sm_100": 1,   # B300: larger tiles already saturate the SM
        "default": 1,  # Conservative fallback for other architectures
    }),
    opt_level=3        # Maximum compiler optimization level
)
def fmha_kernel(...):
    ...
```

## 性能结果

> **注意：** PyTorch SDPA 仅用于正确性验证，不用于性能对比。

### DGX Spark (sm_121) – 序列长度 2048

| 步骤 | 优化 | 延迟 (ms) | TFLOPS |
|---|---|---|---|
| 1 | Basic cuTile | 2.19 | 62.8 |
| 2 | + exp2 | 2.07 | 66.5 |
| 3 | + Load Order | 2.07 | 66.3 |
| 4 | + Latency Hints | 2.07 | 66.5 |
| 5 | + Occupancy=2 | 1.73 | 79.5 |
| 6 | + Approx Div (Final) | 1.69 | 81.1 |

### B300 (sm_103) – 不同序列长度

| Seq Len | 延迟 (ms) | TFLOPS | 相对 Spark |
|---|---|---|---|
| 1024 | 0.074 | 465 | 5.7x |
| 2048 | 0.178 | 770 | 9.5x |
| 4096 | 0.550 | 999 | 15.1x |
| 8192 | 1.897 | 1159 | 14.6x |
| 16384 | 7.014 | 1254 | 14.2x |

## 常见问题

| 问题 | 解决方案 |
|---|---|
| ct.mma 中 shape 不匹配 | 确保 A 是 (M,K)，B 是 (K,N)，C 是 (M,N) |
| dtype 错误 | 在调用 mma 前使用 `.astype()`；累加器应为 float32 |
| 启用 causal 时结果不正确 | 检查 mask_start 的计算与 `offs_m >= offs_n` 的逻辑 |
| 性能偏低 | 尝试不同的 TILE_M/N，检查 occupancy，确认 latency hint 是否生效 |

## 配套脚本

以下脚本随本 playbook 一同提供，可在 DGX Spark 或 B300 上运行：

- **[`assets/fmha_optimization_tutorial.py`](assets/fmha_optimization_tutorial.py)** —— 分步优化教程。从基础版本逐步构建 FMHA kernel，直至完全优化版，与本指南的优化路径一一对应。
- **[`assets/fmha_scaling_analysis.py`](assets/fmha_scaling_analysis.py)** —— 跨序列长度的扩展性分析。对每一级优化进行基准测试并生成性能数据。

```bash
# Run the optimization tutorial (DGX Spark)
python assets/fmha_optimization_tutorial.py --correctness-check

# Run the scaling analysis
python assets/fmha_scaling_analysis.py --iterations 100
```

## 参考资料

- [cuTile Python Documentation](https://docs.nvidia.com/cuda/cutile-python/)
- [Tile IR Specification](https://docs.nvidia.com/cuda/tile-ir/)
- [TileGym (pre-optimized kernels)](https://github.com/NVIDIA/TileGym)
- [NVIDIA Blog: Tuning Flash Attention for Peak Performance in CUDA Tile](https://developer.nvidia.com/blog/tuning-flash-attention-for-peak-performance-in-nvidia-cuda-tile/)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)

<a id="performance-comparison"></a>
## 性能对比

本节汇总了 DGX Spark（GB10）与 B300 之间，在 kernel 基准与端到端 LLM 推理上的性能扩展表现。

## Kernel Benchmark 扩展性

下表展示了 kernel 性能从 DGX Spark（GB10）到 B300 的扩展比例，可作为参考。

| Kernel | 指标 | B300 / GB10 |
|--------|--------|-------------|
| FMHA (causal, 8192) | TFLOPS | 13.7x |
| FMHA (non-causal, 8192) | TFLOPS | 15.1x |
| MatMul (8192) | TFLOPS | 18.9x |
| BMM (batch8, 4096) | TFLOPS | 19.4x |
| Group GEMM (4096) | TFLOPS | 23.9x |
| RMSNorm (4096) | GB/s | 33.1x |
| RoPE (16384) | GB/s | 22.8x |

**主要观察：**
- 计算密集型 kernel 通常从 GB10 到 B300 扩展 14-24 倍
- 内存受限 kernel 因 HBM 带宽优势可扩展 20-33 倍

## Qwen2-7B 性能

## 端到端吞吐

| 配置 | DGX Spark | B300 | 平台加速比 |
|---------------|-----------|------|------------------|
| **cuTile** | 18.52 tok/s | 257.33 tok/s | **13.9x** |

## CUDA Kernel 时间

| 配置 | DGX Spark | B300 | 平台加速比 |
|---------------|-----------|------|------------------|
| **cuTile** | 43,080 ms | 2,954 ms | **14.6x** |

## cuTile Kernel 拆解

**DGX Spark (GB10)：**

| Kernel | CUDA 时间 (ms) | 调用次数 |
|--------|----------------|-------|
| `fmha_kernel` | 4,185.9 | 28 |
| `swiglu_forward_kernel` | 2,459.8 | 1,400 |
| `attention_decode_kernel_grouped` | 2,271.8 | 1,372 |
| `rms_norm_kernel_static_persistent` | 634.7 | 57 |
| `rope_kernel` | 355.6 | 1,400 |

**B300：**

| Kernel | CUDA 时间 (ms) | 相对 Spark 加速 |
|--------|----------------|------------------|
| `fmha_kernel` | 337.9 | 12.4x |
| `swiglu_forward_kernel` | 226.3 | 10.9x |
| `attention_decode_kernel_grouped` | 111.0 | 20.5x |
| `rms_norm_kernel_static_persistent` | 29.7 | 21.4x |
| `rope_kernel` | 16.7 | 21.3x |

**同样的代码，不同的架构** —— cuTile 会分别为 sm_121（Spark）和 sm_103（B300）进行 JIT 编译。

## 平台规格

| 规格 | DGX Spark (GB10) | B300 |
|---------------|------------------|------|
| Compute Capability | sm_121 (12.1) | sm_103 (10.3) |
| SM 数量 | 48 | 132 |
| 内存 | 128 GB LPDDR5x | 192 GB HBM3e |
| 内存带宽 | 273 GB/s | 8 TB/s |

<a id="troubleshooting"></a>
## 故障排查

| 现象 | 原因 | 解决方法 |
|---------|-------|-----|
| `docker: permission denied` | 用户不在 docker 组中 | `sudo usermod -aG docker $USER && newgrp docker` |
| `401 Client Error: Unauthorized` | 缺少 HuggingFace token | `export HF_TOKEN=<your_token>` |
| `ModuleNotFoundError: tilegym` | 未安装 TileGym | `cd TileGym && pip install .` |
| `RuntimeError: CUDA out of memory` | 模型过大 | 减小 batch size 或使用更小的模型 |
| 模型加载时被 `Killed` | 系统内存不足 | 清理缓存：`sync; echo 3 > /proc/sys/vm/drop_caches` |
| 首次运行较慢 | JIT 编译 | 正常现象 —— cuTile 在首次运行时会编译 kernel |
| `FileNotFoundError: input_prompt_small.txt` | 缺少输入文件 | 在 `modeling/transformers` 目录下运行 |
| `torch.cuda.OutOfMemoryError` | GPU 显存不足 | 减小 `--batch_size` 参数 |
| `ImportError: cuda.tile` | 缺少 Tile IR | 安装：`apt-get install cuda-tile-ir-13-1` |
| 基准测试卡住 | GPU 被占用或锁定 | 检查 `nvidia-smi` 中是否有其他进程 |

> [!NOTE] 
> DGX Spark 采用统一内存架构（UMA），可在 GPU 与 CPU 之间动态共享内存。
> 由于许多应用仍在适配 UMA，即使在 DGX Spark 的内存容量范围之内，你也可能遇到内存相关问题。
> 如果出现这种情况，可手动清理 buffer cache：

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

> [!TIP]
> 首次运行 cuTile kernel 包含 JIT 编译开销。后续运行会因为编译后的 kernel 已被缓存而更快。

最新的已知问题请参阅 [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)。
