---
id: txt2kg-assets-deploy-services-vllm
title: vllm
sidebar_label: vllm
---

# 法学硕士服务

该服务使用带有 FP8 量化的 vLLM 提供先进的 GPU 加速 LLM 推理，为生产工作负载提供比 Ollama 更高的吞吐量。

## 概述

vLLM 是一项可选服务，通过提供以下功能来补充 Ollama：
- 并发请求的吞吐量更高
- 高级量化 (FP8)
- PagedAttention 可有效利用内存
- OpenAI 兼容 API

## 快速入门

### 使用完整的堆栈

运行 vLLM 最简单的方法是使用完整的堆栈：

```bash
# From project root
./start.sh --complete
```

这将启动 vLLM 以及所有其他可选服务。

### 手动 Docker 撰写

```bash
# From project root
docker compose -f deploy/compose/docker-compose.complete.yml up -d vllm
```

### 测试部署

```bash
# Check health
curl http://localhost:8001/v1/models

# Test chat completion
curl -X POST "http://localhost:8001/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "messages": [{"role": "user", "content": "Hello! How are you?"}],
    "max_tokens": 100
  }'
```

## 默认配置

- **型号**：`meta-llama/Llama-3.2-3B-Instruct`
- **量化**：FP8（针对计算效率进行了优化）
- **端口**：8001
- **API**：OpenAI 兼容端点

## 配置选项

`docker-compose.complete.yml` 中配置的环境变量：

- `VLLM_MODEL`：要加载的模型（默认值：`meta-llama/Llama-3.2-3B-Instruct`）
- `VLLM_TENSOR_PARALLEL_SIZE`：要使用的 GPU 数量（默认值：1）
- `VLLM_MAX_MODEL_LEN`：最大序列长度（默认值：4096）
- `VLLM_GPU_MEMORY_UTILIZATION`：GPU 内存使用情况（默认值：0.9）
- `VLLM_QUANTIZATION`：量化方法（默认：fp8）
- `VLLM_KV_CACHE_DTYPE`：KV缓存数据类型（默认：fp8）

## 前端集成

txt2kg 前端会自动检测并使用可用的 vLLM：

1. 三重提取：`/api/vllm` 端点
2. RAG 查询：如果配置，则自动使用 vLLM
3. 模型选择：在 UI 中选择 vLLM 模型

## 使用不同的模型

要使用不同的模型，请编辑 `docker-compose.complete.yml` 中的 `VLLM_MODEL` 环境变量：

```yaml
environment:
  - VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
```

然后重启服务：

```bash
docker compose -f deploy/compose/docker-compose.complete.yml restart vllm
```

## 性能技巧

1. **单 GPU**：设置 `VLLM_TENSOR_PARALLEL_SIZE=1` 以获得最佳单 GPU 性能
2. **多GPU**：增加`VLLM_TENSOR_PARALLEL_SIZE`以使用多个GPU
3. **内存**：根据可用 VRAM 调整 `VLLM_GPU_MEMORY_UTILIZATION`
4. **吞吐量**：为了获得高吞吐量，请使用较小的模型或增加量化

## 要求

- 支持 CUDA 的 NVIDIA GPU（推荐 Ampere 架构或更新版本）
- CUDA驱动535或以上
- Docker 与 NVIDIA 容器工具包
- 默认型号至少 8GB VRAM
- 用于门控模型的 HuggingFace 令牌（可选，缓存在 `~/.cache/huggingface` 中）

## 故障排除

### 检查服务状态
```bash
# View logs
docker compose -f deploy/compose/docker-compose.complete.yml logs -f vllm

# Check health
curl http://localhost:8001/v1/models
```

### GPU问题
```bash
# Check GPU availability
nvidia-smi

# Check vLLM container GPU access
docker exec vllm-service nvidia-smi
```

### 模型加载问题
- 确保模型有足够的 VRAM
- 检查 HuggingFace 缓存：`ls ~/.cache/huggingface/hub`
- 对于门控模型，设置 HF_TOKEN 环境变量

## 与奥拉马的比较

| 特征 | 奥拉马 | 法学硕士 |
|---------|--------|------|
| **易于使用** | ✅ 非常简单 | ⚠️更复杂 |
| **模型管理** | ✅ 内置拉/推 | ❌ 手动下载 |
| **吞吐量** | ⚠️中等 | ✅ 高 |
| **量化** | Q4_K_M | FP8、GPTQ |
| **内存效率** | ✅ 好 | ✅ 优秀（PagedAttention） |
| **用例** | 开发，小规模 | 生产，高通量 |

## 何时使用 vLLM

在以下情况下使用 vLLM：
- 处理大批量请求
- 需要最大吞吐量
- 使用多个 GPU
- 高负载部署到生产环境

在以下情况下使用 Ollama：
- 项目入门
- 单用户开发
- 需要更简单的模型管理
- 不需要最高性能
