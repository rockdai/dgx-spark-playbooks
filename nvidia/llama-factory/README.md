# 骆驼工厂

> 使用 LLaMA Factory 安装和微调模型

## 目录

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思路
LLaMA Factory 是一个开源框架，可以简化训练和精细化过程
调整大型语言模型。它为各种尖端技术提供了统一的接口
方法，例如 SFT、RLHF 和 QLoRA 技术。它还支持广泛的LLM
LLaMA、Mistral 和 Qwen 等架构。本剧本演示了如何进行微调
在 NVIDIA Spark 设备上使用 LLaMA Factory CLI 的大型语言模型。

## 你将完成什么

您将在具有 Blackwell 架构的 NVIDIA Spark 上设置 LLaMA Factory，以微调大型
使用 LoRA、QLoRA 和完整微调方法的语言模型。这使得高效
针对特定领域的模型适应，同时利用特定于硬件的优化。

## 开始之前需要了解什么

- 编辑配置文件和故障排除的基本 Python 知识
- 用于运行 shell 命令和管理环境的命令行用法
- 熟悉 PyTorch 和 Hugging Face Transformers 生态系统
- GPU 环境设置，包括 CUDA/cuDNN 安装和 VRAM 管理
- 微调概念：了解 LoRA、QLoRA 和完全微调之间的权衡
- 数据集准备：将文本数据格式化为JSON结构以进行指令调优
- 资源管理：针对 GPU 限制调整批量大小和内存设置

## 先决条件

- 采用 Blackwell 架构的 NVIDIA Spark 设备

- 安装的 CUDA 12.9 或更高版本：`nvcc --version`

- 安装的 Git：`git --version`

- Python 3 与 venv 和 pip：`python3 --version && pip3 --version`

- 足够的存储空间（>50GB用于模型和检查点）：`df -h`

- 用于从 Hugging Face Hub 下载模型的互联网连接

## 附属文件

- 官方 LLaMA Factory 存储库：https://github.com/hiyouga/LLaMA-Factory

- 带有 CUDA 13 的 PyTorch：通过 `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130` 安装

- 训练配置示例：`examples/train_lora/qwen3_lora_sft.yaml`（来自存储库）

- 文档：https://llamafactory.readthedocs.io/en/latest/getting_started/data_preparation.html

## 时间与风险

* **持续时间：** 初始设置 30-60 分钟，训练 1-7 小时，具体取决于模型大小和数据集。
* **风险：** 模型下载需要大量带宽和存储空间。训练可能会消耗大量 GPU 内存，并且需要针对硬件限制进行参数调整。
* **回滚：** 停用虚拟环境并删除 `factoryEnv` 和 `LLaMA-Factory` 目录。训练检查点保存在本地，可以删除以回收存储空间。
* **最后更新：** 2026 年 2 月 18 日
  * 使用 PyTorch CUDA 13 更新为基于 venv 的设置（无 Docker）。 Qwen3 LoRA 微调工作流程。

## 指示

## 步骤 1. 验证系统先决条件

检查您的 NVIDIA Spark 系统是否已安装且可访问所需的组件。

```bash
nvcc --version
nvidia-smi
python3 --version
git --version
```

## 步骤2.创建并激活Python虚拟环境

创建虚拟环境并激活它以进行 LLaMA Factory 安装。

```bash
python3 -m venv factoryEnv
source ./factoryEnv/bin/activate
```

## 步骤 3. 安装支持 CUDA 13 的 PyTorch

从官方 PyTorch 索引安装具有 CUDA 13.0 支持的 PyTorch、torchvision 和 torchaudio。

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

## 步骤 4.验证 PyTorch CUDA 支持

确认 PyTorch 可以看到 GPU。

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## 步骤 5. 克隆 LLaMA Factory 存储库

从官方存储库下载 LLaMA Factory 源代码。

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
```

## 步骤 6. 安装 LLaMA Factory 及其依赖项

在具有指标支持的可编辑模式下安装 LLaMA Factory。

```bash
pip install -e ".[metrics]"
```

## 步骤 7. 准备训练配置

检查为 Qwen3 提供的 LoRA 微调配置。

```bash
cat examples/train_lora/qwen3_lora_sft.yaml
```

## 步骤 8. 启动微调训练

> [！笔记]
> 如果模型被门控，请登录您的 Hugging Face Hub 下载模型。

使用预配置的 LoRA 设置执行训练过程。

```bash
hf auth login   # if the model is gated
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
```

输出示例：
```
***** train metrics *****
  epoch                    =        3.0
  total_flos               = 11076559GF
  train_loss               =     0.9993
  train_runtime            = 0:14:32.12
  train_samples_per_second =      3.749
  train_steps_per_second   =      0.471
Figure saved at: saves/qwen3-4b/lora/sft/training_loss.png
```

## 步骤 9. 验证培训完成情况

验证训练是否成功完成并保存了检查点。

```bash
ls -la saves/qwen3-4b/lora/sft/
```

预期输出应显示：
- 最终检查点目录（`checkpoint-411` 或类似目录）
- 模型配置文件 (`adapter_config.json`)
- 显示损失值下降的训练指标
- 训练损失图另存为 PNG 文件

## 步骤 10. 使用微调模型测试推理

使用自定义提示测试您的微调模型：

```bash
llamafactory-cli chat examples/inference/qwen3_lora_sft.yaml
## Type: "Hello, how can you help me today?"
## Expect: Response showing fine-tuned behavior
```

## 步骤 11. 对于生产部署，导出模型

```bash
llamafactory-cli export examples/merge_lora/qwen3_lora_sft.yaml
```

## 步骤 12. 清理和回滚

> [！警告]
> 这将删除所有训练进度和检查点。

要删除虚拟环境和克隆存储库：

```bash
deactivate
cd ..
rm -rf LLaMA-Factory/
rm -rf factoryEnv/
```

## 故障排除

| 症状 | 原因 | 使固定 |
|---------|--------|-----|
| 训练期间 CUDA 内存不足 | 批量大小对于 GPU VRAM 来说太大 | 减少 `per_device_train_batch_size` 或增加 `gradient_accumulation_steps` |
| 无法访问 URL 的门禁存储库 | 某些 HuggingFace 模型的访问受到限制 | 重新生成你的 [HuggingFace token](https://huggingface.co/docs/hub/en/security-tokens);并请求在您的网络浏览器上访问 [gated model](https://huggingface.co/docs/hub/en/models-gated#customize-requested-information) |
| 模型下载失败或缓慢 | 网络连接或 Hugging Face Hub 问题 | 检查互联网连接，尝试使用 `HF_HUB_OFFLINE=1` 来缓存模型 |
| 训练损失没有减少 | 学习率过高/过低或数据不足 | 调整 `learning_rate` 参数或检查数据集质量 |

> [！笔记]
> DGX Spark 使用统一内存架构 (UMA)，可实现 GPU 和 CPU 之间的动态内存共享。 
> 由于许多应用程序仍在更新以利用 UMA，因此即使在 
> DGX Spark 的内存容量。如果发生这种情况，请使用以下命令手动刷新缓冲区缓存：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
