# 使用 NeMo 进行微调

> 使用 NVIDIA NeMo 在本地微调模型

## 目录

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思路

本手册将Instruct您设置和使用 NVIDIA NeMo AutoModel 在 NVIDIA Spark 设备上微调大型语言模型和视觉语言模型。 NeMo AutoModel 通过原生 PyTorch 支持为 Hugging Face 模型提供 GPU 加速的端到端训练，从而实现即时微调，无需转换延迟。该框架支持跨单 GPU 到多节点集群的分布式训练，具有专​​为 ARM64 架构和 Blackwell GPU 系统设计的优化内核和内存高效配方。

## 你将完成什么

您将在 NVIDIA Spark 设备上使用 NeMo AutoModel 为大型语言模型（1-70B 参数）和视觉语言模型建立完整的微调环境。最后，您将拥有一个支持参数高效微调 (PEFT)、监督微调 (SFT) 和具有 FP8 精度优化的分布式训练功能的工作安装，同时保持与 Hugging Face 生态系统的兼容性。

## 开始之前需要了解什么

- 在 Linux 终端环境和 SSH 连接中工作
- 对 Python 虚拟环境和包管理有基本了解
- 熟悉GPU计算概念和CUDA工具包使用
- 具有容器化工作流程和 Docker/Podman 操作经验
- 了解机器学习模型训练概念和微调工作流程

## 先决条件

- 具有 Blackwell 架构 GPU 访问权限的 NVIDIA Spark 设备
- 安装并配置 CUDA 工具包 12.0+：`nvcc --version`
- Python 3.10+ 可用环境：`python3 --version`
- 最低 32GB 系统 RAM，可实现高效的模型加载和训练
- 用于下载模型和包的有效互联网连接
- 安装 Git 用于存储库克隆：`git --version`
- 已配置对 NVIDIA Spark 设备的 SSH 访问

## 附属文件

该剧本的所有必需文件都可以在 [here on GitHub](https://github.com/NVIDIA-NeMo/Automodel) 中找到

## 时间与风险

* **持续时间：** 完成设置和初始模型微调需要 45-90 分钟
* **风险：** 模型下载可能很大（几个 GB），ARM64 包兼容性问题可能需要进行故障排除，分布式训练设置复杂性随着多节点配置而增加
* **回滚：**可以彻底删除虚拟环境；除了软件包安装之外，不会对主机系统进行任何系统级更改。
* **最后更新：** 2026 年 3 月 4 日
  * 建议通过 Docker 运行 Nemo Finetune 工作流程

## 指示

## 步骤 1. 验证系统要求

检查您的 NVIDIA Spark 设备是否满足 [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) 安装的先决条件。此步骤在主机系统上运行，以确认 CUDA 工具包的可用性和 Python 版本兼容性。

```bash
## Verify CUDA installation
nvcc --version

## Check Python version (3.10+ required)
python3 --version

## Verify GPU accessibility
nvidia-smi

## Check available system memory
free -h

## Docker permission:
docker ps

## if there is permission issue, (e.g., permission denied while trying to connect to the Docker daemon socket), then do:
sudo usermod -aG docker $USER
newgrp docker
```

## 步骤2.配置Docker权限

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

## 步骤 3. 使用 NeMo AutoModel 获取容器镜像

```bash
docker pull nvcr.io/nvidia/nemo-automodel:26.02
```

## 步骤 4. 启动 Docker

启动具有 GPU 访问权限的交互式容器。 `--rm` 标志确保容器在退出时被删除。

```bash
docker run \
  --gpus all \
  --ulimit memlock=-1 \
  -it --ulimit stack=67108864 \
  --entrypoint /usr/bin/bash \
  --rm nvcr.io/nvidia/nemo-automodel:26.02
```

## 第 5 步：探索可用示例

查看适用于不同模型类型和训练场景的预配置训练方案。这些配方为 ARM64 和 Blackwell 架构提供了优化的配置。

```bash
## Navigate to /opt/Automodel
cd /opt/Automodel

## List LLM fine-tuning examples
ls examples/llm_finetune/

## View example recipe configuration
cat examples/llm_finetune/finetune.py | head -20
```

## 步骤 6. 运行示例微调
以下命令显示如何使用 LoRA 和 QLoRA 执行完全微调 (SFT)、参数高效微调 (PEFT)。

首先，导出您的 HF_TOKEN 以便可以下载门控模型。

```bash
## Run basic LLM fine-tuning example
export HF_TOKEN=<your_huggingface_token>
```
> [！笔记]
> 将 `<your_huggingface_token>` 替换为您的个人 Hugging Face 访问令牌。下载任何门控模型都需要有效的令牌。
>
> - 生成令牌：[Hugging Face tokens](https://huggingface.co/settings/tokens)，引导可用[here](https://huggingface.co/docs/hub/en/security-tokens)。
> - 在尝试下载之前，请求并接收每个模型页面的访问权限（并接受许可/条款）。
>   - 骆驼-3.1-8B：[meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)
>   - Qwen3-8B：[Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
>   - 元羊驼-3-70B：[meta-llama/Meta-Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B)
>
> 相同的步骤适用于您使用的任何其他门禁模型：访问 Hugging Face 上的模型卡，请求访问权限，接受许可证，然后等待批准。

**LoRA微调示例：**

执行基本的微调示例来验证完整的设置。这展示了使用适合测试的小模型进行参数高效的微调。
对于下面的示例，我们使用 YAML 进行配置，参数覆盖作为命令行参数传递。

```bash
## Run basic LLM fine-tuning example
cd /opt/Automodel
python3 examples/llm_finetune/finetune.py \
-c examples/llm_finetune/llama3_2/llama3_2_1b_squad_peft.yaml \
--model.pretrained_model_name_or_path meta-llama/Llama-3.1-8B \
--packed_sequence.packed_sequence_size 1024 \
--step_scheduler.max_steps 20
```

这些覆盖可确保 Llama-3.1-8B LoRA 运行按预期运行：
- `--model.pretrained_model_name_or_path`：选择 Llama-3.1-8B 模型以从 Hugging Face 模型中心进行微调（通过 Hugging Face 令牌获取权重）。
- `--packed_sequence.packed_sequence_size`：将打包序列大小设置为 1024 以启用打包序列训练。
- `--step_scheduler.max_steps`：设置最大训练步数。我们将其设置为 20 用于演示目的，请根据您的需要进行调整。

> [！笔记]
> YAML `llama3_2_1b_squad_peft.yaml` 配方定义了可在 Llama 模型大小之间重复使用的训练超参数（LoRA 排名、学习率等）。 `--model.pretrained_model_name_or_path` 覆盖确定实际加载的模型权重。

**QLoRA 微调示例：**

我们可以使用 QLoRA 以节省内存的方式微调大型模型。

```bash
cd /opt/Automodel
python3 examples/llm_finetune/finetune.py \
-c examples/llm_finetune/llama3_1/llama3_1_8b_squad_qlora.yaml \
--model.pretrained_model_name_or_path meta-llama/Meta-Llama-3-70B \
--loss_fn._target_ nemo_automodel.components.loss.te_parallel_ce.TEParallelCrossEntropy \
--step_scheduler.local_batch_size 1 \
--packed_sequence.packed_sequence_size 1024 \
--step_scheduler.max_steps 20
```

这些覆盖可确保 70B QLoRA 运行按预期运行：
- `--model.pretrained_model_name_or_path`：选择 70B 基本模型进行微调（通过 Hugging Face 令牌获取权重）。
- `--loss_fn._target_`：使用与大型 LLM 的张量并行训练兼容的 TransformerEngine 并行交叉熵损失变体。
- `--step_scheduler.local_batch_size`：将每个 GPU 微批量大小设置为 1 以适应内存中的 70B；总体有效批量大小仍然由配方中的梯度累积和数据/张量并行设置驱动。
- `--step_scheduler.max_steps`：设置最大训练步数。我们将其设置为 20 用于演示目的，请根据您的需要进行调整。
- `--packed_sequence.packed_sequence_size`：将打包序列大小设置为 1024 以启用打包序列训练。

**完整微调示例：**

运行以下命令来执行完整 (SFT) 微调：

```bash
cd /opt/Automodel
python3 examples/llm_finetune/finetune.py \
-c examples/llm_finetune/qwen/qwen3_8b_squad_spark.yaml \
--model.pretrained_model_name_or_path Qwen/Qwen3-8B \
--step_scheduler.local_batch_size 1 \
--step_scheduler.max_steps 20 \
--packed_sequence.packed_sequence_size 1024
```

这些覆盖可确保 Qwen3-8B SFT 运行按预期运行：
- `--model.pretrained_model_name_or_path`：选择要从 Hugging Face 模型中心进行微调的 Qwen/Qwen3-8B 模型（通过 Hugging Face 令牌获取的权重）。如果您想微调不同的模型，请调整此项。
- `--step_scheduler.max_steps`：设置最大训练步数。我们将其设置为 20 用于演示目的，请根据您的需要进行调整。
- `--step_scheduler.local_batch_size`：将每个 GPU 微批量大小设置为 1 以适合内存；总体有效批量大小仍然由配方中的梯度累积和数据/张量并行设置驱动。
- `--packed_sequence.packed_sequence_size`：将打包序列大小设置为 1024 以启用打包序列训练。


## 步骤 7. 验证培训是否成功完成

通过检查检查点目录中包含的工件来验证微调后的模型。

```bash
## Inspect logs and checkpoint output.
## The LATEST is a symlink pointing to the latest checkpoint.
## The checkpoint is the one that was saved during training.
## below is an example of the expected output (username and domain-users are placeholders).
ls -lah checkpoints/LATEST/

## $ ls -lah checkpoints/LATEST/
## total 32K
## drwxr-xr-x 6 username domain-users 4.0K Oct 16 22:33 .
## drwxr-xr-x 4 username domain-users 4.0K Oct 16 22:33 ..
## -rw-r--r-- 1 username domain-users 1.6K Oct 16 22:33 config.yaml
## drwxr-xr-x 2 username domain-users 4.0K Oct 16 22:33 dataloader
## drwxr-xr-x 2 username domain-users 4.0K Oct 16 22:33 model
## drwxr-xr-x 2 username domain-users 4.0K Oct 16 22:33 optim
## drwxr-xr-x 2 username domain-users 4.0K Oct 16 22:33 rng
## -rw-r--r-- 1 username domain-users 1.3K Oct 16 22:33 step_scheduler.pt
```

## 步骤 8. 清理（可选）

该容器是使用 `--rm` 标志启动的，因此当您退出时它会自动删除。要回收 Docker 映像使用的磁盘空间，请运行：

> [！警告]
> 这将删除 NeMo AutoModel 图像。如果稍后想使用它，则需要再次拉动它。

```bash
docker rmi nvcr.io/nvidia/nemo-automodel:26.02
```
## 步骤 9. 可选：在 Hugging Face Hub 上发布微调后的模型检查点

在 Hugging Face Hub 上发布您经过微调的模型检查点。
> [！笔记]
> 这是一个可选步骤，对于使用微调模型来说不是必需的。
> 如果您想与其他人共享微调后的模型或在其他项目中使用它，它会很有用。
> 您还可以通过克隆存储库并使用检查点在其他项目中使用微调后的模型。
> 要在其他项目中使用微调后的模型，您需要安装 Hugging Face CLI。
> 您可以通过运行 `pip install huggingface_hub` 安装 Hugging Face CLI。
> 欲了解更多信息，请参阅[Hugging Face CLI documentation](https://huggingface.co/docs/huggingface_hub/en/guides/cli)。

> [！提示]
> 您可以使用 `hf` 命令将微调后的模型检查点上传到 Hugging Face Hub。
> 欲了解更多信息，请参阅[Hugging Face CLI documentation](https://huggingface.co/docs/huggingface_hub/en/guides/cli)。

```bash
## Publish the fine-tuned model checkpoint to Hugging Face Hub
## will be published under the namespace <your_huggingface_username>/my-cool-model, adjust name as needed.
hf upload my-cool-model checkpoints/LATEST/model
```

> [！提示]
> 如果您没有 Hugging Face Hub 以及您使用的 HF_TOKEN 的写入权限，则上述命令可能会失败。
> 错误消息示例：
> ````重击
> user@host:/opt/Automodel$ hf 上传 my-cool-model 检查点/LATEST/model
> 回溯（最近一次调用最后一次）：
>   文件“/home/user/.local/lib/python3.10/site-packages/huggingface_hub/utils/_http.py”，第 409 行，hf_raise_for_status
>     响应.raise_for_status()
>   文件“/home/user/.local/lib/python3.10/site-packages/requests/models.py”，第 1024 行，在 raise_for_status 中
>     引发 HTTPError(http_error_msg, response=self)
> requests.exceptions.HTTPError：403 客户端错误：禁止 url：https://huggingface.co/api/repos/create
> ````
> 要解决此问题，您需要创建具有*写入*权限的访问令牌，请参阅 Hugging Face 指南 [here](https://huggingface.co/docs/hub/en/security-tokens) 了解说明。

## 步骤 10. 后续步骤

开始使用 NeMo AutoModel 来执行您的特定微调任务。从提供的配方开始，并根据您的模型要求和数据集进行定制。

```bash
## Copy a recipe for customization
cp examples/llm_finetune/finetune.py my_custom_training.py

## Edit configuration for your specific model and data, then run:
python3 my_custom_training.py
```

探索 [NeMo AutoModel GitHub repository](https://github.com/NVIDIA-NeMo/Automodel) 以获取更多食谱、文档和社区示例。考虑设置自定义数据集，尝试不同的模型架构，并扩展到更大模型的多节点分布式训练。

## 故障排除

| 症状 | 原因 | 使固定 |
|---------|--------|-----|
| `nvcc: command not found` | CUDA 工具包不在 PATH 中 | 将 CUDA 工具包添加到路径：`export PATH=/usr/local/cuda/bin:$PATH` |
| `pip install uv` 权限被拒绝 | 系统级点限制 | 使用 `pip3 install --user uv` 并更新 PATH |
| 训练中未检测到 GPU | CUDA 驱动程序/运行时不匹配 | 验证驱动程序兼容性：`nvidia-smi` 并根据需要重新安装 CUDA |
| 训练期间内存不足 | 模型对于可用 GPU 内存来说太大 | 减少批量大小、启用梯度检查点或使用模型并行性 |
| ARM64 封装兼容性问题 | 软件包不适用于 ARM 架构 | 使用源安装或使用 ARM64 标志从源构建 |
| 无法访问 URL 的门禁存储库 | 某些 HuggingFace 模型的访问受到限制 | 重新生成你的 [HuggingFace token](https://huggingface.co/docs/hub/en/security-tokens);并请求在您的网络浏览器上访问 [gated model](https://huggingface.co/docs/hub/en/models-gated#customize-requested-information) |

> [！笔记]
> DGX Spark 使用统一内存架构 (UMA)，可实现 GPU 和 CPU 之间的动态内存共享。
> 由于许多应用程序仍在更新以利用 UMA，因此即使在
> DGX Spark 的内存容量。如果发生这种情况，请使用以下命令手动刷新缓冲区缓存：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
