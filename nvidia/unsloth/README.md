# DGX Spark 上的 Unsloth

> 使用 Unsloth 进行优化微调

## 目录

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思路

- **性能第一**：与标准方法相比，它声称可以加快训练速度（例如，在单 GPU 上快 2 倍，在多 GPU 设置中快 30 倍）并减少内存使用。   
- **内核级优化**：核心计算是使用自定义内核（例如 Triton）和手动优化的数学构建的，以提高吞吐量和效率。  
- **量化和模型格式**：支持动态量化（4 位、16 位）和 GGUF 格式以减少占用空间，同时旨在保持准确性。    
- **广泛的模型支持**：可与许多 LLM（LLaMA、Mistral、Qwen、DeepSeek 等）配合使用，并允许训练、微调、导出为 Ollama、vLLM、GGUF、Hugging Face 等格式。   
- **简化的界面**：提供易于使用的笔记本和工具，以便用户可以使用最少的样板来微调模型。  

## 你将完成什么

您将设置 Unsloth 以在 NVIDIA Spark 设备上对大型语言模型进行优化微调， 
通过高效的减少内存使用量，实现高达 2 倍的更快训练速度 
参数高效的微调方法，如 LoRA 和 QLoRA。

## 开始之前需要了解什么

- 使用 pip 和虚拟环境进行 Python 包管理
- Hugging Face Transformers 库基础知识（加载模型、分词器、数据集）
- GPU 基础知识（CUDA/GPU 与 CPU、VRAM 限制、设备可用性）
- 对 LLM 培训概念（损失函数、检查点）的基本了解
- 熟悉即时工程和基础模型交互
- 可选：LoRA/QLoRA 参数高效微调知识

## 先决条件

- 采用 Blackwell GPU 架构的 NVIDIA Spark 设备
- `nvidia-smi` 显示 GPU 信息摘要
- 安装的 CUDA 13.0：`nvcc --version`
- 用于下载模型和数据集的互联网访问

## 附属文件

Python测试脚本可以找到[here on GitHub](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/unsloth/assets/test_unsloth.py)


## 时间与风险

* **持续时间**：初始设置和测试运行需要 30-60 分钟
* **风险**： 
  * Triton编译器版本不匹配可能会导致编译错误
  * CUDA 工具包配置问题可能会阻止内核编译
  * 较小模型的内存限制需要调整批量大小
* **回滚**：卸载带有 `pip uninstall unsloth torch torchvision` 的软件包。
* **最后更新：** 2025 年 12 月 15 日
  * 将pytorch容器和python依赖项升级到最新版本

## 指示

## 步骤 1. 验证先决条件

确认您的 NVIDIA Spark 设备具有所需的 CUDA 工具包和可用的 GPU 资源。

```bash
nvcc --version
```
输出应显示 CUDA 13.0。

```bash
nvidia-smi
```
输出应显示 GPU 信息的摘要。

## 步骤2.获取容器镜像
```bash
docker pull nvcr.io/nvidia/pytorch:25.11-py3
```

## 步骤 3. 启动 Docker
```bash
docker run --gpus all --ulimit memlock=-1 -it --ulimit stack=67108864 --entrypoint /usr/bin/bash --rm nvcr.io/nvidia/pytorch:25.11-py3
```

## 步骤 4. 在 Docker 内安装依赖项

```bash
pip install transformers peft hf_transfer "datasets==4.3.0" "trl==0.26.1"
pip install --no-deps unsloth unsloth_zoo bitsandbytes
```

## 步骤 5. 创建 Python 测试脚本

将测试脚本 [here](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/unsloth/assets/test_unsloth.py) 卷曲到容器中。

```bash
curl -O https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/unsloth/assets/test_unsloth.py
```

我们将使用此测试脚本通过简单的微调任务来验证安装。


## 步骤 6. 运行验证测试

执行测试脚本以验证 Unsloth 是否正常工作。

```bash
python test_unsloth.py
```

终端窗口中的预期输出：
- “Unsloth：将为您的计算机打补丁，使免费微调速度提高 2 倍”
- 训练进度条显示损失在 60 步内减少
- 显示完成情况的最终培训指标

## 步骤 7. 后续步骤

通过更新 `test_unsloth.py` 文件来测试您自己的模型和数据集：

```python
## Replace line 32 with your model choice
model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"

## Load your custom dataset in line 8
dataset = load_dataset("your_dataset_name")

## Adjust training parameter args at line 61
per_device_train_batch_size = 4
max_steps = 1000
```

访问https://github.com/unslothai/unsloth/wiki
高级使用说明，包括：
- [Saving models in GGUF format for vLLM](https://github.com/unslothai/unsloth/wiki#saving-to-gguf)
- [Continued training from checkpoints](https://github.com/unslothai/unsloth/wiki#loading-lora-adapters-for-continued-finetuning)
- [Using custom chat templates](https://github.com/unslothai/unsloth/wiki#chat-templates)
- [Running evaluation loops](https://github.com/unslothai/unsloth/wiki#evaluation-loop---also-fixes-oom-or-crashing)

## 故障排除

> [!NOTE]
> DGX Spark 使用统一内存架构 (UMA)，可实现 GPU 和 CPU 之间的动态内存共享。 
> 由于许多应用程序仍在更新以利用 UMA，因此即使在 
> DGX Spark 的内存容量。如果发生这种情况，请使用以下命令手动刷新缓冲区缓存：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
