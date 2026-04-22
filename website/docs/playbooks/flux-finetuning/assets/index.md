---
id: flux-finetuning-assets
title: assets
sidebar_label: assets
---

# FLUX.1 使用 LoRA 进行微调

该项目演示了使用 Dreambooth LoRA（低阶适应）微调 FLUX.1-dev 12B 模型以生成自定义图像。该演示包括通过命令行脚本和 ComfyUI 进行自定义概念和推理的培训。 

## 概述

该项目包括：
- **FLUX.1-dev Fine-tuning**：基于LoRA的微调
- **自定义概念训练**：在“tjtoy”玩具和“sparkgpu”GPU 上进行训练
- **命令行推理**：使用经过训练的 LoRA 权重生成图像
- **ComfyUI 集成**：使用自定义模型进行推理的直观工作流程
- **Docker 支持**：完整的容器化环境

## 内容
1. [Model Download](#1-model-download)
2. [Base Model Inference](#2-base-model-inference)
3. [Dataset Preparation](#3-dataset-preparation)
4. [Training](#4-training)
5. [Finetuned Model Inference](#5-finetuned-model-inference)

## 1. 模型下载

### 1.1 抱脸认证

您必须被授予访问 FLUX.1-dev 模型的权限，因为它是门控的。前往他们的 [model card](https://huggingface.co/black-forest-labs/FLUX.1-dev)，接受条款并获得进入检查站的权限。

如果您还没有 `HF_TOKEN`，请按照 [here](https://huggingface.co/docs/hub/en/security-tokens) 说明生成一个。通过替换以下命令中生成的令牌来验证您的系统。

```bash
export HF_TOKEN=`<YOUR_HF_TOKEN>`
```

### 1.2 下载预训练的检查点

```bash
cd flux-finetuning/assets

# script to download (can take about a total of 15-60 mins, based on your internet speed)
sh download.sh
```

以下代码片段下载训练和推理所需的 FLUX 模型。
- `flux1-dev.safetensors` (~23.8GB)
- `ae.safetensors` (~335MB) 
- `clip_l.safetensors` (~246MB)
- `t5xxl_fp16.safetensors` (~9.8GB)

下载检查点后，验证您的 `models/` 目录是否遵循此结构。

```
models/
├── checkpoints/
│   └── flux1-dev.safetensors
├── loras/
├── text_encoders/
│   ├── clip_l.safetensors
│   └── t5xxl_fp16.safetensors
└── vae/
    └── ae.safetensors
```

### 1.3（可选）使用微调检查点

如果您已经有经过微调的 LoRA，请将它们放入 `models/loras` 中。如果您还没有，请前往 [Training](#training) 部分了解更多详细信息。

## 2. 基础模型推理

首先，我们使用基本 FLUX.1 模型针对我们感兴趣的 2 个概念（Toy Jensen 和 DGX Spark）生成图像。 

### 2.1 启动docker容器

```bash
# Build the inference docker image
docker build -f Dockerfile.inference -t flux-comfyui .

# Launch the ComfyUI container (ensure you are inside flux-finetuning/assets)
# You can ignore any import errors for `torchaudio`
sh launch_comfyui.sh
```
访问 `http://localhost:8188` 处的 ComfyUI 以使用基本模型生成图像。不要选择任何预先存在的模板。

### 2.2 加载基础工作流程

在 ComfyUI 左侧面板上找到工作流程部分（或按 `w`）。打开它后，您应该会发现已加载两个现有工作流程。对于基本 Flux 模型，我们加载 `base_flux.json` 工作流程。加载 json 后，您应该看到 ComfyUI 加载工作流程。

### 2.3 填写您生成的提示

在 `CLIP Text Encode (Prompt)` 块中提供提示。例如，我们将使用 `Toy Jensen holding a DGX Spark in a datacenter`。您预计生成过程大约需要 3 分钟，因为创建高分辨率 1024 像素图像需要大量计算。

对于提供的提示和随机种子，基本 Flux 模型生成了以下图像。尽管这一代具有良好的质量，但它无法理解我们想要生成的自定义角色和概念。

<figure>
  `<img src="flux_assets/before_workflow.png" alt="Base model workflow" width="1000"/>`
  <figcaption>基本 FLUX.1 模型工作流程，无需自定义概念知识</figcaption>
</figure>

使用基本模型后，您有 2 个可能的后续步骤。
* 如果您已经在 `models/loras/` 内放置了经过微调的 LoRA，请跳至 [Load the finetuned workflow](#52-load-the-finetuned-workflow) 部分。
* 如果您希望针对您的自定义概念训练 LoRA，请首先确保 ComfyUI 推理容器已关闭，然后再继续训练。您可以通过使用 `Ctrl+C` 按键中断终端来调用它。

> **注意**：要清除系统中任何额外占用的内存，请在中断 ComfyUI 服务器后在容器外部执行以下命令。
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

## 3. 数据集准备

让我们准备数据集以在 FLUX.1-dev 12B 模型上执行 Dreambooth LoRA 微调。但是，如果您希望继续使用 Toy Jensen 和 DGX Spark 提供的数据集，请随时跳至 [Training](#training) 部分。该数据集是可通过 Google 图片访问的公共资产的集合。

### 3.1 数据收集

您需要准备一个包含您想要生成的所有概念的数据集，并为每个概念准备大约 5-10 张图像。对于此示例，我们希望生成具有 2 个概念的图像。

#### TJ玩具概念
- **触发短语**：`tjtoy toy`
- **训练图像**：6 张定制玩具人偶的高质量图像
- **用例**：生成具有各种场景中特定玩具角色的图像

#### SparkGPU概念  
- **触发短语**：`sparkgpu gpu`
- **训练图像**：7 张自定义 GPU 硬件图像
- **用例**：生成具有不同上下文中特定 GPU 设计的图像

### 3.2 格式化数据集

为每个概念创建一个具有相应名称的文件夹，并将其放置在 `flux_data` 目录中。在我们的例子中，我们使用 `sparkgpu` 和 `tjtoy` 作为我们的概念，并在每个概念中放置了一些图像。准备好数据集后，`flux_data` 内部的结构应模仿以下内容。

```
flux_data/
├── data.toml
├── concept_1/
│   ├── 1.png
│   ├── 2.jpg
    └── ...
└── concept_2/
    ├── 1.jpeg
    ├── 2.jpg
    └── ...
```

### 3.3 更新数据配置

现在，让我们修改 `flux_data/data.toml` 文件以反映所选的概念。通过修改 `[[datasets.subsets]]` 下的 `image_dir` 和 `class_tokens` 字段，确保为每个概念更新/创建条目。为了获得更好的微调性能，最好将类标记附加到概念名称（例如 `toy` 或 `gpu`）。

## 4. 培训

### 4.1 构建docker镜像

```bash
# Build the inference docker image
docker build -f Dockerfile.train -t flux-train .
```

### 4.2 设置训练命令

通过执行以下命令启动训练。训练脚本设置为使用默认配置，可以在大约 90 分钟的训练中为您的数据集生成合理的图像。此训练命令将自动将检查点存储在 `models/loras/` 目录中。

```bash
sh launch_train.sh
```

如果您希望根据自定义概念生成非常高质量的图像（如我们在自述文件中显示的图像），您将需要训练更长时间（约 4 小时）。要实现此目的，请将 `launch_train.sh` 脚本中的 num epochs 修改为 100。

```bash
--max_train_epochs=100
```

请随意使用 `launch_train.sh` 脚本中的其他超参数来找到适合您的数据集的最佳设置。一些需要调整的值得注意的参数包括：
- **网络类型**：LoRA，维度 256
- **学习率**：1.0（使用 Prodigy 优化器）
- **纪元**：100（每 25 纪元保存一次）
- **分辨率**：1024x1024
- **混合精度**：bfloat16
- **优化**：Torch 编译、梯度检查点、缓存潜在变量

## 5. 微调模型推理

现在让我们使用经过微调的 LoRA 生成图像！

### 5.1 启动docker容器

```bash
# Build the inference docker image, if you haven't already
docker build -f Dockerfile.inference -t flux-comfyui .

# Launch the ComfyUI container (ensure you are inside flux-finetuning/assets)
# You can ignore any import errors for `torchaudio`
sh launch_comfyui.sh
```
访问 `http://localhost:8188` 处的 ComfyUI 以使用微调后的模型生成图像。不要选择任何预先存在的模板。

### 5.2 加载微调后的工作流程

在 ComfyUI 左侧面板上找到工作流程部分（或按 `w`）。打开它后，您应该会发现已加载两个现有工作流程。对于微调后的 Flux 模型，我们加载 `finetuned_flux.json` 工作流程。加载 json 后，您应该看到 ComfyUI 加载工作流程。

### 5.3 填写您生成的提示

在 `CLIP Text Encode (Prompt)` 块中提供提示。现在，让我们将自定义概念合并到微调模型的提示中。例如，我们将使用 `tjtoy toy holding sparkgpu gpu in a datacenter`。您预计生成过程大约需要 3 分钟，因为创建高分辨率 1024 像素图像需要大量计算。

对于提供的提示和随机种子，微调后的 Flux 模型生成了以下图像。与基本模型不同，我们可以看到微调模型可以在单个图像中生成多个概念。

<figure>
  `<img src="flux_assets/after_workflow.png" alt="After Fine-tuning" width="1000"/>`
  <figcaption>具有自定义概念知识的微调 FLUX.1 模型</figcaption>
</figure>

### 5.4（可选）调整你的世代

ComfyUI 公开了多个字段来调整和更改生成图像的外观和感觉。以下是工作流程中需要注意的一些参数。

1. **LoRA 权重**：在 `Load LoRA` 插件中更改训练好的 LoRA 文件，甚至调整其强度
2. **调整分辨率**：修改`Empty Latent Image`插件中的宽度和高度为其他分辨率
3. **随机种子**：更改 `RandomNoise` 插件中的噪声种子，以获取具有相同提示的替代图像
4. **调整采样**：根据需要修改采样器、调度器和步骤


## 制作人员

该项目使用以下开源存储库：
- `kohya-ss` 的 [sd-scripts](https://github.com/kohya-ss/sd-scripts) 存储库用于 FLUX.1 微调。
- `comfyanonymous` 的 [ComfyUI](https://github.com/comfyanonymous/ComfyUI.git) 存储库用于 FLUX.1 推理。
