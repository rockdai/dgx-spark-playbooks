# FLUX.1 Dreambooth LoRA 微调

> 使用 Dreambooth LoRA 微调 FLUX.1-dev 12B 模型，以生成自定义图像

## 目录

- [概述](#overview)
- [操作步骤](#instructions)
- [故障排查](#troubleshooting)

---

<a id="overview"></a>
## 概述

## 基本思路

本 playbook 演示如何在 DGX Spark 上使用多概念 Dreambooth LoRA（Low-Rank Adaptation）对 FLUX.1-dev 12B 模型进行微调，以实现自定义图像生成。
DGX Spark 拥有 128GB 统一内存和强大的 GPU 加速能力，非常适合在内存中同时加载多个模型进行训练，例如 Diffusion Transformer、CLIP Text Encoder、T5 Text Encoder 和 Autoencoder。

多概念 Dreambooth LoRA 微调允许你为 FLUX.1 教会新的概念、角色和风格。训练得到的 LoRA 权重可以轻松集成到现有 ComfyUI 工作流中，非常适合做原型验证和实验。
此外，这个 playbook 还展示了 DGX Spark 不仅可以在内存中加载多个模型，还能够训练并生成 1024px 及以上的高分辨率图像。

## 你将完成的内容

你将得到一个已完成微调的 FLUX.1 模型，能够生成包含自定义概念的图像，并可直接用于 ComfyUI 工作流。
本环境包括：
- 使用 Dreambooth LoRA 技术对 FLUX.1-dev 模型进行微调
- 基于自定义概念（"tjtoy" 玩具和 "sparkgpu" GPU）进行训练
- 进行 1K 高分辨率扩散训练与推理
- 集成 ComfyUI 以支持直观的可视化工作流
- 通过 Docker 容器化实现可复现环境

## 前置条件

-  DGX Spark 设备已完成设置并可访问
-  DGX Spark 的 GPU 上没有其他进程在运行
-  有足够磁盘空间用于下载模型
-  已安装并配置 NVIDIA Docker


## 时间与风险

* **耗时：**
  * 初始环境准备和模型下载约 30-45 分钟
  * Dreambooth LoRA 训练约 1-2 小时
* **风险：**
  * Docker 权限问题可能需要修改用户组并重启会话
  * 若想获得最佳效果，该方案通常需要进行超参数调优并准备高质量数据集
* **回滚：** 停止并删除 Docker 容器；如有需要，可删除已下载模型。
* **最后更新：** 11/07/2025
  * 文案小幅修订

<a id="instructions"></a>
## 操作步骤

## 第 1 步：配置 Docker 权限

如果希望在不使用 sudo 的情况下方便地管理容器，你必须属于 `docker` 组。如果跳过这一步，后续 Docker 命令就需要加 sudo。

打开一个新终端并测试 Docker 访问。在终端中运行：

```bash
docker ps
```

如果你看到权限拒绝错误（例如 permission denied while trying to connect to the Docker daemon socket），请将当前用户加入 docker 组，这样就无需使用 sudo 运行 Docker 命令。

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## 第 2 步：克隆仓库

在终端中克隆仓库并进入 flux-finetuning 目录。

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
```

## 第 3 步：下载模型

由于 FLUX.1-dev 模型是受限访问的，你需要先获得访问权限。请前往其[模型卡](https://huggingface.co/black-forest-labs/FLUX.1-dev)接受条款并获取 checkpoint 访问权限。
如果你还没有 `HF_TOKEN`，请按照[这里](https://huggingface.co/docs/hub/en/security-tokens)的说明生成一个。然后将下面命令中的 token 替换为你生成的值，以完成系统认证。

```bash
export HF_TOKEN=<YOUR_HF_TOKEN>
cd dgx-spark-playbooks/nvidia/flux-finetuning/assets
sh download.sh
```
下载脚本耗时大约 30-45 分钟，具体取决于你的网络速度。

如果你已经有微调好的 LoRA，请将其放到 `models/loras` 中。如果还没有，请继续阅读 `Step 6. Training` 部分。

## 第 4 步：基础模型推理

先使用基础版 FLUX.1 模型针对我们关注的两个概念 Toy Jensen 和 DGX Spark 生成一张图像。

```bash
## Build the inference docker image
docker build -f Dockerfile.inference -t flux-comfyui .

## Launch the ComfyUI container (ensure you are inside flux-finetuning/assets)
## You can ignore any import errors for `torchaudio`
sh launch_comfyui.sh
```
访问 `http://localhost:8188` 打开 ComfyUI，并使用基础模型生成图像。不要选择任何预置模板。

在 ComfyUI 左侧面板找到 workflow 区域（或按 `w`）。打开后，你应该会看到两个已预加载的工作流。对于基础 Flux 模型，请加载 `base_flux.json` 工作流。加载该 json 后，你应该能看到 ComfyUI 已载入对应工作流。

在 `CLIP Text Encode (Prompt)` 模块中输入提示词。例如，这里我们使用 `Toy Jensen holding a DGX Spark in a datacenter`。由于生成的是 1024px 高分辨率图像，计算量较大，预计生成约需 3 分钟。

体验基础模型后，你有两个后续选择。
* 如果你已经将微调后的 LoRA 放入 `models/loras/`，请直接跳到 `Step 7. Fine-tuned model inference`。
* 如果你希望为自己的概念训练 LoRA，请先确保在继续训练前关闭 ComfyUI 推理容器。你可以按 `Ctrl+C` 中断终端来停止它。

> [！笔记]
>  若想清理系统中额外占用的内存，请在中断 ComfyUI 服务器后，在容器外执行以下命令。
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

## 第 5 步：准备数据集

接下来准备数据集，以便对 FLUX.1-dev 12B 模型执行 Dreambooth LoRA 微调。

在这个 playbook 中，我们已经预先准备好了包含 2 个概念的数据集：Toy Jensen 和 DGX Spark。该数据集由可通过 Google Images 获取的公开素材组成。如果你想直接生成这些概念的图像，则无需修改 `data.toml` 文件。

**TJ玩具概念**
- **触发短语**：`tjtoy toy`
- **Training images**: 6 张高质量 Toy Jensen 公版图像
- **Use case**: 在不同场景中生成包含该特定玩具角色的图像

**SparkGPU 概念**
- **触发短语**：`sparkgpu gpu`
- **Training images**: 7 张公开可用的 DGX Spark GPU 图像
- **Use case**: 在不同语境中生成带有该特定 GPU 设计的图像

如果你希望生成自定义概念图像，就需要为所有目标概念准备数据集，并为每个概念准备大约 5-10 张图像。

为每个概念创建对应名称的文件夹，并将其放入 `flux_data` 目录。就本例而言，我们使用 `sparkgpu` 和 `tjtoy` 作为概念名称，并分别在其中放入若干图片。

然后修改 `flux_data/data.toml` 文件，使其反映你所选择的概念。请通过修改 `[[datasets.subsets]]` 下的 `image_dir` 和 `class_tokens` 字段，为每个概念更新或创建对应条目。为了获得更好的微调效果，通常建议在概念名后附加一个类别 token（例如 `toy` 或 `gpu`）。

## 第 6 步：开始训练

执行以下命令启动训练。训练脚本使用默认配置，通常在大约 90 分钟后就能生成较好地捕捉 DreamBooth 概念的图像。该训练命令会自动将 checkpoint 保存到 `models/loras/` 目录中。

```bash
## Build the inference docker image
docker build -f Dockerfile.train -t flux-train .

## Trigger the training
sh launch_train.sh
```

## 第 7 步：使用微调模型进行推理

现在开始使用我们微调好的 LoRA 生成图像。

```bash
## Launch the ComfyUI container (ensure you are inside flux-finetuning/assets)
## You can ignore any import errors for `torchaudio`
sh launch_comfyui.sh
```
访问 `http://localhost:8188` 打开 ComfyUI，并使用微调后的模型生成图像。不要选择任何预置模板。

在 ComfyUI 左侧面板找到 workflow 区域（或按 `w`）。打开后，你应该会看到两个已预加载的工作流。对于微调后的 Flux 模型，请加载 `finetuned_flux.json` 工作流。加载该 json 后，你应该能看到 ComfyUI 已载入对应工作流。

在 `CLIP Text Encode (Prompt)` 模块中输入提示词。现在，把自定义概念加入到微调模型的提示词中。例如，我们使用 `tjtoy toy holding sparkgpu gpu in a datacenter`。由于生成的是 1024px 高分辨率图像，计算量较大，预计生成约需 3 分钟。

与基础模型不同，微调后的模型可以在一张图像中同时生成多个概念。此外，ComfyUI 还提供了多个可调字段，用于调整生成图像的风格和效果。

<a id="troubleshooting"></a>
## 故障排查

| 现象 | 原因 | 解决方法 |
|---------|--------|-----|
| 无法访问受限仓库 URL | 某些 HuggingFace 模型具有访问限制 | 重新生成你的 [HuggingFace token](https://huggingface.co/docs/hub/en/security-tokens)；并在浏览器中申请访问该[受限模型](https://huggingface.co/docs/hub/en/models-gated#customize-requested-information) |

> [！笔记]
> DGX Spark 使用统一内存架构（UMA），可在 GPU 和 CPU 之间动态共享内存。 
> 由于许多应用仍在逐步适配 UMA，即使看起来尚未达到 DGX Spark 的内存上限，也可能遇到内存问题。 
> 如果发生这种情况，请手动刷新 buffer cache：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
