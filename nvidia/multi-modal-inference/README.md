# 多模态推理

> 使用 TensorRT 设置多模态推理

## 目录

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思路

多模态推理将不同的数据类型（例如文本、图像和音频）组合在单个模型管道中，以生成或解释更丰富的输出。  
多模态系统不是一次处理一种输入类型，而是共享**文本到图像生成**、**图像字幕**或**视觉语言推理**的表示。  

在 GPU 上，这可以实现**跨模式并行处理**，从而为结合语言和视觉的任务提供更快、更高保真度的结果。

## 你将完成什么

您将使用 TensorRT 在 NVIDIA Spark 上部署 GPU 加速的多模态推理功能来运行 
Flux.1 和 SDXL 扩散模型在多种精度格式（FP16、 
FP8、FP4）。

## 开始之前需要了解什么

- 使用 Docker 容器和 GPU 直通
- 使用 TensorRT 进行模型优化
- Hugging Face 模型中心认证和下载  
- 适用于 GPU 工作负载的命令行工具
- 对扩散模型和图像生成的基本了解

## 先决条件

- 采用 Blackwell GPU 架构的 NVIDIA Spark 设备
- Docker 已安装并且当前用户可以访问
- 配置 NVIDIA 容器运行时
- Hugging Face 帐户可访问 Hugging Face 上的 Black Forest Labs 模型 [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) 和 [FLUX.1-dev-onnx](https://huggingface.co/black-forest-labs/FLUX.1-dev-onnx)
- Hugging Face [token](https://huggingface.co/settings/tokens) 配置为可以访问两个 FLUX.1 模型存储库
- 至少 48GB VRAM 可用于 FP16 Flux.1 Schnell 操作
- 验证 GPU 访问：`nvidia-smi`
- 检查 Docker GPU 集成：`docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.11-py3 nvidia-smi`

## 附属文件

所有必需的文件都可以在 TensorRT 存储库 [here on GitHub](https://github.com/NVIDIA/TensorRT) 中找到
- [**requirements.txt**](https://github.com/NVIDIA/TensorRT/blob/main/demo/Diffusion/requirements.txt) - TensorRT 演示环境的 Python 依赖项
- [**demo_txt2img_flux.py**](https://github.com/NVIDIA/TensorRT/blob/main/demo/Diffusion/demo_txt2img_flux.py) - Flux.1 模型推理脚本  
- [**demo_txt2img_xl.py**](https://github.com/NVIDIA/TensorRT/blob/main/demo/Diffusion/demo_txt2img_xl.py) - SDXL 模型推理脚本
- **TensorRT 存储库** - 包含扩散演示代码和优化工具

## 时间与风险

- **持续时间**：45-90 分钟，具体取决于模型下载和优化步骤

- **风险**： 
  - 大模型下载可能会超时
  - 高 VRAM 要求可能会导致 OOM 错误
  - 量化模型可能会显示质量下降

- **回滚**： 
  - 从 HuggingFace 缓存中删除下载的模型
  - 然后退出容器环境

* **最后更新：** 2025 年 12 月 22 日
  * 升级到最新的 pytorch 容器版本 nvcr.io/nvidia/pytorch:25.11-py3
  * 添加 HuggingFace 令牌设置说明以进行模型访问
  * 添加docker容器权限设置说明

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

## 步骤2.启动TensorRT容器环境

启动具有 GPU 访问和 HuggingFace 缓存安装的 NVIDIA PyTorch 容器。这提供了 
预安装了所有必需依赖项的 TensorRT 开发环境。

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 \
--ulimit stack=67108864 -it --rm --ipc=host \
-v $HOME/.cache/huggingface:/root/.cache/huggingface \
nvcr.io/nvidia/pytorch:25.11-py3
```

## 步骤 3. 克隆并设置 TensorRT 存储库

下载 TensorRT 存储库并配置扩散模型演示的环境。

```bash
git clone https://github.com/NVIDIA/TensorRT.git -b main --single-branch && cd TensorRT
export TRT_OSSPATH=/workspace/TensorRT/
cd $TRT_OSSPATH/demo/Diffusion
```

## 步骤 4. 安装所需的依赖项

安装 NVIDIA ModelOpt 和其他依赖项以进行模型量化和优化。

```bash
## Install OpenGL libraries
apt update
apt install -y libgl1 libglu1-mesa libglib2.0-0t64 libxrender1 libxext6 libx11-6 libxrandr2 libxss1 libxcomposite1 libxdamage1 libxfixes3 libxcb1

pip install nvidia-modelopt[torch,onnx]
sed -i '/^nvidia-modelopt\[.*\]=.*/d' requirements.txt
pip3 install -r requirements.txt
pip install onnxconverter_common
```

设置您的 HuggingFace 令牌以访问开放模型。
```bash
export HF_TOKEN = <YOUR_HUGGING_FACE_TOKEN>
```

## 步骤 5. 运行 Flux.1 开发模型推理

使用具有不同精度格式的 Flux.1 Dev 模型测试多模态推理。

**子步骤 A. BF16 量化精度**

```bash
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" \
  --hf-token=$HF_TOKEN --download-onnx-models --bf16
```

**子步骤 B. FP8 量化精度**

```bash
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" \
  --hf-token=$HF_TOKEN --quantization-level 4 --fp8 --download-onnx-models
```

**子步骤 C. FP4 量化精度**

```bash
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" \
  --hf-token=$HF_TOKEN --fp4 --download-onnx-models
```

## 步骤 6. 运行 Flux.1 Schnell 模型推理

使用不同的精度格式测试更快的 Flux.1 Schnell 变体。

> [！警告]
> FP16 Flux.1 Schnell 需要 >48GB VRAM 才能进行本机导出

**子步骤 A. FP16 精度（高 VRAM 要求）**

```bash
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" \
  --hf-token=$HF_TOKEN --version="flux.1-schnell"
```

**子步骤 B. FP8 量化精度**

```bash
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" \
  --hf-token=$HF_TOKEN --version="flux.1-schnell" \
  --quantization-level 4 --fp8 --download-onnx-models
```

**子步骤 C. FP4 量化精度**

```bash
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" \
  --hf-token=$HF_TOKEN --version="flux.1-schnell" \
  --fp4 --download-onnx-models
```

## 步骤 7. 运行 SDXL 模型推理

测试 SDXL 模型以与不同精度格式进行比较。

**子步骤 A. BF16 精度**

```bash
python3 demo_txt2img_xl.py "a beautiful photograph of Mt. Fuji during cherry blossom" \
  --hf-token=$HF_TOKEN --version xl-1.0 --download-onnx-models
```

**子步骤 B. FP8 量化精度**

```bash
python3 demo_txt2img_xl.py "a beautiful photograph of Mt. Fuji during cherry blossom" \
  --hf-token=$HF_TOKEN --version xl-1.0 --download-onnx-models --fp8
```

## 步骤 8. 验证推理输出

检查模型是否成功生成图像并测量性能差异。

```bash
## Check for generated images in output directory
ls -la *.png *.jpg 2>/dev/null || echo "No image files found"

## Verify CUDA is accessible
nvidia-smi

## Check TensorRT version
python3 -c "import tensorrt as trt; print(f'TensorRT version: {trt.__version__}')"
```

## 步骤 9. 清理和回滚

删除下载的模型并退出容器环境以释放磁盘空间。

> [！警告]
> 这将删除所有缓存的模型和生成的图像

```bash
## Exit container
exit

## Remove HuggingFace cache (optional)
rm -rf $HOME/.cache/huggingface/
```

## 步骤 10. 后续步骤

使用经过验证的设置生成自定义图像或将多模态推理集成到您的 
应用程序。尝试不同的提示或探索使用已建立的 TensorRT 进行模型微调 
环境。

## 故障排除

| 症状 | 原因 | 使固定 |
|---------|-------|-----|
| “CUDA 内存不足”错误 | 模型 VRAM 不足 | 使用 FP8/FP4 量化或更小的模型 |
| “HF 令牌无效”错误 | HuggingFace 令牌丢失或过期 | 设置有效令牌：`export HF_TOKEN=<YOUR_TOKEN>` |
| 无法访问 URL 的门禁存储库 | 某些 HuggingFace 模型的访问受到限制 | 重新生成你的 [HuggingFace token](https://huggingface.co/docs/hub/en/security-tokens);并请求在您的网络浏览器上访问 [gated model](https://huggingface.co/docs/hub/en/models-gated#customize-requested-information) |
| 模型下载超时 | 网络问题或速率限制 | 重试命令或预下载模型 |

> [！笔记]
> DGX Spark 使用统一内存架构 (UMA)，可实现 GPU 和 CPU 之间的动态内存共享。 
> 由于许多应用程序仍在更新以利用 UMA，因此即使在 
> DGX Spark 的内存容量。如果发生这种情况，请使用以下命令手动刷新缓冲区缓存：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
