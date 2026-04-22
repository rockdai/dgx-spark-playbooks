# 舒适的用户界面

> 安装并使用 Comfy UI 生成图像

## 目录

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思路

ComfyUI 是一款开源 Web 服务器应用程序，用于使用 SDXL、Flux 等基于扩散的模型生成 AI 图像。它具有基于浏览器的 UI，可让您通过多个步骤创建、编辑和运行图像生成和编辑工作流程。这些生成和编辑步骤（例如，加载模型、添加文本或采样）可在 UI 中配置为节点，并且您可以使用电线连接节点以形成工作流程。

ComfyUI 使用主机的 GPU 进行推理，因此您可以将其安装在 DGX Spark 上，并直接在设备上进行所有图像生成和编辑。  

工作流程保存为 JSON 文件，因此您可以对它们进行版本控制，以供将来的工作、协作和可重复性使用。

## 你将完成什么

您将在 NVIDIA DGX Spark 设备上安装并配置 ComfyUI，以便可以使用统一内存来处理大型模型。

## 开始之前需要了解什么

- 使用 Python 虚拟环境和包管理的经验
- 熟悉命令行操作和终端使用
- 对深度学习模型部署和检查点有基本了解
- 了解容器工作流程和 GPU 加速概念
- 了解访问 Web 服务的网络配置

## 先决条件

**硬件要求：**
-  NVIDIA Grace Blackwell GB10 超级芯片系统
-  稳定扩散模型至少 8GB GPU 内存
-  至少 20GB 可用存储空间

**软件要求：**
- 安装了 Python 3.8 或更高版本：`python3 --version`
- pip 包管理器可用：`pip3 --version`
- 与 Blackwell 兼容的 CUDA 工具包：`nvcc --version`
- Git版本控制：`git --version`
- 网络访问从 Hugging Face 下载模型
- Web 浏览器访问`<SPARK_IP>:8188` 端口

## 附属文件

所有需要的资源都可以在[in the ComfyUI repository on GitHub](https://github.com/comfyanonymous/ComfyUI)找到

- `requirements.txt` - ComfyUI 安装的 Python 依赖项
- `main.py` - 主要 ComfyUI 服务器应用程序入口点
- `v1-5-pruned-emaonly-fp16.safetensors` - 稳定扩散 1.5 检查点模型

## 时间与风险

* **Estimated time:** 30-45分钟（含模型下载）
* **Risk level:** 中
  * 模型下载很大（~2GB），可能会因网络问题而失败
  * 端口 8188 必须可访问以用于 Web 界面功能
* **Rollback:** 可以删除虚拟环境以删除所有已安装的软件包。可以从检查点目录中手动删除下载的模型。
* **最后更新：** 2025 年 10 月 11 日
  * 将 ComfyUI PyTorch 更新至 CUDA 13.0

## 指示

## 步骤 1. 验证系统先决条件

在继续安装之前，请检查您的 NVIDIA DGX Spark 设备是否满足要求。

```bash
python3 --version
pip3 --version
nvcc --version
nvidia-smi
```

预期输出应显示 Python 3.8+、pip 可用、CUDA 工具包和 GPU 检测。

## 步骤2.创建Python虚拟环境

您将在主机系统上安装 ComfyUI，因此您应该创建一个隔离的环境以避免与系统软件包发生冲突。

```bash
python3 -m venv comfyui-env
source comfyui-env/bin/activate
```

通过检查命令提示符显示 `(comfyui-env)` 来验证虚拟环境是否处于活动状态。

## 步骤 3. 安装支持 CUDA 的 PyTorch

安装支持 CUDA 13.0 的 PyTorch。

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

此安装的目标是 CUDA 13.0 与 Blackwell 架构 GPU 兼容。

## 步骤 4. 克隆 ComfyUI 存储库

从官方存储库下载 ComfyUI 源代码。

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI/
```

## 步骤 5. 安装 ComfyUI 依赖项

安装 ComfyUI 操作所需的 Python 包。

```bash
pip install -r requirements.txt
```

这将安装所有必需的依赖项，包括 Web 界面组件和模型处理库。

## 步骤 6. 下载稳定扩散检查点

导航到检查点目录并下载 Stable Diffusion 1.5 模型。

```bash
cd models/checkpoints/
wget https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors
cd ../../
```

下载大小约为 2GB，可能需要几分钟时间，具体取决于网络速度。

## 步骤 7. 启动 ComfyUI 服务器

启动 ComfyUI Web 服务器并启用网络访问。

```bash
python main.py --listen 0.0.0.0
```

服务器将绑定到端口 8188 上的所有网络接口，以便从其他设备进行访问。

## 步骤 8. 验证安装

检查 ComfyUI 是否正确运行并可以通过网络浏览器访问。

```bash
curl -I http://localhost:8188
```

预期输出应显示 HTTP 200 响应，表明 Web 服务器正在运行。

打开网络浏览器并导航至`http://<SPARK_IP>:8188`，其中`<SPARK_IP>` 是您设备的IP 地址。

## 步骤 9. 可选 - 清理和回滚

如果您需要完全删除安装，请按照下列步骤操作：

> [！警告]
> 这将删除所有已安装的软件包和下载的模型。

```bash
deactivate
rm -rf comfyui-env/
rm -rf ComfyUI/
```

要在安装过程中回滚，请按`Ctrl+C` 停止服务器并删除虚拟环境。

## 步骤 10. 可选 - 后续步骤

使用基本图像生成工作流程测试安装：

1. 通过`http://<SPARK_IP>:8188` 访问网络界面
2. 加载默认工作流程（应该自动出现）
3. 单击“运行”生成您的第一张图像
4. 在单独的终端中使用 `nvidia-smi` 监视 GPU 使用情况

图像生成应在 30-60 秒内完成，具体取决于您的硬件配置。

## 故障排除

| 症状 | 原因 | 使固定 |
|---------|-------|-----|
| PyTorch CUDA 不可用 | CUDA 版本不正确或缺少驱动程序 | 验证`nvcc --version`与cu129匹配，重新安装PyTorch |
| 模型下载失败 | 网络连接或存储空间 | 检查互联网连接，验证 20GB+ 可用空间 |
| 网页界面无法访问 | 防火墙屏蔽8188端口 | 配置防火墙允许8188端口，检查IP地址 |
| 手动刷新缓冲区缓存后出现 GPU 内存不足错误 | 模型 VRAM 不足 | 使用较小的型号或启用 CPU 回退模式 |

> [！笔记] 
> DGX Spark 使用统一内存架构 (UMA)，可实现 GPU 和 CPU 之间的动态内存共享。 
> 由于许多应用程序仍在更新以利用 UMA，因此即使在 
> DGX Spark 的内存容量。如果发生这种情况，请使用以下命令手动刷新缓冲区缓存：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```


有关最新的已知问题，请查看[DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)。
