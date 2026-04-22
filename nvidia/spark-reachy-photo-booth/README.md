# Spark & Reachy 照相亭

> 使用 DGX Spark 和 Reachy Mini 的 AI 增强照相亭。

## 目录

- [Overview](#overview)
- [Instructions](#instructions)
  - [Guides](#guides)
  - [Service Configuration](#service-configuration)
- [Development](#development)
  - [Customize configuration parameters](#customize-configuration-parameters)
  - [Extend the demo with new tools](#extend-the-demo-with-new-tools)
  - [Create your own service](#create-your-own-service)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思路

![Teaser](assets/teaser.jpg)

Spark & Reachy Photo Booth 是一个交互式、事件驱动的照相亭演示，它将 **DGX Spark™** 与 **Reachy Mini** 机器人相结合，创造出引人入胜的multimodal AI 体验。系统展示：

- **使用 `NeMo Agent Toolkit` 构建的multimodal代理**
- **ReAct 循环** 由 `openai/gpt-oss-20b` LLM 驱动，由 `TensorRT-LLM` 提供支持
- **语音交互**基于`nvidia/riva-parakeet-ctc-1.1B`和`hexgrad/Kokoro-82M`
- **图像生成** 使用 `black-forest-labs/FLUX.1-Kontext-dev` 进行图像到图像的重新设计
- **用户位置跟踪** 使用 `facebookresearch/detectron2` 和 `FoundationVision/ByteTrack` 构建
- **MinIO** 用于存储捕获/生成的图像并通过 QR 码共享它们

该演示基于通过消息总线进行通信的多个服务。

![Architecture diagram](assets/architecture-diagram.png)

另请参阅此剧本的演练视频：[Video](https://www.youtube.com/watch?v=6f1x8ReGLjc)

> [!NOTE]
> 本手册适用于 Reachy Mini Lite。 Reachy Mini（带有板载 Raspberry Pi）可能需要进行细微调整。为简单起见，我们在本手册中将机器人称为 Reachy。

## 你将完成什么

您将在 DGX Spark 上部署完整的照相亭系统，在本地运行多个推理模型 - LLM、图像生成、语音识别、语音生成和计算机视觉 - 所有这些都无需依赖云。 Reachy 机器人通过自然对话与用户互动、拍摄照片并根据提示生成自定义图像，在边缘硬件上演示实时多模态 AI 处理。

## 开始之前需要了解什么

- 基本的 Docker 和 Docker Compose 知识
- 基本网络配置技能

## 先决条件

**硬件要求：**
- [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
- 用于直接在 DGX Spark 上运行此剧本的显示器、键盘和鼠标。
- [Reachy Mini or Reachy Mini Lite robot](https://pollen-robotics-reachy-mini.hf.space/)

> [!TIP]
> 确保您的 Reachy 机器人固件是最新的。您可以找到更新它的说明 [here](https://huggingface.co/spaces/pollen-robotics/Reachy_Mini)。
**软件要求：**
- 官方 [DGX Spark OS](https://docs.nvidia.com/dgx/dgx-spark/dgx-os.html) 映像，包括所有必需的实用程序，例如 Git、Docker、NVIDIA 驱动程序和 NVIDIA 容器工具包
- DGX Spark 的互联网连接
- NVIDIA NGC 个人 API 密钥 (**`NVIDIA_API_KEY`**)。 [Create a key](https://org.ngc.nvidia.com/setup/api-keys) 如有必要。确保在创建密钥时启用 `NGC Catalog` 范围。
- 拥抱面部访问令牌 (**`HF_TOKEN`**)。 [Create a token](https://huggingface.co/settings/tokens) 如有必要。确保创建一个具有_读取您可以访问的所有公共门控存储库内容的权限_的令牌。


## 附属文件

所有必需的资产都可以在 [Spark & Reachy Photo Booth repository](https://github.com/NVIDIA/spark-reachy-photo-booth) 中找到。

- Docker Compose 应用程序
- 各种配置文件
- 所有服务的源代码
- 详细文档

## 时间与风险

* **预计时间：** 2 小时，包括硬件设置、容器构建和模型下载
* **风险级别：** 中
* **回滚：** Docker 容器可以停止并删除以释放资源。可以从缓存目录中删除下载的模型。机器人和外围设备的连接可以安全地断开。可以通过删除自定义设置来恢复网络配置。
* **最后更新：** 2026 年 4 月 1 日
  * 1.0.0 首次发布
  * 1.0.1 文档改进

## 管辖条款
您对 Spark Playbook 脚本的使用受 [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) 管辖，并允许使用受各自许可证管辖的单独开源和专有软件：[Flux.1-Kontext NIM](https://catalog.ngc.nvidia.com/orgs/nim/teams/black-forest-labs/containers/flux.1-kontext-dev?version=1.1)、[Parakeet 1.1b CTC en-US ASR NIM](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/parakeet-1-1b-ctc-en-us?version=1.4)、[TensorRT-LLM](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release?version=1.3.0rc1)、[minio/minio](https://hub.docker.com/r/minio/minio)、[arizephoenix/phoenix](https://hub.docker.com/r/arizephoenix/phoenix)、[grafana/otel-lgtm](https://hub.docker.com/r/grafana/otel-lgtm)、[Python](https://hub.docker.com/_/python)、[Node.js](https://hub.docker.com/_/node)、[nginx](https://hub.docker.com/_/nginx)、[busybox](https://hub.docker.com/_/busybox)、[UV Python Packager](https://docs.astral.sh/uv/)、 [Redpanda](https://www.redpanda.com/)、[Redpanda Console](https://www.redpanda.com/)、[gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)、[FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)、[FLUX.1-Kontext-dev-onnx](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev-onnx)。

> [!NOTE]
> FLUX.1-Kontext-dev 和 FLUX.1-Kontext-dev-onnx 是针对非商业用途发布的模型。请联系 sales@blackforestlabs.ai 了解商业条款。您有责任接受适用的许可协议和可接受的使用政策，并确保您的 HF 令牌具有正确的权限。

## 指示

## 步骤 1. 克隆存储库

要在没有 `sudo` 的情况下轻松管理容器，您必须位于 `docker` 组中。如果您选择跳过此步骤，则需要使用 `sudo` 运行 Docker 命令。

打开新终端并测试 Docker 访问。在终端中，运行：

```bash
docker ps
```

如果您看到权限被拒绝错误（例如尝试连接到 Docker 守护进程套接字时权限被拒绝），请将您的用户添加到 docker 组，这样您就不需要使用 `sudo` 运行命令。

```bash
sudo usermod -aG docker $USER
newgrp docker
```

```bash
git clone https://github.com/NVIDIA/spark-reachy-photo-booth.git
cd spark-reachy-photo-booth
```

> [!WARNING]
> 该剧本预计将直接在 DGX Spark 上并使用附带的 Web 浏览器运行。

## 步骤 2. 创建您的环境

```bash
cp .env.example .env
```

编辑 `.env` 并设置：

- **`NVIDIA_API_KEY`**：您的 NVIDIA API 密钥（必须以 `nvapi-...` 开头）
- **`HF_TOKEN`**：您的 Hugging Face 令牌（必须以 `hf_...` 开头）
- **`EXTERNAL_MINIO_BASE_URL`**：保持不变，除非您愿意（请参阅“在本地网络上启用 QR 码共享”部分）

要访问 FLUX.1-Kontext-dev 模型，请登录您的 Hugging Face 帐户，然后查看并接受 [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) 和 [FLUX.1-Kontext-dev-onnx](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev-onnx) 许可协议和可接受的使用政策。

其余值配置为本地开发 (MinIO) 的合理默认值。对于生产部署或不受信任的环境，应更改并安全存储这些值。

## 步骤 3. 设置 Reachy

- 将电源线插入 Reachy 底座和电源插座。
- 将 USB-C 电缆插入 Reachy 底座和 DGX Spark。
- 打开 Reachy 底部的电源开关。开关旁边的 LED 应变为红色。

您可以通过运行以下命令来验证是否检测到机器人：

```bash
lsusb | grep Reachy
```

您应该会在终端中看到类似于 `Bus 003 Device 003: ID 38fb:1001 Pollen Robotics Reachy Mini Audio` 的设备打印。

运行以下命令，确保 Reachy 扬声器能够达到最大音量。

```bash
./robot-controller-service/scripts/speaker_setup.sh
```

![Setup](assets/setup.jpg)

## 步骤 4. 启动堆栈

登录 nvcr.io 注册表：
```bash
docker login nvcr.io -u "\$oauthtoken"
```

当提示输入密码时，输入您的 NGC 个人 API 密钥。

```bash
docker compose up --build -d
```

此命令提取并构建容器映像，并下载所需的模型工件。第一次运行可能需要 30 分钟到 2 小时，具体取决于您的互联网速度。后续运行通常在大约 5 分钟内完成。

## 步骤 5. 在浏览器中打开 UI

在 DGX Spark 上，打开 Firefox（预装）并浏览至 **Web UI**：[http://127.0.0.1:3001](http://127.0.0.1:3001)。

> [!TIP]
> 仅当所有容器均已启动并运行时，才能访问 Web UI。
> 您还可以使用 `docker compose ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"` 检查所有容器的状态。
> 如果一个或多个容器出现故障，请使用 `docker compose logs -f <container_name>` 检查日志。

> [!TIP]
> 您可以通过打开启用 X11 转发 (`ssh -X <USER>@<SPARK_IP>`) 的 ssh 会话来远程**观察**正在进行的交互。
> 您应该能够从此会话打开 Firefox 并连接到 [http://127.0.0.1:3001](http://127.0.0.1:3001)。

> [!NOTE]
> UI 对图像生成的性能影响很小。为了优化体验中图像生成步骤的性能，您可以安装和使用 Chromium 而不是 Firefox，以及降低显示分辨率。

## 步骤 6. 可选：在本地网络上启用 QR 码共享

Reachy 可以拍摄人物照片并根据照片生成图像。 Web UI 显示生成的图像以及用于下载图像的二维码。本节介绍如何设置系统以便可以从用户的手机访问 QR 码。

要在手机上打开二维码，您的 DGX Spark 和手机必须位于同一本地网络。确保您的路由器允许网络内的设备到设备通信。

#### 1.找到你的Spark的本地IP地址

在 Spark 上，运行以下命令：

```bash
ip -f inet addr show enP7s7 | grep inet
```

或者如果您的 Spark 通过 Wi-Fi 连接，则使用此命令

```bash
ip -f inet addr show wlP9s9 | grep inet
```

找到 LAN 上的 IPv4（通常类似于 `192.168.x.x` 或 `10.x.x.x`）。

#### 2. 确保您的手机可以访问 MinIO

- **同一网络**：将您的手机连接到与 DGX Spark 相同的 Wi‑Fi/LAN。
- **防火墙**：默认情况下，DGX Spark 不会阻止传入请求。如果您安装了防火墙，请允许入站流量流向 **`9010` (MinIO API)** 上的 DGX Spark。

#### 3.更新`.env`并重启

编辑 `.env` 并替换：

- **`EXTERNAL_MINIO_BASE_URL=127.0.0.1:9010`** → **`EXTERNAL_MINIO_BASE_URL=<SPARK_LAN_IP>:9010`**

然后重新启动：

```bash
docker compose down
docker compose up --build -d
```

## 步骤 7. 可选：进一步发展并自定义应用程序

### 指南

- [Getting Started](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/docs/getting-started.md) – 深入的设置和配置演练
- [Writing Your First Service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/docs/writing-your-first-service.md) – 如何创建和集成新服务

### 服务配置

每个服务都有自己的自述文件，其中包含有关自定义、环境变量和特定于服务的配置的详细信息：

| 服务 | 描述 |
|---------|-------------|
| [agent-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/agent-service/README.md) | LLM 支持的代理工作流程和决策逻辑 |
| [animation-compositor-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/animation-compositor-service/README.md) | 结合动画剪辑和音频混合 |
| [animation-database-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/animation-database-service/README.md) | 动画库和程序动画生成 |
| [camera-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/camera-service/README.md) | 相机捕捉和图像采集 |
| [interaction-manager-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/interaction-manager-service/README.md) | 事件编排和机器人话语管理 |
| [metrics-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/metrics-service/README.md) | 指标收集和监控 |
| [remote-control-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/remote-control-service/README.md) | 基于网络的远程控制界面 |
| [robot-controller-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/robot-controller-service/README.md) | 直接机器人硬件控制 |
| [speech-to-text-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/speech-to-text-service/README.md) | 音频转录（NVIDIA Riva/Parakeet） |
| [text-to-speech-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/text-to-speech-service/README.md) | 语音合成 |
| [tracker-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/tracker-service/README.md) | 人员检测和跟踪 |
| [ui-server-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/ui-server-service/README.md) | Web UI 的后端 |

有关自定义服务配置、使用新工具扩展演示或创建您自己的服务的详细指南，请参阅 [Development](development) 选项卡。

## 发展

## 发展

本节提供有关自定义和开发 Reachy Photo Booth 应用程序的全面说明。如果您希望按原样部署和运行应用程序，请参阅 [Instructions](instructions) 选项卡 — 本开发指南专门针对那些需要对应用程序进行修改的人。

## 步骤1.系统依赖

为了使用存储库的 Python 开发设置，请安装以下软件包：

```bash
sudo apt install python3.12-dev portaudio19-dev
```

要创建 Python **venv**，请按照说明 [here](https://docs.astral.sh/uv/getting-started/installation/) 安装 uv。

然后运行以下命令生成Python **venv**：

```bash
uv sync --all-packages
```

## 步骤 2. 熟悉构建和开发过程

每个以 `-service` 为后缀的文件夹都是一个独立的 Python 程序，在其自己的容器中运行。您必须始终通过与存储库根目录下的 `docker-compose.yaml` 交互来启动服务。您可以通过运行以下命令为所有 Python 服务启用代码热重载：

```bash
docker compose up --build --watch
```

每当您更改存储库中的某些 Python 代码时，关联的容器都会更新并自动重新启动。

[Getting Started](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/docs/getting-started.md) 指南提供了构建系统、开发工作流程、调试策略和监控基础设施的全面演练。

## 步骤 3. 对应用程序进行更改

现在您的开发环境已经设置完毕，下面是开发人员通常探索的最常见的自定义项。

### 自定义配置参数

每个服务都有可配置的参数，包括系统提示、音频设备、模型设置等。检查各个服务自述文件和 `src/configuration.py` 文件以获取详细的配置选项。请注意，`src/configuration.py` 中的默认配置也可能在 `compose.yaml` 文件中被覆盖。查看以下服务以开始使用：

- [speech-to-text-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/speech-to-text-service/README.md) - 配置音频设备和转录设置
- [text-to-speech-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/text-to-speech-service/README.md) - 调整语音合成参数
- [agent-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/agent-service/README.md) - 自定义LLM系统提示、代理行为和决策逻辑

有关所有服务及其自述文件的完整列表，请参阅 [instructions](instructions)。

### 使用新工具扩展演示

代理服务和交互管理器服务是使用新功能扩展演示的核心服务：

- [agent-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/agent-service/README.md) - 在此处添加新的代理工具和功能
- [interaction-manager-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/interaction-manager-service/README.md) - 管理事件编排和机器人话语

### 创建您自己的服务

[Writing Your First Service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/docs/writing-your-first-service.md) 指南提供了有关搭建、实施新微服务并将其集成到系统中的分步教程。按照本指南创建扩展照相亭功能的自定义服务。

## 故障排除

| 症状 | 原因 | 使固定 |
|---------|-------|-----|
| 机器人没有声音（音量小） | 默认情况下，Reachy 扬声器音量设置得太低 | 将 Reachy 扬声器音量调至最大 |
| 机器人没有声音（设备冲突） | 另一个捕获 Reachy 扬声器的应用程序 | 检查 `animation-compositor` 日志中的“错误查询设备（-1）”，验证 Reachy 扬声器未在 Ubuntu 声音设置中设置为系统默认值，确保没有其他应用程序捕获扬声器，然后重新启动演示 |
| 首次启动时图像生成失败 | 瞬态初始化问题 | 重新运行 `docker compose up --build -d` 以解决问题 |

如果您对 Reachy 有任何本指南未涵盖的问题，请阅读 [Hugging Face's official troubleshooting guide](https://huggingface.co/docs/reachy_mini/troubleshooting)。

> [!NOTE] 
> DGX Spark 使用统一内存架构 (UMA)，可实现 GPU 和 CPU 之间的动态内存共享。 
> 由于许多应用程序仍在更新以利用 UMA，因此即使在 
> DGX Spark 的内存容量。如果发生这种情况，请使用以下命令手动刷新缓冲区缓存：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```


有关最新的已知问题，请查看 [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)。
