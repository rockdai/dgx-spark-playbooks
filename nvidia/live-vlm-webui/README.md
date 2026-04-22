# 实时 VLM WebUI

> 与网络摄像头流的实时视觉语言模型交互

## 目录

- [Overview](#overview)
- [Instructions](#instructions)
  - [Command Line Options](#command-line-options)
  - [Accept the SSL Certificate](#accept-the-ssl-certificate)
  - [Grant Camera Permissions](#grant-camera-permissions)
  - [Performance Optimization Tips](#performance-optimization-tips)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思路

Live VLM WebUI 是用于实时视觉语言模型 (VLM) 交互和基准测试的通用 Web 界面。它使您能够将网络摄像头直接流式传输到任何 VLM 后端（Ollama、vLLM、SGLang 或云 API）并接收实时 AI 支持的分析。该工具非常适合测试 VLM 模型、对不同硬件配置的性能进行基准测试以及探索视觉 AI 功能。

该界面提供基于 WebRTC 的视频流、集成 GPU 监控、可定制提示以及对多个 VLM 后端的支持。它与 DGX Spark 中强大的 Blackwell GPU 无缝协作，以令人印​​象深刻的速度实现实时视觉推理。

## 你将完成什么

您将在 DGX Spark 上设置完整的实时视觉 AI 测试环境，使您能够：

- 通过网络浏览器传输网络摄像头视频并获得即时 VLM 分析
- 测试和比较不同的视觉语言模型（Gemma 3、Llama Vision、Qwen VL 等）
- 在模型处理视频帧时实时监控 GPU 和系统性能
- 为各种用例定制提示（物体检测、场景描述、OCR、安全监控）
- 使用网络浏览器从网络上的任何设备访问该界面

## 开始之前需要了解什么

- 基本熟悉Linux命令行和终端操作
- 使用 pip 安装 Python 包的基础知识
- REST API 的基本知识以及服务如何通过 HTTP 进行通信
- 熟悉网络浏览器和网络访问（IP 地址、端口）
- 可选：了解视觉语言模型及其功能（有帮助，但不是必需的）

## 先决条件

**硬件要求：**
- 网络摄像头（笔记本电脑内置摄像头、USB 摄像头或带摄像头的远程浏览器）
- 至少 10GB 可用存储空间用于 Python 包和模型下载

**软件要求：**
- 安装了 DGX 操作系统的 DGX Spark
- Python 3.10 或更高版本（使用 `python3 --version` 验证）
- pip 包管理器（使用 `pip --version` 验证）
- 从 PyPI 下载 Python 包的网络访问
- 本地运行的 VLM 后端（Ollama 最简单）或云 API 访问
- Web 浏览器访问 `https://<SPARK_IP>:8090`

**VLM 后端选项：**
1. **Ollama**（推荐初学者）-易于安装和使用
2. **vLLM** - 生产工作负载性能更高
3. **SGLang** - 替代高性能后端
4. **NIM** - 用于优化性能的 NVIDIA 推理微服务
5. **云 API** - NVIDIA API Catalog、OpenAI 或其他 OpenAI 兼容 API

## 附属文件

所有源代码和文档都可以在 [Live VLM WebUI GitHub repository](https://github.com/NVIDIA-AI-IOT/live-vlm-webui) 中找到。

该软件包将直接通过 pip 安装，因此基本安装不需要额外的文件。

## 时间与风险

* **预计时间：** 20-30分钟（包括Ollama安装和模型下载）
  * 5 分钟通过 pip 安装 Live VLM WebUI
  * 安装 Ollama 并下载模型需要 10-15 分钟（因模型大小而异）
  * 5 分钟配置和测试
* **风险级别：**低
  * Python包安装在用户空间，与系统隔离
  * 无需进行系统级更改
  * 端口 8090 必须可访问 Web 界面功能
  * 自签名 SSL 证书需要浏览器安全例外
* **回滚：** 使用 `pip uninstall live-vlm-webui` 卸载 Python 包。 Ollama 可以通过标准软件包删除来卸载。 DGX Spark 配置没有持久性更改。
* **最后更新：** 2026 年 1 月 2 日
  * 首次出版

## 指示

## 步骤 1. 安装 Ollama 作为 VLM 后端

首先，安装 Ollama 来服务视觉语言模型。 Ollama 是在 DGX Spark 上本地运行/服务模型的最简单选项之一。

```bash
## Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

## Verify installation
ollama --version
```

Ollama 将自动作为系统服务启动并检测您的 Blackwell GPU。

现在下载视觉语言模型。我们建议从 `gemma3:4b` 开始进行快速测试：

```bash
## Download a lightweight model (recommended for testing)
ollama pull gemma3:4b

## Alternative models you can try:
## ollama pull llama3.2-vision:11b    # Sometime better quality, slower
## ollama pull qwen2.5-vl:7b          #
```

模型下载可能需要 5-15 分钟，具体取决于您的网络速度和模型大小。

验证 Ollama 是否正常工作：

```bash
## Check if Ollama API is accessible
curl http://localhost:11434/v1/models
```

预期输出应显示一个 JSON 响应，列出您下载的模型。

## 步骤 2. 安装 Live VLM WebUI

使用 pip 安装 Live VLM WebUI：

```bash
pip install live-vlm-webui
```

安装将下载所有必需的 Python 依赖项并安装 `live-vlm-webui` 命令。

现在启动服务器：

```bash
## Launch the web server
live-vlm-webui
```

服务器将：
- 自动生成 HTTPS SSL 证书（网络摄像头访问所需）
- 在端口 8090 上启动 WebRTC 服务器
- 自动检测您的 Blackwell GPU

服务器将启动并显示如下输出：

```
Starting Live VLM WebUI...
Generating SSL certificates...
GPU detected: NVIDIA GB10 Blackwell

Access the WebUI at:
  Local URL:   https://localhost:8090
  Network URL: https://<YOUR_SPARK_IP>:8090

Press Ctrl+C to stop the server
```

### 命令行选项

Live VLM WebUI 支持多种命令行选项进行自定义：

```bash
## Specify a different port
live-vlm-webui --port 8091

## Use custom SSL certificates
live-vlm-webui --ssl-cert /path/to/cert.pem --ssl-key /path/to/key.pem

## Change default API endpoint
live-vlm-webui --api-base http://localhost:8000/v1

## Run in background (optional)
nohup live-vlm-webui > live-vlm.log 2>&1 &
```

## 步骤 3. 访问 Web 界面

打开您的网络浏览器并导航至：

```
https://<YOUR_SPARK_IP>:8090
```

将 `<YOUR_SPARK_IP>` 替换为您的 DGX Spark 的 IP 地址。您可以通过以下方式找到它：

```bash
hostname -I | awk '{print $1}'
```

**重要提示：** 您必须使用 `https://`（而不是 `http://`），因为现代浏览器需要安全连接才能访问网络摄像头。

### 接受 SSL 证书

由于该应用程序使用自签名 SSL 证书，因此您的浏览器将显示安全警告。这是预期的且安全的。

**在 Chrome/Edge 中：**
1. 单击“**高级**”按钮
2. 点击“**继续\<YOUR_SPARK_IP\>（不安全）**”

**在火狐浏览器中：**
1. 单击“**高级...**”
2. 单击“**接受风险并继续**”

### 授予相机权限

出现提示时，允许网站访问您的相机。网络摄像头流应出现在界面中。

> [！提示]
> **推荐远程访问：** 为了获得最佳体验，请从同一网络上的笔记本电脑或 PC 访问 Web 界面。与在 DGX Spark 上本地访问相比，这提供了更好的浏览器性能和内置网络摄像头访问。

## 步骤 4. 配置 VLM 设置

该接口自动检测本地 VLM 后端。验证左侧边栏 **VLM API 配置** 部分中的配置：

**API 端点：** 应显示 `http://localhost:11434/v1` (Ollama)

**模型选择：** 单击下拉菜单并选择您下载的模型（例如 `gemma3:4b`）

**可选设置：**
- **最大令牌：** 控制响应长度（默认值：512，减少到 100-200 以获得更快的响应）
- **帧处理间隔：** 分析之间要跳过多少帧（默认值：30 帧，速度较慢则增加）

### 性能优化技巧

为了在 DGX Spark Blackwell GPU 上获得最佳性能：

- **模型选择：** `gemma3:4b` 给出 1-2 秒/帧，`llama3.2-vision:11b` 给出较慢的速度。
- **帧间隔：** 设置为 60 帧（30 fps 时为 2 秒）或更大，以便舒适观看
- **最大令牌：** 减少到 100 以加快响应速度

> [！笔记]
> DGX Spark 使用统一内存架构 (UMA)，可实现 GPU 和 CPU 之间的动态内存共享。
> 由于许多应用程序仍在更新以利用 UMA，因此即使在
> DGX Spark 的内存容量。如果发生这种情况，请使用以下命令手动刷新缓冲区缓存：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

## 第 5 步：开始分析视频

单击绿色的“**启动相机并开始 VLM 分析**”按钮。

该界面将：
1. 开始通过 WebRTC 传输您的网络摄像头
2. 开始处理帧并将其发送到 VLM
3. 实时显示AI分析结果
4. 在底部显示 GPU/CPU/RAM 指标

你应该看到：
- **右侧实时视频源**（带镜像切换）
- **VLM 分析结果** 覆盖在视频或信息框中
- **性能指标**显示延迟和帧数
- **GPU 监控** 显示 Blackwell GPU 利用率和 VRAM 使用情况

使用 DGX Spark 中的 Blackwell GPU，您应该会看到 `gemma3:4b` 的推理时间为 **每帧 1-2 秒**，而 `llama3.2-vision:11b` 的推理速度类似。

## 步骤 6. 自定义提示

左侧边栏底部的 **提示编辑器** 允许您自定义 VLM 分析的内容。

**快速提示** - 8 个可供使用的预设：
- **场景描述** - “用一句话描述您在此图像中看到的内容。”
- **对象检测** - “列出您可以在此图像中看到的所有对象，以逗号分隔。”
- **活动识别** - “描述此人的活动以及他们正在做什么。”
- **安全监控** - “是否存在任何可见的安全隐患？回答‘警报：描述’或‘安全’。”
- **OCR/文本识别** - “读取并转录图像中可见的任何文本。”
- 还有更多...

**自定义提示** - 输入您自己的提示：

尝试实时 CSV 输出（对下游应用程序有用）：

```
List all objects you can see in this image, separated by commas.
Do not include explanatory text. Output only the comma-separated list.
```

VLM 将立即开始使用新提示进行下一帧分析。这可以实现实时“提示工程”，您可以在观看实时结果的同时迭代和完善提示。

## 步骤 7. 测试不同的模型（可选）

想要比较型号吗？下载另一个模型并切换：

```bash
## Download another model
ollama pull llama3.2-vision:11b

## The model will appear in the Model dropdown in the web interface
```

在网络界面中：
1. 停止 VLM 分析（如果正在运行）
2. 从 **型号** 下拉列表中选择新型号
3. 再次开始VLM分析

比较 DGX Spark Blackwell GPU 上模型之间的推理速度和质量。

## 步骤 8. 监控性能

底部部分显示实时系统指标：

- **GPU 使用率** - Blackwell GPU 利用率百分比
- **VRAM 使用** - GPU 内存消耗
- **CPU 使用率** - 系统 CPU 利用率
- **系统 RAM** - 内存使用情况

使用这些指标可以：
- 在相同硬件上对不同模型进行基准测试
- 识别性能瓶颈
- 针对您的使用案例优化设置

## 步骤 9. 清理

完成后，在运行服务器的终端中使用 `Ctrl+C` 停止服务器。

要完全删除 Live VLM WebUI：

```bash
pip uninstall live-vlm-webui
```

您的 Ollama 安装和下载的模型仍然可供将来使用。

也删除 Ollama（可选）：

```bash
## Uninstall Ollama
sudo systemctl stop ollama
sudo rm /usr/local/bin/ollama
sudo rm -rf /usr/share/ollama

## Remove Ollama models (optional)
rm -rf ~/.ollama
```

## 步骤 10. 后续步骤

现在您已经运行了 Live VLM WebUI，请探索以下用例：

**模型基准测试：**
- 在 DGX Spark 上测试多个模型（Gemma 3、Llama Vision、Qwen VL）
- 比较推理延迟、准确性和 GPU 利用率
- 评估结构化输出功能（JSON、CSV）

**应用程序原型设计：**
- 使用 Web 界面作为构建您自己的 VLM 应用程序的参考
- 与 ROS 2 集成以实现机器人视觉
- 连接到 RTSP IP 摄像机进行安全监控（测试版功能）

**云API集成：**
- 从本地 Ollama 切换到云 API（NVIDIA API Catalog、OpenAI）
- 比较边缘与云推理性能和成本
- 测试混合部署

要使用 NVIDIA API Catalog 或其他云 API：

1. 在 **VLM API 配置** 部分中，将 **API 基本 URL** 更改为：
   - NVIDIA API 目录：`https://integrate.api.nvidia.com/v1`
   - OpenAI：`https://api.openai.com/v1`
   - 其他：您的自定义端点

2. 在出现的字段中输入您的 **API 密钥**

3. 从下拉列表中选择您的型号（列表是从 API 获取的）

**高级配置：**
- 使用 vLLM、SGLang 或 NIM 后端来提高吞吐量
- 设置 NIM 以优化 NVIDIA 特定性能
- 为您的特定用例定制 Python 后端

有关更高级的用法，请参阅 GitHub 上的 [full documentation](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/tree/main/docs)。

有关最新的已知问题，请查看 [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html) 和 [Live VLM WebUI Troubleshooting Guide](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/blob/main/docs/troubleshooting.md)。

## 故障排除

| 症状 | 原因 | 使固定 |
|---------|-------|-----|
| pip install 显示“错误：外部管理环境” | Python 3.12+ 阻止系统范围的 pip 安装 | 使用虚拟环境：`python3 -m venv live-vlm-env && source live-vlm-env/bin/activate && pip install live-vlm-webui` |
| 浏览器显示“您的连接不是私人连接”警告 | 应用程序使用自签名 SSL 证书 | 单击“高级”→“继续到 \<IP\>（不安全）” - 这是安全且预期的行为 |
| 相机无法访问或“权限被拒绝” | 浏览器需要 HTTPS 才能访问网络摄像头 | 确保您使用的是 `https://` （而不是 `http://`）。接受自签名证书警告并在出现提示时授予相机权限 |
| “无法连接到 VLM”或“连接被拒绝” | Ollama 或 VLM 后端未运行 | 验证 Ollama 是否正在使用 `curl http://localhost:11434/v1/models` 运行。如果没有运行，则从 `sudo systemctl start ollama` 开始 |
| VLM 响应非常慢（每帧 >5 秒） | 模型对于可用 VRAM 来说太大或配置不正确 | 尝试使用较小的模型（`gemma3:4b` 而不是较大的模型）。将帧处理间隔增加到 60 帧以上。将最大代币数量减少到 100-200 |
| GPU 统计信息显示所有指标均“N/A” | NVML 不可用或 GPU 驱动程序问题 | 使用 `nvidia-smi` 验证 GPU 访问。确保 NVIDIA 驱动程序已正确安装 |
| 模型下拉列表中“没有可用模型” | API端点不正确或模型未下载 | 验证 Ollama 的 API 端点是否为 `http://localhost:11434/v1`。下载带有 `ollama pull gemma3:4b` 的模型 |
| 服务器无法启动，“端口已在使用中” | 8090端口已被其他服务占用 | 停止冲突的服务或使用 `--port` 标志指定不同的端口：`live-vlm-webui --port 8091` |
| 无法从网络上的远程浏览器访问 | 防火墙阻止端口 8090 或错误的 IP 地址 | 验证防火墙允许端口 8090：`sudo ufw allow 8090`。使用 `hostname -I` 命令中的正确 IP |
| 视频流滞后或冻结 | 网络问题或浏览器性能 | 使用 Chrome 或 Edge 浏览器。从网络上的单独 PC（而不是本地）进行访问。检查网络带宽 |
| 分析结果采用意想不到的语言 | 模型支持多语言和提示中检测到的语言 | 在提示中明确指定输出语言：“用英语回答：描述您所看到的内容” |
| pip 安装失败并出现依赖性错误 | Python 包版本冲突 | 尝试使用 `--user` 标志安装：`pip install --user live-vlm-webui` |
| 安装后未找到命令 `live-vlm-webui` | 二进制路径不在 PATH 中 | 将 `~/.local/bin` 添加到 PATH: `export PATH="$HOME/.local/bin:$PATH"` 然后运行 ​​`source ~/.bashrc` |
| 相机可以工作，但没有出现 VLM 分析结果，浏览器显示 InvalidStateError | 从远程机器通过 SSH 端口转发访问 | WebRTC 需要直接网络连接，不能通过 SSH 隧道工作（SSH 仅转发 TCP，WebRTC 需要 UDP）。 **解决方案 1**：直接从与服务器位于同一网络上的浏览器访问 Web UI。 **解决方案2**：直接使用服务器机器的浏览器。 **解决方案3**：使用X11转发（`ssh -X`）远程显示浏览器 |
