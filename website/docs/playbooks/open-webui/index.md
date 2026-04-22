---
id: open-webui
title: open-webui
sidebar_label: open-webui
---

# 使用 Ollama 打开 WebUI

> 安装 Open WebUI 并使用 Ollama 与 Spark 上的模型聊天

## 目录

- [Overview](#overview)
- [Set up Open WebUI on Remote Spark with NVIDIA Sync](#set-up-open-webui-on-remote-spark-with-nvidia-sync)
- [Set Up Manually](#set-up-manually)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思路

Open WebUI 是一个可扩展、自托管的 AI 界面，完全离线运行。
本手册向您展示了如何在 DGX Spark 设备上部署带有集成 Ollama 服务器的 Open WebUI，该设备允许您在模型在 Spark 的 GPU 上运行时从本地浏览器访问 Web 界面。

## 你将完成什么

您将在 DGX Spark 上运行功能齐全的 Open WebUI 安装。这可以通过您的本地 Web 浏览器通过 **NVIDIA Sync 的托管 SSH 隧道（推荐）** 或通过手动设置进行访问。该设置包括用于模型管理的集成 Ollama、持久数据存储和用于模型推理的 GPU 加速。

## 开始之前需要了解什么

- 如何 [Set Up Local Network Access](/spark/connect-to-your-spark) 到您的 DGX Spark 设备

## 先决条件

-  DGX Spark [device is set up](https://docs.nvidia.com/dgx/dgx-spark/first-boot.html) 且可访问
-  [Local Network Access](/spark/connect-to-your-spark) 到您的 DGX Spark
-  有足够的磁盘空间用于 Open WebUI 容器映像和模型下载

## 时间与风险

* **持续时间**：初始设置需要 15-20 分钟，加上模型下载时间（因模型大小而异）
* **风险**：
  * Docker 权限问题可能需要更改用户组并重新启动会话
  * 大型模型下载可能需要大量时间，具体取决于网络速度
* **最后更新：** 2025 年 10 月 28 日
  * 少量文案编辑

## 使用 NVIDIA Sync 在远程 Spark 上设置 Open WebUI

> [！提示]
> 如果您尚未安装 NVIDIA Sync，[learn how here.](/spark/connect-to-your-spark/sync)

## 步骤1.配置Docker权限

要使用 NVIDIA Sync 轻松管理容器，您必须能够在不使用 sudo 的情况下运行 Docker 命令。 

从 NVIDIA Sync 打开终端应用程序以启动交互式 SSH 会话并测试 Docker 访问。在终端中，运行：

```bash
docker ps
```

如果您看到权限被拒绝错误（例如尝试连接到 Docker 守护程序套接字时权限被拒绝），请将您的用户添加到 docker 组，这样您就不需要使用 sudo 运行命令。

```bash
sudo usermod -aG docker $USER
newgrp docker
```

再次测试 Docker 访问。在终端中，运行：

```bash
docker ps
```

## 步骤 2. 验证 Docker 设置并拉取容器

从 NVIDIA Sync 打开一个新的终端应用程序，并在 DGX Spark 上提取带有集成 Ollama 的 Open WebUI 容器映像：

```bash
docker pull ghcr.io/open-webui/open-webui:ollama
```

下载容器映像后，继续设置 NVIDIA Sync。

## 步骤 3. 打开 NVIDIA 同步设置

- 单击系统托盘或任务栏中的 NVIDIA Sync 图标以打开主应用程序窗口。
- 单击右上角的齿轮图标打开“设置”窗口。
- 单击“自定义”选项卡以访问自定义端口配置。

## 步骤 4. 添加 Open WebUI 自定义端口配置

自定义端口用于自动启动 Open WebUI 容器并设置端口转发。

- 单击“自定义”选项卡上的“添加新项”按钮。

使用以下值填写表格：

- **名称**：打开WebUI
- **端口**：12000
- **在浏览器中自动打开以下路径**：选中此复选框
- **启动脚本**：复制并粘贴整个脚本：

```bash
#!/usr/bin/env bash
set -euo pipefail

NAME="open-webui"
IMAGE="ghcr.io/open-webui/open-webui:ollama"

cleanup() {
  echo "Signal received; stopping ${NAME}..."
  docker stop "${NAME}" >/dev/null 2>&1 || true
  exit 0
}
trap cleanup INT TERM HUP QUIT EXIT

## Ensure Docker CLI and daemon are available
if ! docker info >/dev/null 2>&1; then
  echo "Error: Docker daemon not reachable." >&2
  exit 1
fi

## Already running?
if [ -n "$(docker ps -q --filter "name=^${NAME}$" --filter "status=running")" ]; then
  echo "Container ${NAME} is already running."
else
#  # Exists but stopped? Start it.
  if [ -n "$(docker ps -aq --filter "name=^${NAME}$")" ]; then
    echo "Starting existing container ${NAME}..."
    docker start "${NAME}" >/dev/null
  else
#    # Not present: create and start it.
    echo "Creating and starting ${NAME}..."
    docker run -d -p 12000:8080 --gpus=all \
      -v open-webui:/app/backend/data \
      -v open-webui-ollama:/root/.ollama \
      --name "${NAME}" "${IMAGE}" >/dev/null
  fi
fi

echo "Running. Press Ctrl+C to stop ${NAME}."
## Keep the script alive until a signal arrives
while :; do sleep 86400; done
```

- 单击“添加”按钮将配置保存到 DGX Spark。

## 步骤 5. 启动打开 WebUI

- 单击系统托盘或任务栏中的 NVIDIA Sync 图标以打开主应用程序窗口。
- 在“自定义”部分下，单击“打开 WebUI”。

您的默认 Web 浏览器应自动打开至位于 `http://localhost:12000` 的 Open WebUI 界面。

> [！提示]
> 首次运行时，Open WebUI 将下载模型。这可能会延迟服务器启动并导致页面无法在浏览器中加载。只需等待并刷新页面即可。
> 在未来的发布中，它将快速打开。

## 步骤 6. 创建管理员帐户

要开始使用 Open WebUI，您必须创建一个初始管理员帐户。这是一个本地帐户，您将用它来访问 Open WebUI 界面。

- 在打开的WebUI界面中，单击屏幕底部的“开始”按钮。
- 使用您首选的凭据填写管理员帐户创建表单。
- 单击注册按钮创建您的帐户并访问主界面。

## 步骤 7. 下载并配置模型

接下来，使用 Ollama 下载语言模型并将其配置为在
打开网络用户界面。此下载发生在您的 DGX Spark 设备上，可能需要几分钟时间。

- 单击 Open WebUI 界面左上角的“选择模型”下拉菜单。
- 在搜索字段中输入 `gpt-oss:20b`。
- 单击出现的 `Pull "gpt-oss:20b" from Ollama.com` 按钮。
- 等待模型下载完成。您可以在界面中监控进度。
- 完成后，从模型下拉列表中选择“gpt-oss:20b”。

## 步骤 8. 测试模型

您可以通过测试模型来验证设置是否正常工作。

- 在 Open WebUI 界面底部的聊天文本区域中，输入：**给我写一首关于 GPU 的俳句**。
- 按 Enter 发送消息并等待模型的响应。

## 步骤 9. 停止打开 WebUI 

当您完成会话并想要停止 Open WebUI 服务器并回收资源时，请从 NVIDIA Sync 关闭 Open WebUI。

- 单击系统托盘或任务栏中的 NVIDIA Sync 图标以打开主应用程序窗口。
- 在“自定义”部分下，单击“打开 WebUI”条目右侧的 `x` 图标。
- 这将关闭隧道并停止 Open WebUI docker 容器。

## 步骤 10. 后续步骤

尝试从 Ollama 库（https://ollama.com/library.）下载不同的模型

当您尝试不同的模型时，您可以通过 NVIDIA Sync 中提供的 DGX 仪表板监控 GPU 和内存使用情况。

如果 Open WebUI 报告有可用更新，您可以通过在终端中运行以下命令来拉取容器映像：

```bash
docker stop open-webui
docker rm open-webui 
docker pull ghcr.io/open-webui/open-webui:ollama
```

更新后，再次从 NVIDIA Sync 启动 Open WebUI。

## 步骤 11. 清理和回滚

完全删除 Open WebUI 安装并释放资源的步骤。

> [！警告]
> 这些命令将永久删除所有 Open WebUI 数据和下载的模型。

停止并删除 Open WebUI 容器：

```bash
docker stop open-webui
docker rm open-webui
```

删除下载的图像：

```bash
docker rmi ghcr.io/open-webui/open-webui:ollama
```

删除持久数据卷：

```bash
docker volume rm open-webui open-webui-ollama
```

打开“设置”>“自定义”选项卡并删除该条目，从 NVIDIA Sync 中删除自定义应用程序。

## 手动设置

## 步骤1.配置Docker权限

要在不使用 sudo 的情况下轻松管理容器，您必须位于 `docker` 组中。如果您选择跳过此步骤，则需要使用 sudo 运行 Docker 命令。

打开新终端并测试 Docker 访问。在终端中，运行：

```bash
docker ps
```

如果您看到权限被拒绝错误（例如尝试连接到 Docker 守护程序套接字时权限被拒绝），请将您的用户添加到 docker 组，这样您就不需要使用 sudo 运行命令。

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## 步骤 2. 验证 Docker 设置并拉取容器

使用集成的 Ollama 拉取 Open WebUI 容器映像：

```bash
docker pull ghcr.io/open-webui/open-webui:ollama
```

## 步骤 3. 启动 Open WebUI 容器

通过运行以下命令启动 Open WebUI 容器：

```bash
docker run -d -p 8080:8080 --gpus=all \
  -v open-webui:/app/backend/data \
  -v open-webui-ollama:/root/.ollama \
  --name open-webui ghcr.io/open-webui/open-webui:ollama
```

这将启动 Open WebUI 容器并使其可在 `http://localhost:8080` 处访问。您可以从本地 Web 浏览器访问 Open WebUI 界面。

> [！笔记]
> 应用程序数据将存储在 `open-webui` 卷中，模型数据将存储在 `open-webui-ollama` 卷中。

## 步骤4.创建管理员帐户

设置 Open WebUI 的初始管理员帐户。这是一个本地帐户，您将用它来访问 Open WebUI 界面。

- 在打开的WebUI界面中，单击屏幕底部的“开始”按钮。
- 使用您首选的凭据填写管理员帐户创建表单。
- 单击注册按钮创建您的帐户并访问主界面。

## 步骤 5. 下载并配置模型

然后，您将通过 Ollama 下载语言模型并将其配置为在
打开网络用户界面。此下载发生在您的 DGX Spark 设备上，可能需要几分钟时间。

- 单击 Open WebUI 界面左上角的“选择模型”下拉菜单。
- 在搜索字段中输入 `gpt-oss:20b`。
- 单击出现的“从 Ollama.com 拉取‘gpt-oss:20b’”按钮。
- 等待模型下载完成。您可以在界面中监控进度。
- 完成后，从模型下拉列表中选择“gpt-oss:20b”。

## 步骤 6. 测试模型

您可以通过测试模型来验证设置是否正常工作
通过网络界面进行推理。

- 在 Open WebUI 界面底部的聊天文本区域中，输入：**给我写一首关于 GPU 的俳句**。
- 按 Enter 发送消息并等待模型的响应。

## 步骤 7. 后续步骤

尝试从 Ollama 库（https://ollama.com/library.）下载不同的模型

您可以尝试这个 [set up with NVIDIA Sync](/spark/open-webui/sync) ，以便在尝试不同模型时通过 DGX 仪表板监控 GPU 和内存使用情况。

如果 Open WebUI 报告有可用更新，您可以通过运行以下命令来更新容器映像：

```bash
docker pull ghcr.io/open-webui/open-webui:ollama
```

## 步骤 8. 清理和回滚

完全删除 Open WebUI 安装并释放资源的步骤。

> [！警告]
> 这些命令将永久删除所有 Open WebUI 数据和下载的模型。

停止并删除 Open WebUI 容器：

```bash
docker stop open-webui
docker rm open-webui
```

删除下载的图像：

```bash
docker rmi ghcr.io/open-webui/open-webui:ollama
```

删除持久数据卷：

```bash
docker volume rm open-webui open-webui-ollama
```

## 故障排除

## 通过 NVIDIA Sync 设置的常见问题

| 症状 | 原因 | 使固定 |
|---------|-------|-----|
| docker ps 上的权限被拒绝 | 用户不在docker组中 | 完全运行步骤 1，包括终端重启 |
| 浏览器不会自动打开 | 自动打开设置已禁用 | 手动导航到 localhost:12000 |
| 模型下载失败 | 网络连接问题 | 检查互联网连接，重试下载 |
| 容器中未检测到 GPU | 缺少 `--gpus=all flag` | 使用正确的启动脚本重新创建容器 |
| 端口 12000 已被使用 | 使用端口的另一个应用程序 | 更改自定义应用程序设置中的端口或停止冲突的服务 |

## 手动设置的常见问题

| 症状 | 原因 | 使固定 |
|---------|-------|-----|
| docker ps 上的权限被拒绝 | 用户不在docker组中 | 完全运行步骤 1，包括注销并重新登录或使用 sudo|
| 模型下载失败 | 网络连接问题 | 检查互联网连接，重试下载 |
| 容器中未检测到 GPU | 缺少 `--gpus=all flag` | 使用正确的命令重新创建容器 |
| 8080端口已被使用 | 使用端口的另一个应用程序 | 更改 docker 命令中的端口或停止冲突的服务 |

> [！笔记]
> DGX Spark 使用统一内存架构 (UMA)，可实现 GPU 和 CPU 之间的动态内存共享。 
> 由于许多应用程序仍在更新以利用 UMA，因此即使在 
> DGX Spark 的内存容量。如果发生这种情况，请使用以下命令手动刷新缓冲区缓存：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
