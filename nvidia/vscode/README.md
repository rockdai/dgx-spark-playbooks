# VS代码

> 本地或远程安装和使用 VS Code

## 目录

- [Overview](#overview)
- [Direct Installation](#direct-installation)
- [Access with NVIDIA Sync](#access-with-nvidia-sync)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思路
本演练将帮助您设置 Visual Studio Code（具有扩展功能的全功能 IDE、集成终端和 Git 集成），同时利用 DGX Spark 设备进行开发和测试。使用 VS Code 有两种不同的方法：

 * **直接安装**：直接在基于 ARM64 的 Spark 系统上安装 VS Code 开发环境，以便在目标硬件上进行本地开发，无需远程开发开销。

 * **使用 NVIDIA Sync 进行访问**：设置 NVIDIA Sync 以通过 SSH 远程连接到 Spark，并将 VS Code 配置为您的开发工具之一。

## 你将完成什么
您将设置 VS Code 以在 DGX Spark 设备上进行开发，并可以访问系统的 ARM64 架构和 GPU 资源。此设置支持直接代码开发、调试和执行。

## 开始之前需要了解什么
您应该具有使用 VS Code 界面和功能的基本经验；您选择的方法需要一些额外的理解：

* **直接安装**：
  * 熟悉Linux系统上的包管理
  * 了解 Linux 上的文件权限和身份验证

* **通过 NVIDIA Sync 访问**：
  * 熟悉 SSH 概念

## 先决条件
您的 DGX Spark [device is set up](https://docs.nvidia.com/dgx/dgx-spark/first-boot.html)。您还需要以下内容：

* **直接安装**：
  * DGX Spark 设置为具有管理权限
  * 用于下载 VS Code 安装程序的有效互联网连接

* **通过 NVIDIA Sync 访问**：
  * 安装在笔记本电脑上的 VS Code，从 https://code.visualstudio.com/download. 下载

## 时间与风险

* **持续时间：** 10-15 分钟
* **风险级别：**低 - 安装使用带有标准回滚的官方软件包
* **回滚：** 通过系统包管理器删除标准包
* **最后更新：** 2025 年 11 月 21 日
  * 澄清选项和较小的文案编辑

## 直接安装

## 步骤 1. 验证系统要求

在安装 VS Code 之前，请确认您的 DGX Spark 系统满足要求并具有 GUI 支持。

```bash
## Verify ARM64 architecture
uname -m
## Expected output: aarch64

## Check available disk space (VS Code requires ~200MB)
df -h /

## Verify desktop environment is running
ps aux | grep -E "(gnome|kde|xfce)"

## Verify GUI desktop environment is available
  echo $DISPLAY
## Should return display information like :0 or :10.0
```

## 步骤 2. 下载 VS Code ARM64 安装程序

导航到 VS Code [download](https://code.visualstudio.com/download) 页面并下载适合您的系统的 ARM64 `.deb` 软件包。 

或者，您可以使用以下命令下载安装程序：

```bash
wget https://code.visualstudio.com/sha/download?build=stable\&os=linux-deb-arm64 -O vscode-arm64.deb
```

## 步骤3.安装VS Code包

使用系统包管理器安装下载的包。 

您可以直接单击安装程序文件或使用命令行。 

```bash
## Install the downloaded .deb package
sudo dpkg -i vscode-arm64.deb

## Fix any dependency issues if they occur
sudo apt-get install -f
```

## 步骤 4. 验证安装

确认 VS Code 应用程序已成功安装并且可以启动。 

您可以直接从应用程序列表打开应用程序或使用命令行。 

```bash
## Check if VS Code is installed
which code

## Verify version
code --version

## Test launch (will open VS Code GUI)
code &
```

VS Code 应启动并显示欢迎屏幕。

## 步骤 5. 配置 Spark 开发

设置 VS Code 以在 DGX Spark 平台上进行开发。

```bash
## Launch VS Code if not already running
code

## Or create a new project directory and open it
mkdir ~/spark-dev-workspace
cd ~/spark-dev-workspace
code .
```

从 VS Code 中：

* 打开 **文件** > **首选项** > **设置**
* 搜索“terminal Integrated shell”以配置默认终端
* 通过 **扩展** 选项卡（左侧边栏）安装推荐的扩展

## 步骤 6. 验证设置和测试功能

测试核心 VS Code 功能以确保在 ARM64 上正常运行。

创建测试文件：
```bash
## Create test directory and file
mkdir ~/vscode-test
cd ~/vscode-test
echo 'print("Hello from DGX Spark!")' > test.py
code test.py
```

在 VS 代码中：
* 验证语法高亮是否有效
* 打开集成终端（**终端** > **新终端**）
* 运行测试脚本：`python3 test.py`
* 通过在终端中运行 `git status` 测试 Git 集成

## 步骤 8. 卸载 VS Code

> [!WARNING]
> 卸载 VS Code 将删除所有用户设置和扩展。

如果需要删除 VS Code：
```bash
## Remove VS Code package
sudo apt-get remove code

## Remove configuration files (optional)
rm -rf ~/.config/Code
rm -rf ~/.vscode
```

## 通过 NVIDIA Sync 进行访问

## 步骤 1. 安装并配置 NVIDIA Sync

按照 [NVIDIA Sync setup guide](https://build.nvidia.com/spark/connect-to-your-spark/sync) 进行：
- 为您的操作系统安装 NVIDIA Sync
- 配置您要使用的开发工具（VS Code、光标、终端等）
- 通过提供主机名/IP 和凭据来添加您的 DGX Spark 设备

NVIDIA Sync 将自动配置基于 SSH 密钥的身份验证，以实现安全、无密码的访问。

## 步骤 2. 通过 NVIDIA Sync 启动 VS Code

- 单击系统托盘/任务栏中的 NVIDIA Sync 图标
- 确保您的设备已连接（如果需要，请单击“连接”）
- 单击“VS Code”以通过与 DGX Spark 的自动 SSH 连接来启动它
- 等待建立远程连接（您的本地计算机可能会要求输入密码或授权连接）
- 成功 SSH 连接后首次登陆主目录时，系统可能会提示您“信任此文件夹中文件的作者”

## 步骤 3. 验证和跟进

- 验证您是否可以使用 VS Code 作为文本编辑器访问 DGX Spark 的文件系统
- 在 VS Code 中打开集成终端并运行 `hostnamectl` 和 `whoami` 等测试命令，以确保您正在远程访问 DGX Spark
- 导航到特定文件路径或目录并开始编辑/写入文件
- 为您的开发工作流程安装 VS Code 扩展（Python、Docker、GitLens 等）
- 从 GitHub 或其他版本控制系统克隆存储库
- 如果需要，配置并在本地托管 LLM 代码助理

## 故障排除

| 症状 | 原因 | 使固定 |
|---------|-------|-----|
| 安装期间的 `dpkg: dependency problems` | 缺少依赖项 | 运行 `sudo apt-get install -f` |
| VS Code 无法启动并出现 GUI 错误 | 无显示服务器/X11 | 验证 GUI 桌面是否正在运行：`echo $DISPLAY` |
| 扩展无法安装 | 网络连接或 ARM64 兼容性 | 检查互联网连接，验证扩展 ARM64 支持 |


有关最新的已知问题，请查看 [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)。
