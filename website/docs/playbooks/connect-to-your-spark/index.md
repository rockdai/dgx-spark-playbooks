---
id: connect-to-your-spark
title: connect-to-your-spark
sidebar_label: connect-to-your-spark
---

# 设置本地网络访问

> NVIDIA Sync 可帮助设置并配置 SSH 访问

## 目录

- [概述](#overview)
- [使用 NVIDIA Sync 连接](#connect-with-nvidia-sync)
- [使用手动 SSH 连接](#connect-with-manual-ssh)
- [故障排查](#troubleshooting)

---

<a id="overview"></a>
## 概述

## 基本思路

如果你主要在另一台系统上工作，例如笔记本电脑，并希望将 DGX Spark 作为远程资源来使用，本 playbook 将向你展示如何通过 SSH 连接并开展工作。借助 SSH，你可以安全地打开终端会话，或通过隧道访问本地机器上的 Web 应用和 API，而这些服务实际上运行在 DGX Spark 上。

这里有两种方式：**NVIDIA Sync（推荐）** 适合更顺畅的设备管理体验，**手动 SSH** 则提供直接的命令行控制能力。

开始之前，有几个重要概念需要了解：

**Secure Shell (SSH)** 是一种加密协议，用于通过不可信网络安全连接到远程计算机。它可以让你像坐在 DGX Spark 设备前一样打开终端、执行命令、传输文件和管理服务，且通信全程加密。

**SSH tunneling**（也称端口转发）可以将你笔记本上的一个端口（例如 localhost:8888）安全映射到 DGX Spark 上某个应用监听的端口（例如运行在 8888 端口上的 JupyterLab）。你的浏览器连接的是 localhost，而 SSH 会通过加密连接把流量转发到远端服务，无需将该端口暴露到更广泛的网络中。

**mDNS (Multicast DNS)** 允许设备在本地网络中通过名称相互发现，而不需要中心化 DNS 服务器。你的 DGX Spark 会通过 mDNS 广播自己的主机名，因此你可以直接使用类似 `spark-abcd.local`（注意 `.local` 后缀）这样的名称进行连接，而不必手动查找 IP 地址。

## 你将完成的内容

你将通过 NVIDIA Sync 或手动 SSH 配置，在 DGX Spark 设备上建立安全的 SSH 访问。NVIDIA Sync 提供图形化的设备管理界面，并集成应用启动能力；手动 SSH 则提供直接的命令行控制和端口转发能力。两种方式都允许你从笔记本远程执行终端命令、访问 Web 应用并管理 DGX Spark。


## 开始前需要了解

- 基础终端/命令行使用
- 理解 SSH 基本概念和基于密钥的认证
- 熟悉主机名、IP 地址、端口转发等网络概念

## 前置条件

- 你的 DGX Spark [设备已完成设置](https://docs.nvidia.com/dgx/dgx-spark/first-boot.html)，并已创建本地用户账户
- 你的笔记本和 DGX Spark 处于同一网络中
- 你知道自己的 DGX Spark 用户名和密码
- 你拥有设备的 mDNS 主机名（印在 Quick Start Guide 上）或 IP 地址

## 时间与风险

- **预计耗时：** 5-10 分钟
- **风险等级：** 低 - SSH 设置涉及凭据配置，但不会对 DGX Spark 设备进行系统级更改
- **回滚：** 可通过编辑 DGX Spark 上的 `~/.ssh/authorized_keys` 删除 SSH 密钥。
- **最后更新：** 10/28/2025
  * 文案小幅修订

<a id="connect-with-nvidia-sync"></a>
## 使用 NVIDIA Sync 连接

## 第 1 步：安装 NVIDIA Sync

NVIDIA Sync 是一款桌面应用，可通过本地网络将你的电脑连接到 DGX Spark。
它为你提供一个统一界面，用于管理 SSH 访问并在 DGX Spark 上启动开发工具。

请先在你的电脑上下载并安装 NVIDIA Sync。

::spark-download

**对于 macOS**

- 下载后，打开 `nvidia-sync.dmg`
- 将应用拖放到 Applications 文件夹中
- 在 Applications 文件夹中打开 `NVIDIA Sync`

**对于 Windows**

- 下载后，运行安装程序 `.exe`
- 安装完成后，NVIDIA Sync 会自动启动


**对于 Debian/Ubuntu**

* 配置软件包仓库：

  ```
  curl -fsSL  https://workbench.download.nvidia.com/stable/linux/gpgkey  |  sudo tee -a /etc/apt/trusted.gpg.d/ai-workbench-desktop-key.asc
  echo "deb https://workbench.download.nvidia.com/stable/linux/debian default proprietary" | sudo tee -a /etc/apt/sources.list
  ```
* 更新软件包列表：

  ```
  sudo apt update
  ```
* 安装 NVIDIA Sync：

  ```
  sudo apt install nvidia-sync
  ```

## 第 2 步：配置 Apps

Apps 是安装在笔记本上的桌面程序，NVIDIA Sync 可以为其自动配置到 Spark 的连接并直接启动。

你可以随时在 Settings 窗口中更改所选应用。标记为 "unavailable" 的应用需要先安装，才能使用。

**默认应用：**
- **DGX Dashboard**：预装在 DGX Spark 上的 Web 应用，用于系统管理并集成 JupyterLab 访问
- **Terminal**：系统自带终端，可自动建立 SSH 连接

**可选应用（需要单独安装）：**
- **VS Code**：从 https://code.visualstudio.com/download 下载
- **Cursor**：从 https://cursor.com/downloads 下载
- **NVIDIA AI Workbench**：从 https://www.nvidia.com/workbench 下载

## 第 3 步：添加你的 DGX Spark 设备

> [!NOTE]
> 你必须知道主机名或 IP 地址中的至少一种，才能完成连接。
>
> - 默认主机名可以在包装盒内附带的 Quick Start Guide 上找到，例如 `spark-abcd.local`
> - 如果你的设备连接了显示器，也可以在 [DGX Dashboard](http://localhost:11000) 的 Settings 页面中查看主机名
> - 如果你的网络不支持 `.local`（mDNS）主机名，则必须使用 IP 地址。这可以在 Ubuntu 的网络设置中查看，或通过登录路由器管理控制台获取。

最后，在表单中填写以下信息以连接你的 DGX Spark：

- **Name**：描述性名称（例如 "My DGX Spark"）
- **Hostname or IP**：Spark 的 mDNS 主机名（例如 `spark-abcd.local`）或 IP 地址
- **Username**：你的 DGX Spark 用户名
- **Password**：你的 DGX Spark 用户密码

> [!NOTE]
> 你的密码只会在首次设置期间用于配置基于 SSH 密钥的认证。设置完成后不会被存储或再次传输。NVIDIA Sync 会通过 SSH 登录你的设备，并配置本地生成的 SSH 密钥对。

点击 "Add" 按钮后，NVIDIA Sync 会自动：

1. 在你的笔记本上生成 SSH 密钥对
2. 使用你提供的用户名和密码连接到 DGX Spark
3. 将公钥添加到设备上的 `~/.ssh/authorized_keys`
4. 在本地创建一个 SSH 别名，供后续连接使用
5. 丢弃你的用户名和密码信息

> [!IMPORTANT]
> 设备首次完成系统初始化后，可能需要几分钟时间进行更新并重新出现在网络中。如果 NVIDIA Sync 连接失败，请等待 3-4 分钟后再试。

## 第 4 步：访问你的 DGX Spark

连接建立后，NVIDIA Sync 会以系统托盘/任务栏应用的形式运行。点击 NVIDIA Sync 图标即可打开设备管理界面。

- **SSH connection**：点击大号的 "Connect" 和 "Disconnect" 按钮，用于控制与你设备之间的整体 SSH 连接状态。
- **Set working directory**（可选）：选择 Apps 通过 NVIDIA Sync 启动时默认打开的目录。默认值为远程设备上的 home 目录。
- **Launch applications**：点击任意已配置应用，即可在自动连接到 DGX Spark 的同时启动它。
- **Customize ports**（可选）："Custom Ports" 可在 Settings 页面中配置，用于访问运行在你设备上的自定义 Web 应用或 API。

## 第 5 步：验证 SSH 设置

NVIDIA Sync 会为你的设备创建一个 SSH 别名，方便你手动使用或在其他支持 SSH 的应用中使用。

通过该 SSH 别名验证本地 SSH 配置是否正确。使用别名时，你不应再被要求输入密码：

```bash
## Configured if you use mDNS hostname
ssh `SPARK_HOSTNAME`.local
```

或

```bash
## Configured if you use IP address
ssh `<IP>`
```

在 DGX Spark 上，验证你已成功连接：

```bash
hostname
whoami
```

退出 SSH 会话：

```bash
exit
```

## 第 6 步：后续操作

通过启动一个开发工具来测试你的设置：
- 点击 NVIDIA Sync 的系统托盘图标。
- 选择 "Terminal" 打开连接到 DGX Spark 的终端会话。
- 选择 "DGX Dashboard" 使用 JupyterLab 并管理更新。
- 试试[配合 Open WebUI 的自定义端口示例](/spark/open-webui/sync)。

<a id="connect-with-manual-ssh"></a>
## 使用手动 SSH 连接

## 第 1 步：确认 SSH 客户端可用

确认你的系统已安装 SSH 客户端。大多数现代操作系统都默认内置 SSH。请在终端中运行：

```bash
## Check SSH client version
ssh -V
```

预期输出应显示 OpenSSH 版本信息。

## 第 2 步：收集连接信息

收集连接 DGX Spark 所需的信息：

- **Username**：你的 DGX Spark 用户名
- **Password**：你的 DGX Spark 账户密码
- **Hostname**：设备的 mDNS 主机名（来自 Quick Start Guide，例如 `spark-abcd.local`）
- **IP Address**：仅当你的网络不支持 mDNS 时才需要使用，见下文说明

在某些网络配置下，例如复杂的企业网络环境，mDNS 可能无法正常工作，此时你需要直接使用设备 IP 地址进行连接。如果你尝试 SSH 时命令一直卡住不返回，或者看到类似下面的错误，就说明你很可能处于这种情况：

```
ssh: Could not resolve hostname spark-abcd.local: Name or service not known
```

**测试 mDNS 解析**

使用 `ping` 工具测试 mDNS 是否工作正常：

```bash
ping spark-abcd.local
```

如果 mDNS 工作正常，并且你可以通过主机名进行 SSH，应该会看到类似下面的输出：

```
$ ping -c 3 spark-abcd.local
PING spark-abcd.local (10.9.1.9): 56 data bytes
64 bytes from 10.9.1.9: icmp_seq=0 ttl=64 time=6.902 ms
64 bytes from 10.9.1.9: icmp_seq=1 ttl=64 time=116.335 ms
64 bytes from 10.9.1.9: icmp_seq=2 ttl=64 time=33.301 ms
```

如果 mDNS **无法**工作，意味着你需要直接使用 IP 连接，那么你会看到类似下面的结果：

```
$ ping -c 3 spark-abcd.local
ping: cannot resolve spark-abcd.local: Unknown host
```

如果上述方法都不奏效，你需要：
- 登录路由器管理面板，查找设备的 IP 地址
- 连接显示器、键盘和鼠标，并在 Ubuntu 桌面中查看

## 第 3 步：测试初始连接

首次连接到 DGX Spark，以验证基础连通性：

```bash
## Connect using mDNS hostname (preferred)
ssh `YOUR_USERNAME`@`SPARK_HOSTNAME`.local
```

或

```bash
## Alternative: Connect using IP address
ssh `YOUR_USERNAME`@`<DEVICE_IP_ADDRESS>`
```

将占位符替换为你的实际值：
- `YOUR_USERNAME`：你的 DGX Spark 账户名
- `SPARK_HOSTNAME`：不带 `.local` 后缀的设备主机名
- `<DEVICE_IP_ADDRESS>`：设备的 IP 地址

首次连接时，你会看到主机指纹警告。输入 `yes` 并按回车，然后在提示时输入密码。

## 第 4 步：验证远程连接

连接成功后，确认你已经登录到 DGX Spark：

```bash
## Check hostname
hostname
## Check system information
uname -a
## Exit the session
exit
```

## 第 5 步：使用 SSH 隧道访问 Web 应用

如果要访问运行在 DGX Spark 上的 Web 应用，请使用 SSH 端口转发。以下示例将访问 DGX Dashboard Web 应用。

> [!NOTE]
> DGX Dashboard 运行在 localhost 的 11000 端口。

打开隧道：

```bash
## local port 11000 → remote port 11000
ssh -L 11000:localhost:11000 `YOUR_USERNAME`@`SPARK_HOSTNAME`.local
```

建立隧道后，在浏览器中访问转发后的 Web 应用：[http://localhost:11000](http://localhost:11000)

## 第 6 步：后续操作

配置好 SSH 访问后，你可以：
- 打开持久化终端会话：`ssh `YOUR_USERNAME`@`SPARK_HOSTNAME`.local`。
- 转发 Web 应用端口：`ssh -L <local_port>:localhost:<remote_port> `YOUR_USERNAME`@`SPARK_HOSTNAME`.local`。

<a id="troubleshooting"></a>
## 故障排查

## 通过 NVIDIA Sync 连接时的常见问题

| 现象 | 原因 | 解决方法 |
|---------|--------|-----|
| 设备名称无法解析 | 网络阻止了 mDNS | 使用 IP 地址代替 hostname.local |
| 连接被拒绝/超时 | DGX Spark 尚未完成启动或 SSH 尚未就绪 | 等待设备启动完成；更新完成后 SSH 才会可用 |
| 认证失败 | SSH 密钥设置未完成 | 在 NVIDIA Sync 中重新执行设备设置；检查凭据 |

## 通过手动 SSH 连接时的常见问题

| 现象 | 原因 | 解决方法 |
|---------|--------|-----|
| 设备名称无法解析 | 网络阻止了 mDNS | 使用 IP 地址代替 hostname.local |
| 连接被拒绝/超时 | DGX Spark 尚未完成启动或 SSH 尚未就绪 | 等待设备启动完成；更新完成后 SSH 才会可用 |
| 端口转发失败 | 服务未运行或端口冲突 | 确认远端服务已启动；尝试其他本地端口 |
