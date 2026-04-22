---
id: tailscale
title: tailscale
sidebar_label: tailscale
---

# 在 Spark 上设置 Tailscale

> 无论您身在何处，都可以使用 Tailscale 连接到家庭网络上的 Spark


## 目录

- [Overview](#overview)
- [Instructions](#instructions)
  - [Step 1. Verify system requirements](#step-1-verify-system-requirements)
  - [Step 2. Install SSH server (if needed)](#step-2-install-ssh-server-if-needed)
  - [Step 3. Install Tailscale on NVIDIA DGX Spark](#step-3-install-tailscale-on-nvidia-dgx-spark)
  - [Step 4. Verify Tailscale installation](#step-4-verify-tailscale-installation)
  - [Step 5. Connect your DGX Spark to Tailscale network](#step-5-connect-your-dgx-spark-to-tailscale-network)
  - [Step 6. Install Tailscale on client devices](#step-6-install-tailscale-on-client-devices)
  - [Step 7. Connect client devices to tailnet](#step-7-connect-client-devices-to-tailnet)
  - [Step 8. Verify network connectivity](#step-8-verify-network-connectivity)
  - [Step 9. Configure SSH authentication](#step-9-configure-ssh-authentication)
  - [Step 10. Test SSH connection](#step-10-test-ssh-connection)
  - [Step 11. Validate installation](#step-11-validate-installation)
  - [Step 13. Cleanup and rollback](#step-13-cleanup-and-rollback)
  - [Step 14. Next steps](#step-14-next-steps)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思路

Tailscale 创建一个加密的点对点网状网络，允许安全访问
从任何地方连接到您的 NVIDIA DGX Spark 设备，无需复杂的防火墙配置
或端口转发。通过在 DGX Spark 和客户端设备上安装 Tailscale，
您建立一个私有“尾网”，其中每个设备都获得稳定的私有IP
地址和主机名，无论您在家、工作、
或咖啡店。

## 你将完成什么

您将在 DGX Spark 设备和客户端计算机上设置 Tailscale
创建安全的远程访问。完成后，您将能够通过 SSH 访问您的
使用 `ssh user@spark-hostname` 等简单命令从任何地方使用 DGX Spark
所有流量自动加密，NAT 穿越透明处理。

## 开始之前需要了解什么

- 使用终端/命令行界面
- SSH 基本概念和用法
- 在 Ubuntu 上使用 `apt` 安装软件包
- 了解用户帐户和身份验证
- 熟悉systemd服务管理

## 先决条件

**硬件要求：**
-  NVIDIA Grace Blackwell GB10 超级芯片系统

**软件要求：**
- NVIDIA DGX 操作系统
- 用于远程访问的客户端设备（Mac、Windows 或 Linux）
- 测试连接时客户端设备和 DGX Spark 不在同一网络上
- 两台设备上的互联网连接
- 用于 Tailscale 身份验证的有效电子邮件帐户（Google、GitHub、Microsoft）
- SSH 服务器可用性检查：`systemctl status ssh`
- 包管理器工作：`sudo apt update`
- 在 DGX Spark 设备上具有 sudo 权限的用户帐户

## 时间与风险

* **持续时间**：初始设置 15-30 分钟，每个附加设备 5 分钟
* **风险**：中
  * 潜在的 SSH 服务配置冲突
  * 初始设置期间的网络连接问题
  * 身份验证提供程序服务依赖项
* **回滚**：可以使用 `sudo apt remove tailscale` 完全删除 Tailscale，所有网络路由自动恢复为默认设置。
* **最后更新：** 11/07/2025
  * 少量文案编辑

## 指示

### 步骤 1. 验证系统要求

检查您的 NVIDIA DGX Spark 设备是否正在运行受支持的 Ubuntu 版本，并且
有互联网连接。此步骤在 DGX Spark 设备上运行以确认
先决条件。

```bash
## Check Ubuntu version (should be 20.04 or newer)
lsb_release -a

## Test internet connectivity
ping -c 3 google.com

## Verify you have sudo access
sudo whoami
```

### 步骤 2. 安装 SSH 服务器（如果需要）

确保 SSH 服务器在您的 DGX Spark 设备上运行，因为 Tailscale 提供了
网络连接，但需要 SSH 进行远程访问。此步骤运行于
DGX Spark 设备。

```bash
## Check if SSH is running
systemctl status ssh --no-pager
```

**如果未安装或运行 SSH：**

```bash
## Install OpenSSH server
sudo apt update
sudo apt install -y openssh-server

## Enable and start SSH service
sudo systemctl enable ssh --now --no-pager

## Verify SSH is running
systemctl status ssh --no-pager
```

### 步骤 3. 在 NVIDIA DGX Spark 上安装 Tailscale

使用官方 Ubuntu 在 DGX Spark 上安装 Tailscale
存储库。此步骤添加 Tailscale 软件包存储库并安装
客户。

```bash
## Update package list
sudo apt update

## Install required tools for adding external repositories
sudo apt install -y curl gnupg

## Add Tailscale signing key
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.noarmor.gpg | \
  sudo tee /usr/share/keyrings/tailscale-archive-keyring.gpg > /dev/null

## Add Tailscale repository
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.tailscale-keyring.list | \
  sudo tee /etc/apt/sources.list.d/tailscale.list

## Update package list with new repository
sudo apt update

## Install Tailscale
sudo apt install -y tailscale
```

### 步骤 4. 验证 Tailscale 安装

在继续之前，请确认 Tailscale 已正确安装在您的 DGX Spark 设备上
与认证。

```bash
## Check Tailscale version
tailscale version

## Check Tailscale service status
sudo systemctl status tailscaled --no-pager
```

### 步骤 5. 将 DGX Spark 连接到 Tailscale 网络

使用您选择的身份通过 Tailscale 验证您的 DGX Spark 设备
提供者。这将创建您的私有尾网并分配稳定的 IP 地址。

```bash
## Start Tailscale and begin authentication
sudo tailscale up

## Follow the URL displayed to complete login in your browser
## Choose from: Google, GitHub, Microsoft, or other supported providers
```

### 步骤 6. 在客户端设备上安装 Tailscale

在您将用于远程连接到 DGX Spark 的设备上安装 Tailscale。

选择适合您的客户端操作系统的方法：

**在 macOS 上：**
- 选项 1：从 Mac App Store 搜索“Tailscale”然后单击“获取”→“安装”进行安装
- 选项 2：从 [Tailscale website](https://tailscale.com/download) 下载 .pkg 安装程序


**在 Windows 上：**
- 从 [Tailscale website](https://tailscale.com/download) 下载安装程序
- 运行 .msi 文件并按照安装提示进行操作
- 从“开始”菜单或系统托盘启动 Tailscale


**在 Linux 上：**

请遵循用于 DGX Spark 安装的相同说明。

```bash
## Update package list
sudo apt update

## Install required tools for adding external repositories
sudo apt install -y curl gnupg

## Add Tailscale signing key
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.noarmor.gpg | \
  sudo tee /usr/share/keyrings/tailscale-archive-keyring.gpg > /dev/null

## Add Tailscale repository
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.tailscale-keyring.list | \
  sudo tee /etc/apt/sources.list.d/tailscale.list

## Update package list with new repository
sudo apt update

## Install Tailscale
sudo apt install -y tailscale
```

### 步骤 7. 将客户端设备连接到 tailnet

使用相同的身份提供商在每个客户端设备上登录 Tailscale
您用于 DGX Spark 的帐户。

**在 macOS/Windows (GUI) 上：**
- 启动 Tailscale 应用程序
- 点击“登录”按钮
- 使用 DGX Spark 上使用的相同帐户登录

**在 Linux (CLI) 上：**

```bash
## Start Tailscale on client
sudo tailscale up

## Complete authentication in browser using same account
```

### 步骤 8. 验证网络连接

之前测试设备是否可以通过 Tailscale 网络进行通信
尝试 SSH 连接。

```bash
## On any device, check tailnet status
tailscale status

## Test ping to Spark device (use hostname or IP from status output)
tailscale ping `SPARK_HOSTNAME`

## Example output should show successful pings
```

### 步骤 9. 配置 SSH 身份验证

设置 SSH 密钥身份验证以安全访问 DGX Spark。这
步骤在您的客户端设备和 DGX Spark 设备上运行。

**在客户端上生成 SSH 密钥（如果尚未完成）：**

```bash
## Generate new SSH key pair
ssh-keygen -t ed25519 -f ~/.ssh/tailscale_spark

## Display public key to copy
cat ~/.ssh/tailscale_spark.pub
```

**将公钥添加到 DGX Spark：**

```bash
## On Spark device, add client's public key
echo "`<YOUR_PUBLIC_KEY>`" >> ~/.ssh/authorized_keys

## Set correct permissions
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh
```

### 步骤 10. 测试 SSH 连接

使用 SSH 通过 Tailscale 网络连接到 DGX Spark 进行验证
完整的设置有效。

```bash
## Connect using Tailscale hostname (preferred)
ssh -i ~/.ssh/tailscale_spark `USERNAME`@`SPARK_HOSTNAME`

## Or connect using Tailscale IP address
ssh -i ~/.ssh/tailscale_spark `USERNAME`@`TAILSCALE_IP`

## Example:
## ssh -i ~/.ssh/tailscale_spark nvidia@my-spark-device
```

### 步骤 11. 验证安装

验证 Tailscale 是否正常工作以及您的 SSH 连接是否稳定。

```bash
## From client device, check connection status
tailscale status

## Create a test file on the client device
echo "test file for the spark" > test.txt

## Test file transfer over SSH
scp -i ~/.ssh/tailscale_spark test.txt `USERNAME`@`SPARK_HOSTNAME`:~/

## Verify you can run commands remotely
ssh -i ~/.ssh/tailscale_spark `USERNAME`@`SPARK_HOSTNAME` 'nvidia-smi'
```

预期输出：
- Tailscale 状态将两个设备显示为“活动”
- 文件传输成功
- 远程命令执行工作

### 步骤 13. 清理和回滚

如果需要，请完全移除尾鳞。这将断开设备与
tailnet 并删除所有网络配置。

> [!WARNING]
> 这将从您的 Tailscale 网络中永久删除该设备，并需要重新身份验证才能重新加入。

```bash
## Stop Tailscale service
sudo tailscale down

## Remove Tailscale package
sudo apt remove --purge tailscale

## Remove repository and keys (optional)
sudo rm /etc/apt/sources.list.d/tailscale.list
sudo rm /usr/share/keyrings/tailscale-archive-keyring.gpg

## Update package list
sudo apt update
```

要恢复：重新运行安装步骤 3-5。

### 步骤 14. 后续步骤

您的 Tailscale 设置已完成。您现在可以：

- 通过以下方式从任何网络访问您的 DGX Spark 设备：`ssh `USERNAME`@`SPARK_HOSTNAME`
- 安全地传输文件：`scp file.txt `USERNAME`@`SPARK_HOSTNAME`:~/`
- 打开 DGX 仪表板并启动 JupyterLab，然后连接：
  `ssh -L 8888:localhost:1102 `USERNAME`@`SPARK_HOSTNAME`

## 故障排除

| 症状 | 原因 | 使固定 |
|---------|-------|-----|
| `tailscale up` 身份验证失败 | 网络问题 | 检查互联网，尝试 `curl -I login.tailscale.com` |
| SSH 连接被拒绝 | SSH 未运行 | 在 Spark 上运行 `sudo systemctl start ssh --no-pager` |
| SSH 身份验证失败 | SSH 密钥错误 | 检查 `~/.ssh/authorized_keys` 中的公钥 |
| 无法 ping 通主机名 | DNS问题 | 使用 `tailscale status` 的 IP 代替 |
| 设备丢失 | 不同的账户 | 为所有设备使用相同的身份提供商 |


有关最新的已知问题，请查看 [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)。
