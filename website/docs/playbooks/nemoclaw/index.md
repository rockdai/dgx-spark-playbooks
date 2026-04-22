---
id: nemoclaw
title: nemoclaw
sidebar_label: nemoclaw
---

# NemoClaw 与 Nemotron 3 Super 和 DGX Spark 上的 Telegram

> 在 DGX Spark 上安装 NemoClaw，并集成本地 Ollama 推理和 Telegram 机器人


## 目录

- [Overview](#overview)
  - [Overview](#overview)
  - [Basic idea](#basic-idea)
  - [What you'll accomplish](#what-youll-accomplish)
  - [Notice and disclaimers](#notice-and-disclaimers)
  - [Isolation layers (OpenShell)](#isolation-layers-openshell)
  - [What to know before starting](#what-to-know-before-starting)
  - [Prerequisites](#prerequisites)
  - [Have ready before you begin](#have-ready-before-you-begin)
  - [Ancillary files](#ancillary-files)
  - [Time and risk](#time-and-risk)
- [Instructions](#instructions)
  - [Step 1. Configure Docker and the NVIDIA container runtime](#step-1-configure-docker-and-the-nvidia-container-runtime)
  - [Step 2. Install Ollama](#step-2-install-ollama)
  - [Step 3. Pull the Nemotron 3 Super model](#step-3-pull-the-nemotron-3-super-model)
  - [Step 4. Install NemoClaw](#step-4-install-nemoclaw)
  - [Step 5. Connect to the sandbox and verify inference](#step-5-connect-to-the-sandbox-and-verify-inference)
  - [Step 6. Talk to the agent (CLI)](#step-6-talk-to-the-agent-cli)
  - [Step 7. Interactive TUI](#step-7-interactive-tui)
  - [Step 8. Exit the sandbox and access the Web UI](#step-8-exit-the-sandbox-and-access-the-web-ui)
  - [Step 9. Create a Telegram bot](#step-9-create-a-telegram-bot)
  - [Step 10. Configure and start the Telegram bridge](#step-10-configure-and-start-the-telegram-bridge)
  - [Step 11. Stop services](#step-11-stop-services)
  - [Step 12. Uninstall NemoClaw](#step-12-uninstall-nemoclaw)
- [Troubleshooting](#troubleshooting)

---

## 概述

### 概述

### 基本思路

**NVIDIA NemoClaw** 是一个开源参考堆栈，可更安全地简化 OpenClaw 始终在线助手的运行。它安装 **NVIDIA OpenShell** 运行时（一种专为执行具有额外安全性的代理而设计的环境）以及 NVIDIA Nemotron 等开源模型。单个安装程序命令即可处理 Node.js、OpenShell 和 NemoClaw CLI，然后引导您通过板载向导使用 Ollama 和 Nemotron 3 Super 在 DGX Spark 上创建沙盒代理。

在本剧本结束时，您将在 OpenShell 沙箱内拥有一个可工作的 AI 代理，可通过 Web 仪表板和 Telegram 机器人进行访问，并将推理路由到 Spark 上的本地 Nemotron 3 Super 120B 模型 - 所有这些都无需将您的主机文件系统或网络暴露给代理。

### 你将完成什么

- 在 DGX Spark 上为 OpenShell 配置 Docker 和 NVIDIA 容器运行时
- 安装 Ollama，拉动 Nemotron 3 Super 120B，并将其配置为沙盒访问
- 使用单个命令安装 NemoClaw（处理 Node.js、OpenShell 和 CLI）
- 运行板载向导来创建沙箱并配置本地推理
- 通过 CLI、TUI 和 Web UI 与客服人员聊天
- 设置一个 Telegram 机器人，将消息转发给您的沙盒代理

### 通知和免责声明

以下部分描述了运行此演示时的安全、风险和您的责任。

#### 快速启动安全检查

**仅使用干净的环境。** 在没有个人数据、机密信息或敏感凭据的新设备或虚拟机上运行此演示。像沙箱一样将其隔离。

通过安装此演示，您对所有第三方组件承担责任，包括审查其许可证、条款和安全状况。安装或使用之前请阅读并接受。

#### 你得到什么

此体验“按原样”提供，仅用于演示目的——不做任何保证。这是一个演示，而不是生产就绪的解决方案。您将需要针对您的环境和用例实施适当的安全控制。

#### 人工智能代理的主要风险

- **数据泄露**——代理访问的任何材料都可能被暴露、泄露或被盗。
- **恶意代码执行** - 代理或其连接的工具可能会使您的系统遭受恶意代码或网络攻击。
- **意外操作** - 代理可能会在未经明确批准的情况下修改或删除文件、发送消息或访问服务。
- **及时注入和操纵** - 外部输入或连接的内容可能会以意想不到的方式劫持代理的行为。

#### 参与者致谢

通过参与此演示，您承认您对您的配置以及您连接的任何数据、帐户和工具承担全部责任。在法律允许的最大范围内，对于因配置或使用 NemoClaw 演示材料（包括 OpenClaw 或任何连接的工具或服务）而导致的任何数据丢失、设备损坏、安全事件或其他损害，NVIDIA 概不负责。

### 隔离层 (OpenShell)

| 层      | 它保护什么                                   | 当它适用时             |
|------------|----------------------------------------------------|-----------------------------|
| 文件系统 | 防止在允许的路径之外进行读/写。       | 锁定在沙箱创建时。  |
| 网络    | 阻止未经授权的出站连接。          | 运行时可热重载。  |
| 过程    | 阻止权限升级和危险的系统调用。| 锁定在沙箱创建时。  |
| 推理  | 将模型 API 调用重新路由到受控后端。   | 运行时可热重载。  |

### 开始之前需要了解什么

- Linux终端和SSH的基本使用
- 熟悉 Docker（权限，`docker run`）
- 对上述安全和风险部分的认识

### 先决条件

**硬件和访问：**

- 具有键盘和显示器或 SSH 访问权限的 DGX Spark (GB10)
- 来自 [build.nvidia.com](https://build.nvidia.com/settings/api-keys) 的 **NVIDIA API 密钥**（Telegram 桥需要）
- 来自 [@BotFather](https://t.me/BotFather) 的 **Telegram 机器人令牌**（使用 `/newbot` 创建一个）

**软件：**

- 全新安装带有最新更新的 DGX 操作系统

开始之前验证您的系统：

```bash
head -n 2 /etc/os-release
nvidia-smi
docker info --format '{{.ServerVersion}}'
```

预计：Ubuntu 24.04、NVIDIA GB10 GPU、Docker 28.x+。

### 开始前做好准备

| 物品 | 哪里可以得到它 |
|------|----------------|
| NVIDIA API 密钥 | [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys) |
| Telegram 机器人令牌 | Telegram 上的 [@BotFather](https://t.me/BotFather) -- 使用 `/newbot` 创建 |

### 附属文件

所有必需的资源均由 NemoClaw 安装程序处理。无需手动克隆。

### 时间和风险

- **预计时间：** 20--30 分钟（Ollama 和模型已下载）。首次模型下载会增加约 15--30 分钟，具体取决于网络速度。
- **风险级别：**中——您正在沙箱中运行人工智能代理；隔离可以降低风险，但不能消除风险。使用干净的环境，不要连接敏感数据或生产帐户。
- **最后更新：** 2026 年 3 月 31 日
  * 首次出版

## 指示

## 第一阶段：先决条件

这些步骤为 NemoClaw 准备一个新的 DGX Spark。如果 Docker、NVIDIA 运行时和 Ollama 已配置，请跳至第 2 阶段。

### 步骤 1. 配置 Docker 和 NVIDIA 容器运行时

OpenShell 的网关在 Docker 内运行 k3s。在 DGX Spark（Ubuntu 24.04，cgroup v2）上，必须使用 NVIDIA 运行时和主机 cgroup 命名空间模式配置 Docker。

为 Docker 配置 NVIDIA 容器运行时：

```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

在 DGX Spark 上设置 OpenShell 所需的 cgroup 命名空间模式：

```bash
sudo python3 -c "
import json, os
path = '/etc/docker/daemon.json'
d = json.load(open(path)) if os.path.exists(path) else {}
d['default-cgroupns-mode'] = 'host'
json.dump(d, open(path, 'w'), indent=2)
"
```

重新启动 Docker：

```bash
sudo systemctl restart docker
```

验证 NVIDIA 运行时是否正常工作：

```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

如果您在 `docker` 上收到权限被拒绝错误，请将您的用户添加到 Docker 组并在当前会话中激活新组：

```bash
sudo usermod -aG docker $USER
newgrp docker
```

这会立即应用组更改。或者，您可以注销并重新登录，而不是运行 `newgrp docker`。

> [！笔记]
> DGX Spark 使用 cgroup v2。 OpenShell 的网关将 k3s 嵌入到 Docker 中，并且需要主机 cgroup 命名空间访问。如果没有 `default-cgroupns-mode: host`，网关可能会失败并出现“无法启动 ContainerManager”错误。

### 步骤2.安装Ollama

安装奥拉玛：

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

配置 Ollama 以侦听所有接口，以便沙箱容器可以访问它：

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\nEnvironment="OLLAMA_HOST=0.0.0.0"\n' | sudo tee /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

验证它正在运行并且在所有接口上均可访问：

```bash
curl http://0.0.0.0:11434
```

预期：`Ollama is running`。如果没有，则以 `sudo systemctl start ollama` 开头。

> [！重要的]
> 始终通过 systemd (`sudo systemctl restart ollama`) 启动 Ollama — 不要使用 `ollama serve &`。手动启动的 Ollama 进程不会采用上面的 `OLLAMA_HOST=0.0.0.0` 设置，并且 NemoClaw 沙箱将无法到达推理服务器。

### 步骤3.拉动 Nemotron 3 Super 模型

下载 Nemotron 3 Super 120B（~87 GB；可能需要 15--30 分钟，具体取决于网络速度）：

```bash
ollama pull nemotron-3-super:120b
```

短暂运行它以将权重预加载到内存中（输入 `/bye` 退出）：

```bash
ollama run nemotron-3-super:120b
```

验证模型是否可用：

```bash
ollama list
```

您应该在输出中看到 `nemotron-3-super:120b`。

---

## 第 2 阶段：安装并运行 NemoClaw

### 步骤 4. 安装 NemoClaw

这个命令可以处理所有事情：安装 Node.js（如果需要）、安装 OpenShell、克隆最新稳定的 NemoClaw 版本、构建 CLI 以及运行板载向导来创建沙箱。

```bash
curl -fsSL https://www.nvidia.com/nemoclaw.sh | bash
```

板载向导将引导您完成设置：

1. **沙箱名称** -- 选择一个名称（例如 `my-assistant`）。名称必须是小写字母数字，仅包含连字符。
2. **推理提供程序** - 选择**本地 Ollama**（选项 7）。
3. **型号** - 选择**nemotron-3-super:120b**（选项 1）。
4. **策略预设** - 出现提示时接受建议的预设（点击 **Y**）。

完成后您将看到如下输出：

```text
──────────────────────────────────────────────────
Dashboard    http://localhost:18789/
Sandbox      my-assistant (Landlock + seccomp + netns)
Model        nemotron-3-super:120b (Local Ollama)
──────────────────────────────────────────────────
Run:         nemoclaw my-assistant connect
Status:      nemoclaw my-assistant status
Logs:        nemoclaw my-assistant logs --follow
──────────────────────────────────────────────────
```

> [！重要的]
> 保存最后打印的标记化 Web UI URL——您将在第 8 步中需要它。它看起来像：
> `http://127.0.0.1:18789/#token=<long-token-here>`

> [！笔记]
> 如果安装后未找到 `nemoclaw`，请运行 `source ~/.bashrc` 重新加载 shell 路径。

### 步骤 5. 连接到沙箱并验证推理

连接到沙箱：

```bash
nemoclaw my-assistant connect
```

您将看到 `sandbox@my-assistant:~$` ——您现在位于沙盒环境中。

验证推理路由是否正常工作：

```bash
curl -sf https://inference.local/v1/models
```

预期：JSON 列表 `nemotron-3-super:120b`。

### 步骤 6. 与代理交谈 (CLI)

仍在沙箱内，发送测试消息：

```bash
openclaw agent --agent main --local -m "hello" --session-id test
```

代理将使用 Nemotron 3 Super 进行响应。对于本地运行的 120B 参数模型，首次响应可能需要 30--90 秒。

### 步骤 7. 交互式 TUI

启动交互式聊天会话的终端 UI：

```bash
openclaw tui
```

按 **Ctrl+C** 退出 TUI。

### 步骤8.退出沙箱并访问Web UI

退出沙箱返回主机：

```bash
exit
```

**如果直接在 Spark 上访问 Web UI**（连接键盘和显示器），请打开浏览器并导航到步骤 4 中的标记化 URL：

```text
http://127.0.0.1:18789/#token=<long-token-here>
```

**如果从远程计算机访问 Web UI**，您需要设置端口转发。

首先，找到 Spark 的 IP 地址。在 Spark 上运行：

```bash
hostname -I | awk '{print $1}'
```

这将打印主 IP 地址（例如 `192.168.1.42`）。您还可以在 Spark 桌面上的 **设置 > Wi-Fi** 或 **设置 > 网络** 中找到它，或者检查路由器的连接设备列表。

在 Spark 主机上启动端口转发：

```bash
openshell forward start 18789 my-assistant --background
```

然后从远程计算机创建到 Spark 的 SSH 隧道（将 `<your-spark-ip>` 替换为上面的 IP 地址）：

```bash
ssh -L 18789:127.0.0.1:18789 <your-user>@<your-spark-ip>
```

现在在远程计算机的浏览器中打开标记化的 URL：

```text
http://127.0.0.1:18789/#token=<long-token-here>
```

> [！重要的]
> 使用 `127.0.0.1`，而不是 `localhost` —— 网关来源检查需要完全匹配。

---

## 第三阶段：Telegram 机器人

> [！笔记]
> 如果您已在 NemoClaw 入门向导（步骤 5/8）期间配置了 Telegram，则可以跳过此阶段。这些步骤包括在初始设置后添加 Telegram。

### 第 9 步：创建 Telegram 机器人

打开 Telegram，找到 [@BotFather](https://t.me/BotFather)，发送 `/newbot`，然后按照提示操作。复制它为您提供的机器人令牌。

### 步骤 10. 配置并启动 Telegram 桥

确保您位于 **主机**（不在沙箱内）。如果您位于沙箱内，请先运行 `exit`。

设置所需的环境变量。将占位符替换为您的实际值。 `SANDBOX_NAME` 必须与您在板载向导中选择的沙箱名称匹配：

```bash
export TELEGRAM_BOT_TOKEN=<your-bot-token>
export SANDBOX_NAME=my-assistant
export NVIDIA_API_KEY=<your-nvidia-api-key>
```

将 Telegram 网络策略添加到沙箱中：

```bash
nemoclaw my-assistant policy-add
```

出现提示时，选择 `telegram` 并点击 **Y** 进行确认。

启动电报桥。

```bash
export TELEGRAM_BOT_TOKEN=<your-bot-token>
nemoclaw start
```

仅当设置 `TELEGRAM_BOT_TOKEN` 环境变量时，Telegram 桥才会启动。验证服务正在运行：

```bash
nemoclaw status
```

打开 Telegram，找到您的机器人，然后向其发送消息。机器人将其转发给代理并回复。

> [！笔记]
> 对于本地运行的 120B 参数模型，第一次响应可能需要 30--90 秒。

> [！笔记]
> 如果桥未出现在 `nemoclaw status` 中，请确保在运行 `nemoclaw start` 的同一 shell 会话中导出 `TELEGRAM_BOT_TOKEN`。您还可以尝试停止并重新启动：
> ````重击
> 尼莫爪停止
> 导出 TELEGRAM_BOT_TOKEN=<your-bot-token>
> 尼莫爪开始
> ````

> [！笔记]
> 有关限制哪些 Telegram 聊天可以与代理交互的详细信息，请参阅 [NemoClaw Telegram bridge documentation](https://docs.nvidia.com/nemoclaw/latest/deployment/set-up-telegram-bridge.html)。

---

## 第 4 阶段：清理和卸载

### 步骤11.停止服务

停止任何正在运行的辅助服务（Telegram 桥、cloudflared 隧道）：

```bash
nemoclaw stop
```

停止端口转发：

```bash
openshell forward list          # find active forwards
openshell forward stop 18789    # stop the dashboard forward
```

### 步骤12.卸载NemoClaw

从克隆的源目录运行卸载程序。它会删除所有沙箱、OpenShell 网关、Docker 容器/映像/卷、CLI 和所有状态文件。 Docker、Node.js、npm 和 Ollama 均被保留。

```bash
cd ~/.nemoclaw/source
./uninstall.sh
```

**卸载程序标志：**

| 旗帜 | 影响 |
|------|--------|
| `--yes` | 跳过确认提示 |
| `--keep-openshell` | 将 `openshell` 二进制文件保留在原处 |
| `--delete-models` | 同时删除 NemoClaw 拉出的 Ollama 模型 |

要删除包括 Ollama 模型在内的所有内容：

```bash
./uninstall.sh --yes --delete-models
```

卸载程序运行 6 个步骤：
1. 停止 NemoClaw 帮助程序服务和端口转发进程
2. 删除所有 OpenShell 沙箱、NemoClaw 网关和提供商
3. 删除全局 `nemoclaw` npm 包
4. 删除 NemoClaw/OpenShell Docker 容器、镜像和卷
5. 删除 Ollama 模型（仅适用于 `--delete-models`）
6. 删除状态目录（`~/.nemoclaw`、`~/.config/openshell`、`~/.config/nemoclaw`）和 OpenShell 二进制文件

> [！笔记]
> 作为步骤 6 中状态清理的一部分，将删除 `~/.nemoclaw/source` 处的源克隆。如果要保留本地副本，请在运行卸载程序之前将其移动或备份。

## 有用的命令

| 命令 | 描述 |
|---------|-------------|
| `nemoclaw my-assistant connect` | 将 shell 放入沙箱中 |
| `nemoclaw my-assistant status` | 显示沙箱状态和推理配置 |
| `nemoclaw my-assistant logs --follow` | 实时流式传输沙箱日志 |
| `nemoclaw list` | 列出所有已注册的沙箱 |
| `nemoclaw start` | 启动辅助服务（Telegrambridge、cloudflared） |
| `nemoclaw stop` | 停止辅助服务 |
| `openshell term` | 打开主机上的监控TUI |
| `openshell forward list` | 列出活动端口转发 |
| `openshell forward start 18789 my-assistant --background` | 重新启动 Web UI 的端口转发 |
| `cd ~/.nemoclaw/source && ./uninstall.sh` | 删除 NemoClaw（保留 Docker、Node.js、Ollama） |
| `cd ~/.nemoclaw/source && ./uninstall.sh --delete-models` | 删除 NemoClaw 和 Ollama 模型 |

## 故障排除

| 症状 | 原因 | 使固定 |
|---------|-------|-----|
| 安装后`nemoclaw: command not found` | Shell 路径未更新 | 运行 `source ~/.bashrc` （或 zsh 的 `source ~/.zshrc` ），或打开一个新的终端窗口。 |
| 安装程序因 Node.js 版本错误而失败 | Node.js 版本低于 20 | 安装 Node.js 20+：`curl -fsSL https://deb.nodesource.com/setup_22.x \| sudo -E bash - && sudo apt-get install -y nodejs` 然后重新运行安装程序。 |
| npm 安装失败并出现 `EACCES` 权限错误 | npm 全局目录不可写 | `mkdir -p ~/.npm-global && npm config set prefix ~/.npm-global && export PATH=~/.npm-global/bin:$PATH` 然后重新运行安装程序。将 `export` 行添加到 `~/.bashrc` 以使其永久化。 |
| Docker 权限被拒绝 | 用户不在docker组中 | `sudo usermod -aG docker $USER`，然后注销并重新登录。 |
| 网关失败并出现 cgroup /“无法启动 ContainerManager”错误 | 未为 DGX Spark 上的主机 cgroup 命名空间配置 Docker | 运行 cgroup 修复：`sudo python3 -c "import json, os; path='/etc/docker/daemon.json'; d=json.load(open(path)) if os.path.exists(path) else {}; d['default-cgroupns-mode']='host'; json.dump(d, open(path,'w'), indent=2)"` 然后是 `sudo systemctl restart docker`。或者，运行 `sudo nemoclaw setup-spark` 来自动应用此修复。 |
| 网关失败并显示“端口 8080 由容器占用...” | 另一个 OpenShell 网关或容器正在使用端口 8080 | 停止冲突的容器：`openshell gateway destroy -g <old-gateway-name>` 或 `docker stop <container-name> && docker rm <container-name>`，然后重试 `nemoclaw onboard`。 |
| 沙箱创建失败 | 过时的网关状态或 DNS 未传播 | 运行 `openshell gateway destroy && openshell gateway start`，然后重新运行安装程序或 `nemoclaw onboard`。 |
| CoreDNS 崩溃循环 | 某些 DGX Spark 配置的已知问题 | 从 NemoClaw 存储库目录运行 `sudo ./scripts/fix-coredns.sh`。 |
| 板载期间“未检测到 GPU” | DGX Spark GB10 以不同方式报告统一内存 | 预计在 DGX Spark 上。向导仍然工作并使用 Ollama 进行推理。 |
| 推理超时或挂起 | Ollama 未运行或无法访问 | 检查奥拉玛：`curl http://localhost:11434`。如果未运行：`ollama serve &`。如果正在运行但无法从沙箱访问，请确保将 Ollama 配置为侦听 `0.0.0.0`（请参阅说明中的步骤 2）。 |
| 代理没有响应或速度非常慢 | 120B型号本地运行正常 | Nemotron 3 Super 120B 每次响应可能需要 30--90 秒。验证推理路线：`nemoclaw my-assistant status`。 |
| 端口 18789 已被使用 | 另一个进程绑定到该端口 | `lsof -i :18789` 然后 `kill `PID`。如果需要，`kill -9 `PID` 强制终止。 |
| Web UI 端口转发失败或仪表板无法访问 | 端口转发未激活 | `openshell forward stop 18789 my-assistant` 然后 `openshell forward start 18789 my-assistant --background`。 |
| Web UI 显示 `origin not allowed` | 通过 `localhost` 而不是 `127.0.0.1` 访问 | 在浏览器中使用 `http://127.0.0.1:18789/#token=...`。网关来源检查完全需要 `127.0.0.1`。 |
| 电报桥未启动 | 缺少环境变量 | 确保在主机上设置 `TELEGRAM_BOT_TOKEN` 和 `SANDBOX_NAME`。 `SANDBOX_NAME` 必须与入职时的沙箱名称匹配。 |
| Telegram 网桥需要重新启动，但 `nemoclaw stop` 不起作用 | `nemoclaw stop` 中的已知错误 | 从 `nemoclaw start` 输出中找到 PID，使用 `kill -9 `PID` 强制终止，然后再次运行 `nemoclaw start`。 |
| Telegram 机器人收到消息但不回复 | Telegram 策略未添加到沙箱 | 运行 `nemoclaw my-assistant policy-add`，输入 `telegram`，按 Y。然后使用 `nemoclaw start` 重新启动桥。 |

> [！笔记]
> DGX Spark 使用统一内存架构 (UMA)，可实现 GPU 和 CPU 之间的动态内存共享。由于许多应用程序仍在更新以利用 UMA，即使在 DGX Spark 的内存容量范围内，您也可能会遇到内存问题。如果发生这种情况，请使用以下命令手动刷新缓冲区缓存：

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

有关最新的已知问题，请查看 [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)。
