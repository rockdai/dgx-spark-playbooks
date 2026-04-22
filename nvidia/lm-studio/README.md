# DGX Spark 上的 LM Studio

> 在 Spark 设备上部署 LM Studio 并为 LLM 提供服务；使用 LM Link 远程访问模型。


## 目录

- [Overview](#overview)
- [Instructions](#instructions)
  - [JavaScript](#javascript)
  - [Python](#python)
  - [Bash](#bash)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思路

LM Studio 是一款完全在您自己的硬件上发现、运行和服务大型语言模型的应用程序。您可以私下免费运行本地 LLM，例如 gpt-oss、Qwen3、Gemma3、DeepSeek 以及更多模型。

本手册向您展示了如何在 NVIDIA DGX Spark 设备上部署 LM Studio，以通过 GPU 加速在本地运行 LLM。在 DGX Spark 上运行 LM Studio 使 Spark 能够充当您自己的私有高性能 LLM 服务器。

**LM Link**（可选）可让您使用另一台计算机上的 Spark 模型，就好像它们是本地的一样。您可以通过端到端加密连接来链接 DGX Spark 和笔记本电脑（或其他设备），这样您就可以从笔记本电脑在 Spark 上加载和运行模型，而无需位于同一 LAN 上或打开网络访问。请参阅说明中的 [LM Link](https://lmstudio.ai/link) 和步骤 3b。


## 你将完成什么

您将在 NVIDIA DGX Spark 设备上部署 LM Studio 以运行 gpt-oss 120B，并使用笔记本电脑中的模型。更具体地说，您将：

- 在 Spark 上安装 **llmster**，一个完全无头的、终端本机 LM Studio
- 通过 API 在 DGX Spark 上本地运行 LLM 推理
- 使用 LM Studio SDK 与笔记本电脑上的模型进行交互
- 可以选择使用 **LM Link** 通过加密链接连接 Spark 和笔记本电脑，以便远程模型显示为本地模型（无需相同网络或绑定设置）


## 开始之前需要了解什么

- [Set Up Local Network Access](https://build.nvidia.com/spark/connect-to-your-spark) 到您的 DGX Spark 设备
- 使用终端/命令行界面
- 了解 REST API 概念

## 先决条件

**硬件要求：**
- 具有 ARM64 处理器和 Blackwell GPU 架构的 DGX Spark 设备
- 最低 65GB GPU 内存，建议 70GB 或以上
- 至少65GB可用存储空间，建议70GB或以上 

**软件要求：**
- NVIDIA DGX 操作系统
- 客户端设备（Mac、Windows 或 Linux） 
- 笔记本电脑和 DGX Spark 必须位于同一本地网络
- 网络访问下载包和模型

## LM 链接（可选）

[LM Link](https://lmstudio.ai/link) 让您**远程使用本地模型**。您链接机器（例如 DGX Spark 和笔记本电脑），然后在 Spark 上加载模型并从笔记本电脑使用它们，就像它们在本地一样。

- **端到端加密** — 基于 Tailscale 网状 VPN 构建；设备不暴露于公共互联网。
- **与本地服务器配合使用** - 连接到 LM Studio 本地 API 的任何工具（例如 `localhost:1234`）都可以使用您链接中的模型，包括 Codex、Claude Code、OpenCode 和 LM Studio SDK。
- **预览版** — 最多 2 个用户免费，每个用户 5 台设备（总共 10 台设备）。在 [lmstudio.ai/link](https://lmstudio.ai/link) 创建您的链接。

如果您使用LM Link，则可以跳过将服务器绑定到`0.0.0.0`并使用Spark的IP；链接设备后，将笔记本电脑指向 `localhost:1234`，远程模型就会出现在模型加载器中。

## 附属文件

所有必需的资产都可以在下面找到。这些示例脚本可在说明的步骤 6 中使用。

- [run.js](https://github.com/lmstudio-ai/docs/blob/main/_assets/nvidia-spark-playbook/js/run.js) - 用于向 Spark 发送测试提示的 JavaScript 脚本
- [run.py](https://github.com/lmstudio-ai/docs/blob/main/_assets/nvidia-spark-playbook/py/run.py) - 用于向 Spark 发送测试提示的 Python 脚本
- [run.sh](https://github.com/lmstudio-ai/docs/blob/main/_assets/nvidia-spark-playbook/bash/run.sh) - 用于向 Spark 发送测试提示的 Bash 脚本

## 时间与风险

* **预计时间：** 15-30 分钟（包括模型下载时间，这可能会根据您的互联网连接和模型大小而有所不同）
* **风险级别：**低
  * 大型模型下载可能需要大量时间，具体取决于网络速度
* **回滚：**
  * 可以从模型目录中手动删除下载的模型。
  * 卸载 LM Studio 或 llmster
* **最后更新：** 2026 年 3 月 12 日
  * 添加 LM Link 功能说明

## 指示

## 步骤 1. 在 DGX Spark 上安装 llmster

**llmster** 是 LM Studio 的终端本机、无头 LM Studio“守护进程”。

您可以将其安装在服务器、云实例、没有 GUI 的计算机上，或者仅安装在您的计算机上。这对于在 DGX Spark 上以无头模式运行 LM Studio，然后通过 API 从笔记本电脑连接到它非常有用。

**在 Spark 上，通过运行安装 llmster：**

```bash
curl -fsSL https://lmstudio.ai/install.sh | bash
```

对于 Windows：
```bash
irm https://lmstudio.ai/install.ps1 | iex
```

安装后，按照终端输出中的说明将 `lms` 添加到您的 PATH。使用 `lms` CLI 或 SDK/LM Studio V1 REST API（[enhanced features](https://lmstudio.ai/docs/developer/rest) 的新功能）/OpenAI 兼容的 REST API 与 LM Studio 进行交互。

## 步骤 2. 下载所需的辅助文件

在本地终端中运行以下curl命令以下载完成本剧本中后续步骤所需的文件。您可以选择 Python、JavaScript 或 Bash。

```bash
## JavaScript
curl -L -O https://raw.githubusercontent.com/lmstudio-ai/docs/main/_assets/nvidia-spark-playbook/js/run.js

## Python
curl -L -O https://raw.githubusercontent.com/lmstudio-ai/docs/main/_assets/nvidia-spark-playbook/py/run.py

## Bash
curl -L -O https://raw.githubusercontent.com/lmstudio-ai/docs/main/_assets/nvidia-spark-playbook/bash/run.sh
```

## 步骤 3. 启动 LM Studio API 服务器

使用 `lms`（LM Studio 的 CLI）从终端启动服务器。启用本地网络访问，这允许同一本地网络上的所有其他设备访问您计算机上运行的 LM Studio API 服务器（确保它们是受信任的设备）。为此，请运行以下命令：

```bash
lms server start --bind 0.0.0.0 --port 1234
```

要测试笔记本电脑和 Spark 之间的连接，请在本地终端中运行以下命令

```bash
curl http://<SPARK_IP>:1234/api/v1/models 
```
其中 `<SPARK_IP>` 是您设备的 IP 地址。您可以通过在 Spark 上运行以下命令来找到 Spark 的 IP 地址：

```bash
hostname -I
```

## 步骤 3b。 （可选）连接LM Link

**LM Link** 允许您通过端到端加密连接从笔记本电脑（或其他设备）使用 Spark 模型，就像它们在本地一样。您不需要位于同一本地网络或将服务器绑定到 `0.0.0.0`。

1. **创建链接** — 转到 [lmstudio.ai/link](https://lmstudio.ai/link) 并按照 **创建您的链接** 设置您的私有 LM Link 网络。
2. **链接两个设备** — 在您的 DGX Spark (llmster) 和笔记本电脑上，登录并加入同一个链接。 LM Link 使用 Tailscale 网状 VPN；设备无需打开互联网端口即可进行通信。
3. **使用远程模型** — 在您的笔记本电脑上，打开 LM Studio（或使用本地服务器）。 Spark 中的远程模型出现在模型加载器中。连接到 `localhost:1234` 的任何工具（包括 LM Studio SDK、Codex、Claude Code、OpenCode 和步骤 6 中的脚本）都可以使用这些模型，而无需更改端点。

LM Link 处于**预览版**状态，最多可供 2 个用户（每个用户 5 台设备）免费使用。有关详细信息和限制，请参阅 [LM Link](https://lmstudio.ai/link)。

## 步骤 4. 将模型下载到您的 Spark

作为示例，让我们下载并运行 gpt-oss 120B，这是 OpenAI 最好的开源模型之一。由于内存限制，该模型对于许多笔记本电脑来说太大了，这使得它成为 Spark 的绝佳用例。

```bash
lms get openai/gpt-oss-120b
```

由于体积较大，此下载将需要一段时间。通过列出您的模型来验证模型是否已成功下载：

```bash
lms ls
```

## 步骤 5. 加载模型 

将模型加载到 Spark 上，以便它准备好响应来自笔记本电脑的请求。

```bash
lms load openai/gpt-oss-120b
```

## 步骤 6. 在笔记本电脑上设置一个使用 LM Studio SDK 的简单程序

安装 LM Studio SDK 并使用简单的脚本向 Spark 发送提示并验证响应。为了快速入门，我们提供了以下适用于 Python、JavaScript 和 Bash 的简单脚本。从本手册的概述页面下载脚本，并从包含该脚本的目录运行相应的命令。

> [！笔记]
> 在每个脚本中，将 `<SPARK_IP>` 替换为本地网络上 DGX Spark 的 IP 地址。

### JavaScript

先决条件：用户已安装 `npm` 和 `node`

```bash
npm install @lmstudio/sdk
node run.js
```

### Python

先决条件：用户已安装 `uv`

```bash
uv run --script run.py
```

### 重击

先决条件：用户已安装 `jq` 和 `curl`

```bash
bash run.sh
```

## 步骤 7. 后续步骤

- 尝试从 [LM Studio model catalog](https://lmstudio.ai/models) 下载并提供不同的模型。
- 使用 [LM Link](https://lmstudio.ai/link) 连接更多设备，并通过端到端加密从任何地方使用 Spark 模型。

## 步骤 8. 清理和回滚
如果需要，请完全删除并卸载 LM Studio。请注意，LM Studio 将模型与应用程序分开存储。卸载 LM Studio 不会删除下载的模型，除非您明确删除它们。

如果要删除整个 LM Studio 应用程序，请先从托盘中退出 LM Studio，然后将该应用程序移至回收站。

要卸载 llmster，请删除文件夹 `~/.lmstudio/llmster`。

要删除下载的模型，请删除 `~/.lmstudio/models/` 的内容。

## 故障排除

| 症状 | 原因 | 使固定 |
|---------|-------|-----|
| API 返回“未找到模型”错误 | 模型未下载或加载到 LM Studio 中 | 运行 `lms ls` 验证下载状态，然后使用 `lms load {model-name}` 加载模型 |
| 未找到 `lms` 命令 | 假设安装成功的 PATH 问题 | 通过运行 `source ~/.bashrc` 刷新您的 shell |
| 模型加载失败 - CUDA 内存不足 | 模型对于可用 VRAM 来说太大 | 切换到较小的模型或不同的量化 |
| LM Link：设备未连接或远程模型不可见 | 设备不在同一链路中，或者两者上均未设置 LM Link | 确保 Spark 和笔记本电脑均已登录并加入位于 [lmstudio.ai/link](https://lmstudio.ai/link) 的同一链接。加入后重新启动LM Studio/llmster。请参阅 [LM Link](https://lmstudio.ai/link) 了解其工作原理。 |


> [！笔记] 
> DGX Spark 使用统一内存架构 (UMA)，可实现 GPU 和 CPU 之间的动态内存共享。 
> 由于许多应用程序仍在更新以利用 UMA，因此即使在 
> DGX Spark 的内存容量。如果发生这种情况，请使用以下命令手动刷新缓冲区缓存：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```


有关最新的已知问题，请查看 [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)。
