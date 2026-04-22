---
id: openclaw
title: openclaw
sidebar_label: openclaw
---

# 开爪🦞

> 使用 LM Studio 或 Ollama 在 DGX Spark 上本地运行 OpenClaw

## 目录

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思路

OpenClaw（以前称为 Clawdbot 和 Moltbot）是在您的计算机上运行的**本地优先** AI 代理。它将多种功能组合到一个助手中：它记住对话、适应您的使用情况、连续运行、使用文件和应用程序中的上下文，并且可以通过社区**技能**进行扩展。

**完全在 DGX Spark 上运行 OpenClaw 及其 LLM** 可保持数据的私密性并避免持续的云 API 成本。 DGX Spark 非常适合这一点：它运行 Linux，设计为持久运行，并拥有 **128GB 内存**，因此您可以运行大型本地模型，以获得更高的准确性和更强大的行为。

## 你将完成什么

您将在 DGX Spark 上安装 OpenClaw 并连接到本地 LLM（通过 LM Studio 或 Ollama）。您可以使用 OpenClaw Web UI 与您的客服人员聊天，并可选择连接沟通渠道和技能。代理和模型完全在您的 Spark 上运行 - 除非您添加云或外部集成，否则数据不会离开您的计算机。

## 热门用例

- **私人秘书**：通过访问您的收件箱、日历和文件，OpenClaw 可以帮助您管理日程、起草回复、发送提醒和查找会议空档。
- **主动项目管理**：通过电子邮件或消息检查项目状态、发送状态更新以及跟进或发送提醒。
- **研究代理**：结合网络搜索和本地文件来生成具有个性化背景的报告。
- **安装助手**：搜索应用程序/库、运行安装以及使用终端访问调试错误（建议使用较大的型号）。

## 开始之前需要了解什么

- Linux 终端和文本编辑器的基本使用
- 可选：如果您计划使用本地模型，则熟悉 Ollama 或 LM Studio
- 了解以下安全注意事项

## 重要提示：安全和风险

人工智能代理可能会带来真正的风险。阅读 OpenClaw 的指南：[OpenClaw Gateway Security](https://docs.openclaw.ai/gateway/security)。

主要风险：

1. **数据暴露**：个人信息或文件可能被泄露或被盗。
2. **恶意代码**：代理或连接的工具可能会让您遭受恶意软件或攻击。

您无法消除所有风险；继续操作的风险由您自行承担。 **关键安全措施：**

- **强烈建议：** 在专用或隔离系统（例如，干净的 DGX Spark 或 VM）上运行 OpenClaw，并且仅复制代理所需的数据。不要在包含敏感数据的主工作站上运行此程序。
- 使用代理的**专用账户**而不是您的主账户；仅授予其所需的最低访问权限。
- 仅启用**您信任的技能**，最好是经过社区审查的技能。提供终端或文件系统访问的技能会显着增加风险。
- **关键：** 确保 OpenClaw Web UI 和任何消息通道在没有强大身份验证的情况下**不会暴露**到公共互联网。如果远程访问，请使用 SSH 隧道或 VPN。
- 在可能的情况下，使用防火墙规则或网络隔离**限制代理的互联网访问**。
- **监控活动**：定期查看代理执行的日志和命令。

## 先决条件

- 运行 Linux 的 DGX Spark，已连接到您的网络
- 终端（SSH 或本地）访问 Spark
- 对于本地法学硕士：为您选择的模型提供足够的 GPU 内存（有关大小Instruct，请参阅说明；DGX Spark 的 128GB 支持大型模型）

## 时间和风险

- **持续时间**：安装和首次模型设置大约需要 30 分钟；模型下载时间取决于大小和网络（gpt-oss-120b 约为 65GB，在较慢的连接上可能需要更长的时间）。
- **风险级别**：**中到高** - 代理可以访问您配置的任何文件、工具和通道。如果您启用终端/命令执行技能或连接外部帐户，风险会显着增加。如果没有适当的隔离，此设置可能会暴露敏感数据或允许代码执行。 **始终遵循上述安全措施。**
- **回滚**：您可以通过相同的安装脚本或删除其目录来停止 OpenClaw 网关并卸载；如果需要，请单独卸载 Ollama 或 LM Studio。
- **最后更新**：2026 年 3 月 11 日
  - 首次出版

## 指示

> [！警告]
> **继续之前，请查看“概述”选项卡中的安全风险。** OpenClaw 是一个 AI 代理，可以访问您的文件、执行命令以及连接到外部服务。数据泄露和恶意代码执行是真正的风险。 **强烈建议：** 在隔离的系统或虚拟机上运行 OpenClaw，使用专用帐户（而不是您的主帐户），并且切勿在未经身份验证的情况下将仪表板暴露于公共互联网。

## 步骤 1. 在 DGX Spark 上安装 OpenClaw

在 DGX Spark 上，打开终端并运行官方安装脚本。这将在您的 Linux 系统上安装 OpenClaw 及其依赖项。

```bash
curl -fsSL https://openclaw.ai/install.sh | bash
```

下载依赖项后，OpenClaw 将显示**安全警告**。阅读风险；如果您接受它们，请使用箭头键选择**是**并按 Enter。

## 步骤 2. 完成 OpenClaw 入门

按照以下提示进行操作。

1. **快速入门与手动**：选择**快速入门**。

2. **模型提供者**：要使用**本地模型**（建议用于 DGX Spark），请转到列表底部并选择 **立即跳过** - 您稍后将配置模型。要使用云模型，请选择提供商并按照其说明进行操作。

3. **按提供商过滤模型**：选择**所有提供商**。在默认模型的下一个提示中，选择 **保持当前**。

4. **通信通道**：您可以连接通道（例如消息传递）以在远离计算机时使用机器人，或者选择 **立即跳过** 并稍后进行配置。

5. **技能**：我们建议暂时选择**否**。测试完基础知识后，您可以稍后从 Web UI 或 Clawhub 添加技能。

6. **Homebrew**：如果系统提示您安装 Homebrew，请选择 **否** - Homebrew 仅适用于 macOS，Linux 上不需要。

7. **挂钩**：我们建议选择全部三个以获得更好的体验。请注意，这可能会在本地记录数据；仅当您对此感到满意时才启用。

8. **仪表板 URL**：终端将打印 OpenClaw 仪表板的 URL。 **保存此 URL**（以及显示的任何访问令牌） - 您将需要它来打开 Web UI。

9. **完成**：在最终提示上选择**是**以完成安装。

您现在可以使用安装程序中的 URL 和令牌在浏览器中打开 OpenClaw 仪表板。

## 步骤 3. 选择并安装本地 LLM 后端

OpenClaw 可以通过 **LM Studio**（最佳原始性能，使用 Llama.cpp）或 **Ollama**（更简单且适合部署）使用本地 LLM。在 DGX Spark 上使用 **单独的终端** 作为后端，以便网关和模型服务器可以并行运行。

**安装以下其中一项：**

**选项 A – LM Studio**

```bash
curl -fsSL https://lmstudio.ai/install.sh | bash
```

**选项 B – 奥拉马**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## 步骤 4. 选择并下载模型

模型质量和能力随规模而变化。释放尽可能多的 GPU 内存（避免其他 GPU 工作负载，仅启用您需要的技能）。 DGX Spark 具有 **128GB 统一内存**，因此您可以运行大型模型并留出空间。

**按 GPU 内存推荐的型号：**

| GPU显存   | 推荐型号                    | 型号尺寸 | 笔记 |
|-------------|-------------------------------------|-----------|-------|
| 8–12 GB     | qwen3-4B-思考-2507             | 〜5GB      | —     |
| 16 GB       | gpt-oss-20b                        | 〜12GB     | 延迟较低，适合交互使用 |
| 24–48 GB    | Nemotron-3-Nano-30B-A3B            | 〜20GB     | —     |
| 128GB      | gpt-oss-120b                       | 〜65GB     | **DGX Spark 上的最佳质量**（量化）；为上下文窗口和其他进程留下约 63GB；如果您喜欢更快的响应，请使用 20B/30B |

**质量与延迟：** 120B 模型提供最佳的准确性和功能，但每个令牌的延迟较高。如果您喜欢更快的回复，请改用 **gpt-oss-20b** （或 30B 型号）；两者都可以在 DGX Spark 上轻松运行，并具有充足的内存空间。

**下载模型：**

**LM工作室**

```bash
lms get openai/gpt-oss-120b
```

**奥拉马**

```bash
ollama pull gpt-oss:120b
```

（使用与表中您的选择相匹配的型号名称；相应地调整 `lms get` 或 `ollama pull` 命令。）

## 步骤 5. 使用大上下文窗口运行模型

OpenClaw 在 **32K 令牌或更多** 的上下文窗口中效果最佳。

**LM工作室**

```bash
lms load openai/gpt-oss-120b --context-length 32768
```

**奥拉马**

```bash
ollama run gpt-oss:120b
```

出现交互式提示后，设置上下文窗口（在 Ollama 提示符下键入以下内容；不要包含任何 `>>>` 前缀）：

```
/set parameter num_ctx 32768
```

保持此终端（或进程）运行，以便模型保持加载状态。现在，您可以与模型聊天或按 Ctrl+D 退出交互模式，同时保持模型服务器运行。

> [！提示]
> **如果您看到内存不足 (OOM) 错误：** 尝试较小的上下文（例如 `16384`）或切换到较小的模型（例如 gpt-oss-20b）。加载模型时使用 `nvidia-smi` 监视内存。

## 步骤 6. 配置 OpenClaw 以使用您的本地模型

**如果您使用 LM Studio：**

1. 在您喜欢的编辑器（例如 `nano`、`vim` 或图形编辑器）中打开 OpenClaw 配置文件。配置路径为：
   ```bash
   ~/.openclaw/openclaw.json
   ```
   纳米示例：
   ```bash
   nano ~/.openclaw/openclaw.json
   ```

2. 添加或更新 `models` 部分，使其包含 LM Studio 提供程序。 **gpt-oss-120b** (DGX Spark) 示例：

```json
"models": {
  "mode": "merge",
  "providers": {
    "lmstudio": {
      "baseUrl": "http://localhost:1234/v1",
      "apiKey": "lmstudio",
      "api": "openai-responses",
      "models": [
        {
          "id": "openai/gpt-oss-120b",
          "name": "openai/gpt-oss-120b",
          "reasoning": false,
          "input": ["text"],
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          },
          "contextWindow": 32768,
          "maxTokens": 4096
        }
      ]
    }
  }
}
```

对于 **gpt-oss-20b** 或其他模型，请使用相同的结构，但设置 `id` 和 `name` 以匹配您加载的模型（例如 `openai/gpt-oss-20b`）。如果需要，调整 `contextWindow` 和 `maxTokens`。

**如果您使用 Ollama：**

> [！笔记]
> `ollama launch openclaw` 需要 **Ollama v0.15 或更高版本**。如果您看到“未知命令”错误，请升级 Ollama (`ollama --version`) 并重试。

跑步：

```bash
ollama launch openclaw
```

如果 OpenClaw 网关已在运行，它应该自动采用新配置。您可以添加 `--config` 进行配置，而无需启动网关。

## 步骤 7. 验证设置

1. 在浏览器中，打开 **OpenClaw 仪表板 URL**（并根据需要使用访问令牌）。
2. 开始**新**对话并发送短信。
3. 如果您收到代理的回复，则说明设置正常。

您还可以询问 OpenClaw 它使用的是哪种型号。在网关聊天 UI 中，您可以通过键入：**`/model MODEL_NAME`** 来切换模型。

## 步骤 8. 可选：添加技能并了解更多信息

- **技能**增加了能力，但也增加了风险；仅启用您信任的技能（例如，经过社区审查的技能）。添加技能：
  - 要求 OpenClaw 配置技能，或者
  - 使用 Web UI 中的侧边栏启用技能，或者
  - 浏览 [Clawhub](https://docs.openclaw.ai/tools/clawhub) 了解社区技能。

- 有关更多使用和配置详细信息，请参阅 [OpenClaw documentation](https://docs.openclaw.ai)。

## 故障排除

| 症状 | 原因 | 使固定 |
|---------|--------|-----|
| OpenClaw 仪表板 URL 未加载 | 网关未运行或主机/端口错误 | **重新启动 OpenClaw 网关：** 对于 Ollama，运行 `ollama launch openclaw` 以重新启动已配置的网关。对于 LM Studio，通过 LM Studio UI 重新启动 OpenClaw 网关或重新启动 OpenClaw 服务/容器。 **验证：** 检查网关进程是否正在使用 `pgrep -f openclaw` 或 `ps aux \ 运行| grep openclaw`. **Find URL/token:** Check the original installer output (scroll up in your terminal) or look in gateway logs (typically `~/.openclaw/logs/`) 获取仪表板 URL 和访问令牌 |
| 模型“连接被拒绝”（例如 localhost:1234 或 Ollama 端口） | LM Studio 或 Ollama 未运行，或端口错误 | 在单独的终端（`lms load ...` 或 `ollama run ...`）中启动模型，并确保 `openclaw.json` 中的端口匹配（LM Studio 为 1234，Ollama 为 11434） |
| OpenClaw 说没有可用的模型 | 模型提供程序未配置或模型未加载 | 将 `models` 部分添加到 LM Studio 的 `~/.openclaw/openclaw.json` 中，或为 Ollama 运行 `ollama launch openclaw` ；确保模型已加载/运行 |
| DGX Spark 内存不足或推理速度非常慢 | 模型对于可用 GPU 内存或其他 GPU 工作负载来说太大 | 释放 GPU 内存（关闭其他应用程序），选择较小的型号，或使用 `nvidia-smi` 检查使用情况 |
| 安装脚本失败或缺少依赖项 | Linux 上缺少系统包 | 安装curl和任何所需的构建工具；有关当前要求，请参阅 [OpenClaw documentation](https://docs.openclaw.ai) |
| 配置更改未应用 | 网关未重新加载 | 重新启动 OpenClaw 网关，使其重新加载 `~/.openclaw/openclaw.json` |
