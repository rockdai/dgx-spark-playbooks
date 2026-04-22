---
id: ollama
title: ollama
sidebar_label: ollama
---

# 奥拉马

> 安装和使用Ollama

## 目录

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思想

本手册演示了如何设置对 NVIDIA 上运行的 Ollama 服务器的远程访问
使用 NVIDIA Sync 的自定义应用程序功能的 Spark 设备。您将在 Spark 设备上安装 Ollama，
配置 NVIDIA Sync 以创建 SSH 隧道，并从本地计算机访问 Ollama API。
这样就无需公开网络上的端口，同时可以从您的网络中进行 AI 推理。
笔记本电脑通过安全的 SSH 隧道。

## 你将完成什么

您将让 Ollama 在采用 Blackwell 架构的 NVIDIA Spark 上运行，并可通过以下方式访问
来自本地笔记本电脑的 API 调用。此设置允许您在您的设备上构建应用程序或使用工具
与 Ollama API 通信以进行大型语言模型推理的本地计算机，利用
无需复杂的网络配置即可使用 Spark 设备强大的 GPU 功能。

## 开始之前需要了解什么

- 使用 SSH 连接和系统托盘应用程序
- 基本熟悉用于 API 测试的终端命令和 cURL
- 了解 REST API 概念和 JSON 格式
- 具有容器环境和 GPU 加速工作负载方面的经验

## 先决条件

- DGX Spark 设备设置并连接到您的网络
- NVIDIA Sync 已安装并连接到您的 Spark
- 终端访问本地计算机以测试 API 调用



## 时间与风险

* **持续时间**：初始设置 10-15 分钟，模型下载 2-3 分钟（因模型大小而异）

* **风险级别**：低 - 没有系统级更改，通过停止自定义应用程序可以轻松逆转

* **回滚**：停止 NVIDIA Sync 中的自定义应用程序并使用标准包卸载 Ollama
如果需要的话移除

* **最后更新：** 2025 年 10 月 12 日
  * 首次发表

## 指示

## 步骤 1. 验证 Ollama 安装状态

**说明**：检查 Ollama 是否已安装在您的 NVIDIA Spark 设备上。这运行于
Spark设备通过NVIDIA Sync终端来确定是否需要安装。

```bash
ollama --version
```

如果您看到版本信息，请跳至步骤 3。如果您看到“未找到命令”，请继续执行步骤 2。

## 步骤 2. 在 Spark 设备上安装 Ollama

**说明**：使用官方安装脚本下载并安装Ollama。这运行于
Spark 设备并安装 Ollama 二进制文件和服务组件。

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

等待安装完成。您应该看到指示安装成功的输出。

## 步骤 3. 下载并验证语言模型

**描述**：将语言模型拉取到您的 Spark 设备。这将下载模型文件并
使它们可用于推理。该示例使用针对 Blackwell GPU 进行优化的 Qwen2.5 30B。

```bash
ollama pull qwen2.5:32b
```

预期输出：
```
pulling manifest
pulling 58574f2e94b9: 100% ████████████████████████████  18 GB
pulling 53e4ea15e8f5: 100% ████████████████████████████ 1.5 KB
pulling d18a5cc71b84: 100% ████████████████████████████  11 KB
pulling cff3f395ef37: 100% ████████████████████████████  120 B
pulling 3cdc64c2b371: 100% ████████████████████████████  494 B
verifying sha256 digest
writing manifest
success
```

## 步骤 4. 访问 NVIDIA Sync 设置

**说明**：在本地机器上打开NVIDIA Sync配置界面，添加新的
自定义应用程序隧道。它在您本地的笔记本电脑/工作站上运行。

1. 单击系统托盘/任务栏中的 NVIDIA Sync 徽标
2. 单击右上角的齿轮图标打开“设置”窗口
3. 单击“自定义”选项卡

## 步骤 5. 在 NVIDIA Sync 中配置 Ollama 自定义应用程序

**描述**：创建一个新的自定义应用程序条目，该条目将建立到
Ollama 服务器在端口 11434 上运行。此配置在您的本地计算机上运行。

1. 单击“添加新的”按钮
2. 使用以下值填写表格：
  - **姓名**：`Ollama Server`
  - **端口**：`11434`
  - **在浏览器中自动打开**：不选中（这是一个 API，而不是 Web 界面）
  - **启动脚本**：留空
3. 点击“添加”

新的 Ollama Server 条目现在应该出现在您的 NVIDIA Sync 自定义应用程序列表中。

## 步骤 6. 启动 SSH 隧道

**描述**：激活 SSH 隧道，使远程 Ollama 服务器可以在本地访问
机器。这将创建从 localhost:11434 到 Spark 设备的安全连接。

1. 单击系统托盘/任务栏中的 NVIDIA Sync 徽标
2. 在“自定义”部分下，单击“Ollama Server”

当您在 NVIDIA Sync 中看到连接状态指示器时，隧道处于活动状态。

## 步骤 7. 验证 API 连接

**描述**：从本地计算机测试 Ollama API 连接以确保隧道正常
工作正常。它在您本地的笔记本电脑终端上运行。

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen2.5:32b",
  "messages": [{
    "role": "user",
    "content": "Write me a haiku about GPUs and AI."
  }],
  "stream": false
}'
```

预期响应格式：
```json
{
  "model": "qwen2.5:32b",
  "created_at": "2024-01-15T12:30:45.123Z",
  "message": {
    "role": "assistant",
    "content": "Silicon power flows\nThrough circuits, dreams become real\nAI awakens"
  },
  "done": true
}
```

## 步骤 8. 测试其他 API 端点

**描述**：验证其他 Ollama API 功能以确保完整运行。这些命令
在本地计算机上运行并测试不同的 API 功能。

测试型号清单：
```bash
curl http://localhost:11434/api/tags
```

测试流响应：
```bash
curl -N http://localhost:11434/api/chat -d '{
  "model": "qwen2.5:32b",
  "messages": [{"role": "user", "content": "Count to 5 slowly"}],
  "stream": true
}'
```

## 步骤 9. 清理和回滚

**说明**：如何删除设置并恢复到原始状态。

停止隧道：
1. 打开 NVIDIA Sync 并单击“Ollama Server”以停用

要删除自定义应用程序：
1. 打开 NVIDIA 同步设置 → 自定义选项卡
2. 选择“Ollama 服务器”并单击“删除”

> [！警告]
> 要从 Spark 设备中完全卸载 Ollama：

```bash
sudo systemctl stop ollama
sudo systemctl disable ollama
sudo rm /usr/local/bin/ollama
sudo rm -rf /usr/share/ollama
sudo userdel ollama
```

这将删除所有 Ollama 文件和下载的模型。

## 步骤 10. 后续步骤

**描述**：探索与您工作的 Ollama 的附加功能和集成选项
设置。

测试 [Ollama library](https://ollama.com/library) 的不同模型：
```bash
ollama pull llama3.1:8b
ollama pull codellama:13b
ollama pull phi3.5:3.8b
```

使用 NVIDIA Sync 提供的 DGX 仪表板在推理期间监控 GPU 和系统使用情况。

通过与您首选的编程语言集成，使用 Ollama API 构建应用程序
HTTP 客户端库。

## 故障排除

| 症状 | 原因 | 使固定 |
|---------|--------|-----|
| 本地主机上的“连接被拒绝”：11434 | SSH 隧道未激活 | 在 NVIDIA Sync 自定义应用程序中启动 Ollama Server |
| 模型下载因磁盘空间错误而失败 | Spark 存储空间不足 | 释放空间或选择较小的型号（例如 qwen2.5:7b） |
| 安装后未找到 Ollama 命令 | 安装路径不在PATH中 | 重新启动终端会话或运行 `source ~/.bashrc` |
| API 返回“未找到模型”错误 | 型号未拉出或名称错误 | 运行 `ollama list` 来验证可用模型 |
| Spark 推理速度慢 | 模型对于 GPU 内存来说太大 | 尝试较小的模型或使用 `nvidia-smi` 检查 GPU 内存 |
