# VS Code 中的 Vibe 编码

> 使用 DGX Spark 作为 Ollama 的本地或远程 Vibe Coding 助手并继续


## 目录

- [Overview](#overview)
  - [What You'll Accomplish](#what-youll-accomplish)
  - [Prerequisites](#prerequisites)
  - [Time & risk](#time-risk)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思路

本手册将引导您将 DGX Spark 设置为 **Vibe 编码助手** — 在本地或通过Continue.dev 将其设置为 VSCode 的远程编码伴侣。  
本指南使用 **Ollama** 和 **GPT-OSS 120B** 来轻松将编码助手部署到 VSCode。其中包括高级说明，允许 DGX Spark 和 Ollama 提供可通过本地网络使用的编码助手。本指南也是针对操作系统的**全新安装**编写的。如果您的操作系统不是新安装的并且遇到问题，请参阅故障排除选项卡。

### 你将完成什么

您将拥有一个完全配置的 DGX Spark 系统，该系统能够：
- 通过 Ollama 运行本地代码帮助。
- 远程服务模型以进行Continue 和 VSCode 集成。
- 使用统一内存托管 GPT-OSS 120B 等大型 LLM。

### 先决条件

- DGX Spark（推荐128GB统一内存）
- **Ollama** 和您选择的法学硕士（例如 `gpt-oss:120b`）
- **VSCode**
- **继续** VSCode 扩展
- 模型下载的互联网接入
- 基本熟悉打开 Linux 终端、复制和粘贴命令。
- 具有 sudo 访问权限。
- 可选：用于远程访问配置的防火墙控制

### 时间与风险
* **持续时间：** 约30分钟 
* **风险：** 由于网络问题导致数据下载缓慢或失败
* **回滚：** 正常使用期间不会进行永久性系统更改。
* **最后更新：** 2025 年 10 月 21 日
  * 首次发表

## 指示

## 步骤1.安装Ollama

使用以下命令安装最新版本的 Ollama：

```bash
curl -fsSL https://ollama.com/install.sh | sh
```
服务运行后，拉取所需的模型：

```bash
ollama pull gpt-oss:120b
```

## 步骤 2.（可选）启用远程访问

要允许远程连接（例如，从使用 VSCode 并继续的工作站），请修改 Ollama systemd 服务：

```bash
sudo systemctl edit ollama
```

在注释部分下方添加以下行：

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_ORIGINS=*"
```

重新加载并重新启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

如果使用防火墙，请打开端口 11434：

```bash
sudo ufw allow 11434/tcp
```

验证工作站是否可以连接到 DGX Spark 的 Ollama 服务器：

  ```bash
  curl -v http://YOUR_SPARK_IP:11434/api/version
  ```
 将 **YOUR_SPARK_IP** 替换为您的 DGX Spark 的 IP 地址。
 如果连接失败，请参阅“故障排除”选项卡。

## 步骤3.安装VSCode

对于 DGX Spark（基于 ARM），下载并安装 VSCode：
  导航到 https://code.visualstudio.com/download 并下载 Linux ARM64 版本的 VSCode。后
  下载完成，记下下载的包名称。在下一个命令中使用它来代替 DOWNLOADED_PACKAGE_NAME。
```bash
sudo dpkg -i DOWNLOADED_PACKAGE_NAME
```

如果使用远程工作站，**安装适合您的系统架构的 VSCode**。

## 步骤 4. 安装Continue.dev 扩展

打开 VSCode 并从 Marketplace 安装 **Continue.dev**：
- 转到 VSCode 中的**扩展视图**
- 搜索 [Continue.dev](https://www.continue.dev/) 发布的 **Continue** 并安装扩展。
安装完成后，单击右侧栏上的继续图标。

## 步骤 5. 本地推理设置
- 单击`Or, configure your own models`
- 单击`Click here to view more providers`
- 选择 `Ollama` 作为提供商
- 对于型号，选择 `Autodetect`
- 通过发送测试提示来测试推理。

您下载的模型现在将成为推理的默认模型（例如 `gpt-oss:120b`）。

## 步骤 6. 设置工作站以连接到 DGX Spark 的 Ollama 服务器

要将运行 VSCode 的工作站连接到远程 DGX Spark 实例，必须在该工作站上完成以下操作：
  - 按照步骤 4 中的说明继续安装
  - 单击左侧窗格中的 `Continue` 图标
  - 单击`Or, configure your own models`
  - 单击`Click here to view more providers`
  - 选择 `Ollama` 作为提供者
  - 选择 `Autodetect` 作为模型。

继续 **将**无法检测到模型，因为它正在尝试连接到本地托管的 Ollama 服务器。
  - 在“继续”窗口的右上角找到 `gear` 图标并单击它。
  - 在左侧窗格中，单击 **模型**
  - 在 **聊天** 下的第一个下拉菜单旁边，单击齿轮图标。
  - 继续的 `config.yaml` 将打开。记下您的 DGX Spark 的 IP 地址。
  - 将配置替换为以下内容。 **YOUR_SPARK_IP** 应替换为您的 DGX Spark 的 IP。


```yaml
name: Config
version: 1.0.0
schema: v1

assistants:
  - name: default
    model: OllamaSpark

models:
  - name: OllamaSpark
    provider: ollama
    model: gpt-oss:120b
    apiBase: http://YOUR_SPARK_IP:11434
    title: gpt-oss:120b
    roles:
      - chat
      - edit
      - autocomplete
```

将 `YOUR_SPARK_IP` 替换为 DGX Spark 的 IP 地址。  
为您希望远程托管的任何其他 Ollama 模型添加其他模型条目。

## 故障排除

| 症状 | 原因 | 使固定 |
|---------|-------|-----|
|奥拉马未开始|GPU 驱动程序可能未正确安装|在终端中运行 `nvidia-smi`。如果命令失败，请检查 DGX 仪表板以获取 DGX Spark 的更新。|
|继续 无法通过网络连接|端口 11434 可能未打开或无法访问|运行命令 `ss -tuln \| grep 11434`. If the output does not reflect ` tcp LISTEN 0 4096 *:11434 *:* `，返回步骤 2 并运行 ufw 命令。|
|继续无法检测到本地运行的 Ollama 模型|未正确设置或检测到配置|检查 `/etc/systemd/system/ollama.service.d/override.conf` 文件中的 `OLLAMA_HOST` 和 `OLLAMA_ORIGINS`。如果 `OLLAMA_HOST` 和 `OLLAMA_ORIGINS` 设置正确，请将这些行添加到 `~/.bashrc` 文件中。|
|内存使用率高|模型尺寸太大|确认没有其他大型模型或容器正在使用 `nvidia-smi` 运行。使用较小的模型（例如 `gpt-oss:20b`）以实现轻量级使用。|

> [!NOTE]
> DGX Spark 使用统一内存架构 (UMA)，可实现 GPU 和 CPU 之间的动态内存共享。
> 由于许多应用程序仍在更新以利用 UMA，因此即使在
> DGX Spark 的内存容量。如果发生这种情况，请使用以下命令手动刷新缓冲区缓存：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
