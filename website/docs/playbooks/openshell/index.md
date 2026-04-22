---
id: openshell
title: openshell
sidebar_label: openshell
---

# 在 DGX Spark 上使用 OpenShell 保护长时间运行的 AI 代理

> 在 DGX Spark 上的 NVIDIA OpenShell 沙箱中使用本地模型运行 OpenClaw

## 目录

- [Overview](#overview)
  - [Notice & Disclaimers](#notice-disclaimers)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思路

OpenClaw 是一个本地优先的 AI 代理，可以在您的计算机上运行，​​将内存、文件访问、工具使用和社区技能结合到一个持久的助手中。直接在您的系统上运行它意味着代理可以访问您的文件、凭据和网络，从而产生真正的安全风险。

**NVIDIA OpenShell** 解决了这个问题。它是一个开源沙箱运行时，使用声明性 YAML 策略将代理包装在内核级隔离中。 OpenShell 控制代理可以在磁盘上读取的内容、可以访问哪些网络端点以及它拥有哪些权限，而无需禁用使代理有用的功能。

通过将 OpenClaw 与 DGX Spark 上的 OpenShell 相结合，您可以获得由大型模型的 128GB 统一内存支持的本地 AI 代理的全部功能，同时对文件系统访问、网络出口和凭证处理实施显式控制。

### 通知及免责声明
#### 快速启动安全检查

仅使用干净的环境。在没有个人数据、机密信息或敏感凭据的新设备或虚拟机上运行此剧本。把它想象成一个沙箱——保持它的隔离。

通过安装此手册，您将对所有第三方组件负责，包括检查其许可证、条款和安全状况。安装或使用之前请阅读并接受。

---

#### 你会得到什么

该手册展示了实验性人工智能代理功能。即使您的工具包中使用了 OpenShell 等尖端开源工具，您也需要针对特定​​威胁模型采取适当的安全措施。

---

#### 人工智能代理的主要风险

使用人工智能代理时请注意以下风险：

1. 数据泄露——代理访问的任何材料都可能被暴露、泄露或被盗。

2. 恶意代码执行 – 代理或其连接的工具可能会使您的系统遭受恶意代码或网络攻击。

3. 意外操作 – 代理可能会在未经明确批准的情况下修改或删除文件、发送消息或访问服务。

4. 及时注入和操纵——外部输入或连接的内容可能会以意想不到的方式劫持代理的行为。

---

#### 安全最佳实践

 没有一个系统是完美的，但这些做法有助于确保您的信息和系统的安全。

1. 隔离您的环境 – 在干净的 PC 或隔离的虚拟机上运行。仅提供您希望代理访问的特定数据。

2. 切勿使用真实帐户 – 不要连接个人、机密或生产帐户。创建具有最小权限的专用测试帐户。

3. 审查您的技能/插件 – 仅启用来自经过社区审查的可信来源的技能。

4. 锁定访问 – 确保在没有正确身份验证的情况下，无法通过网络访问您的 OpenClaw UI 或消息通道。

5. 限制网络访问 – 在可行的情况下，限制代理的互联网连接。

6. 自行清理 – 完成后，删除 OpenClaw 并撤销您授予的所有凭据、API 密钥和帐户访问权限。

---

## 你将完成什么

您将安装 OpenShell CLI (`openshell`)，在 DGX Spark 上部署网关，并使用预构建的 OpenClaw 社区沙箱在沙箱环境中启动 OpenClaw。沙箱默认强制执行文件系统、网络和进程隔离。您还将配置本地推理路由，以便 OpenClaw 使用在 Spark 上运行的模型，而无需外部 API 密钥。

## 热门用例

- **安全代理实验**：测试 OpenClaw 技能和集成，而无需向代理暴露您的主文件系统或凭据。
- **私营企业开发**：将所有推理路由到 DGX Spark 上的本地模型。除非您在策略中明确允许，否则不会有数据离开计算机。
- **可审核的代理访问**：对项目 YAML 策略进行版本控制。在授予访问权限之前，请准确检查代理可以访问的内容。
- **迭代策略调整**：使用 `openshell term` 实时监控被拒绝的连接，然后热重新加载更新的策略，而无需重新创建沙箱。

## 开始之前需要了解什么

- 熟悉 Linux 终端和 SSH
- 对 Docker 的基本了解（OpenShell 在 Docker 内部运行 k3s 集群）
- 熟悉 Ollama 本地模型服务
- 对安全模型的认识：OpenShell通过隔离降低风险，但不能消除所有风险。查看 [OpenShell documentation](https://pypi.org/project/openshell/) 和 [OpenClaw security guidance](https://docs.openclaw.ai/gateway/security)。

## 先决条件

**硬件要求：**
- 具有 128GB 统一内存的 NVIDIA DGX Spark
- 大型本地模型至少需要 70GB 可用内存（例如，gpt-oss:120b，约 65GB 加上开销），或者较小模型需要 25GB+（例如 gpt-oss-20b）

**软件要求：**
- NVIDIA DGX 操作系统（Ubuntu 24.04 基础）
- Docker 桌面或 Docker 引擎正在运行：`docker info`
- Python 3.12 或更高版本：`python3 --version`
- `uv` 包管理器：`uv --version`（使用 `curl -LsSf https://astral.sh/uv/install.sh | sh` 安装）
- Ollama 0.17.0 或更高版本（建议使用最新版本以支持 gpt-oss MXFP4）：`ollama --version`
- 网络访问可从 PyPI 下载 Python 包并从 Ollama 下载模型权重
- 已为您的 DGX Spark 安装并配置 [NVIDIA Sync](https://build.nvidia.com/spark/connect-to-your-spark)

## 时间与风险

* **预计时间：** 20–30 分钟（加上模型下载时间，这取决于模型大小和网络速度）。

> [!警告] **风险级别：** 中
  * OpenShell 沙箱强制执行内核级隔离，与直接在主机上运行 OpenClaw 相比，显着降低了风险。
  * 沙箱默认策略拒绝所有未明确允许的出站流量。错误配置的策略可能会阻止合法的代理流量；使用 `openshell logs` 进行诊断。
  * 在不稳定的网络上，大型模型下载可能会失败。
* **回滚：** 使用 `openshell sandbox delete <sandbox-name>` 删除沙箱，使用 `openshell gateway stop` 停止网关，并可以选择使用 `openshell gateway destroy` 销毁它。 Ollama 模型可以使用 `ollama rm <model>` 删除。
* **最后更新：** 2026 年 3 月 13 日

## 指示

## 步骤1.确认您的环境

在安装任何内容之前，请验证操作系统、GPU、Docker 和 Python 是否可用。

```bash
head -n 2 /etc/os-release
nvidia-smi
docker info --format '{{.ServerVersion}}'
python3 --version
```
确保 [NVIDIA Sync](https://build.nvidia.com/spark/connect-to-your-spark/sync) 配置了自定义端口：使用“OpenClaw”作为名称，使用“18789”作为端口。

预期输出应显示 Ubuntu 24.04 (DGX OS)、检测到的 GPU、Docker 服务器版本和 Python 3.12+。

## 步骤2.Docker配置

首先，使用以下命令验证本地用户是否具有 Docker 权限。
``` bash
docker ps
```
如果您收到权限被拒绝错误 (`permission denied while trying to connect to the docker API at unix:///var/run/docker.sock`)，请将您的用户添加到系统的 Docker 组。这将使您无需 `sudo` 即可运行 Docker 命令。执行此操作的命令如下：

``` bash
sudo usermod -aG docker $USER
newgrp docker
```
请注意，您应该在将用户添加到组后重新启动 Spark，以便在所有终端会话中持久生效。

现在我们已经验证了用户的 Docker 权限，我们必须配置 Docker，以便它可以使用 NVIDIA Container Runtime。
``` bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

运行示例工作负载来验证设置：

``` bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

## 步骤 3. 安装 OpenShell CLI

创建虚拟环境并安装 `openshell` CLI。

```bash
cd ~
uv venv openshell-env && source openshell-env/bin/activate
uv pip install openshell 
openshell --help
```

如果您尚未安装 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

预期输出应显示 `openshell` 命令树，其中包含 `gateway`、`sandbox`、`provider` 和 `inference` 等子命令。

## 步骤 4. 在 DGX Spark 上部署 OpenShell 网关

网关是管理沙箱的控制平面。由于您直接在 Spark 上运行，因此它会在 Docker 内部进行本地部署。

```bash
openshell gateway start
openshell status
```

`openshell status` 应将网关报告为 **已连接**。第一次运行可能需要几分钟时间，同时 Docker 会拉取所需的映像和内部 k3s 集群引导程序。

> [！笔记]
> 远程网关部署需要无密码 SSH 访问。在使用 `--remote` 标志之前，确保您的 SSH 公钥已添加到 DGX Spark 上的 `~/.ssh/authorized_keys` 中。

> [！提示]
> 如果您想从单独的工作站管理 Spark 网关，请从该工作站运行 `openshell gateway start --remote <username>@<spark-ssid>.local`。所有后续命令都将通过 SSH 隧道进行路由。

## 步骤 5. 安装 Ollama 并拉取模型

安装 Ollama（如果尚未存在）并下载用于本地推理的模型。

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
```

DGX Spark的128GB内存可以运行大型模型：

| 可用 GPU 内存 | 推荐型号          | 型号尺寸 | 笔记 |
|---------------------|---------------------------|-----------|-------|
| 25–48 GB            | nemotron-3-nano           | 〜24GB     | 延迟较低，适合交互使用 |
| 48–80 GB            | gpt-oss:120b              | 〜65GB     | 质量和速度的良好平衡 |
| 128GB              | nemotron-3-super:120b     | 〜86GB     | DGX Spark 的最佳质量 |

验证 Ollama 是否正在运行（它在安装后作为服务自动启动）。如果没有，请手动启动：

```bash
ollama serve &
```

配置 Ollama 以侦听所有接口，以便 OpenShell 网关容器可以访问它：

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\nEnvironment="OLLAMA_HOST=0.0.0.0"\n' | sudo tee /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

验证 Ollama 正在运行并且在所有接口上均可访问：

```bash
curl http://0.0.0.0:11434
```

预期：`Ollama is running`。如果没有，则以 `sudo systemctl start ollama` 开头。

接下来，运行 Ollama 中的模型（调整模型名称以匹配您在 [the Ollama model library](https://ollama.com/library) 中的选择）。如果模型尚不存在，`ollama run` 命令将自动拉取模型。在此运行模型可确保您在将其与 OpenClaw 一起使用时已加载并准备就绪，从而减少以后超时的可能性。 nemotron-3-super 的示例：

```bash
ollama run nemotron-3-super:120b
```

输入 `/bye` 退出。

验证模型是否可用：

```bash
ollama list
```

## 步骤 6. 创建推理提供程序

我们将创建一个指向您本地 Ollama 服务器的 OpenShell 提供程序。这允许 OpenShell 将推理请求路由到 Spark 托管的模型。

首先，找到 DGX Spark 的 IP 地址：

```bash
hostname -I | awk '{print $1}'
```

然后创建提供程序，将 `{Machine_IP}` 替换为上述命令中的 IP 地址（例如 `10.110.106.169`）：

```bash
openshell provider create \
    --name local-ollama \
    --type openai \
    --credential OPENAI_API_KEY=not-needed \
    --config OPENAI_BASE_URL=http://{Machine_IP}:11434/v1
```

> [！重要的]
> 不要在这里使用 `localhost` 或 `127.0.0.1`。 OpenShell 网关在 Docker 容器内运行，因此无法通过 `localhost` 到达主机。使用机器的实际 IP 地址。

验证提供者是否已创建：

```bash
openshell provider list
```

## 步骤 7. 配置推理路由

将 `inference.local` 端点（每个沙箱内都可用）指向您的 Ollama 模型。将型号名称替换为您在第 5 步中选择的名称：

```bash
openshell inference set \
    --provider local-ollama \
    --model nemotron-3-super:120b
```

输出应确认路由并显示经过验证的端点 URL，例如：`http://10.110.106.169:11434/v1/chat/completions (openai_chat_completions)`。

> [！笔记]
> 如果您看到 `failed to verify inference endpoint` 或 `failed to connect`（例如，因为网关无法从其容器内部访问主机 IP），请添加 `--no-verify` 以跳过端点验证：`openshell inference set --provider local-ollama --model nemotron-3-super:120b --no-verify`。确保 Ollama 正在所有接口上运行并侦听（请参阅步骤 5）。

验证配置：

```bash
openshell inference get
```

预期输出应显示 `provider: local-ollama` 和 `model: nemotron-3-super:120b` （或您选择的任何模型）。

## 步骤 8. 部署 OpenShell 沙箱

使用预先构建的 OpenClaw 社区沙箱创建沙箱。这将从 OpenShell 社区目录中提取 OpenClaw Dockerfile、默认策略和启动脚本：

``` bash
openshell sandbox create \
  --keep \
  --forward 18789 \
  --name dgx-demo \
  --from openclaw \
  -- openclaw-start
```

> [！笔记]
> 使用 `--from openclaw` 时，请勿将 `--policy` 与本地文件路径（例如 `openclaw-policy.yaml`）一起传递。该策略与社区沙箱捆绑；本地文件路径可能会导致“找不到文件”。

`--keep` 标志使沙箱在初始进程退出后保持运行，以便您可以稍后重新连接。这是默认行为。要在初始进程退出时终止沙箱，请改用 `--no-keep` 标志。

CLI 将：
1. 针对社区目录解析 `openclaw`
2. 拉取并构建容器镜像
3. 应用捆绑沙盒策略
4. 在沙箱内启动 OpenClaw

## 步骤 9. 在 OpenShell Sandbox 中配置 OpenClaw

沙盒容器将启动，OpenClaw 入门向导将在您的终端中自动启动。

> [！重要的]
> 入门向导是**完全交互式的** - 它需要箭头键导航和 Enter 来选择选项。它无法从非交互式会话（例如脚本或自动化工具）完成。您必须从完全支持 TTY 的终端运行 `openshell sandbox create`。
>
> 如果向导在沙箱创建过程中未完成，请重新连接到沙箱以重新运行它：
> ````重击
> openshell 沙盒连接 dgx-demo
> ````

使用箭头键和 Enter 键与安装进行交互。
- 如果您理解并同意，请使用键盘的箭头键选择“是”，然后按 Enter 键。
- 快速启动与手动：选择快速启动并按 Enter 键。
- 模型/身份验证提供商：选择**自定义提供商**，倒数第二个选项。
- API 基本 URL：更新为 https://inference.local/v1
- 您想如何提供此 API 密钥？：暂时粘贴 API 密钥。
- API密钥：请输入“ollama”。
- 端点兼容性：选择**OpenAI-兼容**并按 Enter。
- 型号 ID：输入您在步骤 5 中选择的型号名称（例如 `nemotron-3-super:120b`）。
	- 这可能需要 1-2 分钟，因为 Ollama 模型在后台旋转。
- 端点 ID：保留默认值。
- 别名：输入相同的型号名称（可选）。
- 频道：选择**暂时跳过**。
- 搜索提供商：选择**暂时跳过**。
- 技能：暂时选择**否**。
- 启用挂钩：按空格键选择**立即跳过**，然后按 Enter。

完成最后阶段可能需要 1-2 分钟。然后，您应该会看到一个带有令牌的 URL，您可以使用该令牌连接到网关。 

预期输出将相似，但令牌将是唯一的。
``` bash
OpenClaw gateway starting in background.
  Logs: /tmp/gateway.log
  UI:   http://127.0.0.1:18789/?token=9b4c9a9c9f6905131327ce55b6d044bd53e0ec423dd6189e
```

现在我们已经在 OpenShell 沙箱中配置了 OpenClaw，让我们将 openshell 沙箱的名称设置为环境变量。这将使未来的命令更容易运行。请注意，沙箱的名称是在步骤 8 的 `openshell sandbox create` 命令中使用 `--name` 标志设置的。

```bash
export SANDBOX_NAME=dgx-demo
```

为了验证为您的沙箱启用的默认策略，请运行以下命令：

```bash
openshell sandbox get $SANDBOX_NAME
```
> [！笔记]
> 步骤 8 的 `--forward 18789` 已经设置了从 OpenShell 网关到沙箱的端口转发。对于通常情况，您**不需要**需要带有 `openshell ssh-proxy` 的手动 `ssh` 命令。

要验证转发是否处于活动状态，请使用以下命令：

```bash
openshell forward list
```

您应该会看到您的沙箱名称（例如 `dgx-demo`）和端口 `18789`。如果缺少或 `dead`，则启动它：

```bash
openshell forward start --background 18789 $SANDBOX_NAME
```

路径 A：如果您使用 Spark 作为主设备，请右键单击 UI 部分中的 URL，然后选择“打开链接”。

路径 B：如果您使用的笔记本电脑或工作站*不在 Spark 上（例如，您仅通过 SSH 连接到 Spark）：在 **该** 计算机上安装 OpenShell CLI。

> [！重要的]
> **SSH 必须在 `gateway add` 之前从该计算机工作到 Spark。** 运行 `ssh nvidia@<spark-ip>`（或您的用户/主机）并确认您获得不带 `Permission denied (publickey)` 的 shell。如果失败，请将您的公钥添加到 Spark：`ssh-copy-id nvidia@<spark-ip>`（来自同一台计算机），或将您的 `~/.ssh/id_ed25519.pub` （或 `id_rsa.pub`）粘贴到 Spark 上的 `~/.ssh/authorized_keys` 中。 OpenShell 使用此 SSH 会话来访问远程 Docker API 并提取网关 TLS 证书。如果您使用非默认密钥，请将 `--ssh-key ~/.ssh/your_key` 传递到 `gateway add` （与步骤 4 的远程网关注释相同）。

注册 Spark 的 **已运行** 网关。 **不要**单独使用 `openshell gateway add user@ip`——它被解析为云 URL 并且不会写入 `mtls/ca.crt`。

根据 [OpenShell gateway docs](https://docs.nvidia.com/openshell/latest/sandboxes/manage-gateways.html)，使用 **主机名 `openshell`** 注册，而不是原始 Spark IP，用于 HTTPS。

> [！警告]
> 网关 TLS 证书对 `openshell`、`localhost` 和 `127.0.0.1` 有效 — **不适用于**您的 Spark 的 LAN IP。如果您使用 `https://10.x.x.x:8080` 或 `ssh://user@10.x.x.x:8080`，`openshell status` 可能会失败，并显示 **证书对于名称“10.x.x.x”无效**。

**在您的笔记本电脑/WSL 上**，将 `openshell` 映射到 Spark（每台机器一次）：

```bash
## Replace with your Spark’s IP. Requires sudo on Linux/WSL.
echo "<spark-ip> openshell" | sudo tee -a /etc/hosts
## Example: echo "10.110.17.10 openshell" | sudo tee -a /etc/hosts
```

然后添加网关（SSH目标保留真实IP或主机名；HTTPS URL使用`openshell`）：

```bash
openshell gateway add https://openshell:8080 --remote <user>@<spark-ip>
```

例子：

```bash
openshell gateway add https://openshell:8080 --remote nvidia@10.110.17.10
```

如果您已经使用该 IP 注册并看到证书错误，请删除该条目并重新添加：

```bash
openshell gateway destroy 
openshell gateway add https://openshell:8080 --remote nvidia@10.110.17.10
```

（如果销毁名称不同，请使用 `openshell gateway select`。）

完成任何浏览器或 CLI 提示，直到命令完成（不要过早按 Ctrl+C）。然后：

```bash
openshell status   # should show Connected, not TLS CA errors
openshell forward start --background 18789 dgx-demo
```

然后在 **笔记本电脑** 浏览器上打开（使用 `#token=` 以便 UI 接收网关令牌）：

`http://127.0.0.1:18789/#token=<your-token>`

使用 Spark 上 OpenClaw 向导输出的令牌值。路径 B 需要从笔记本电脑到 Spark 的 SSH，以便 CLI 可以到达 `:8080` 上的网关。

**NVIDIA Sync：** 右键单击​​ UI 中的 URL，然后选择复制链接。同步连接到 Spark，打开 OpenClaw 条目，然后将 URL 粘贴到浏览器地址栏中。

从此页面中，您现在可以在 OpenShell 提供的运行时的受保护范围内与您的 OpenClaw 代理**聊天**。
## 步骤 10. 在沙箱内进行推理

#### 连接到沙箱（终端）

现在 OpenClaw 已在 OpenShell 受保护的运行时中配置，您可以通过以下方式直接连接到沙箱环境：

```bash
openshell sandbox connect $SANDBOX_NAME
```

加载到沙箱终端后，您可以使用以下命令测试与 Ollama 模型的连接：
``` bash
curl https://inference.local/v1/responses \
          -H "Content-Type: application/json" \
          -d '{
        "instructions": "You are a helpful assistant.",
        "input": "Hello!"
      }'
```

## 步骤 11. 验证沙箱隔离

打开第二个终端并检查沙箱状态和实时日志：

```bash
source ~/openshell-env/bin/activate
openshell term
```

终端仪表板显示：
- **沙盒状态** — 名称、阶段、图像、提供商和端口转发
- **实时日志流** — 出站连接、策略决策（`allow`、`deny`、`inspect_for_inference`）和推理拦截

验证 OpenClaw 代理是否可以访问 `inference.local` 进行模型请求，并且未经授权的出站流量被拒绝。

> [！提示]
> 按 `f` 跟踪实时输出，按 `s` 按源过滤，按 `q` 退出终端仪表板。

## 步骤 12. 重新连接到沙箱

如果您退出沙盒会话，请随时重新连接：

```bash
openshell sandbox connect $SANDBOX_NAME
```

> [！笔记]
> `openshell sandbox connect` 是交互式的——它在沙箱内打开一个终端会话。无法传递非交互式执行的命令。使用 `openshell sandbox upload`/`download` 进行文件传输，或使用 `openshell sandbox ssh-config` 进行脚本化 SSH（请参阅步骤 14）。

要将文件传入或传出沙箱，请使用以下命令：

```bash
openshell sandbox upload $SANDBOX_NAME ./local-file /sandbox/destination
openshell sandbox download $SANDBOX_NAME /sandbox/file ./local-destination
```

## 步骤 13. 清理

停止并删除沙箱：

```bash
openshell sandbox delete $SANDBOX_NAME
```

删除您在步骤 6 中创建的推理提供程序：

```bash
openshell provider delete local-ollama
```

停止网关（保留状态供以后使用）：

```bash
openshell gateway stop
```

> [！警告]
> 以下命令将永久删除网关集群及其所有数据。

```bash
openshell gateway destroy
```

要同时删除 Ollama 模型：

```bash
ollama rm nemotron-3-super:120b
```

## 步骤 14. 后续步骤

- **添加更多提供商**：使用 `openshell provider create` 附加 GitHub 令牌、GitLab 令牌或云 API 密钥作为提供商。创建沙箱时，使用 `--provider <name>`（例如 `--provider my-github`）传递提供程序名称，以将这些凭据安全地注入沙箱中。
- **尝试其他社区沙箱**：为其他预构建环境运行 `openshell sandbox create --from base` 或 `--from sdg`。
- **连接 VS Code**：使用 `openshell sandbox ssh-config <sandbox-name>` 并将输出附加到 `~/.ssh/config` 以将 VS Code Remote-SSH 直接连接到沙箱。
- **监控和审计**：使用 `openshell logs <sandbox-name> --tail` 或 `openshell term` 持续监控代理活动和策略决策。

## 故障排除

| 症状 | 原因 | 使固定 |
|---------|-------|-----|
| `openshell gateway start` 因“连接被拒绝”或 Docker 错误而失败 | Docker 未运行 | 使用 `sudo systemctl start docker` 启动 Docker 或启动 Docker Desktop，然后重试 `openshell gateway start` |
| `openshell status` 显示网关不健康 | 网关容器崩溃或初始化失败 | 运行 `openshell gateway destroy`，然后运行 ​​`openshell gateway start` 以重新创建它。使用 `docker ps -a` 和 `docker logs <container-id>` 检查 Docker 日志以获取详细信息 |
| `openshell sandbox create --from openclaw` 构建失败 | 拉取社区沙箱或 Dockerfile 构建失败的网络问题 | 检查互联网连接。重试该命令。如果在特定包上构建失败，请检查基础映像是否与您的 Docker 版本兼容 |
| 沙盒创建后处于 `Error` 阶段 | 策略验证失败或容器启动崩溃 | 运行 `openshell logs <sandbox-name>` 查看错误详细信息。常见原因：策略 YAML 无效、缺少提供商凭据或端口冲突 |
| 代理无法到达沙箱内的 `inference.local` | 未配置推理路由或无法访问提供程序 | 运行 `openshell inference get` 以验证提供程序和模型是否已设置。测试 Ollama 可从主机访问：`curl http://localhost:11434/api/tags`。确保提供程序 URL 使用 `host.docker.internal` 而不是 `localhost` |
| 网关/沙箱访问主机Ollama时503验证失败或超时 | Ollama 仅绑定到本地主机，或主机防火墙阻止端口 11434 | 让 Ollama 监听所有接口，以便网关容器（例如在 Docker 网络 172.17.x.x 上）可以访问它：`OLLAMA_HOST=0.0.0.0 ollama serve &`。允许端口 11434 通过主机防火墙：`sudo ufw allow 11434/tcp comment 'Ollama for OpenShell Gateway'`（然后是 `sudo ufw reload`，如果需要）。 |
| 代理的出站连接全部被拒绝 | 默认策略不包括所需的端点 | 使用 `openshell logs <sandbox-name> --tail --source sandbox` 监视拒绝。使用 `openshell policy get <sandbox-name> --full` 拉取当前策略，在 `network_policies` 下添加所需的主机/端口，并使用 `openshell policy set <sandbox-name> --policy <file> --wait` 推送 |
| 沙箱内出现“权限被拒绝”或 Landlock 错误 | 代理尝试访问不在 `read_only` 或 `read_write` 文件系统策略中的路径 | 拉取当前策略并将路径添加到 `read_write`（如果读取访问权限足够，则添加到 `read_only`）。推送更新的政策。注意：文件系统策略是静态的，需要重新创建沙箱 |
| Ollama OOM 或非常慢的推理 | 模型对于可用内存或 GPU 争用来说太大 | 释放 GPU 内存（关闭其他 GPU 工作负载），尝试较小的模型（例如 `gpt-oss:20b`），或减少上下文长度。使用 `nvidia-smi` 进行监控 |
| `openshell sandbox connect` 挂起或超时 | 沙箱未处于 `Ready` 阶段 | 运行 `openshell sandbox get <sandbox-name>` 检查相位。如果卡在 `Provisioning` 中，请等待或检查日志。如果在 `Error` 中，则删除并重新创建沙箱 |
| 策略推送返回退出代码 1（验证失败） | YAML 格式错误或策略字段无效 | 检查 YAML 语法。常见问题：路径不以 `/` 开头、路径中存在 `..` 遍历、`root` 为 `run_as_user`，或者端点缺少必需的 `host`/`port` 字段。修复并重新推送 |
| `openshell gateway start` 失败，并显示“K8s 命名空间未就绪”/等待命名空间超时 | Docker 容器内的 k3s 集群的引导时间比 CLI 超时允许的时间长。内部组件（TLS 密钥、Helm 图表、命名空间创建）可能需要额外的时间，尤其是在首次运行时，当图像被拉入容器内时。 | 首先，检查容器是否仍在运行并正在进行：`docker ps --filter name=openshell`（查找`health: starting`）。检查容器内的 k3s 状态：`docker exec <container> sh -c "KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl get ns"` 和 `kubectl get pods -A`。如果 Pod 位于 `ContainerCreating` 中并且缺少 TLS 机密（`navigator-server-tls`、`openshell-server-tls`），则集群仍在引导 - 等待几分钟并再次运行 `openshell status`。如果它没有恢复，请使用 `openshell gateway destroy`（如果需要的话，使用 `docker rm -f <container>`）销毁并重试 `openshell gateway start`。确保 Docker 有足够的资源（内存和磁盘）供 k3s 集群使用。 |
| 即使 Docker 容器正在运行，`openshell status` 仍显示“未配置网关” | `gateway start` 命令在将网关配置保存到本地配置存储之前失败或超时 | 容器可能仍然正常 - 请与 `docker ps --filter name=openshell` 检查。如果容器正在运行且运行状况良好，请再次尝试 `openshell gateway start` （它应该检测到现有容器）。如果容器不健康或卡住，请依次使用 `docker rm -f <container>`、`openshell gateway destroy` 和 `openshell gateway start` 将其移除。 |

> [！笔记]
> DGX Spark 使用统一内存架构 (UMA)，可实现 GPU 和 CPU 之间的动态内存共享。
> 由于许多应用程序仍在更新以利用 UMA，因此即使在
> DGX Spark 的内存容量。如果发生这种情况，请使用以下命令手动刷新缓冲区缓存：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

有关最新的已知问题，请查看 [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)。
