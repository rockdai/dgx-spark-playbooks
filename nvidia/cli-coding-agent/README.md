# CLI Coding Agent

> 在 DGX Spark 上使用 Ollama 运行本地编码模型，并连接你选择的 CLI coding agent（Claude Code、OpenCode 或 Codex CLI）

## 目录

- [概述](#overview)
- [操作步骤](#instructions)
  - [选项 A：Claude Code](#option-a-claude-code)
  - [选项 B：OpenCode](#option-b-opencode)
  - [选项 C：Codex CLI](#option-c-codex-cli)
- [故障排查](#troubleshooting)

---

<a id="overview"></a>
## 概述

## 基本思路

在 [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) 上使用 [Ollama](https://ollama.com) 运行一个本地编码模型，并连接一个 CLI coding agent。本 playbook 支持三个选项：**[Claude Code](https://docs.claude.com/en/docs/claude-code)**、**[OpenCode](https://opencode.ai)** 和 **[Codex CLI](https://github.com/openai/codex)**。每个 agent 都通过 Ollama 内置的[启动方式](https://ollama.com/blog/launch)（`ollama launch <agent>`）进行接入，因此你无需配置环境变量、provider 配置文件，也无需依赖外部云端 API。

## 选择你的 CLI agent

根据你想使用的 CLI agent 选择对应的章节：

- **Claude Code**：以最快路径让本地 Ollama 模型搭配可用的 CLI agent 跑起来。
- **OpenCode**：直接通过 Ollama 启动的开源 CLI。
- **Codex CLI**：通过 Ollama 直接启动 OpenAI 的 Codex CLI，对接本地模型。

## 你将完成什么

你将在 DGX Spark 上使用 Ollama 运行一个本地编码模型（[Qwen3.6](https://ollama.com/library/qwen3.6)），用一条命令启动你选择的 CLI agent 与该模型对接，并端到端地完成一个小型编码任务。

## 开始之前需要了解什么

- 熟悉 Linux 命令行基础操作
- 有运行基于终端的工具与编辑器的经验
- 了解 Python，以完成一个简短的编码任务

## 先决条件

- 可访问运行 NVIDIA DGX OS 7.3.1（基于 Ubuntu 24.04.3 LTS）的 DGX Spark
- 可访问互联网以下载模型权重
- [Ollama](https://ollama.com/download) v0.15 或更新版本（[`ollama launch`](https://ollama.com/blog/launch) 所必需）
- GPU 显存需求取决于你选择的 Qwen3.6 变体：
  - `qwen3.6:latest`（35B-a3b，MoE）— 约 24GB，256K 上下文
  - `qwen3.6:35b-a3b-nvfp4` — 约 22GB，针对 Blackwell（DGX Spark）调优的 NVIDIA FP4 构建
  - `qwen3.6:35b-a3b-q8_0` — 约 39GB，质量更高的量化版本
  - `qwen3.6:35b-a3b-bf16` — 约 71GB，全精度（可放入 Spark 的统一内存）

## 时间与风险

* **耗时**：约 15-25 分钟（主要是模型下载时间）
* **风险等级**：低
  * 网络不稳定时大模型下载可能失败
  * 低于 0.15 版本的 Ollama 不支持 `ollama launch`
* **回滚方式**：停止 Ollama 并从 `~/.ollama/models` 删除已下载的模型
* **最近更新**：2026/04/16
  * 切换到 `ollama launch` 方式，并将默认模型升级为 Qwen3.6

---

<a id="instructions"></a>
## 操作步骤

下方分别给出三种 CLI agent 的完整流程。前 4 步（环境确认、安装 Ollama、拉取模型、可选的本地推理测试）在三种选项中是一致的，只有从第 5 步开始的启动命令和后续操作不同。请根据需要选择其中一节执行。

---

<a id="option-a-claude-code"></a>
### 选项 A：Claude Code

#### 步骤 1. 确认你的环境

**说明**：在安装任何东西之前，先确认操作系统版本和 GPU 是否可见。

```bash
cat /etc/os-release | head -n 2
nvidia-smi
```

预期输出应显示 Ubuntu 24.04.3 LTS（DGX OS 7.3.1 基础）以及检测到的 GPU。

#### 步骤 2. 安装或升级 Ollama

**说明**：安装 [Ollama](https://ollama.com/download)，或者确保版本足够新以支持 [`ollama launch`](https://ollama.com/blog/launch)。

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
```

如果 Ollama 已安装，只需确认版本：

```bash
ollama --version
```

预期输出应显示 Ollama v0.15 或更新版本。

#### 步骤 3. 拉取 Qwen3.6

**说明**：将 [Qwen3.6](https://ollama.com/library/qwen3.6) 模型权重下载到你的 Spark 节点上。

```bash
ollama pull qwen3.6
```

如果你需要不同的显存占用或精度，可选地拉取以下变体：

```bash
ollama pull qwen3.6:35b-a3b-nvfp4   # NVIDIA FP4 build tuned for Blackwell (~22GB)
ollama pull qwen3.6:35b-a3b-q8_0    # Higher-quality 8-bit quant (~39GB)
ollama pull qwen3.6:35b-a3b-bf16    # Full precision (~71GB)
```

预期输出应在 `ollama list` 中显示 `qwen3.6`（以及任何可选变体）。

#### 步骤 4. 测试本地推理（可选）

**说明**：通过一个简短的提示词来确认模型可以加载。

```bash
ollama run qwen3.6
```

可以尝试这样的提示词：

```text
Write a short README checklist for a Python project.
```

预期输出应显示模型在终端中作出回复。完成后，输入 `/bye` 或按 `Ctrl+D` 退出交互式会话，然后再继续。

#### 步骤 5. 通过 Ollama 启动 Claude Code

**说明**：使用 Ollama 内置的[启动方式](https://ollama.com/blog/launch)，将 [Claude Code](https://docs.claude.com/en/docs/claude-code) 对接到你的本地模型。无需配置任何环境变量或配置文件。

```bash
ollama launch claude
```

预期输出应显示 Claude Code 启动并使用本地的 Qwen3.6 模型。Qwen3.6 默认提供 256K 上下文窗口；如需进一步调整，可通过 Ollama 的设置来调节上下文长度。

#### 步骤 6. 完成一个小型编码任务

**说明**：创建一个小仓库，让 Claude Code 实现一个函数及对应的测试。

```bash
mkdir -p ~/cli-agent-demo
cd ~/cli-agent-demo

printf 'def add(a, b):\n    """Return the sum of a and b."""\n    pass\n' > math_utils.py
printf 'import math_utils\n\n\ndef test_add():\n    assert math_utils.add(1, 2) == 3\n' > test_math_utils.py
```

如果你还没有安装 pytest：

```bash
python -m pip install -U pytest
```

在 Claude Code 中：

```text
Please implement add() in math_utils.py and make sure the test passes.
```

运行测试：

```bash
python -m pytest -q
```

预期输出应显示测试通过。

#### 步骤 7. 清理与回滚

**说明**：如果你不再需要这些组件，可以删除模型并停止服务。

停止服务：

```bash
sudo systemctl stop ollama
```

> [!WARNING]
> 这将删除已下载的模型文件。

```bash
ollama rm qwen3.6
```

#### 步骤 8. 后续可尝试

- 试试 `qwen3.6:35b-a3b-nvfp4` 或 `bf16` 变体，以体验不同的质量/显存权衡
- 在多文件重构或测试生成等任务中使用 Claude Code
- 在更大的代码库上充分利用完整的 256K 上下文窗口

---

<a id="option-b-opencode"></a>
### 选项 B：OpenCode

#### 步骤 1. 确认你的环境

**说明**：在安装任何东西之前，先确认操作系统版本和 GPU 是否可见。

```bash
cat /etc/os-release | head -n 2
nvidia-smi
```

预期输出应显示 Ubuntu 24.04.3 LTS（DGX OS 7.3.1 基础）以及检测到的 GPU。

#### 步骤 2. 安装或升级 Ollama

**说明**：安装 [Ollama](https://ollama.com/download)，或者确保版本足够新以支持 [`ollama launch`](https://ollama.com/blog/launch)。

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
```

如果 Ollama 已安装，只需确认版本：

```bash
ollama --version
```

预期输出应显示 Ollama v0.15 或更新版本。

#### 步骤 3. 拉取 Qwen3.6

**说明**：将 [Qwen3.6](https://ollama.com/library/qwen3.6) 模型权重下载到你的 Spark 节点上。

```bash
ollama pull qwen3.6
```

如果你需要不同的显存占用或精度，可选地拉取以下变体：

```bash
ollama pull qwen3.6:35b-a3b-nvfp4   # NVIDIA FP4 build tuned for Blackwell (~22GB)
ollama pull qwen3.6:35b-a3b-q8_0    # Higher-quality 8-bit quant (~39GB)
ollama pull qwen3.6:35b-a3b-bf16    # Full precision (~71GB)
```

预期输出应在 `ollama list` 中显示 `qwen3.6`。

#### 步骤 4. 测试本地推理（可选）

**说明**：通过一个简短的提示词来确认模型可以加载。

```bash
ollama run qwen3.6
```

可以尝试这样的提示词：

```text
Write a short README checklist for a Python project.
```

预期输出应显示模型作出回复。完成后，输入 `/bye` 或按 `Ctrl+D` 退出，然后再继续。

#### 步骤 5. 通过 Ollama 启动 OpenCode

**说明**：使用 Ollama 内置的[启动方式](https://ollama.com/blog/launch)，将 [OpenCode](https://opencode.ai) 对接到你的本地模型。无需配置 [`opencode.json`](https://opencode.ai/docs/config/) 中的 provider。

```bash
ollama launch opencode
```

如果你想预先配置 OpenCode 而不立即启动：

```bash
ollama launch opencode --config
```

预期输出应显示 OpenCode 启动，并已自动选定 Ollama 作为 provider、Qwen3.6 作为模型。Qwen3.6 默认提供 256K 上下文窗口。

#### 步骤 6. 完成一个小型编码任务

**说明**：创建一个小仓库，让 OpenCode 实现一个函数及对应的测试。

```bash
mkdir -p ~/cli-agent-demo
cd ~/cli-agent-demo

printf 'def add(a, b):\n    """Return the sum of a and b."""\n    pass\n' > math_utils.py
printf 'import math_utils\n\n\ndef test_add():\n    assert math_utils.add(1, 2) == 3\n' > test_math_utils.py
```

如果你还没有安装 pytest：

```bash
python -m pip install -U pytest
```

在 OpenCode 中：

```text
Please implement add() in math_utils.py and make sure the test passes.
```

运行测试：

```bash
python -m pytest -q
```

预期输出应显示测试通过。

#### 步骤 7. 清理与回滚

**说明**：如果你不再需要这些组件，可以删除模型并停止服务。

停止服务：

```bash
sudo systemctl stop ollama
```

> [!WARNING]
> 这将删除已下载的模型文件。

```bash
ollama rm qwen3.6
```

#### 步骤 8. 后续可尝试

- 试试 `qwen3.6:35b-a3b-nvfp4` 或 `bf16` 变体，以体验不同的质量/显存权衡
- 在多文件改动或测试生成等任务中使用 OpenCode
- 在更大的代码库上充分利用完整的 256K 上下文窗口

---

<a id="option-c-codex-cli"></a>
### 选项 C：Codex CLI

#### 步骤 1. 确认你的环境

**说明**：在安装任何东西之前，先确认操作系统版本和 GPU 是否可见。

```bash
cat /etc/os-release | head -n 2
nvidia-smi
```

预期输出应显示 Ubuntu 24.04.3 LTS（DGX OS 7.3.1 基础）以及检测到的 GPU。

#### 步骤 2. 安装或升级 Ollama

**说明**：安装 [Ollama](https://ollama.com/download)，或者确保版本足够新以支持 [`ollama launch`](https://ollama.com/blog/launch)。

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
```

如果 Ollama 已安装，只需确认版本：

```bash
ollama --version
```

预期输出应显示 Ollama v0.15 或更新版本。

#### 步骤 3. 拉取 Qwen3.6

**说明**：将 [Qwen3.6](https://ollama.com/library/qwen3.6) 模型权重下载到你的 Spark 节点上。

```bash
ollama pull qwen3.6
```

如果你需要不同的显存占用或精度，可选地拉取以下变体：

```bash
ollama pull qwen3.6:35b-a3b-nvfp4   # NVIDIA FP4 build tuned for Blackwell (~22GB)
ollama pull qwen3.6:35b-a3b-q8_0    # Higher-quality 8-bit quant (~39GB)
ollama pull qwen3.6:35b-a3b-bf16    # Full precision (~71GB)
```

预期输出应在 `ollama list` 中显示 `qwen3.6`。

#### 步骤 4. 测试本地推理（可选）

**说明**：通过一个简短的提示词来确认模型可以加载。

```bash
ollama run qwen3.6
```

可以尝试这样的提示词：

```text
Write a short README checklist for a Python project.
```

预期输出应显示模型作出回复。完成后，输入 `/bye` 或按 `Ctrl+D` 退出，然后再继续。

#### 步骤 5. 通过 Ollama 启动 Codex CLI

**说明**：使用 Ollama 内置的[启动方式](https://ollama.com/blog/launch)，将 [Codex CLI](https://github.com/openai/codex) 对接到你的本地模型。不需要 `~/.codex/config.toml`，也不需要手动执行 `npm install -g @openai/codex` —— Ollama 会负责 Codex 的集成。

```bash
ollama launch codex
```

预期输出应显示 Codex CLI 启动，并以 Ollama 为 provider、Qwen3.6 为模型。Qwen3.6 默认提供 256K 上下文窗口，非常适合 Codex 的智能体式工作流。

#### 步骤 6. 完成一个小型编码任务

**说明**：创建一个小仓库，让 Codex 实现一个函数及对应的测试。

```bash
mkdir -p ~/cli-agent-demo
cd ~/cli-agent-demo

printf 'def add(a, b):\n    """Return the sum of a and b."""\n    pass\n' > math_utils.py
printf 'import math_utils\n\n\ndef test_add():\n    assert math_utils.add(1, 2) == 3\n' > test_math_utils.py
```

如果你还没有安装 pytest：

```bash
python -m pip install -U pytest
```

在 Codex 中：

```text
Please implement add() in math_utils.py and make sure the test passes.
```

运行测试：

```bash
python -m pytest -q
```

预期输出应显示测试通过。

#### 步骤 7. 清理与回滚

**说明**：如果你不再需要这些组件，可以删除模型并停止服务。

停止服务：

```bash
sudo systemctl stop ollama
```

> [!WARNING]
> 这将删除已下载的模型文件。

```bash
ollama rm qwen3.6
```

#### 步骤 8. 后续可尝试

- 试试 `qwen3.6:35b-a3b-nvfp4` 或 `bf16` 变体，以体验不同的质量/显存权衡
- 在多文件改动或测试生成等任务中使用 Codex CLI
- 在更大的代码库上充分利用完整的 256K 上下文窗口

---

<a id="troubleshooting"></a>
## 故障排查

| 现象 | 原因 | 解决办法 |
|---------|-------|-----|
| `ollama: command not found` | Ollama 未安装，或 PATH 未更新 | 重新执行 `curl -fsSL https://ollama.com/install.sh \| sh`，并打开新的 shell |
| `ollama launch` 提示未知命令 | Ollama 版本低于 v0.15 | 升级 Ollama：`curl -fsSL https://ollama.com/install.sh \| sh` |
| 模型加载失败，提示版本错误或 HTTP 412 | Ollama 版本对该模型来说过旧 | 升级 Ollama：`curl -fsSL https://ollama.com/install.sh \| sh` |
| 启动 agent 时报 `model not found` | 没有先拉取模型 | 执行 `ollama pull qwen3.6` 后重试 |
| 连接 localhost:11434 时 `connection refused` | Ollama 服务未运行 | 通过 `ollama serve` 启动，或使用 `sudo systemctl start ollama` |
| `ollama launch <agent>` 立即退出 | Agent 集成初始化失败 | 重新执行 `ollama launch <agent>`；如果问题仍然存在，请查看 `journalctl -u ollama` |
| 响应缓慢或出现 OOM 错误 | 选用的模型变体超出了 GPU 显存 | 切换到 `qwen3.6:35b-a3b-nvfp4`，或关闭其他占用 GPU 的工作负载 |

> [!NOTE]
> DGX Spark 采用统一内存架构（UMA），可以让 GPU 与 CPU 之间动态共享内存。
> 如果你看到内存压力较大，可以使用以下命令清空缓冲区缓存：
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```
