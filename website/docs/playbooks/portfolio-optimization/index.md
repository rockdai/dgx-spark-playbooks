---
id: portfolio-optimization
title: portfolio-optimization
sidebar_label: portfolio-optimization
---

# 投资组合优化

> 使用 cuOpt 和 cuML 进行 GPU 加速的投资组合优化

## 目录

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思路

本手册演示了使用 NVIDIA cuOpt 和 NVIDIA cuML 的端到端 GPU 加速工作流程，使用 Mean-CVaR（条件风险价值）模型近乎实时地解决大规模投资组合优化问题。 

投资组合优化 (PO) 涉及解决高维、非线性数值优化问题以平衡风险和回报。现代投资组合通常包含数千个资产，使得传统的基于 CPU 的求解器对于高级工作流程而言速度太慢。通过将繁重的计算工作转移到 GPU，该解决方案极大地减少了计算时间。

## 你将完成什么

您将实现一个提供绩效评估、策略回测、基准测试和可视化工具的管道。工作流程包括：
- **GPU 加速优化：** 利用 NVIDIA cuOpt LP/MILP 求解器 
- **数据驱动的风险建模：** 将 CVaR 实施为基于场景的风险度量，对尾部风险进行建模，而不对资产回报分布做出假设。
- **场景生成：** 通过 NVIDIA cuML 使用 GPU 加速的内核密度估计 (KDE) 对回报分布进行建模。
- **现实世界的约束管理：** 实施约束，包括集中度限制、杠杆约束、营业额限制和基数约束。
- **全面回测：** 使用测试再平衡策略的特定工具评估投资组合表现。


## 开始之前需要了解什么

- **所需技能（你会得到的）：**
  - 基本使用终端和 Linux 命令行
  - 对 Docker 容器的基本了解
  - 使用 Jupyter Notebooks 和 Jupyter Lab 的基础知识
  - 基础Python知识
  - 数据科学和机器学习概念的基础知识
  - 股票市场和股票的基本知识

- **可选技能（你会喜欢的）：**
  - 金融服务背景，特别是量化金融和投资组合管理
  - 适度的知识编程算法和策略，在Python中，使用机器学习概念 

- **须知条款：**
  - **CVaR 与均值方差：** 与传统的均值方差模型不同，此工作流程使用条件风险值 (CVaR) 来捕获风险的细微差别，特别是尾部风险或特定于场景的压力。
  - **线性规划：** CVaR 将风险回报权衡重新表述为基于场景的线性规划，其中问题的规模随着场景的数量而变化，这就是 GPU 加速至关重要的原因。
  - **基准测试：** 该管道包含内置工具，可根据基于标准 CPU 的库简化基准测试流程，以验证性能提升。

## 先决条件

**硬件要求：**
- NVIDIA Grace Blackwell GB10 超级芯片系统 (DGX Spark)
- 至少 40GB 统一内存可用于 docker 容器和 GPU 加速数据处理
- 至少 30GB 可用存储空间用于 docker 容器和数据文件
- 建议使用高速互联网连接

**软件要求：**
- 带有有效 NVIDIA 和 CUDA 驱动程序的 NVIDIA DGX 操作系统
- 码头工人
- git

## 附属文件

所有必需的资产都可以在 [in the Portfolio Optimization repository](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/portfolio-optimization/assets/) 中找到。  在运行的剧本中，它们都可以在 `playbook` 文件夹下找到。

- `cvar_basic.ipynb` - 主要剧本笔记本。  
- `/setup/README.md` - Playbook 环境快速入门指南。
- `/setup/start_playbook.sh` - 在 Docker 容器中开始安装 playbook 的脚本
- `/setup/setup_playbook.sh` - 在用户进入 jupyterlab 环境之前配置 Docker 容器
- `/setup/pyproject.toml` - 用作 setup_playbook 中的命令将安装到 playbook 环境中的库列表 
- `cuDF, cuML, and cuGraph folders` - 更多示例笔记本，可继续您的 GPU 加速数据科学之旅。  当您启动 Docker 容器时，这些将成为 Docker 容器的一部分。

## 时间与风险

* **预计时间** 首次运行约 20 分钟
  - 笔记本总处理时间：整个管道大约需要 7 分钟。

- **风险：**
  - 最小，因为它在 Docker 容器中运行。

* **回滚：** 停止 Docker 容器并删除克隆的存储库以完全删除安装。

* **最后更新：** 2026 年 1 月 21 日
  * 使用正确的项目路径更新 `git clone` 命令。

## 指示

## 步骤 1. 验证您的环境

我们首先验证您是否有可用的 GPU、git 和 Docker。  打开终端，然后复制并粘贴以下命令：

```bash
nvidia-smi
git --version
docker --version
```

- `nvidia-smi` 将输出有关您的 GPU 的信息。  如果没有，则说明您的 GPU 配置不正确。
- `git --version` 将打印类似 `git version 2.43.0` 的内容。  如果出现错误提示未安装 git，请重新安装。
- `docker --version` 将打印类似 `Docker version 28.3.3, build 980b856` 的内容。  如果您收到错误提示 Docker 未安装，请重新安装。

## 步骤2.安装
打开终端，然后复制并粘贴以下命令：

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/portfolio-optimization/assets
bash ./setup/start_playbook.sh
```

start_playbook.sh 将：

1. 拉取 RAPIDS 25.10 笔记本 Docker 容器
2. 使用 `setup_playbook.sh` 在容器中构建剧本所需的所有环境
3. 启动 Jupyterlab

使用剧本时请保持终端窗口打开。

您可以通过三种方式访问​​ Jupyterlab 服务器
1. 如果在 DGX Spark 上本地运行，则位于 `http://127.0.0.1:8888` 处。 
2. 如果通过网络使用无头 DGX Spark，请在 `http://SPARK_IP:8888` 处。
3. 通过在终端中使用 `ssh -L 8888:localhost:8888 username@spark-IP` 创建 SSH 隧道，并在主机上的浏览器中转到 `http://127.0.0.1:8888`

进入 Jupyterlab 后，您将看到一个包含 `cvar_basic.ipynb` 的目录以及文件夹 `cudf`、`cuml` 和 `cugraph`。 

- `cvar_basic.ipynb` 是 playbook 笔记本。  您需要通过双击该文件来打开它。
- `cudf`、`cuml`、`cugraph` 文件夹包含标准 RAPIDS 库示例笔记本，可帮助您继续探索。
- `playbook` 包含剧本文件。  该文件夹的内容在无根 Docker 容器内是只读的。

如果您想在自己的系统上安装任何 playbook 笔记本，请查看笔记本附带的文件夹中的自述文件

## 步骤 3. 运行笔记本

进入 jupyterlab 后，您要做的就是运行 `cvar_basic.ipynb`。 

在开始运行笔记本中的单元之前，**请按照笔记本中的说明将内核更改为“Portfolio Optimization”。** 如果不这样做，将导致第二个代码单元出错。  如果您已经启动，则必须将其设置为正确的内核，然后重新启动内核，然后重试。 

您可以使用 `Shift + Enter` 按照自己的节奏手动运行每个单元，或使用 `Run > Run All` 运行所有单元。

探索完 `cvar_basic` 笔记本后，您可以通过进入文件夹、选择其他笔记本并执行相同的操作来探索其他 RAPIDS 笔记本。

## 第 4 步：下载您的作品

由于 docker 容器没有特权并且无法写回主机系统，因此您可以使用 Jupyterlab 下载 docker 容器关闭后可能想要保留的任何文件。

只需在浏览器中右键单击所需的文件，然后单击下拉列表中的 `Download` 即可。

## 步骤 5. 清理

下载完所有工作后，返回到开始运行剧本的终端窗口。

在终端窗口中：
1. 类型 `Ctrl + C`
2. 快速输入 `y`，然后在提示符下点击 `Enter` 或再次点击 `Ctrl + C`
3. Docker 容器将继续关闭

> [！警告]
> 这将删除尚未从 Docker 容器下载的所有数据。  如果浏览器窗口仍处于打开状态，则可能仍会显示缓存的文件。

## 步骤 6. 后续步骤

一旦您熟悉了这个基础工作流程，请在 **[NVIDIA AI Blueprints](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/)** 以任意顺序探索这些高级投资组合优化主题：

* **[`efficient_frontier.ipynb`](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/tree/main/notebooks/efficient_frontier.ipynb)** - 有效的前沿分析

  本笔记本演示了如何：
  - 通过解决多个优化问题生成有效前沿
  - 可视化不同投资组合配置的风险回报权衡
  - 沿有效边界比较投资组合
  - 利用GPU加速快速计算多个最优投资组合

* **[`rebalancing_strategies.ipynb`](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/tree/main/notebooks/rebalancing_strategies.ipynb)** - 动态投资组合再平衡

  本笔记本介绍了动态投资组合管理技术：
  - 时间序列回测框架
  - 测试各种再平衡策略（定期、基于阈值等）
  - 评估交易成本对投资组合绩效的影响
  - 分析不同市场条件下的策略绩效
  - 比较多种再平衡方法

* 如果您想进一步了解如何使用类似的风险回报框架来制定投资组合优化问题，请查看 **[DLI course: Accelerating Portfolio Optimization](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-DS-09+V1)**

## 步骤 7. 进一步支持

如有疑问或问题，请访问：
- [GitHub Issues](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/issues)

## 故障排除

{/* 故障排除模板：虽然是可选的，但此资源可以显着帮助用户解决常见问题。
将 {} 中的所有占位符内容替换为您的实际故障排除信息。
完成后删除这些注释块。

目的：为用户可能遇到的问题提供快速解决方案。
格式：使用表格格式以便于扫描。需要时添加详细注释。 */}

| 症状 | 原因 | 使固定 |
|---------|-------|-----|
| 未找到 Docker。 | Docker 可能已被卸载，因为它已预安装在您的 DGX Spark 上 | 请使用此处的便捷脚本安装 Docker：`curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh`。系统将提示您输入密码。 |
| Docker 命令意外退出并出现“权限”错误 | 您的用户不属于 `docker` 组 | 打开终端并运行以下命令：`sudo groupadd docker $$ sudo usermod -aG docker $USER`。  系统将提示您输入密码。  然后，关闭终端，打开一个新终端，然后重试 |
| Docker容器下载、环境搭建或数据下载失败 | 存在连接问题或资源可能暂时不可用。 | 您可能需要稍后重试。如果这种情况持续存在，请联系我们！ |




{/* 为可能与您的项目相关的一些常见已知问题保留了空间。在更改或删除之前评估潜在后果。 */}

> [！笔记] 
> DGX Spark 使用统一内存架构 (UMA)，可实现 GPU 和 CPU 之间的动态内存共享。 
> 由于许多应用程序仍在更新以利用 UMA，因此即使在 
> DGX Spark 的内存容量。如果发生这种情况，请使用以下命令手动刷新缓冲区缓存：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

有关最新的已知问题，请查看 [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)。
