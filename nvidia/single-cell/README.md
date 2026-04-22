# 单细胞 RNA 测序

> 使用 RAPIDS 的端到端 GPU 驱动的 scRNA-seq 工作流程

## 目录

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思路

单细胞 RNA 测序 (scRNA-seq) 让研究人员能够单独研究每个细胞中的基因活性，揭示批量方法隐藏的变异、细胞类型和细胞状态。但这些大型高维数据集需要大量计算来处理。

本手册展示了使用 [RAPIDS-singlecell](https://rapids-singlecell.readthedocs.io/en/latest/)（[scverse® ecosystem](https://github.com/scverse) 中的 RAPIDS 支持的库）进行 scRNA-seq 的端到端 GPU 驱动的工作流程。它遵循熟悉的 [Scanpy API](https://scanpy.readthedocs.io/en/stable/) ，通过直接在 GPU 上处理稀疏计数矩阵，研究人员可以比 CPU 工具更快地运行数据预处理、质量控制 (QC) 和清理、可视化和调查步骤。

## 你将完成什么

1. GPU 加速的数据加载和预处理
2. QC 细胞直观地了解数据
3. 过滤异常细胞
4. 消除不需要的变异来源 
5. PCA 和 UMAP 数据的聚类和可视化
6. 使用 Harmony、k 最近邻、UMAP 和 tSNE 进行批量校正和分析
7. 通过差异表达分析和轨迹分析从数据中探索生物信息

自述文件详细说明了这些步骤。

## 开始之前需要了解什么

- Rapids-singlecell 库模仿了 scverse 的 Scanpy API，允许熟悉标准 CPU 工作流程的用户通过 cuPy 和 NVIDIA RAPIDS cuML 和 cuGraph 轻松适应 GPU 加速。
- 算法精度：与 Scanpy 的 CPU 实现使用近似最近邻搜索不同，该 GPU 实现计算精确的图；因此，结果的微小差异是预期的并且是有效的。
- 参数灵敏度：执行t-SNE时，最近邻的数量必须至少为3x以避免失真

## 先决条件
**硬件要求：**
- NVIDIA Grace Blackwell GB10 超级芯片系统 (DGX Spark)
- 至少 40GB 统一内存可用于 docker 容器和 GPU 加速数据处理
- 至少 30GB 可用存储空间用于 docker 容器和数据文件
- 高速网络连接
- 建议使用高速互联网连接

**软件要求：**
- NVIDIA DGX 操作系统
- 码头工人

## 附属文件

所有必需的资产都可以在 [in the Single-cell RNA Sequencing repository](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/single-cell/) 中找到。在运行的剧本中，它们都可以在 `playbook` 文件夹下找到。

- `scRNA_analysis_preprocessing.ipynb` - 主要剧本笔记本。  
- `README.md` - Playbook 环境快速入门指南。  它还可以在 Jupyter Lab 的主目录中找到。  请从那里开始！
- `/setup/start_playbook.sh` - 在 Docker 容器中开始安装 playbook 的脚本
- `/setup/setup_playbook.sh` - 在用户进入 JupyterLab 环境之前配置 Docker 容器
- `/setup/requirements.txt` - 用作 setup_playbook 中的命令将安装到 playbook 环境中的库列表 

## 时间与风险
* **预计时间：** 首次运行约 15 分钟

  - 笔记本总处理时间：整个管道大约需要 2-3 分钟（演示中记录的约为 130 秒）。
  - 数据加载：~1.7 秒。
  - 预处理：~21 秒。
  - 后处理（聚类/差异表达式）：约 104 秒。
  - 数据：通过互联网下载 docker 容器、库和演示数据集 (dli_census.h5ad)。

* **风险**

  - GPU 内存限制：该工作流程非常消耗 GPU 内存。大型数据集可能会触发内存不足 (OOM) 错误。
  - 内核管理：您可能需要终止/重新启动内核以在工作流程阶段之间释放 GPU 资源。
  - 回滚：如果发生 OOM 错误，请终止所有内核以释放 GPU 内存，然后重新启动特定笔记本或整个 Playbook。

* **最后更新：** 2026 年 1 月 2 日
  * 首次出版

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
- `docker --version` 将打印类似 `Docker version 28.3.3, build 980b856` 的内容。  如果您收到错误提示 Docker 未安装，请重新安装。如果您看到权限被拒绝错误，请通过运行 `sudo usermod -aG docker $USER && newgrp docker` 将您的用户添加到 docker 组。

## 步骤2.安装
打开终端，然后复制并粘贴以下命令：

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/single-cell/assets
bash ./setup/start_playbook.sh
```

start_playbook.sh 将：

1. 拉取 RAPIDS 25.10 笔记本 Docker 容器
2. 使用 setup_playbook.sh 在容器中构建 playbook 所需的所有环境
3. 启动 JupyterLab

使用剧本时请保持终端窗口打开。

您可以通过两种方式访问​​ JupyterLab 服务器
1. 如果在 DGX Spark 上本地运行，则位于 `http://127.0.0.1:8888` 处。 
2. 如果通过网络使用无头 DGX Spark，请在 `http://<SPARK_IP>:8888` 处。

进入 JupyterLab 后，您将看到一个包含 scRNA_analysis_preprocessing.ipynb 的目录以及文件夹 `cuDF`、`cuML`、`cuGraph` 和 `playbook`。 

- `scRNA_analysis_preprocessing.ipynb` 是 playbook 笔记本。  您需要通过双击该文件来打开它。
- `cuDF`、`cuML`、`cuGraph` 文件夹包含标准 RAPIDS 库示例笔记本，可帮助您继续探索。
- `playbook` 包含剧本文件。  该文件夹的内容在无根 Docker 容器内是只读的。

如果您想在自己的系统上安装任何 playbook 笔记本，请查看笔记本附带的文件夹中的自述文件

## 步骤 3. 运行笔记本

进入 JupyterLab 后，您所要做的就是运行 `scRNA_analysis_preprocessing.ipynb`。您将获得这些 playbook 笔记本以及标准 RAPIDS 库示例笔记本来帮助您入门。

您可以使用 `Shift + Enter` 按照自己的节奏手动运行每个单元，或使用 `Run > Run All` 运行所有单元。

探索完 `scRNA_analysis_preprocessing` 笔记本后，您可以通过进入文件夹、选择其他笔记本并执行相同的操作来探索其他 RAPIDS 笔记本。

## 第 4 步：下载您的作品

由于 docker 容器无法以特权写回主机系统，因此您可以使用 JupyterLab 下载 docker 容器关闭后可能想要保留的任何文件。

只需在浏览器中右键单击所需的文件，然后单击下拉列表中的 `Download` 即可。

## 步骤 5. 清理

下载完所有工作后，返回到开始运行剧本的终端窗口。

在终端窗口中， 
1. 类型 `Ctrl + C`
2. 快速输入 `y`，然后在提示符下点击 `Enter` 或再次点击 `Ctrl + C`
3. Docker 容器将继续关闭

> [!WARNING]
> 这将删除尚未从 Docker 容器下载的所有数据。  如果浏览器窗口仍处于打开状态，则可能仍会显示缓存的文件。

## 故障排除

<!-- 
故障排除模板：虽然是可选的，但此资源可以显着帮助用户解决常见问题。
将 {} 中的所有占位符内容替换为您的实际故障排除信息。
完成后删除这些注释块。

目的：为用户可能遇到的问题提供快速解决方案。
格式：使用表格格式以便于扫描。需要时添加详细注释。
-->

| 症状 | 原因 | 使固定 |
|---------|-------|-----|
| 未找到 Docker。 | Docker 可能已被卸载，因为它已预安装在您的 DGX Spark 上 | 请使用此处的便捷脚本安装 Docker：`curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh`。系统将提示您输入密码。 |
| Docker 命令意外退出并出现“权限”错误 | 您的用户不属于 `docker` 组 | 打开终端并运行以下命令：`sudo groupadd docker && sudo usermod -aG docker $USER`。  系统将提示您输入密码。  然后，关闭终端，打开一个新终端，然后重试 |
| Docker容器下载、环境搭建或数据下载失败 | 存在连接问题或资源可能暂时不可用。 | 您可能需要稍后重试。如果这种情况仍然存在，请在 Spark 用户论坛上发帖寻求支持 |




<!-- 
为可能与您的项目相关的一些常见已知问题保留了空间。在更改或删除之前评估潜在后果。
-->

> [!NOTE] 
> DGX Spark 使用统一内存架构 (UMA)，可实现 GPU 和 CPU 之间的动态内存共享。 
> 由于许多应用程序仍在更新以利用 UMA，因此即使在 
> DGX Spark 的内存容量。如果发生这种情况，请使用以下命令手动刷新缓冲区缓存：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

有关最新的已知问题，请查看 [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)。
