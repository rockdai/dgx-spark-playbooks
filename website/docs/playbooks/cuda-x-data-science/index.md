---
id: cuda-x-data-science
title: cuda-x-data-science
sidebar_label: cuda-x-data-science
---

# CUDA-X 数据科学

> 安装并使用 NVIDIA cuML 和 NVIDIA cuDF，在 UMAP、HDBSCAN、pandas 等工作负载上实现零代码改动加速


## 目录

- [概述](#overview)
- [操作步骤](#instructions)

---

<a id="overview"></a>
## 概述

## 基本思路
本 playbook 包含两个示例 notebook，演示如何使用 CUDA-X Data Science 库加速关键机器学习算法以及 pandas 的核心操作：

- **NVIDIA cuDF:** 无需修改代码，即可加速数据准备操作以及对 8GB 字符串数据的核心数据处理。
- **NVIDIA cuML:** 无需修改代码，即可加速 sci-kit learn（LinearSVC）、UMAP 和 HDBSCAN 中常见且计算密集的机器学习算法。

CUDA-X Data Science（原 RAPIDS）是一组用于加速数据科学与数据处理生态的开源库。这些库无需改动代码，就能加速 scikit-learn、pandas 等常用 Python 工具。在 DGX Spark 上，你可以直接用现有代码在桌面侧获得更高性能。

## 你将完成的内容
你将使用 GPU 加速常见的机器学习算法和数据分析操作，了解如何为常用 Python 工具启用加速，并理解在 DGX Spark 上运行数据科学工作流的价值。

## 前置条件
- 熟悉 pandas、scikit-learn，以及支持向量机、聚类、降维等机器学习算法
- 已安装 conda
- 已生成 Kaggle API key

## 时间与风险
* **耗时：** 环境准备约 20-30 分钟，每个 notebook 运行约 2-3 分钟。
* **风险：**
  * 由于网络问题，数据下载可能较慢或失败
  * Kaggle API 生成可能失败，需要重试
* **回滚：** 正常使用过程中不会对系统做永久性更改。
* **最后更新：** 11/07/2025
  * 文案小幅修订

<a id="instructions"></a>
## 操作步骤

## 第 1 步：验证系统要求
- 使用 `nvcc --version` 或 `nvidia-smi` 验证系统已安装 CUDA 13
- 按照[这些说明](https://docs.anaconda.com/miniconda/install/)安装 conda
- 按照[这些说明](https://www.kaggle.com/discussions/general/74235)创建 Kaggle API key，并将 **kaggle.json** 文件放在与 notebook 相同的文件夹中

## 第 2 步：安装 Data Science 库
使用以下命令安装 CUDA-X 库（这会创建一个新的 conda 环境）
  ```bash
    conda create -n rapids-test -c rapidsai -c conda-forge -c nvidia  \
    rapids=25.10 python=3.12 'cuda-version=13.0' \
    jupyter hdbscan umap-learn
  ```
## 第 3 步：激活 conda 环境
  ```bash
    conda activate rapids-test
  ```
## 第 4 步：克隆 playbook 仓库
- 克隆 GitHub 仓库，并进入 **cuda-x-data-science** 文件夹中的 assets 目录
  ```bash
    git clone https://github.com/NVIDIA/dgx-spark-playbooks
  ```
- 将第 1 步创建的 **kaggle.json** 放入 assets 文件夹

## 第 5 步：运行 notebooks
GitHub 仓库中包含两个 notebook。
其中一个使用 pandas 代码在 GPU 上演示大规模字符串数据处理工作流。
- 运行 **cudf_pandas_demo.ipynb** notebook，并在浏览器中使用 `localhost:8888` 访问
  ```bash
    jupyter notebook cudf_pandas_demo.ipynb
  ```
另一个演示了包括 UMAP 和 HDBSCAN 在内的机器学习算法。
- 运行 **cuml_sklearn_demo.ipynb** notebook，并在浏览器中使用 `localhost:8888` 访问
  ```bash
    jupyter notebook cuml_sklearn_demo.ipynb
  ```
如果你是远程访问 DGX-Spark，请务必转发所需端口，以便在本地浏览器中访问 notebook。请使用下面的端口转发命令：
```bash
  ssh -N -L YYYY:localhost:XXXX username@remote_host 
```
- `YYYY`：你希望在本地使用的端口（例如 8888）
- `XXXX`：你在远程机器上启动 Jupyter Notebook 时指定的端口（例如 8888）
- `-N`：阻止 SSH 执行远程命令
- `-L`：指定本地端口转发
