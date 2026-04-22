# 优化的 JAX

> 为 Spark 优化 JAX 运行环境

## 目录

- [概述](#overview)
- [操作步骤](#instructions)
- [故障排查](#troubleshooting)

---

<a id="overview"></a>
## 概述

## 基本思路

JAX 让你能够编写**NumPy 风格的 Python 代码**，无需编写 CUDA 就能在 GPU 上高效运行。其实现方式包括：

- **在加速器上使用 NumPy**：像使用 NumPy 一样使用 `jax.numpy`，但数组实际驻留在 GPU 上。  
- **函数变换**：  
  - `jit` → 将函数编译为高性能 GPU 代码  
  - `grad` → 提供自动微分 
  - `vmap` → 对批量数据进行向量化  
  - `pmap` → 在多个 GPU 上并行运行 
- **XLA 后端**：JAX 会将代码交给 XLA（Accelerated Linear Algebra 编译器），由其融合操作并生成优化后的 GPU kernel。

## 你将完成的内容

你将会在采用 Blackwell 架构的 NVIDIA Spark 上搭建 JAX 开发环境，借助熟悉的 NumPy 风格抽象完成高性能机器学习原型开发，并具备 GPU 加速与性能优化能力。

## 开始前需要了解

- 熟悉 Python 和 NumPy 编程
- 对机器学习工作流和常见技术有基本理解
- 有终端使用经验
- 有使用和构建容器的经验
- 熟悉不同版本的 CUDA
- 具备基础线性代数知识（高中数学水平即可）

## 前置条件

- 采用 Blackwell 架构的 NVIDIA Spark 设备
- ARM64（AArch64）处理器架构
- 已安装 Docker 或其他容器运行时
- 已配置 NVIDIA Container Toolkit
- 验证 GPU 可访问：`nvidia-smi`
- 8080 端口可用于访问 marimo notebook

## 相关文件

所有必需资源可在 [GitHub](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main) 上找到

- [**JAX introduction notebook**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/jax/assets/jax-intro.py) — 介绍 JAX 编程模型与 NumPy 的差异及性能评估
- [**NumPy SOM implementation**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/jax/assets/numpy-som.py) — 使用 NumPy 实现自组织映射训练算法的参考实现  
- [**JAX SOM implementations**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/jax/assets/som-jax.py) — 使用 JAX 逐步优化的多版 SOM 算法实现
- [**Environment configuration**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/jax/assets/Dockerfile) — 环境依赖与容器配置说明


## 时间与风险

* **耗时：** 2-3 小时，包括环境搭建、教程完成和验证
* **风险：**
  * Python 环境中的包依赖可能发生冲突
  * 性能验证可能需要针对特定架构做优化
* **回滚：** 容器环境具备隔离性；删除容器并重新启动即可重置状态。
* **最后更新：** 11/07/2025
  * 文案小幅修订

<a id="instructions"></a>
## 操作步骤

## 第 1 步：验证系统前置条件

确认你的 NVIDIA Spark 系统满足要求，并且已配置 GPU 访问。

```bash
## Verify GPU access
nvidia-smi

## Verify ARM64 architecture  
uname -m

## Check Docker GPU support
docker run --gpus all --rm nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04 nvidia-smi
```

如果出现权限拒绝错误（例如 permission denied while trying to connect to the Docker daemon socket），请将当前用户加入 docker 组，这样就无需使用 sudo 运行 Docker 命令。

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## 第 2 步：克隆 playbook 仓库

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
```

## 第 3 步：构建 Docker 镜像


> [！警告]
> 此命令会下载基础镜像，并在本地构建支持该环境的容器。

```bash
cd dgx-spark-playbooks/nvidia/jax/assets
docker build -t jax-on-spark .
```

## 第 4 步：启动 Docker 容器

运行带有 GPU 支持和 marimo 端口转发的 JAX 开发环境容器。

```bash
docker run --gpus all --rm -it \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 8080:8080 \
    jax-on-spark
```

## 第 5 步：访问 marimo 界面

连接到 marimo notebook 服务器，开始 JAX 教程。

```bash
## Access via web browser
## Navigate to: http://localhost:8080
```

界面会加载目录视图以及简要的 marimo 介绍。

## 第 6 步：完成 JAX 入门教程

学习入门内容，理解 JAX 编程模型与 NumPy 的差异。

进入并完成 JAX introduction notebook，其中包括：
- JAX 编程模型基础
- 与 NumPy 的关键差异
- 性能评估方法

## 第 7 步：实现 NumPy 基线版本

完成基于 NumPy 的自组织映射（SOM）实现，建立性能基线。

完成 NumPy SOM notebook，以便：
- 理解 SOM 训练算法
- 使用熟悉的 NumPy 操作实现该算法
- 记录性能指标用于比较

## 第 8 步：使用 JAX 实现进行优化

依次完成逐步优化的 JAX 实现，观察性能提升。

完成 JAX SOM notebook 中的以下部分：
- NumPy 实现的基础 JAX 移植版
- 性能优化版 JAX 实现
- GPU 加速的并行 JAX 实现
- 对比所有版本的性能

## 第 9 步：验证性能提升

这些 notebook 会演示如何检查各个 SOM 训练实现的性能；你将看到 JAX 实现相较于 NumPy 基线具有性能提升，其中一些版本会快很多。

通过观察随机颜色数据上的 SOM 训练输出，确认算法结果正确。

## 第 10 步：后续操作

将 JAX 优化技巧应用到你自己的基于 NumPy 的机器学习代码中。

```bash
## Example: Profile your existing NumPy code
python -m cProfile your_numpy_script.py

## Then adapt to JAX and compare performance
```

尝试将你常用的 NumPy 算法迁移到 JAX，并在 Blackwell GPU 架构上测量性能提升。

<a id="troubleshooting"></a>
## 故障排查

| 现象 | 原因 | 解决方法 |
|---------|--------|-----|
| 找不到 `nvidia-smi` | 缺少 NVIDIA 驱动 | 为 ARM64 安装 NVIDIA 驱动 |
| 容器无法访问 GPU | 缺少 NVIDIA Container Toolkit | 安装 `nvidia-container-toolkit` |
| JAX 只使用 CPU | CUDA/JAX 版本不匹配 | 重新安装支持 CUDA 的 JAX |
| 8080 端口不可用 | 端口已被占用 | 使用 `-p 8081:8080` 或终止占用 8080 的进程 |
| Docker 构建时发生包冲突 | 环境文件已过时 | 为 Blackwell 更新环境文件 |

> [！笔记]
> DGX Spark 使用统一内存架构（UMA），可在 GPU 和 CPU 之间动态共享内存。 
> 由于许多应用仍在逐步适配 UMA，即使看起来尚未达到 DGX Spark 的内存上限，也可能遇到内存问题。 
> 如果发生这种情况，请手动刷新 buffer cache：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
