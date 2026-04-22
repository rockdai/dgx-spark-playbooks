---
id: isaac
title: isaac
sidebar_label: isaac
---

# 安装和使用 Isaac Sim 和 Isaac Lab

> 在 Spark 上从源码构建 Isaac Sim 和 Isaac Lab

## 目录

- [概述](#overview)
- [运行 Isaac Sim](#run-isaac-sim)
- [运行 Isaac Lab](#run-isaac-lab)
- [故障排查](#troubleshooting)

---

<a id="overview"></a>
## 概述

## 基本思路

Isaac Sim 是构建于 NVIDIA Omniverse 之上的机器人仿真平台，可实现对机器人和环境的照片级真实、物理精确仿真。它为机器人开发提供了完整工具集，包括物理仿真、传感器仿真和可视化能力。Isaac Lab 则是构建在 Isaac Sim 之上的强化学习框架，用于训练和部署机器人应用中的 RL 策略。

Isaac Sim 利用 GPU 加速物理仿真，实现快速且逼真的机器人模拟，甚至能够快于实时运行。Isaac Lab 在此基础上进一步提供了预构建的 RL 环境、训练脚本和评估工具，适用于行走、操作、导航等常见机器人任务。二者结合后，可以在真正部署到实体硬件之前，提供一套端到端的解决方案，用于在纯仿真环境中开发、训练和测试机器人应用。

## 你将完成的内容

你将会在 NVIDIA DGX Spark 设备上从源码构建 Isaac Sim，并搭建 Isaac Lab 以进行强化学习实验。这包括编译 Isaac Sim 引擎、配置开发环境，以及运行一个示例 RL 训练任务来验证安装。

## 开始前需要了解

- 具备使用 CMake 和构建系统从源码构建软件的经验
- 熟悉 Linux 命令行操作和环境变量
- 理解 Git 版本控制以及用于大文件管理的 Git LFS
- 具备 Python 包管理和虚拟环境的基础知识
- 了解机器人仿真相关概念（有帮助，但不是必须）

## 前置条件

**硬件要求：**
- NVIDIA Grace Blackwell GB10 超级芯片系统
- 至少 50GB 可用存储空间，用于 Isaac Sim 构建产物和依赖

**软件要求：**
- NVIDIA DGX 操作系统
- GCC/G++ 11 编译器：`gcc --version` 显示 11.x
- 已安装 Git 和 Git LFS：`git --version` 和 `git lfs version` 可正常执行
- 可访问网络以从 GitHub 克隆仓库并下载依赖

## 相关文件

所有必需资源都可在 GitHub 上的 Isaac Sim 和 Isaac Lab 仓库中找到：
- [Isaac Sim repository](https://github.com/isaac-sim/IsaacSim) - Isaac Sim 主源码仓库
- [Isaac Lab repository](https://github.com/isaac-sim/IsaacLab) - Isaac Lab 强化学习框架

## 时间与风险

* **预计耗时：** 30 分钟（其中构建通常需要 10-15 分钟）
* **风险等级：** 中
  * 使用 Git LFS 克隆大型仓库可能因网络问题失败
  * 构建过程需要较长编译时间，并可能遇到依赖问题
  * 构建产物会占用大量磁盘空间
* **回滚：** 可删除 Isaac Sim 构建目录以释放空间；如有需要，也可删除 Git 仓库后重新克隆。
* **最后更新：** 01/02/2026
  * 首次发布

<a id="run-isaac-sim"></a>
## 运行 Isaac Sim

## 第 1 步：安装 gcc-11 和 git-lfs

构建前，请使用以下命令确认正在使用 GCC/G++ 11：
```bash
sudo apt update && sudo apt install -y gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 200
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 200
sudo apt install git-lfs
gcc --version
g++ --version
```

## 第 2 步：将 Isaac Sim 仓库克隆到你的工作区

从 NVIDIA GitHub 仓库克隆 Isaac Sim，并配置 Git LFS 以拉取大文件。

> **Note:** 对于 Isaac Sim 6.0.0 Early Developer Release，请使用：
> ````重击
> git克隆--深度= 1 --递归--分支=开发https://github.com/isaac-sim/IsaacSim
> ````

```bash
git clone --depth=1 --recursive https://github.com/isaac-sim/IsaacSim
cd IsaacSim
git lfs install
git lfs pull
```

## 第 3 步：构建 Isaac Sim

构建 Isaac Sim，并接受许可协议。

```bash
./build.sh
```

当构建成功时，你会看到如下消息：**BUILD (RELEASE) SUCCEEDED (Took 674.39 seconds)**


## 第 4 步：让系统识别 Isaac Sim

运行以下命令前，请确保你当前位于 Isaac Sim 目录中。

```bash
export ISAACSIM_PATH="${PWD}/_build/linux-aarch64/release"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
```

## 第 5 步：运行 Isaac Sim

使用提供的 Python 可执行环境启动 Isaac Sim。

```bash
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
${ISAACSIM_PATH}/isaac-sim.sh
```

<a id="run-isaac-lab"></a>
## 运行 Isaac Lab

## 第 1 步：安装 Isaac Sim
如果你尚未安装，请先安装 [Isaac Sim](build.nvidia.com/spark/isaac/isaac-sim)。

## 第 2 步：将 Isaac Lab 仓库克隆到你的工作区

从 NVIDIA GitHub 仓库克隆 Isaac Lab。

> **Note:** 对于 Isaac Lab Early Developer Release，请使用：
> ````重击
> git 克隆 --recursive --branch=develop https://github.com/isaac-sim/IsaacLab
> ````

```bash
git clone --recursive https://github.com/isaac-sim/IsaacLab
cd IsaacLab
```

## 第 3 步：为 Isaac Sim 安装创建符号链接

运行以下命令前，请确保你已经按照 [Isaac Sim](build.nvidia.com/spark/isaac/isaac-sim) 完成安装。

```bash
echo "ISAACSIM_PATH=$ISAACSIM_PATH"
```
为 Isaac Sim 安装目录创建一个符号链接。
```bash
ln -sfn "${ISAACSIM_PATH}" "${PWD}/_isaac_sim"
ls -l "${PWD}/_isaac_sim/python.sh"
```

## 第 4 步：安装 Isaac Lab

安装 Newton 所需依赖，其中需要 X11 开发库来从源码构建 imgui_bundle。

```bash
sudo apt update
sudo apt install -y libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl1-mesa-dev
```

然后安装 Isaac Lab。

```bash
./isaaclab.sh --install
```

## 第 5 步：运行 Isaac Lab 并验证类人机器人强化学习训练

使用提供的 Python 可执行环境启动 Isaac Lab。你可以使用以下任一模式运行训练：

**选项 1：无界面模式（推荐用于更快训练）**

不进行可视化，日志会直接输出到终端。

```bash
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-H1-v0 --headless
```

**选项 2：启用可视化**

在 Isaac Sim 中启用实时可视化，以便交互式观察训练过程。

```bash
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-H1-v0
```

<a id="troubleshooting"></a>
## 故障排查

## Isaac Sim 常见问题

| 现象                     | 原因                    | 解决方法                               |
|-----------------------------|--------------------------|-----------------------------------|
| Isaac Sim 编译报错 | 默认 gcc/g++ 不是 11 | 确保 gcc/g++ 11 被设置为默认版本 |
| Isaac Sim 无法执行      | `libgomp.so.1` 错误       | 添加 `export LD_PRELOAD`             |
| 构建时报错              | 旧安装残留         | 删除 `.cache` 文件夹              |

## Isaac Lab 常见问题
| 现象                          | 原因 | 解决方法 |
|----------------------------------|--------|-----|
| Isaac Lab 无法执行           | `libgomp.so.1` 错误       | 添加 `export LD_PRELOAD`     |
