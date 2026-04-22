# NCCL 两火花

> 在两个 Spark 上安装并测试 NCCL

## 目录

- [Overview](#overview)
- [Run on two Sparks](#run-on-two-sparks)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思路

NCCL（NVIDIA Collective Communication Library）实现高性能 GPU 到 GPU 通信
跨多个节点。本演练设置 NCCL 进行多节点分布式训练
采用 Blackwell 架构的 DGX Spark 系统。您将配置网络、构建 NCCL
Blackwell 支持源，并验证节点之间的通信。

## 你将完成什么

您将拥有一个可运行的多节点 NCCL 环境，可实现高带宽 GPU 通信
跨 DGX Spark 系统进行分布式训练工作负载，并具有经过验证的网络性能
以及正确的 GPU 拓扑检测。

## 开始之前需要了解什么

- 使用 Linux 网络配置和 netplan
- 对 MPI（消息传递接口）概念的基本了解
- SSH 密钥管理和无密码身份验证设置

## 先决条件

- 两个 DGX Spark 系统
- 完成《连接两个 Sparks》手册
- 安装的 NVIDIA 驱动程序：`nvidia-smi`
- 可用的 CUDA 工具包：`nvcc --version`
- Root/sudo 权限：`sudo whoami`

## 时间与风险

* **持续时间**：30 分钟用于设置和验证
* **风险级别**：中 - 涉及网络配置更改
* **回滚**：可以从 DGX Spark 中删除 NCCL 和 NCCL 测试存储库
* **最后更新：** 2025 年 12 月 15 日
  * 使用nccl最新版本v2.28.9-1

## 在两个 Spark 上运行

## 步骤 1. 配置网络连接

按照 [Connect two Sparks](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks) 手册中的网络设置说明在 DGX Spark 节点之间建立连接。

这包括：
- 物理 QSFP 电缆连接
- 网络接口配置（自动或手动IP分配）
- 无密码 SSH 设置
- 网络连接验证

## 步骤 2. 在 Blackwell 支持下构建 NCCL

在两个节点上执行这些命令以使用 Blackwell 从源代码构建 NCCL
架构支持：

```bash
## Install dependencies and build NCCL
sudo apt-get update && sudo apt-get install -y libopenmpi-dev
git clone -b v2.28.9-1 https://github.com/NVIDIA/nccl.git ~/nccl/
cd ~/nccl/
make -j src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"

## Set environment variables
export CUDA_HOME="/usr/local/cuda"
export MPI_HOME="/usr/lib/aarch64-linux-gnu/openmpi"
export NCCL_HOME="$HOME/nccl/build/"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64/:$MPI_HOME/lib:$LD_LIBRARY_PATH"
```

## 步骤 3. 构建 NCCL 测试套件

在**两个节点**上编译 NCCL 测试套件：

```bash
## Clone and build NCCL tests
git clone https://github.com/NVIDIA/nccl-tests.git ~/nccl-tests/
cd ~/nccl-tests/
make MPI=1
```

## 步骤 4. 查找活动网络接口和 IP 地址

首先，确定哪些网络端口可用且已启动：

```bash
## Check network port status
ibdev2netdev
```

输出示例：
```
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Down)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)
rocep1s0f0 port 1 ==> enp1s0f0np0 (Down)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)
```

使用在输出中显示为“(Up)”的界面。在此示例中，我们将使用 **enp1s0f1np1**。您可以忽略以前缀 `enP2p<...>` 开头的接口，而只考虑以 `enp1<...>` 开头的接口。

您需要查找已启动的 CX-7 接口的 IP 地址。在两个节点上，运行以下命令来查找 IP 地址并记下它们以供下一步使用。
```bash
  ip addr show enp1s0f0np0
  ip addr show enp1s0f1np1
```

输出示例：
```
## In this example, we are using interface enp1s0f1np1.
nvidia@dgx-spark-1:~$ ip addr show enp1s0f1np1
    4: enp1s0f1np1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
        link/ether 3c:6d:66:cc:b3:b7 brd ff:ff:ff:ff:ff:ff
        inet **169.254.35.62**/16 brd 169.254.255.255 scope link noprefixroute enp1s0f1np1
          valid_lft forever preferred_lft forever
        inet6 fe80::3e6d:66ff:fecc:b3b7/64 scope link
          valid_lft forever preferred_lft forever
```

在此示例中，节点 1 的 IP 地址是 **169.254.35.62**。对节点 2 重复该过程。

## 步骤 5. 运行 NCCL 通信测试

> [！笔记] 
> 只需一根 QSFP 电缆即可实现全带宽。
> 当连接两根 QSFP 电缆时，必须为所有四个接口分配 IP 地址以获得完整带宽。

在两个节点上执行以下命令，运行NCCL通信测试。将 IP 地址和接口名称替换为您在上一步中找到的地址和接口名称。

```bash
## Set network interface environment variables (use your Up interface from the previous step)
export UCX_NET_DEVICES=enp1s0f1np1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export OMPI_MCA_btl_tcp_if_include=enp1s0f1np1

## Run the all_gather performance test across both nodes (replace the IP addresses with the ones you found in the previous step)
mpirun -np 2 -H <IP for Node 1>:1,<IP for Node 2>:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  $HOME/nccl-tests/build/all_gather_perf
```

您还可以使用更大的缓冲区大小来测试 NCCL 设置，以使用更多 200Gbps 带宽。

```bash
## Set network interface environment variables (use your active interface)
export UCX_NET_DEVICES=enp1s0f1np1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export OMPI_MCA_btl_tcp_if_include=enp1s0f1np1

## Run the all_gather performance test across both nodes
mpirun -np 2 -H <IP for Node 1>:1,<IP for Node 2>:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  $HOME/nccl-tests/build/all_gather_perf -b 16G -e 16G -f 2
```

注意：`mpirun` 命令中的 IP 地址后跟 `:1`。例如，`mpirun -np 2 -H 169.254.35.62:1,169.254.35.63:1`

## 步骤 7. 清理和回滚

```bash
## Rollback network configuration (if needed)
rm -rf ~/nccl/
rm -rf ~/nccl-tests/
```

## 步骤 8. 后续步骤
您的 NCCL 环境已准备好在 DGX Spark 上进行多节点分布式训练工作负载。
现在您可以尝试运行更大的分布式工作负载，例如 TRT-LLM 或 vLLM 推理。

## 故障排除

## 在两个 Spark 上运行的常见问题

| 问题 | 原因 | 解决方案 |
|-------|-------|----------|
| mpirun 挂起或超时 | SSH 连接问题 | 1. 测试基本 SSH 连接：`ssh <remote_ip>` 应该可以在没有密码提示的情况下工作<br>2。尝试一个简单的 mpirun 测试：`mpirun -np 2 -H <IP for Node 1>:1,<IP for Node 2>:1 hostname`<br>3。验证所有节点的 SSH 密钥设置正确 |
| 未找到网络接口 | 接口名称错误或状态为down | 使用 `ibdev2netdev` 检查接口状态并验证 IP 配置 |
| NCCL 构建失败 | 缺少 OpenMPI 等依赖项或 CUDA 版本不正确 | 验证 CUDA 安装和所需的库是否存在 |
