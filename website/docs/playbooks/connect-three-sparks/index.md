---
id: connect-three-sparks
title: connect-three-sparks
sidebar_label: connect-three-sparks
---

# 以环形拓扑连接三个 DGX Spark

> 以环形拓扑连接并配置三台 DGX Spark 设备

## 目录

- [概述](#overview)
- [在三台 Spark 上进行配置](#run-on-three-sparks)
  - [选项 1：自动配置 SSH](#option-1-automatically-configure-ssh)
  - [选项 2：手动发现并配置 SSH](#option-2-manually-discover-and-configure-ssh)
- [故障排查](#troubleshooting)

---

<a id="overview"></a>
## 概述

## 基本思路

通过 200GbE 直连 QSFP 连接，为三台 DGX Spark 系统配置环形拓扑的高速节点间通信。该配置通过建立网络连通性并配置 SSH 认证，使三台 DGX Spark 节点能够运行分布式工作负载。

## 你将完成的内容

你将使用 QSFP 线缆物理连接三台 DGX Spark 设备，为集群通信配置网络接口，并在节点之间建立免密 SSH，从而搭建一个可用的分布式计算环境。

## 开始前需要了解

- 对分布式计算概念有基本理解
- 了解网络接口配置和 netplan 的使用
- 具备 SSH 密钥管理经验

## 前置条件

- 三台 DGX Spark 系统
- 三根用于设备间环形直连 200GbE 的 QSFP 线缆。请使用[推荐线缆](https://marketplace.nvidia.com/en-us/enterprise/personal-ai-supercomputers/qsfp-cable-0-4m-for-dgx-spark/)或同类产品。
- 所有系统都可通过 SSH 访问
- 所有系统都具备 root 或 sudo 权限：`sudo whoami`
- 所有系统使用相同的用户名
- 将所有系统更新到最新 OS 和 Firmware。请参阅 DGX Spark 文档 https://docs.nvidia.com/dgx/dgx-spark/os-and-component-update.html

## 相关文件

本 playbook 的相关文件可在 [GitHub](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-three-sparks/) 上找到

- [**discover-sparks.sh**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/discover-sparks) 脚本，用于自动发现节点并分发 SSH 密钥
- [**Cluster setup scripts**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/multi-sparks-through-switch/assets/spark_cluster_setup) 脚本，用于自动配置网络、执行验证并运行 NCCL 基础测试

## 时间与风险

- **耗时：** 约 1 小时（包含验证）

- **风险等级：** 中 - 涉及网络重新配置

- **回滚：** 删除 netplan 配置或 IP 分配即可撤销网络更改

- **最后更新：** 3/19/2026
  * 首次发布

<a id="run-on-three-sparks"></a>
## 在三台 Spark 上进行配置

## 第 1 步：确保所有系统使用相同用户名

在所有系统上检查用户名，并确保一致：

```bash
## Check current username
whoami
```

如果用户名不一致，请在所有系统上创建一个新用户（例如 `nvidia`），然后使用新用户登录：

```bash
## Create nvidia user and add to sudo group
sudo useradd -m nvidia
sudo usermod -aG sudo nvidia

## Set password for nvidia user
sudo passwd nvidia

## Switch to nvidia user
su - nvidia
```

## 第 2 步：物理硬件连接

以环形拓扑方式连接三台 DGX Spark 系统之间的 QSFP 线缆。其中，Port0 是紧邻 Ethernet 端口的 CX7 端口，Port1 是距离它更远的 CX7 端口。
1. Node1 (Port0) 连接到 Node2 (Port1)
2. Node2 (Port0) 连接到 Node3 (Port1)
3. Node3 (Port0) 连接到 Node1 (Port1)

> [!NOTE]
> 请再次确认连接是否正确，否则网络配置可能失败。

这会建立高速节点间通信所需的 200GbE 直连链路。三台节点连接完成后，所有节点上都应看到类似下面的输出：在此示例中，显示为 'Up' 的接口是 **enp1s0f0np0** / **enP2p1s0f0np0** 和 **enp1s0f1np1** / **enP2p1s0f1np1**（每个物理端口对应两个逻辑接口）。

示例输出：
```bash
## Check QSFP interface availability on all nodes
nvidia@dgx-spark-1:~$ ibdev2netdev
rocep1s0f0 port 1 ==> enp1s0f0np0 (Up)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Up)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)
```

> [!NOTE] 
> 如果并非所有接口都显示为 'Up'，请检查 QSFP 线缆连接，重启系统后再试。

## 第 3 步：配置网络接口

请选择一种方式来配置网络接口。各选项互斥。建议优先使用选项 1，以降低网络配置复杂度。

> [!NOTE] 
> 每个 CX7 端口都能提供完整的 200GbE 带宽。
> 在三节点环形拓扑中，每个节点上的四个接口都必须分配 IP 地址，才能构成对称集群。

**选项 1：使用脚本自动分配 IP**

我们在 [GitHub](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/multi-sparks-through-switch/assets/spark_cluster_setup) 上提供了一个脚本，用于自动完成以下操作：
1. 为所有 DGX Spark 配置接口网络
2. 在 DGX Spark 之间建立免密认证
3. 验证多节点通信
4. 运行 NCCL 带宽测试

> [!NOTE]
> 如果你使用下面的脚本步骤，则可以跳过本 playbook 中后续的大部分设置说明。

按照下面的步骤运行脚本：

```bash
## Clone the repository
git clone https://github.com/NVIDIA/dgx-spark-playbooks

## Enter the script directory
cd dgx-spark-playbooks/nvidia/multi-sparks-through-switch/assets/spark_cluster_setup

## Check the README.md for steps to run the script and configure the cluster networking
```

**选项 2：使用 netplan 配置文件手动分配 IP**

在节点 1 上：
```bash
## Create the netplan configuration file
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      dhcp4: false
      addresses:
        - 192.168.0.1/24
    enP2p1s0f0np0:
      dhcp4: false
      addresses:
        - 192.168.1.1/24
    enp1s0f1np1:
      dhcp4: false
      addresses:
        - 192.168.2.1/24
    enP2p1s0f1np1:
      dhcp4: false
      addresses:
        - 192.168.3.1/24
EOF

## Set appropriate permissions
sudo chmod 600 /etc/netplan/40-cx7.yaml

## Apply the configuration
sudo netplan apply
```

在节点 2 上：
```bash
## Create the netplan configuration file
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      dhcp4: false
      addresses:
        - 192.168.4.1/24
    enP2p1s0f0np0:
      dhcp4: false
      addresses:
        - 192.168.5.1/24
    enp1s0f1np1:
      dhcp4: false
      addresses:
        - 192.168.0.2/24
    enP2p1s0f1np1:
      dhcp4: false
      addresses:
        - 192.168.1.2/24
EOF

## Set appropriate permissions
sudo chmod 600 /etc/netplan/40-cx7.yaml

## Apply the configuration
sudo netplan apply
```

在节点 3 上：
```bash
## Create the netplan configuration file
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      dhcp4: false
      addresses:
        - 192.168.2.2/24
    enP2p1s0f0np0:
      dhcp4: false
      addresses:
        - 192.168.3.2/24
    enp1s0f1np1:
      dhcp4: false
      addresses:
        - 192.168.4.2/24
    enP2p1s0f1np1:
      dhcp4: false
      addresses:
        - 192.168.5.2/24
EOF

## Set appropriate permissions
sudo chmod 600 /etc/netplan/40-cx7.yaml

## Apply the configuration
sudo netplan apply
```

## 第 4 步：配置免密 SSH 认证

<a id="option-1-automatically-configure-ssh"></a>
### 选项 1：自动配置 SSH

在任意一个节点上运行 DGX Spark 的 [**discover-sparks.sh**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/discover-sparks) 脚本，以自动发现节点并配置 SSH：

```bash
curl -O https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/connect-two-sparks/assets/discover-sparks
bash ./discover-sparks
```

预期输出类似如下，但 IP 和节点名称会有所不同。由于四个接口（**enp1s0f0np0**、**enP2p1s0f0np0**、**enp1s0f1np1** 和 **enP2p1s0f1np1**）都分配了 IP，因此你可能会在每个节点上看到不止一个 IP。这是正常现象，不会造成问题。首次运行该脚本时，系统会提示你为每个节点输入密码。
```
Found: 192.168.0.1 (dgx-spark-1.local)
Found: 192.168.0.2 (dgx-spark-2.local)
Found: 192.168.3.2 (dgx-spark-3.local)

Setting up bidirectional SSH access (local <-> remote nodes)...
You may be prompted for your password for each node.

SSH setup complete! All nodes can now SSH to each other without passwords.
```

> [!NOTE]
> 如果遇到任何错误，请按照下面的选项 2 手动配置 SSH，并借此排查问题。

<a id="option-2-manually-discover-and-configure-ssh"></a>
### 选项 2：手动发现并配置 SSH

你需要找到处于 up 状态的 CX-7 接口 IP 地址。在所有节点上运行以下命令，找到对应 IP，并记下来供下一步使用。
```bash
  ip addr show enp1s0f0np0
  ip addr show enp1s0f1np1
```

示例输出：
```
## In this example, we are using interface enp1s0f1np1.
nvidia@dgx-spark-1:~$ ip addr show enp1s0f1np1
    4: enp1s0f1np1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
        link/ether 3c:6d:66:cc:b3:b7 brd ff:ff:ff:ff:ff:ff
        inet **192.168.1.1**/24 brd 192.168.1.255 scope link noprefixroute enp1s0f1np1
          valid_lft forever preferred_lft forever
        inet6 fe80::3e6d:66ff:fecc:b3b7/64 scope link
          valid_lft forever preferred_lft forever
```

在这个示例中，节点 1 的 IP 地址是 **192.168.1.1**。请对其他节点重复此过程。

在所有节点上，运行以下命令启用免密 SSH：
```bash
## Copy your SSH public key to all nodes. Please replace the IP addresses with the ones you found in the previous step.
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<IP for Node 1>
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<IP for Node 2>
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<IP for Node 3>
```

## 第 5 步：验证多节点通信

测试基本的多节点功能：

```bash
## Test hostname resolution across nodes
ssh <IP for Node 1> hostname
ssh <IP for Node 2> hostname
ssh <IP for Node 3> hostname
```

## 第 6 步：运行 NCCL 测试

现在你的集群已经可以在三台节点上运行分布式工作负载了。接下来尝试运行 NCCL 带宽测试。

按照下面的步骤运行脚本，该脚本会在集群上执行 NCCL 测试：

```bash
## Clone the repository
git clone https://github.com/NVIDIA/dgx-spark-playbooks

## Enter the script directory
cd dgx-spark-playbooks/nvidia/multi-sparks-through-switch/assets/spark_cluster_setup

## Check the README.md in the script directory for steps to run the NCCL tests with "--run-nccl-test" option
```

## 第 7 步：清理与回滚

> [!WARNING]
> 以下步骤会重置网络配置。

```bash
## Rollback network configuration
sudo rm /etc/netplan/40-cx7.yaml
sudo netplan apply
```

<a id="troubleshooting"></a>
## 故障排查

| 现象 | 原因 | 解决方法 |
|---------|-------|-----|
| 出现 "Network unreachable" 错误 | 网络接口未配置 | 检查 netplan 配置并执行 `sudo netplan apply` |
| SSH 认证失败 | SSH 密钥未正确分发 | 重新运行 `./discover-sparks` 并输入密码 |
| 集群中看不到节点 | 网络连接问题 | 检查 QSFP 线缆连接并核对 IP 配置 |
| 出现 "APT update" 错误（例如 E: The list of sources could not be read.） | APT 源、冲突源或签名密钥存在问题 | 查阅 APT 和 Ubuntu 文档，修复 APT 源或密钥冲突 |
