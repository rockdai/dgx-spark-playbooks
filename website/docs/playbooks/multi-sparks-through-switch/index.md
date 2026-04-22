---
id: multi-sparks-through-switch
title: multi-sparks-through-switch
sidebar_label: multi-sparks-through-switch
---

# 通过交换机连接多个 DGX Spark

> 设置通过 Switch 连接的 DGX Spark 设备集群

## 目录

- [Overview](#overview)
- [Run on Four Sparks](#run-on-four-sparks)
  - [Step 3.1. Verify negotiated Link speed](#step-31-verify-negotiated-link-speed)
  - [4.1 Script for Cluster networking configuration](#41-script-for-cluster-networking-configuration)
  - [4.2 Manual Cluster networking configuration](#42-manual-cluster-networking-configuration)
  - [Option 1: Automatically configure SSH](#option-1-automatically-configure-ssh)
  - [Option 2: Manually discover and configure SSH](#option-2-manually-discover-and-configure-ssh)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思路

配置四个 DGX Spark 系统，通过 QSFP 交换机使用 200Gbps QSFP 连接进行高速节点间通信。此设置通过建立网络连接和配置 SSH 身份验证来实现跨多个 DGX Spark 节点的分布式工作负载。

## 你将完成什么

在此手册中，您将使用 QSFP 电缆和 QSFP 交换机物理连接四个 DGX Spark 设备，配置用于集群通信的网络接口，并在节点之间建立无密码 SSH，以创建功能齐全的分布式计算环境。相同的设置可以扩展到通过同一交换机连接的更多 DGX Spark 设备。

## 开始之前需要了解什么

- 对分布式计算概念的基本了解
- 使用网络接口配置和网络规划
- 具有 SSH 密钥管理经验
- 对配置您计划使用的托管 QSFP 网络交换机有基本的了解和经验。请参阅使用说明书：
  - 了解如何连接到交换机以管理端口和功能
  - 了解如何启用/禁用 QSFP 端口并在交换机上创建软件桥
  - 了解如何在端口上手动配置链路速度并在需要时禁用自动协商

## 先决条件

- 四个 DGX Spark 系统（这些说明适用于与交换机连接的任意数量的 DGX Spark 设备）
- 具有至少 4 个 QSFP56-DD 端口（每个端口至少 200Gbps）的 QSFP 交换机
- QSFP 电缆用于从交换机到设备的 200Gbps 连接。使用 [recommended cable](https://marketplace.nvidia.com/en-us/enterprise/personal-ai-supercomputers/qsfp-cable-0-4m-for-dgx-spark/) 或类似的。
  - 每个火花一根电缆
  - 如果交换机有 400Gbps 端口，那么您还可以使用分支电缆将它们分成两个 200Gbps 端口
- 所有系统均可使用 SSH 访问
- 所有系统上的 root 或 sudo 访问权限：`sudo whoami`
- 所有系统上的用户名相同
- 将所有系统更新到最新的操作系统和固件。请参阅 DGX Spark 文档 https://docs.nvidia.com/dgx/dgx-spark/os-and-component-update.html

## 附属文件

此剧本所需的所有文件都可以在 [here on GitHub](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/multi-sparks-through-switch/) 中找到

- 用于自动节点发现和 SSH 密钥分发的 [**discover-sparks.sh**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/discover-sparks) 脚本
- [**Cluster setup script**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/multi-sparks-through-switch/assets/spark_cluster_setup) 用于自动网络配置、验证和运行 NCCL 健全性测试

## 时间与风险

- **持续时间：** 2 小时（包括验证）

- **风险级别：** 中 - 涉及网络重新配置

- **回滚：** 可以通过删除网络规划配置或 IP 分配来撤销网络更改

- **最后更新：** 2026 年 3 月 19 日
  * 首次发表

## 靠四个火花奔跑

## 步骤 1. 确保所有四个系统上的用户名相同

在所有四个系统上检查并确保用户名相同：

```bash
## Check current username
whoami
```

如果用户名不匹配，请在所有四个系统上创建一个新用户（例如 nvidia）并使用新用户登录：

```bash
## Create nvidia user and add to sudo group
sudo useradd -m nvidia
sudo usermod -aG sudo nvidia

## Set password for nvidia user
sudo passwd nvidia

## Switch to nvidia user
su - nvidia
```

## 步骤 2. 交换机管理

大多数 QSFP 交换机通过 CLI 或 UI 提供某种形式的管理界面。参考文档并连接到管理界面。确保交换机上的端口已启用。要连接四个 Spark，您需要确保交换机配置为向每个 DGX Spark 提供 200Gbps 连接。如果尚未完成，请参阅本手册的 [Overview](https://build.nvidia.com/spark/multi-sparks-through-switch/overview)，了解本手册所需的先验知识和先决条件。

## 步骤 3. 物理硬件连接

使用每个 Spark 系统上的一个 CX7 端口连接 DGX Spark 系统和交换机（QSFP56-DD/QSFP56 端口）之间的 QSFP 电缆。建议在所有 Spark 系统上使用相同的 CX7 端口，以便更轻松地进行网络配置并避免 NCCL 测试失败。在此剧本中，使用第二个端口（距离以太网端口较远的一个）。这应该建立高速节点间通信所需的 200Gbps 连接。您将在所有四个火花上看到如下所示的输出。在此示例中，显示为“Up”的接口是 **enp1s0f1np1** 和 **enP2p1s0f1np1**（每个物理端口有两个逻辑接口）。

输出示例：
```bash
## Check QSFP interface availability on all nodes
nvidia@dxg-spark-1:~$ ibdev2netdev
rocep1s0f0 port 1 ==> enp1s0f0np0 (Down)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Down)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)
```

> [！笔记]
> 如果没有接口显示为“Up”，请检查 QSFP 电缆连接，重新启动系统并重试。
> 显示为“Up”的接口取决于您用于将节点连接到交换机的端口。每个物理端口有两个逻辑接口，例如，端口 1 有两个接口 - enp1s0f1np1 和 enP2p1s0f1np1。请忽略 enp1s0f0np0 和 enP2p1s0f0np0，仅使用 enp1s0f1np1 和 enP2p1s0f1np1。

### 步骤 3.1。验证协商的链接速度

自动协商的链路速度可能不会默认为 200Gbps。要进行确认，请在所有 Spark 上运行以下命令并检查速度是否显示为 200000Mb/s。如果显示小于该值，则需要在交换机端口配置中手动将链路速度设置为 200Gbps，并且应禁用自动协商。请参阅交换机的手册/文档以禁用自动协商并将链路速度手动设置为 200Gbps（例如 200G-baseCR4）

输出示例：
```bash
nvidia@dxg-spark-1:~$ sudo ethtool enp1s0f1np1 | grep Speed
	Speed: 100000Mb/s

nvidia@dxg-spark-1:~$ sudo ethtool enP2p1s0f1np1 | grep Speed
	Speed: 100000Mb/s
```

在交换机端口上设置正确的速度后。再次验证所有 DGX Spark 上的链接速度。

输出示例：
```bash
nvidia@dxg-spark-1:~$ sudo ethtool enp1s0f1np1 | grep Speed
	Speed: 200000Mb/s

nvidia@dxg-spark-1:~$ sudo ethtool enP2p1s0f1np1 | grep Speed
	Speed: 200000Mb/s
```

## 步骤 4. 网络接口配置

> [！笔记]
> 只需一根 QSFP 电缆即可实现全带宽。

对于集群设置，所有 DGX 都会触发：
1. 应该可供管理访问（例如 SSH 和运行命令）
2. 应该能够访问互联网（例如下载模型/实用程序）
3. 应该能够使用 TCP/IP over CX7 相互通信。以下步骤可帮助进行配置。

建议使用以太网/WiFi 网络进行管理和互联网流量，并将其与 CX7 网络分开，以避免 CX7 带宽用于非工作负载流量。

使用交换机配置集群的受支持方法需要在交换机上配置网桥（或使用默认网桥），并通过交换机管理界面将所有感兴趣的端口（连接到 DGX Spark 的端口）添加到其中。
1. 这样，所有端口都是集群网络配置所需的单个第 2 层域的一部分
2. 某些交换机有限制，只能在一个网桥上启用硬件卸载，因此需要将所有端口保留在单个网桥中

完成创建/添加桥接端口后，您应该准备好在 DGX Spark 端配置网络。

### 4.1 集群网络配置脚本

我们创建了一个脚本 [here on GitHub](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/multi-sparks-through-switch/assets/spark_cluster_setup) ，它可以自动执行以下操作：
1. 所有 DGX Spark 的接口网络 IP 配置
2. 在 DGX Sparks 之间设置无密码身份验证
3. 验证多节点通信
4. 运行 NCCL 带宽测试

> [！笔记]
> 您可以使用该脚本或继续以下部分中的手动配置。如果您使用该脚本，则可以跳过本手册中的其余设置部分。

使用以下步骤运行脚本：

```bash
## Clone the repository
git clone https://github.com/NVIDIA/dgx-spark-playbooks

## Enter the script directory
cd dgx-spark-playbooks/nvidia/multi-sparks-through-switch/assets/spark_cluster_setup

## Check the README.md in the script directory for steps to run the script and configure the cluster networking with "--run-setup" argument
```

### 4.2 手动集群网络配置

在这种情况下，您可以选择其中一个选项来将 IP 分配给 CX7 逻辑接口。选项 1、2 和 3 是互斥的。
1. 交换机上的 DHCP 服务器（推荐，如果支持）
2. 链接本地 IP 寻址（所有节点的网络规划都相同）
3. 手动 IP 寻址（每个节点上的网络规划都不同，但提供更多控制和确定性 IP）

#### 选项 1：在交换机上配置 DHCP 服务器

1. 在交换机上配置 DHCP 服务器，其子网足够大，以便为所有 Spark 分配 IP。 /24 子网应该可以很好地进行配置和任何未来的扩展。
2. 配置 DGX Spark 中的“UP”CX7 接口以使用 DHCP 获取 IP。例如。如果逻辑接口 **enp1s0f1np1** / **enP2p1s0f1np1** 为“UP”，则在所有 Spark 上创建如下所示的网络规划。

```bash
## Create the netplan configuration file
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f1np1:
      dhcp4: true
    enP2p1s0f1np1:
      dhcp4: true
EOF

## Set appropriate permissions
sudo chmod 600 /etc/netplan/40-cx7.yaml

## Apply the configuration
sudo netplan apply
```

3. 确认接口已分配 IP

```bash
## In this example, we are using interface enp1s0f1np1. Similarly check enP2p1s0f1np1.
nvidia@dgx-spark-1:~$ ip addr show enp1s0f1np1 | grep -w inet
    inet 100.100.100.4/24 brd 100.100.100.255 scope global noprefixroute enp1s0f1np1
```

#### 选项 2：自动链接本地 IP 分配

在所有 DGX Spark 节点上使用 netplan 配置网络接口，以实现自动链路本地寻址：

```bash
## Create the netplan configuration file
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f1np1:
      link-local: [ ipv4 ]
    enP2p1s0f1np1:
      link-local: [ ipv4 ]
EOF

## Set appropriate permissions
sudo chmod 600 /etc/netplan/40-cx7.yaml

## Apply the configuration
sudo netplan apply
```

#### 选项 3：使用 netplan 配置文件手动分配 IP

在节点 1 上：
```bash
## Create the netplan configuration file
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f1np1:
      addresses:
        - 192.168.100.10/24
      dhcp4: no
    enP2p1s0f1np1:
      addresses:
        - 192.168.100.11/24
      dhcp4: no
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
    enp1s0f1np1:
      addresses:
        - 192.168.100.12/24
      dhcp4: no
    enP2p1s0f1np1:
      addresses:
        - 192.168.100.13/24
      dhcp4: no
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
    enp1s0f1np1:
      addresses:
        - 192.168.100.14/24
      dhcp4: no
    enP2p1s0f1np1:
      addresses:
        - 192.168.100.15/24
      dhcp4: no
EOF

## Set appropriate permissions
sudo chmod 600 /etc/netplan/40-cx7.yaml

## Apply the configuration
sudo netplan apply
```

在节点 4 上：
```bash
## Create the netplan configuration file
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f1np1:
      addresses:
        - 192.168.100.16/24
      dhcp4: no
    enP2p1s0f1np1:
      addresses:
        - 192.168.100.17/24
      dhcp4: no
EOF

## Set appropriate permissions
sudo chmod 600 /etc/netplan/40-cx7.yaml

## Apply the configuration
sudo netplan apply
```

## 步骤 5. 设置无密码 SSH 身份验证

### 选项 1：自动配置 SSH

从节点之一运行 DGX Spark [**discover-sparks.sh**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/discover-sparks) 脚本以自动发现和配置 SSH：

```bash
curl -O https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/connect-two-sparks/assets/discover-sparks
bash ./discover-sparks
```

预期输出类似于以下内容，具有不同的 IP 和节点名称。您可能会看到每个节点最多有两个 IP，因为两个接口（例如 **enp1s0f1np1** 和 **enP2p1s0f1np1**）已分配 IP 地址。这是预期的结果，不会导致任何问题。第一次运行该脚本时，系统会提示您输入每个节点的密码。
```
Found: 169.254.35.62 (dgx-spark-1.local)
Found: 169.254.35.63 (dgx-spark-2.local)
Found: 169.254.35.64 (dgx-spark-3.local)
Found: 169.254.35.65 (dgx-spark-4.local)

Setting up bidirectional SSH access (local <-> remote nodes)...
You may be prompted for your password for each node.

SSH setup complete! All local and remote nodes can now SSH to each other without passwords.
```

> [！笔记]
> 如果遇到任何错误，请按照下面的选项 2 手动配置 SSH 并调试问题。

### 选项 2：手动发现和配置 SSH

您需要查找已启动的 CX-7 接口的 IP 地址。在所有节点上，运行以下命令来查找 IP 地址并记下它们以供下一步使用。
```bash
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

在此示例中，节点 1 的 IP 地址是 **169.254.35.62**。对其他节点重复该过程。

在所有节点上运行以下命令启用无密码 SSH：
```bash
## Copy your SSH public key to all nodes. Replace the IP addresses with the ones you found in the previous step.
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<IP for Node 1>
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<IP for Node 2>
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<IP for Node 3>
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<IP for Node 4>
```

## 步骤 6. 验证多节点通信

从头节点测试基本的多节点功能：

```bash
## Test hostname resolution across nodes
ssh <IP for Node 1> hostname
ssh <IP for Node 2> hostname
ssh <IP for Node 3> hostname
ssh <IP for Node 4> hostname
```

## 步骤 7. 运行测试和工作负载

现在，您的集群已设置为跨四个节点运行分布式工作负载。尝试运行 [NCCL playbook](https://build.nvidia.com/spark/nccl/stacked-sparks)。

> [！笔记]
> 无论剧本要求在**两个节点**上运行命令，只需在**所有四个节点**上运行它即可。
> 确保调整您在 **头节点** 上运行的 *mpirun* NCCL 命令以容纳 **四个节点**

NCCL 的 mpirun 命令示例：
```bash
## Set network interface environment variables (use your Up interface from the previous step)
export UCX_NET_DEVICES=enp1s0f1np1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export OMPI_MCA_btl_tcp_if_include=enp1s0f1np1

## Run the all_gather performance test across four nodes (replace the IP addresses with the ones you found in the previous step)
mpirun -np 4 -H <IP for Node 1>:1,<IP for Node 2>:1,<IP for Node 3>:1,<IP for Node 4>:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  $HOME/nccl-tests/build/all_gather_perf
```

## 步骤 8. 清理和回滚

> [！警告]
> 这些步骤将重置网络配置。

```bash
## Rollback network configuration
sudo rm /etc/netplan/40-cx7.yaml
sudo netplan apply
```

> [！笔记]
> 如果断开开关，请确保执行以下操作
> 1. 如果交换机用于不同目的，请重新启用自动协商以避免以后出现问题。
> 2. 如果您使用交换机上的 DHCP 服务器配置为 Sparks 分配 IP，请删除该配置。
> 3. 如果您创建了新网桥，请将端口移回默认网桥并删除新网桥。

## 故障排除

| 症状 | 原因 | 使固定 |
|---------|-------|-----|
| “网络无法访问”错误 | 网络接口未配置 | 验证 netplan 配置和 `sudo netplan apply` |
| SSH 身份验证失败 | SSH 密钥未正确分配 | 重新运行 `./discover-sparks` 并输入密码 |
| 集群中节点不可见 | 网络连接问题 | 验证 QSFP 电缆连接，检查 IP 配置 |
| “APT 更新”错误（例如 E：无法读取源列表。） | APT 源错误、源冲突或签名密钥 | 检查 APT 和 Ubuntu 文档以修复 APT 源或密钥冲突 |
| NCCL 测试失败（例如 libnccl.so.2：无法打开共享对象文件） | 未在所有节点上完成 NCCL 配置 | 在运行 NCCL 测试之前，请确保按照 NCCL playbook 配置**所有**节点|
