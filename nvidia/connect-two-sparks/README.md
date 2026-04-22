# 连接两个火花

> 连接两台 Spark 设备，并将其配置为可进行推理和微调

## 目录

- [概述](#overview)
- [在两台 Spark 上进行配置](#run-on-two-sparks)
- [故障排查](#troubleshooting)

---

<a id="overview"></a>
## 概述

## 基本思路

通过 200GbE 直连 QSFP 连接，为两台 DGX Spark 系统配置高速节点间通信。该配置通过建立网络连通性并配置 SSH 认证，使多台 DGX Spark 节点能够运行分布式工作负载。

## 你将完成的内容

你将使用一根 QSFP 线缆物理连接两台 DGX Spark 设备，为集群通信配置网络接口，并在节点之间建立免密 SSH，从而搭建一个可用的分布式计算环境。

## 开始前需要了解

- 对分布式计算概念有基本理解
- 了解网络接口配置和 netplan 的使用
- 具备 SSH 密钥管理经验

## 前置条件

- 两台 DGX Spark 系统
- 一根用于两台设备间直连 200GbE 的 QSFP 线缆
- 两台系统都可通过 SSH 访问
- 两台系统都具备 root 或 sudo 权限：`sudo whoami`
- 两台系统使用相同的用户名

## 相关文件

本 playbook 所需的所有文件都可在 [GitHub](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/) 上找到

- [**discover-sparks.sh**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/discover-sparks) 脚本，用于自动发现节点并分发 SSH 密钥

## 时间与风险

- **耗时：** 约 1 小时（包含验证）

- **风险等级：** 中 - 涉及网络重新配置

- **回滚：** 删除 netplan 配置或 IP 分配即可撤销网络更改

- **最后更新：** 11/24/2025
  * 文案小幅修订

<a id="run-on-two-sparks"></a>
## 在两台 Spark 上进行配置

## 第 1 步：确保两台系统使用相同用户名

在两台系统上检查用户名，并确保一致：

```bash
## Check current username
whoami
```

如果用户名不一致，请在两台系统上创建一个新用户（例如 `nvidia`），然后使用新用户登录：

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

在两台 DGX Spark 系统之间使用任意 QSFP 接口连接 QSFP 线缆。这会建立高速节点间通信所需的 200GbE 直连链路。两台节点连接完成后，你会看到类似下面的输出：在此示例中，显示为 'Up' 的接口是 **enp1s0f1np1** / **enP2p1s0f1np1**（每个物理端口都有两个名称）。

示例输出：
```bash
## Check QSFP interface availability on both nodes
nvidia@dxg-spark-1:~$ ibdev2netdev
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Down)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)
rocep1s0f0 port 1 ==> enp1s0f0np0 (Down)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)
```

> [!NOTE] 
> 如果所有接口都没有显示为 'Up'，请检查 QSFP 线缆连接，重启系统后再试。
> 哪个接口显示为 'Up' 取决于你使用哪一个端口连接两台节点。每个物理端口有两个名称，例如 enp1s0f1np1 和 enP2p1s0f1np1 指的是同一个物理端口。请忽略 enP2p1s0f0np0 和 enP2p1s0f1np1，只使用 enp1s0f0np0 和 enp1s0f1np1。

## 第 3 步：配置网络接口

请选择一种方式来配置网络接口。选项 1 和选项 2 互斥。

> [!NOTE] 
> 只使用一根 QSFP 线缆也可以达到满带宽。
> 当连接两根 QSFP 线缆时，必须为四个接口都分配 IP 地址，才能获得满带宽。
> 下方的选项 1 仅适用于连接了 1 根 QSFP 线缆的情况。

**选项 1：自动分配 IP（仅在连接 1 根 QSFP 线缆时可用）**

在两台 DGX Spark 节点上使用 netplan 配置网络接口，以实现自动链路本地地址分配：

```bash
## Create the netplan configuration file
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      link-local: [ ipv4 ]
    enp1s0f1np1:
      link-local: [ ipv4 ]
EOF

## Set appropriate permissions
sudo chmod 600 /etc/netplan/40-cx7.yaml

## Apply the configuration
sudo netplan apply
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
      addresses:
        - 192.168.100.10/24
      dhcp4: no
    enp1s0f1np1:
      addresses:
        - 192.168.200.12/24
      dhcp4: no
    enP2p1s0f0np0:
      addresses:
        - 192.168.100.14/24
      dhcp4: no
    enP2p1s0f1np1:
      addresses:
        - 192.168.200.16/24
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
    enp1s0f0np0:
      addresses:
        - 192.168.100.11/24
      dhcp4: no
    enp1s0f1np1:
      addresses:
        - 192.168.200.13/24
      dhcp4: no
    enP2p1s0f0np0:
      addresses:
        - 192.168.100.15/24
      dhcp4: no
    enP2p1s0f1np1:
      addresses:
        - 192.168.200.17/24
      dhcp4: no
EOF

## Set appropriate permissions
sudo chmod 600 /etc/netplan/40-cx7.yaml

## Apply the configuration
sudo netplan apply
```


**选项 3：通过命令行手动分配 IP**

> [!NOTE]
> 使用此选项时，接口上的 IP 分配会在系统重启后丢失。

首先，确认哪些网络端口可用并处于 up 状态：

```bash
## Check network port status
ibdev2netdev
```

示例输出：
```
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Down)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)
rocep1s0f0 port 1 ==> enp1s0f0np0 (Down)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)
```

请使用输出中显示为 "(Up)" 的接口。在本示例中，我们使用 **enp1s0f1np1**。你可以忽略以 `enP2p<...>` 为前缀的接口，只使用以 `enp1<...>` 为前缀的接口。

在节点 1 上：
```bash
## Assign static IP and bring up interface.
sudo ip addr add 192.168.100.10/24 dev enp1s0f1np1
sudo ip link set enp1s0f1np1 up
```

在节点 2 上重复相同步骤，但使用 IP **192.168.100.11/24**。请务必使用 `ibdev2netdev` 命令确认正确的接口名。
```bash
## Assign static IP and bring up interface.
sudo ip addr add 192.168.100.11/24 dev enp1s0f1np1
sudo ip link set enp1s0f1np1 up
```

你可以在两个节点上分别运行以下命令，以验证 IP 分配：
```bash
## Replace enp1s0f1np1 with the interface showing as "(Up)" in your output, either enp1s0f0np0 or enp1s0f1np1
ip addr show enp1s0f1np1
```

## 第 4 步：配置免密 SSH 认证

#### 选项 1：自动配置 SSH

在其中一个节点上运行 DGX Spark 的 [**discover-sparks.sh**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/discover-sparks) 脚本，以自动发现节点并配置 SSH：

```bash
bash ./discover-sparks
```

预期输出会类似如下，但 IP 和节点名称会不同。首次运行该脚本时，系统会提示你为每个节点输入密码。
```
Found: 169.254.35.62 (dgx-spark-1.local)
Found: 169.254.35.63 (dgx-spark-2.local)

Setting up bidirectional SSH access (local <-> remote nodes)...
You may be prompted for your password for each node.

SSH setup complete! Both local and remote nodes can now SSH to each other without passwords.
```

> [!NOTE]
> 如果遇到任何错误，请按照下面的选项 2 手动配置 SSH，并借此排查问题。

#### 选项 2：手动发现并配置 SSH

你需要找到处于 up 状态的 CX-7 接口 IP 地址。在两个节点上运行以下命令，找到对应 IP，并记下来供下一步使用。
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
        inet **169.254.35.62**/16 brd 169.254.255.255 scope link noprefixroute enp1s0f1np1
          valid_lft forever preferred_lft forever
        inet6 fe80::3e6d:66ff:fecc:b3b7/64 scope link
          valid_lft forever preferred_lft forever
```

在这个示例中，节点 1 的 IP 地址是 **169.254.35.62**。对节点 2 重复相同步骤。

在两个节点上，运行以下命令启用免密 SSH：
```bash
## Copy your SSH public key to both nodes. Please replace the IP addresses with the ones you found in the previous step.
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<IP for Node 1>
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<IP for Node 2>
```

## 第 5 步：验证多节点通信

测试基本的多节点功能：

```bash
## Test hostname resolution across nodes
ssh <IP for Node 1> hostname
ssh <IP for Node 2> hostname
```

## 第 6 步：清理与回滚

> [!WARNING]
> 以下步骤会重置网络配置。

```bash
## Rollback network configuration (if using Option 1)
sudo rm /etc/netplan/40-cx7.yaml
sudo netplan apply

## Rollback network configuration (if using Option 2)
sudo ip addr del 192.168.100.10/24 dev enp1s0f0np0  # Adjust the interface name to the one you used in step 3.
sudo ip addr del 192.168.100.11/24 dev enp1s0f0np0  # Adjust the interface name to the one you used in step 3.
```

<a id="troubleshooting"></a>
## 故障排查

| 现象 | 原因 | 解决方法 |
|---------|-------|-----|
| 出现 "Network unreachable" 错误 | 网络接口未配置 | 检查 netplan 配置并执行 `sudo netplan apply` |
| SSH 认证失败 | SSH 密钥未正确分发 | 重新运行 `./discover-sparks` 并输入密码 |
| 集群中看不到节点 2 | 网络连接问题 | 检查 QSFP 线缆连接并核对 IP 配置 |
