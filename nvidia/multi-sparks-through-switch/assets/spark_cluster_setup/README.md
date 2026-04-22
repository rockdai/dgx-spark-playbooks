# 多 Spark 集群设置脚本

## 用法

### 步骤 1. 克隆存储库

从 GitHub 克隆 dgx-spark-playbooks 存储库

### 步骤 2. 切换到 multi Spark 集群设置脚本目录

```bash
cd dgx-spark-playbooks/nvidia/multi-sparks-through-switch/assets/spark_cluster_setup
```

### 步骤 3. 使用集群信息创建或编辑 JSON 配置文件

```bash
# Create or edit JSON config file under the `config` directory with the ssh credentials for your nodes.
# Adjust the number of nodes in "nodes_info" list based on the number of nodes in your cluster

# Example: (config/spark_config_b2b.json):
# {
#     "nodes_info": [
#         {
#             "ip_address": "10.0.0.1",
#             "port": 22,
#             "user": "nvidia",
#             "password": "nvidia123"
#         },
#         {
#             "ip_address": "10.0.0.2",
#             "port": 22,
#             "user": "nvidia",
#             "password": "nvidia123"
#         }
#
```

### 步骤 4. 使用 json 配置文件运行集群设置脚本

该脚本可以使用不同的选项运行，如下所述

```bash
# To run validation, cluster setup and NCCL bandwidth test (all steps)

bash spark_cluster_setup.sh -c <JSON config file> --run-setup

# To only run pre-setup validation steps

bash spark_cluster_setup.sh -c <JSON config file> --pre-validate-only

# To run NCCL test and skip cluster setup (use this after cluster is already set up)

bash spark_cluster_setup.sh -c <JSON config file> --run-nccl-test

```

> [!NOTE]
> 完整的集群设置（上面的第一个命令）将执行以下操作
> 1. 创建一个python虚拟环境并安装所需的包
> 2. 验证环境和集群配置
> 3. 检测拓扑并配置IP地址
> 4. 在集群节点之间配置无密码 ssh
> 5. 运行 NCCL BW 测试
