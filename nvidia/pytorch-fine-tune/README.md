# 使用 Pytorch 进行微调

> 使用 Pytorch 在本地微调模型

## 目录

- [Overview](#overview)
- [Instructions](#instructions)
- [Run on two Sparks](#run-on-two-sparks)
  - [Step 1. Configure network connectivity](#step-1-configure-network-connectivity)
  - [Step 2. Configure Docker permissions](#step-2-configure-docker-permissions)
  - [Step 3. Install NVIDIA Container Toolkit & setup Docker environment](#step-3-install-nvidia-container-toolkit-setup-docker-environment)
  - [Step 4. Enable resource advertising](#step-4-enable-resource-advertising)
  - [Step 5. Initialize Docker Swarm](#step-5-initialize-docker-swarm)
  - [Step 6. Join worker nodes and deploy](#step-6-join-worker-nodes-and-deploy)
  - [Step 7. Find your Docker container ID](#step-7-find-your-docker-container-id)
  - [Step 9. Adapt the configuration files](#step-9-adapt-the-configuration-files)
  - [Step 10. Run finetuning scripts](#step-10-run-finetuning-scripts)
  - [Step 14. Cleanup and rollback](#step-14-cleanup-and-rollback)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思路

本手册将Instruct您设置和使用 Pytorch 在 NVIDIA Spark 设备上微调大型语言模型。

## 你将完成什么

您将在 NVIDIA Spark 设备上为大型语言模型（1-70B 参数）建立完整的微调环境。 
最后，您将拥有一个支持参数高效微调 (PEFT) 和监督微调 (SFT) 的有效安装。

## 开始之前需要了解什么

- 具有 Pytorch 微调经验
- 使用 Docker



## 先决条件
配方专门针对 DGX SPARK。请确保操作系统和驱动程序是最新的。


## 附属文件

微调所需的所有文件都包含在 [the GitHub repository here](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/pytorch-fine-tune) 文件夹中。

## 时间与风险

* **时间估计：** 30-45 分钟用于设置和运行微调。微调运行时间根据模型大小而变化 
* **风险：** 模型下载可能很大（数 GB），ARM64 软件包兼容性问题可能需要进行故障排除。
* **最后更新：** 2025 年 1 月 15 日
  * 添加两个Spark分布式finetuning示例
  * 添加详细说明以在 Llama3 3B、8B 和 70B 型号上运行完整的 SFT、LoRA 和 qLoRA 工作流程。

## 指示

## 步骤1.配置Docker权限

要在不使用 sudo 的情况下轻松管理容器，您必须位于 `docker` 组中。如果您选择跳过此步骤，则需要使用 sudo 运行 Docker 命令。

打开新终端并测试 Docker 访问。在终端中，运行：

```bash
docker ps
```

如果您看到权限被拒绝错误（例如尝试连接到 Docker 守护进程套接字时权限被拒绝），请将您的用户添加到 docker 组，这样您就不需要使用 sudo 运行命令。

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## 步骤 2. 拉取最新的 Pytorch 容器

```bash
docker pull nvcr.io/nvidia/pytorch:25.11-py3
```

## 步骤 3. 启动 Docker

```bash
docker run --gpus all -it --rm --ipc=host \
-v $HOME/.cache/huggingface:/root/.cache/huggingface \
-v ${PWD}:/workspace -w /workspace \
nvcr.io/nvidia/pytorch:25.11-py3
```

## 步骤 4. 在容器内安装依赖项

```bash
pip install transformers peft datasets trl bitsandbytes
```

## 第 5 步：使用 Huggingface 进行身份验证

```bash
hf auth login
##<input your huggingface token.
##<Enter n for git credential>
```

## 第 6 步：克隆 git 存储库并微调配方

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/pytorch-fine-tune/assets
```

## Step7：运行微调配方

#### 可用的微调脚本

提供了以下微调脚本，每个脚本针对不同的模型大小和训练方法进行了优化：

| 脚本 | 模型 | 微调型 | 描述 |
|--------|-------|------------------|-------------|
| `Llama3_3B_full_finetuning.py` | 羊驼 3.2 3B | 完整的SFT | 完全监督微调（所有参数均可训练） |
| `Llama3_8B_LoRA_finetuning.py` | 羊驼3.1 8B | 洛拉 | 低阶适应（参数有效） |
| `Llama3_70B_LoRA_finetuning.py` | 羊驼3.1 70B | 洛拉 | 具有 FSDP 支持的低秩适应 |
| `Llama3_70B_qLoRA_finetuning.py` | 羊驼3.1 70B | QLoRA | 量化 LoRA（4 位量化以提高内存效率） |

#### 基本用法

使用默认设置运行任何脚本：

```bash
## Full fine-tuning on Llama 3.2 3B
python Llama3_3B_full_finetuning.py

## LoRA fine-tuning on Llama 3.1 8B
python Llama3_8B_LoRA_finetuning.py

## qLoRA fine-tuning on Llama 3.1 70B
python Llama3_70B_qLoRA_finetuning.py
```

#### 常见命令行参数

所有脚本都支持以下命令行参数进行自定义：

##### 型号配置
- `--model_name`：模型名称或路径（默认值：因脚本而异）
- `--dtype`：模型精度 - `float32`、`float16` 或 `bfloat16`（默认值：`bfloat16`）

##### 训练配置
- `--batch_size`：每设备训练批量大小（默认值：因脚本而异）
- `--seq_length`：最大序列长度（默认值：`2048`）
- `--num_epochs`：训练纪元数（默认值：`1`）
- `--gradient_accumulation_steps`：梯度累积步长（默认：`1`）
- `--learning_rate`：学习率（默认值：因脚本而异）
- `--gradient_checkpointing`：启用梯度检查点以节省内存（标志）

##### LoRA 配置（仅限 LoRA 和 QLoRA 脚本）
- `--lora_rank`：LoRA 等级 - 较高的值 = 更多可训练参数（默认值：`8`）

##### 数据集配置
- `--dataset_size`：羊驼数据集中使用的样本数（默认值：`512`）

##### 日志记录配置
- `--logging_steps`：每 N 步记录一次指标（默认值：`1`）
- `--log_dir`：TensorBoard 日志目录（默认：`logs`）

##### 模型保存
- `--output_dir`：保存微调模型的目录（默认：`None` - 模型未保存）

#### 使用示例
```bash
python Llama3_8B_LoRA_finetuning.py \
  --dataset_size 100 \
  --num_epochs 1 \
  --batch_size 2
  ```

## 在两个 Spark 上运行

### 步骤 1. 配置网络连接

按照 [Connect two Sparks](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks) 手册中的网络设置说明在 DGX Spark 节点之间建立连接。

这包括：
- 物理 QSFP 电缆连接
- 网络接口配置（自动或手动IP分配）
- 无密码 SSH 设置
- 网络连接验证

### 步骤2.配置Docker权限

要在不使用 sudo 的情况下轻松管理容器，您必须位于 `docker` 组中。如果您选择跳过此步骤，则需要使用 sudo 运行 Docker 命令。

打开新终端并测试 Docker 访问。在终端中，运行：

```bash
docker ps
```

如果您看到权限被拒绝错误（例如尝试连接到 Docker 守护进程套接字时权限被拒绝），请将您的用户添加到 docker 组，这样您就不需要使用 sudo 运行命令。

```bash
sudo usermod -aG docker $USER
newgrp docker
```
### 步骤 3. 安装 NVIDIA Container Toolkit 并设置 Docker 环境

确保在将提供 GPU 资源的每个节点（管理器和工作节点）上安装 NVIDIA 驱动程序和 NVIDIA Container Toolkit。该包使 Docker 容器能够访问主机的 GPU 硬件。确保您完成 [installation steps](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)，包括 NVIDIA 容器工具包的 [Docker configuration](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker)。

### 步骤 4. 启用资源广告

首先，通过运行以下命令找到您的 GPU UUID：
```bash
nvidia-smi -a | grep UUID
```

接下来，修改 Docker 守护进程配置以将 GPU 通告给 Swarm。编辑 **/etc/docker/daemon.json**：

```bash
sudo nano /etc/docker/daemon.json
```

添加或修改文件以包含 nvidia 运行时和 GPU UUID（将 **GPU-45cbf7b3-f919-7228-7a26-b06628ebefa1** 替换为您的实际 GPU UUID）：

```json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia",
  "node-generic-resources": [
    "NVIDIA_GPU=GPU-45cbf7b3-f919-7228-7a26-b06628ebefa1"
    ]
}
```

通过取消 **config.toml** 文件中 swarm-resource 行的注释，修改 NVIDIA Container Runtime 以将 GPU 通告给 Swarm。您可以使用您喜欢的文本编辑器（例如 vim、nano...）或使用以下命令来执行此操作：
```bash
sudo sed -i 's/^#\s*\(swarm-resource\s*=\s*".*"\)/\1/' /etc/nvidia-container-runtime/config.toml
```

最后，重新启动 Docker 守护进程以应用所有更改：
```bash
sudo systemctl restart docker
```

在所有节点上重复这些步骤。

### 步骤 5. 初始化 Docker Swarm

在要用作主节点的任何节点上，运行以下 swarm 初始化命令
```bash
docker swarm init --advertise-addr $(ip -o -4 addr show enp1s0f0np0 | awk '{print $4}' | cut -d/ -f1) $(ip -o -4 addr show enp1s0f1np1 | awk '{print $4}' | cut -d/ -f1)
```

上述的典型输出类似于以下内容：
```
Swarm initialized: current node (node-id) is now a manager.

To add a worker to this swarm, run the following command:

    docker swarm join --token <worker-token> <advertise-addr>:<port>

To add a manager to this swarm, run 'docker swarm join-token manager' and follow the instructions.
```

### 步骤 6. 加入工作节点并部署

现在我们可以继续设置集群的工作节点。在所有工作节点上重复这些步骤。

在每个工作节点上运行 docker swarm init 建议的命令来加入 Docker swarm
```bash
docker swarm join --token <worker-token> <advertise-addr>:<port>
```

在两个节点上，将 [**pytorch-ft-entrypoint.sh**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/pytorch-fine-tune/assets/pytorch-ft-entrypoint.sh) 脚本下载到包含微调脚本和配置文件的目录中，然后运行以下命令以使其可执行：

```bash
chmod +x $PWD/pytorch-ft-entrypoint.sh
```

在主节点上，通过将 [**docker-compose.yml**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/pytorch-fine-tune/assets/docker-compose.yml) 文件下载到与上一步相同的目录并运行以下命令来部署 Finetuning 多节点堆栈：
```bash
docker stack deploy -c $PWD/docker-compose.yml finetuning-multinode
```
> [！笔记]
> 确保将这两个文件下载到运行命令的同一目录中。

您可以使用以下命令验证工作节点的状态
```bash
docker stack ps finetuning-multinode
```

如果一切正常，您应该会看到类似以下内容的输出：
```
nvidia@spark-1b3b:~$ docker stack ps finetuning-multinode
ID             NAME                                IMAGE                              NODE         DESIRED STATE   CURRENT STATE            ERROR     PORTS
vlun7z9cacf9   finetuning-multinode_finetunine.1   nvcr.io/nvidia/pytorch:25.10-py3   spark-1d84   Running         Starting 2 seconds ago             
tjl49zicvxoi   finetuning-multinode_finetunine.2   nvcr.io/nvidia/pytorch:25.10-py3   spark-1b3b   Running         Starting 2 seconds ago             

```

> [！笔记]
> 如果您的“当前状态”不是“正在运行”，请参阅故障排除部分以获取更多信息。

### 步骤 7. 查找您的 Docker 容器 ID

您可以使用 `docker ps` 查找您的 Docker 容器 ID。您可以将容器 ID 保存在变量中，如下所示。在两个节点上运行此命令。
```bash
export FINETUNING_CONTAINER=$(docker ps -q -f name=finetuning-multinode)
```

### 步骤 9. 调整配置文件

对于多节点运行，我们提供了2个配置文件：
- [**config_finetuning.yaml**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/pytorch-fine-tune/assets/configs/config_finetuning.yaml) 用于 Llama3 3B 的全面微调。
- [**config_fsdp_lora.yaml**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/pytorch-fine-tune/assets/configs/config_fsdp_lora.yaml) 用于 Llama3 8B 和 Llama3 70B 的 LoRa 和 FSDP 微调。

这些配置文件需要调整：
- 根据每个节点的等级设置 `machine_rank`。您的主节点应具有等级 `0`。第二个节点的等级为 `1`。
- 使用主节点的 IP 地址设置 `main_process_ip`。确保两个配置文件具有相同的值。在主节点上使用 `ifconfig` 查找 CX-7 IP 地址的正确值。
- 设置可在主节点上使用的端口号。

YAML 文件中需要填写的字段：

```bash
machine_rank: 0
main_process_ip: < TODO: specify IP >
main_process_port: < TODO: specify port >
```

所有脚本和配置文件都在此 [**repository**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/pytorch-fine-tune/assets) 中可用。

### 步骤 10. 运行微调脚本

成功运行前面的步骤后，您可以使用此 [**repository**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/pytorch-fine-tune/assets) 中提供的 `run-multi-llama_*` 脚本之一进行微调。以下是 Llama3 70B 使用 LoRa 进行微调和 FSDP2 的示例。

```bash
## Need to specify huggingface token for model download.
export HF_TOKEN=<your-huggingface-token>

docker exec \
  -e HF_TOKEN=$HF_TOKEN \
  -it $FINETUNING_CONTAINER bash -c '
  bash /workspace/install-requirements;
  accelerate launch --config_file=/workspace/configs/config_fsdp_lora.yaml /workspace/Llama3_70B_LoRA_finetuning.py'
```

在运行期间，微调的进度条将仅出现在主节点的标准输出上。这是预期行为，因为 `accelerate` 使用 `tqdm` 周围的包装器仅按照 [**here**](https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/tqdm.py#L25) 的说明显示主进程的进度。在工作节点上使用 `nvidia-smi` 应显示 GPU 已被使用。

### 步骤 14. 清理和回滚

在领导节点上使用以下命令停止并删除容器：

```bash
docker stack rm finetuning-multinode
```

删除下载的模型以释放磁盘空间：

```bash
rm -rf $HOME/.cache/huggingface/hub/models--meta-llama* $HOME/.cache/huggingface/hub/datasets*
```

## 故障排除

| 症状 | 原因 | 使固定 |
|---------|--------|-----|
| 无法访问 URL 的门禁存储库 | 某些 HuggingFace 模型的访问受到限制 | 重新生成你的 [HuggingFace token](https://huggingface.co/docs/hub/en/security-tokens);并请求在您的网络浏览器上访问 [gated model](https://huggingface.co/docs/hub/en/models-gated#customize-requested-information) |
| 多次 Spark 运行中的错误和超时 | 各种原因 | 我们建议设置以下变量以启用额外的日志记录和运行时一致性检查 <br> `ACCELERATE_DEBUG_MODE=1`<br> `ACCELERATE_LOG_LEVEL=DEBUG`<br> `TORCH_CPP_LOG_LEVEL=INFO`<br> `TORCH_DISTRIBUTED_DEBUG=DETAIL`|
| 任务：非零退出 (255) | 容器退出，错误代码为 255 | 使用 `docker ps -a --filter "name=finetuning-multinode"` 检查容器日志以获取容器 ID，然后使用 `docker logs <container_id>` 查看详细的错误消息 |
|无法连接到位于 unix:///var/run/docker.sock 的 Docker 守护程序。 docker 守护进程是否正在运行？ | 由于 Docker Swarm 尝试绑定到过时或无法访问的本地链路 IP 地址而导致 Docker 守护进程崩溃 | 停止 Docker `sudo systemctl stop docker`<br> 删除 Swarm 状态 `sudo rm -rf /var/lib/docker/swarm`<br> 重新启动 Docker `sudo systemctl start docker`<br> 使用活动接口上的有效广告地址重新初始化 Swarm|

> [！笔记]
> DGX Spark 使用统一内存架构 (UMA)，可实现 GPU 和 CPU 之间的动态内存共享。 
> 由于许多应用程序仍在更新以利用 UMA，因此即使在 
> DGX Spark 的内存容量。如果发生这种情况，请使用以下命令手动刷新缓冲区缓存：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
