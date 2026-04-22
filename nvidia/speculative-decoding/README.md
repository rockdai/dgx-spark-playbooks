# 推测性解码

> 了解如何设置推测解码以在 Spark 上进行快速推理

## 目录

- [Overview](#overview)
- [Instructions](#instructions)
  - [Option 1: EAGLE-3](#option-1-eagle-3)
  - [Option 2: Draft Target](#option-2-draft-target)
- [Run on Two Sparks](#run-on-two-sparks)
  - [Step 1. Configure Docker Permissions](#step-1-configure-docker-permissions)
  - [Step 2. Network Setup](#step-2-network-setup)
  - [Step 3. Set Container Name Variable](#step-3-set-container-name-variable)
  - [Step 4. Start the TRT-LLM Multi-Node Container](#step-4-start-the-trt-llm-multi-node-container)
  - [Step 5. Configure OpenMPI Hostfile](#step-5-configure-openmpi-hostfile)
  - [Step 6. Launch Eagle3 Speculative Decoding](#step-6-launch-eagle3-speculative-decoding)
  - [Step 7. Validate the API](#step-7-validate-the-api)
  - [Step 8. Cleanup](#step-8-cleanup)
  - [Step 9. Next Steps](#step-9-next-steps)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思路

推测性解码通过使用 **小而快速的模型** 提前起草多个标记，然后让 **更大的模型** 快速验证或调整它们，从而加速文本生成。
这样，大模型不需要逐步预测每个令牌，从而在保持输出质量的同时减少延迟。

## 你将完成什么

您将使用两种方法在 NVIDIA Spark 上使用 TensorRT-LLM 探索推测解码：EAGLE-3 和 Draft-Target。
这些示例演示了如何在保持输出质量的同时加速大型语言模型推理。

## 为什么是两个火花？

单个 DGX Spark 具有在 CPU 和 GPU 之间共享的 128 GB 统一内存。这足以运行带有 EAGLE-3 的 GPT-OSS-120B 或带有 Draft-Target 的 Llama-3.3-70B 等模型，如 **Instructions** 选项卡中所示。

**Qwen3-235B-A22B** 等较大的模型超出了单个 Spark 在内存中的容量 - 即使使用 FP4 量化，模型权重、KV 缓存和 Eagle3 草稿头总共也需要超过 128 GB。通过连接两个 Spark，您可以将可用内存增加一倍，达到 256 GB，从而可以为这些更大的模型提供服务。

**在两个 Sparks 上运行**选项卡将逐步完成此设置。两个 Spark 通过 QSFP 电缆连接，并使用 **张量并行性 (TP=2)** 来分割模型 - 每个 Spark 保存每层权重矩阵的一半，并计算每个前向传递的其部分。节点使用 NCCL 和 OpenMPI 通过高带宽链路传送中间结果，因此该模型作为跨两个设备的单个逻辑实例运行。

简而言之：两个 Spark 可以让您运行对于一个来说太大的模型，而顶部的推测解码 (Eagle3) 通过并行起草和验证多个标记进一步加速推理。

## 开始之前需要了解什么

- Docker 和容器化应用程序的经验
- 了解推测性解码概念
- 熟悉 TensorRT-LLM 服务和 API 端点
- 了解大型语言模型的 GPU 内存管理

## 先决条件

- 具有足够可用 GPU 内存的 NVIDIA Spark 设备
- 启用 GPU 支持的 Docker

  ```bash
  docker run --gpus all nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc12 nvidia-smi
  ```
- 用于模型访问的主动 HuggingFace 令牌
- 用于模型下载的网络连接


## 时间与风险

* **持续时间：** 10-20 分钟用于设置，额外时间用于模型下载（因网络速度而异）
* **风险：** 大型模型的 GPU 内存耗尽、容器注册表访问问题、下载期间网络超时
* **回滚：** 停止 Docker 容器并可选择清理下载的模型缓存。
* **最后更新：** 2026 年 4 月 20 日
  * 升级到最新容器1.3.0rc12
  * 添加在两个 Spark 上使用 Qwen3-235B-A22B 进行推测解码的示例

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

## 步骤2.设置环境变量

设置下游服务的环境变量：

 ```bash
export HF_TOKEN=<your_huggingface_token>
 ```

## 步骤 3. 运行推测解码方法

### 选项 1：EAGLE-3

通过执行以下命令来运行 EAGLE-3 推测解码：

```bash
docker run \
  -e HF_TOKEN=$HF_TOKEN \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host --network host \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc12 \
  bash -c '
    hf download openai/gpt-oss-120b && \
    hf download nvidia/gpt-oss-120b-Eagle3-long-context \
        --local-dir /opt/gpt-oss-120b-Eagle3/ && \
    cat > /tmp/extra-llm-api-config.yml <<EOF
enable_attention_dp: false
disable_overlap_scheduler: false
enable_autotuner: false
cuda_graph_config:
    max_batch_size: 1
speculative_config:
    decoding_type: Eagle
    max_draft_len: 5
    speculative_model_dir: /opt/gpt-oss-120b-Eagle3/

kv_cache_config:
    free_gpu_memory_fraction: 0.9
    enable_block_reuse: false
EOF
    export TIKTOKEN_ENCODINGS_BASE="/tmp/harmony-reqs" && \
    mkdir -p $TIKTOKEN_ENCODINGS_BASE && \
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken && \
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
    trtllm-serve openai/gpt-oss-120b \
      --backend pytorch --tp_size 1 \
      --max_batch_size 1 \
      --extra_llm_api_options /tmp/extra-llm-api-config.yml'
```

服务器运行后，通过从另一个终端进行 API 调用来测试它：

```bash
## Test completion endpoint
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "prompt": "Solve the following problem step by step. If a train travels 180 km in 3 hours, and then slows down by 20% for the next 2 hours, what is the total distance traveled? Show all intermediate calculations and provide a final numeric answer.",
    "max_tokens": 300,
    "temperature": 0.7
  }'
```

**EAGLE-3 推测解码的主要特点**

- **更简单的部署** - EAGLE-3 没有管理单独的草稿模型，而是使用内置的草稿头在内部生成推测令牌。

- **更高的准确性** - 通过融合模型多层的特征，草稿令牌更有可能被接受，从而减少浪费的计算。

- **更快的生成** - 每个前向传递并行验证多个令牌，从而减少自回归推理的延迟。

### 选项 2：草案目标

执行以下命令来设置并运行草稿目标推测解码：

```bash
docker run \
  -e HF_TOKEN=$HF_TOKEN \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host --network host nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc12 \
  bash -c "
#    # Download models
    hf download nvidia/Llama-3.3-70B-Instruct-FP4 && \
    hf download nvidia/Llama-3.1-8B-Instruct-FP4 \
    --local-dir /opt/Llama-3.1-8B-Instruct-FP4/ && \

#    # Create configuration file
    cat <<EOF > extra-llm-api-config.yml
print_iter_log: false
disable_overlap_scheduler: true
speculative_config:
  decoding_type: DraftTarget
  max_draft_len: 4
  speculative_model_dir: /opt/Llama-3.1-8B-Instruct-FP4/
kv_cache_config:
  enable_block_reuse: false
EOF

#    # Start TensorRT-LLM server
    trtllm-serve nvidia/Llama-3.3-70B-Instruct-FP4 \
      --backend pytorch --tp_size 1 \
      --max_batch_size 1 \
      --kv_cache_free_gpu_memory_fraction 0.9 \
      --extra_llm_api_options ./extra-llm-api-config.yml
  "
```

服务器运行后，通过从另一个终端进行 API 调用来测试它：

```bash
## Test completion endpoint
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Llama-3.3-70B-Instruct-FP4",
    "prompt": "Explain the benefits of speculative decoding:",
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

**草稿目标的主要特点：**

- **高效的资源利用**：8B草稿模型加速70B目标模型
- **灵活配置**：可调整草案令牌长度以进行优化
- **内存效率**：使用 FP4 量化模型来减少内存占用
- **兼容模型**：使用具有一致标记化的 Llama 系列模型

## 步骤 4. 清理

完成后停止 Docker 容器：

```bash
## Find and stop the container
docker ps
docker stop <container_id>

## Optional: Clean up downloaded models from cache
## rm -rf $HOME/.cache/huggingface/hub/models--*gpt-oss*
```

## 步骤 5. 后续步骤

- 尝试不同的 `max_draft_len` 值（1、2、3、4、8）
- 监控令牌接受率和吞吐量改进
- 使用不同的提示长度和生成参数进行测试
- 阅读有关推测解码 [here](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html) 的更多信息。

## 靠两个火花奔跑

### 步骤1.配置Docker权限

**在 Spark A 和 Spark B 上运行：**

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### 步骤 2. 网络设置

按照 **[Connect Two Sparks](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks)** 手册中的网络设置说明进行操作。

> [!NOTE]
> 在继续之前，请先完成《连接两个 Spark》手册中的步骤 1-3：
>
> - **第 1 步**：确保两个系统上的用户名相同
> - **步骤 2**：物理硬件连接（QSFP 电缆）
> - **步骤 3**：网络接口配置
>   - 使用 **选项 2：通过 netplan 配置文件手动分配 IP**
>   - 每个 Spark 有两对网络端口。当您在两个 Spark 之间物理连接电缆时，连接的端口将显示为 **Up**。您可以使用任一对 - **`enp1s0f0np0`** 和 **`enP2p1s0f0np0`**，或 **`enp1s0f1np1`** 和 **`enP2p1s0f1np1`**
>   - 本剧本假设您使用 **`enp1s0f1np1`** 和 **`enP2p1s0f1np1`**。如果您的 Up 接口不同，请在以下命令中替换您的接口名称

**对于本剧本，我们将使用以下 IP 地址：**

**Spark A（节点 1）：**
- `enp1s0f1np1`：192.168.200.12/24
- `enP2p1s0f1np1`：192.168.200.14/24

**Spark B（节点 2）：**
- `enp1s0f1np1`：192.168.200.13/24
- `enP2p1s0f1np1`：192.168.200.15/24

完成连接两个 Spark 设置后，返回此处继续 TRT-LLM 容器设置。

### 步骤 3. 设置容器名称变量

**在 Spark A 和 Spark B 上运行：**

```bash
export TRTLLM_MN_CONTAINER=trtllm-multinode
```

### 步骤4.启动TRT-LLM多节点容器

**在 Spark A 和 Spark B 上运行：**

```bash
docker run -d --rm \
  --name $TRTLLM_MN_CONTAINER \
  --gpus '"device=all"' \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --device /dev/infiniband:/dev/infiniband \
  -e UCX_NET_DEVICES="enp1s0f1np1,enP2p1s0f1np1" \
  -e NCCL_SOCKET_IFNAME="enp1s0f1np1,enP2p1s0f1np1" \
  -e OMPI_MCA_btl_tcp_if_include="enp1s0f1np1,enP2p1s0f1np1" \
  -e OMPI_MCA_orte_default_hostfile="/etc/openmpi-hostfile" \
  -e OMPI_MCA_rmaps_ppr_n_pernode="1" \
  -e OMPI_ALLOW_RUN_AS_ROOT="1" \
  -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1" \
  -e CPATH="/usr/local/cuda/include" \
  -e TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas" \
  -v ~/.cache/huggingface/:/root/.cache/huggingface/ \
  -v ~/.ssh:/tmp/.ssh:ro \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc12 \
  bash -c "curl https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/trt-llm/assets/trtllm-mn-entrypoint.sh | bash"
```

核实：

```bash
docker logs -f $TRTLLM_MN_CONTAINER
```

最后的预期输出：

```
total 56K
drwx------ 2 root root 4.0K Jan 13 05:13 .
drwx------ 1 root root 4.0K Jan 13 05:12 ..
-rw------- 1 root root  100 Jan 13 05:13 authorized_keys
-rw------- 1 root root   45 Jan 13 05:13 config
-rw------- 1 root root  411 Jan 13 05:13 id_ed25519
-rw-r--r-- 1 root root  102 Jan 13 05:13 id_ed25519.pub
-rw------- 1 root root  411 Jan 13 05:13 id_ed25519_shared
-rw-r--r-- 1 root root  100 Jan 13 05:13 id_ed25519_shared.pub
-rw------- 1 root root 3.4K Jan 13 05:13 id_rsa
-rw-r--r-- 1 root root  743 Jan 13 05:13 id_rsa.pub
-rw------- 1 root root 5.0K Jan 13 05:13 known_hosts
-rw------- 1 root root 3.2K Jan 13 05:13 known_hosts.old
Starting SSH
```

### 步骤 5. 配置 OpenMPI 主机文件

主机文件告诉 MPI 哪些节点参与分布式执行。使用步骤 2 中配置的 `enp1s0f1np1` 接口的 IP。

**在 Spark A 和 Spark B** 上，创建主机文件：

```bash
cat > ~/openmpi-hostfile <<EOF
192.168.200.12
192.168.200.13
EOF
```

**在 Spark A 和 Spark B 上运行**以将主机文件复制到每个容器中：

```bash
docker cp ~/openmpi-hostfile $TRTLLM_MN_CONTAINER:/etc/openmpi-hostfile
```

验证连接：

```bash
docker exec -it $TRTLLM_MN_CONTAINER bash -c "mpirun -np 2 hostname"
```

预期输出：

```
nvidia@spark-afe0:~$ docker exec -it $TRTLLM_MN_CONTAINER bash -c "mpirun -np 2 hostname"
Warning: Permanently added '[192.168.200.13]:2233' (ED25519) to the list of known hosts.
spark-afe0
spark-ae11
nvidia@spark-afe0:~$
```

### 步骤 6. 启动 Eagle3 推测解码

Eagle3 推测性解码通过提前预测多个标记，然后并行验证它们来加速推理。与标准自回归生成相比，这可以提供显着的加速。

#### 设置您的拥抱脸标记

```bash
export HF_TOKEN=your_huggingface_token_here
```

#### 在两个节点上下载 Eagle3 推测模型

```bash
docker exec \
  -e HF_TOKEN=$HF_TOKEN \
  -it $TRTLLM_MN_CONTAINER bash -c "
    mpirun -x HF_TOKEN -np 2 bash -c 'hf download nvidia/Qwen3-235B-A22B-Eagle3 --local-dir /opt/Qwen3-235B-A22B-Eagle3/'
"
```

#### 创建 Eagle3 推测解码配置

此配置支持使用 3 个草案令牌和保守的内存设置进行 Eagle 推测解码。

```bash
docker exec -it $TRTLLM_MN_CONTAINER bash -c "cat > /tmp/extra-llm-api-config.yml <<EOF
enable_attention_dp: false
disable_overlap_scheduler: false
enable_autotuner: false
enable_chunked_prefill: false
cuda_graph_config:
    max_batch_size: 1
speculative_config:
    decoding_type: Eagle
    max_draft_len: 3
    speculative_model_dir: /opt/Qwen3-235B-A22B-Eagle3/
kv_cache_config:
    free_gpu_memory_fraction: 0.9
    enable_block_reuse: false
EOF
"
```

#### 使用 Eagle3 推测解码启动服务器

**仅在 Spark A 上运行。** 这将使用启用了 Eagle3 推测解码的 FP4 基本模型启动 TensorRT-LLM API 服务器。 `mpirun` 命令协调两个节点之间的执行，因此只需要从 Spark A 启动。最大令牌长度设置为 1024（根据需要调整）。

```bash
docker exec \
  -e MODEL="nvidia/Qwen3-235B-A22B-FP4" \
  -e HF_TOKEN=$HF_TOKEN \
  -it $TRTLLM_MN_CONTAINER bash -c '
    mpirun -x CPATH=/usr/local/cuda/include \
           -x TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
           -x HF_TOKEN \
           trtllm-llmapi-launch \
           trtllm-serve \
           $MODEL \
           --backend pytorch \
           --tp_size 2 \
           --max_num_tokens 1024 \
           --extra_llm_api_options /tmp/extra-llm-api-config.yml \
           --port 8355 --host 0.0.0.0
'
```

端点就绪时的预期输出：

```
[01/13/2026-06:16:56] [TRT-LLM] [I] get signal from executor worker
INFO:     Started server process [2011]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 步骤 7. 验证 API

**仅在 Spark A 上运行。** 服务器正在侦听 Spark A，因此从那里测试端点：

```bash
curl -s http://localhost:8355/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Qwen3-235B-A22B-FP4",
    "messages": [{"role": "user", "content": "Paris is great because"}],
    "max_tokens": 64
  }'
```

预期：带有生成文本的 JSON 响应。这证实了具有 Eagle3 推测解码功能的多节点 TensorRT-LLM 服务器正常工作。

### 步骤 8. 清理

#### 停止容器

**在 Spark A 和 B 上运行：**

```bash
docker stop $TRTLLM_MN_CONTAINER
```

由于 `--rm` 标志，容器将被自动删除。

#### （可选）删除下载的模型

如果您需要释放磁盘空间：

**在 Spark A 和 B 上运行：**

```bash
rm -rf $HOME/.cache/huggingface/hub/models--nvidia--Qwen3*
```

这将删除模型文件（约数百 GB）。如果您打算再次运行安装程序，请跳过此步骤。

### 步骤 9. 后续步骤

现在您已经运行了 Eagle3 推测解码，请考虑以下优化和实验：

- **调整草稿长度：** 修改配置中的 `max_draft_len` （尝试 2-5 之间的值）以平衡推测速度与准确性
- **尝试不同的模型：** 尝试支持 Eagle 推测解码的其他模型对
- **优化批量大小：** 调整 `cuda_graph_config` 中的 `max_batch_size` 以实现吞吐量-延迟权衡
- **了解更多：** 查看 [TensorRT-LLM Speculative Decoding documentation](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html) 以获取高级调整选项
- **基准性能：** 比较有和没有推测解码的推理速度，以测量加速增益

## 故障排除

| 症状 | 原因 | 使固定 |
|---------|--------|-----|
| “CUDA 内存不足”错误 | GPU显存不足 | 将 `kv_cache_free_gpu_memory_fraction` 减少到 0.9 或使用具有更多 VRAM 的设备 |
| 容器无法启动 | Docker GPU 支持问题 | 验证 `nvidia-docker` 已安装并且支持 `--gpus=all` 标志 |
| 模型下载失败 | 网络或身份验证问题 | 检查 HuggingFace 身份验证和网络连接 |
| 无法访问 URL 的门禁存储库 | 某些 HuggingFace 模型的访问受到限制 | 重新生成你的 [HuggingFace token](https://huggingface.co/docs/hub/en/security-tokens);并请求在您的网络浏览器上访问 [gated model](https://huggingface.co/docs/hub/en/models-gated#customize-requested-information) |
| 服务器没有响应 | 端口冲突或防火墙 | 检查8000端口是否可用且未被阻塞 |
| `mpirun` 失败并拒绝 SSH 连接 | 容器或节点之间未配置 SSH | 从 Connect Two Sparks playbook 完成 SSH 设置；验证 `ssh <node_ip>` 无需密码即可从两个节点正常工作 |
| `mpirun` 与远程节点的连接挂起或超时 | 主机文件 IP 与实际节点 IP 不匹配 | 验证 `/etc/openmpi-hostfile` 中的 IP 与分配给具有 `ip addr show` 的网络接口的 IP 匹配 |
| NCCL 错误：“非套接字上的套接字操作” | 指定的网络接口错误 | 检查 `ibdev2netdev` 输出并确保 `NCCL_SOCKET_IFNAME` 和 `UCX_NET_DEVICES` 与活动接口 `enp1s0f1np1,enP2p1s0f1np1` 匹配 |
| mpirun 期间的 `Permission denied (publickey)` | 容器之间不交换 SSH 密钥 | 从 Connect Two Sparks playbook 重新运行 SSH 设置或手动验证 `/root/.ssh/authorized_keys` 包含来自两个节点的公钥 |
| 在多节点设置中模型下载失败且无提示 | HF_TOKEN 未传播到 mpirun | 将 `-e HF_TOKEN=$HF_TOKEN` 添加到 `docker exec` 命令，将 `-x HF_TOKEN` 添加到 `mpirun` 命令 |

> [!NOTE]
> DGX Spark 使用统一内存架构 (UMA)，可实现 GPU 和 CPU 之间的动态内存共享。
> 由于许多应用程序仍在更新以利用 UMA，因此即使在
> DGX Spark 的内存容量。如果发生这种情况，请使用以下命令手动刷新缓冲区缓存：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
