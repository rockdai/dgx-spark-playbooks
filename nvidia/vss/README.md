# 构建视频搜索和摘要 (VSS) 代理

> 在 Spark 上运行 VSS 蓝图

## 目录

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思路

部署 NVIDIA 的视频搜索和摘要 (VSS) AI 蓝图来构建结合了视觉语言模型、大型语言模型和检索增强生成的智能视频分析系统。该系统通过视频摘要、问答和实时警报将原始视频内容转化为实时可操作的见解。您将设置完全本地的 Event Reviewer 部署或使用远程模型端点的混合部署。

## 你将完成什么

您将在采用 Blackwell 架构的 NVIDIA Spark 硬件上部署 NVIDIA 的 VSS AI 蓝图，并在两种部署方案之间进行选择：VSS Event Reviewer（使用 VLM 管道完全本地化）或标准 VSS（使用远程 LLM/嵌入端点的混合部署）。这包括设置警报桥、VLM 管道、警报检查器 UI、视频存储工具包和可选的 DeepStream CV 管道，以进行自动视频分析和事件审查。

## 开始之前需要了解什么

- 使用 NVIDIA Docker 容器和容器注册表
- 使用共享网络设置 Docker Compose 环境
- 管理环境变量和身份验证令牌
- 对视频处理和分析工作流程的基本了解

## 先决条件

- 具有 ARM64 架构和 Blackwell GPU 的 NVIDIA Spark 设备
- DGX 操作系统（建议：7.4.0 或更高版本）
- 安装的驱动程序版本 580.95.05 或更高版本：`nvidia-smi | grep "Driver Version"`
- 安装的 CUDA 版本 13.0：`nvcc --version`
- Docker 安装并运行：`docker --version && docker compose version`
- 使用 [NGC API Key](https://org.ngc.nvidia.com/setup/api-keys) 访问 NVIDIA 容器注册表
- NVIDIA 容器工具包
- [可选] 用于远程模型端点的 NVIDIA API 密钥（仅限混合部署）
- 足够的视频处理存储空间（`/tmp/`中建议>10GB）

## 附属文件

- [VSS Blueprint GitHub Repository](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization) - 主要代码库和 Docker Compose 配置
- [VSS Official Documentation](https://docs.nvidia.com/vss/latest/index.html) - 完整的系统文档

## 时间与风险

* **持续时间：** 初始设置 30-45 分钟，视频处理验证需要额外时间
* **风险：**
  * 由于模型下载量较大，容器启动可能会占用大量资源且耗时
  * 如果共享网络已存在，网络配置会发生冲突
  * 远程 API 端点可能存在速率限制或连接问题（混合部署）
* **回滚：** 使用 `scripts/dev-profile.sh down` 停止所有容器
* **最后更新：** 2026 年 3 月 16 日
  * 更新所需的操作系统和驱动程序版本
  * 通过 Cosmos Reason 2 VLM 支持 VSS 3.1.0

## 指示

## 步骤 1. 验证环境要求

检查您的系统是否满足硬件和软件 [prerequisites](https://docs.nvidia.com/vss/latest/prerequisites.html) 的要求。

```bash
## Verify driver version
nvidia-smi | grep "Driver Version"
## Expected output: Driver Version: 580.126.09 or higher

## Verify CUDA version
nvcc --version
## Expected output: release 13.0

## Verify Docker is running
docker --version && docker compose version
```

## 步骤 2. 配置 Docker

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


此外，配置 Docker 以便它可以使用 NVIDIA 容器运行时。

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

##Run a sample workload to verify the setup
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

## 步骤 3. 克隆 VSS 存储库

从 NVIDIA 的公共 GitHub 克隆视频搜索和摘要存储库。

```bash
## Clone the VSS AI Blueprint repository
git clone https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization.git
cd video-search-and-summarization
```

## 步骤 4. 运行缓存清理脚本

启动系统缓存清理器以优化容器操作期间的内存使用。

在下面提到的 /usr/local/bin/sys-cache-cleaner.sh 创建缓存清理脚本

```bash
sudo tee /usr/local/bin/sys-cache-cleaner.sh << 'EOF'
#!/bin/bash
## Exit immediately if any command fails
set -e

## Disable hugepages
echo "disable vm/nr_hugepage"
echo 0 | tee /proc/sys/vm/nr_hugepages

## Notify that the cache cleaner is running
echo "Starting cache cleaner - Running"
echo "Press Ctrl + C to stop"
## Repeatedly sync and drop caches every 3 seconds
while true; do
     sync && echo 3 | tee /proc/sys/vm/drop_caches > /dev/null
     sleep 3
done
EOF

sudo chmod +x /usr/local/bin/sys-cache-cleaner.sh
```
在后台运行

```bash
## In another terminal, start the cache cleaner script.
sudo -b /usr/local/bin/sys-cache-cleaner.sh
```

> [!NOTE]
+> 以上仅在当前会话中运行缓存清理器；它不会在重新启动后持续存在。要让缓存清理器在重新启动后运行，请创建一个 systemd 服务。
+> 
+> 要停止后台缓存清理器：
+> ```bash
+> sudo pkill -f sys-cache-cleaner.sh
+> ```


## 步骤 5. 使用 NVIDIA 容器注册表进行身份验证

使用 [NGC API Key](https://org.ngc.nvidia.com/setup/api-keys) 登录 NVIDIA 的容器注册表。

> [!NOTE]
> 如果您还没有 NVIDIA 帐户，则必须创建一个帐户并注册 [developer program](https://developer.nvidia.com/nvidia-developer-program)。

```bash
## Log in to NVIDIA Container Registry
docker login nvcr.io
## Username: $oauthtoken
## Password: <PASTE_NGC_API_KEY_HERE>
```

## 步骤6.选择部署场景

根据您的要求在两个部署选项之间进行选择：

| 部署场景                       | VLM（Cosmos-Reason2-8B）| 法学硕士                           | 
|-------------------------------------------|------------------------|-------------------------------|
| 标准 VSS（基础）                       | 当地的           | 偏僻的                               |
| 标准 VSS（警报验证）         | 当地的           | 偏僻的                               |
| 标准VSS部署（实时警报）| 当地的           | 偏僻的                               |


## 步骤 7. 标准 VSS 

**[Standard VSS](https://docs.nvidia.com/vss/latest/#architecture-overview)（混合部署）**

在此混合部署中，我们将使用 [build.nvidia.com](https://build.nvidia.com/) 中的 NIM。或者，您可以按照 [VSS remote LLM deployment guide](https://docs.nvidia.com/vss/latest/vss-agent/configure-llm.html) 中的说明配置您自己的托管端点。


**7.1 获取 NVIDIA API 密钥**

- 登录 https://build.nvidia.com/explore/discover.
- 在页面中搜索“获取API Key”并点击。


**7.2 启动标准 VSS 部署**

[Standard VSS deployment (Base)](https://docs.nvidia.com/vss/latest/quickstart.html#deploy)
[Standard VSS deployment (Alert Verification)](https://docs.nvidia.com/vss/latest/agent-workflow-alert-verification.html)
[Standard VSS deployment (Real-Time Alerts)](https://docs.nvidia.com/vss/latest/agent-workflow-rt-alert.html#real-time-alert-workflow)

```bash
## Start Standard VSS (Base)
export NGC_CLI_API_KEY='your_ngc_api_key'
export LLM_ENDPOINT_URL=https://your-llm-endpoint.com
scripts/dev-profile.sh up -p base -H DGX-SPARK --use-remote-llm --llm <REMOTE LLM MODEL NAME>

## Start Standard VSS (Alert Verification)
export NGC_CLI_API_KEY='your_ngc_api_key'
export LLM_ENDPOINT_URL=https://your-llm-endpoint.com
scripts/dev-profile.sh up -p alerts -m verification -H DGX-SPARK --use-remote-llm --llm <REMOTE LLM MODEL NAME>

## Start Standard VSS (Real-Time Alerts)
export NGC_CLI_API_KEY='your_ngc_api_key'
export LLM_ENDPOINT_URL=https://your-llm-endpoint.com
scripts/dev-profile.sh up -p alerts -m real-time -H DGX-SPARK --use-remote-llm --llm <REMOTE LLM MODEL NAME>
```

> [!NOTE]
> 随着容器的拉取和服务的初始化，此步骤将需要几分钟的时间。 VSS 后端需要额外的启动时间。
> 部署前设置以下环境变量：
> • **NGC_CLI_API_KEY** —（必需）用于拉取映像和部署的 NGC API 密钥
> • **LLM_ENDPOINT_URL** —（使用 `--use-remote-llm` 时需要）远程 LLM 的基本 URL
> • **NVIDIA_API_KEY** —（可选）适用于需要它的远程 LLM/VLM 端点
> • **OPENAI_API_KEY** —（可选）对于需要它的远程 LLM/VLM 端点
> • **VLM_CUSTOM_WEIGHTS** —（可选）自定义权重目录的绝对路径
>
> 将这些附加标志传递给 **`scripts/dev-profile.sh`** 以实现远程 LLM 模式：
> • **`--use-remote-llm`** —（必需）使用远程 LLM，基本 URL 是从环境中的 **`LLM_ENDPOINT_URL`** 读取的
> • **`--llm`** —（必需）远程LLM 模型名称（例如：`nvidia/nvidia-nemotron-nano-9b-v2`）。 **强烈建议**对于警报工作流程（验证和实时）：使用 `nvidia/nvidia-nemotron-nano-9b-v2`。省略 `--llm` 可能会导致脚本使用远程端点返回的任何模型。
>
> 运行 **`scripts/dev-profile.sh -h`** 以获得受支持参数的完整列表。


**7.3 验证标准 VSS 部署**

访问 VSS UI 以确认部署成功。
[Common VSS Endpoints](https://docs.nvidia.com/vss/latest/agent-workflow-alert-verification.html#service-endpoints)

```bash
## Test Agent UI accessibility
## If running locally on your Spark device, use localhost:
curl -I http://localhost:3000
## Expected: HTTP 200 response

## If your Spark is running in Remote/Accessory mode, replace 'localhost' with the IP address or hostname of your Spark device.
## To find your Spark's IP address, run the following command on the Spark terminal:
hostname -I
## Or to get the hostname:
hostname
## Then test accessibility (replace <SPARK_IP_OR_HOSTNAME> with the actual value):
curl -I http://<SPARK_IP_OR_HOSTNAME>:3000
```

在浏览器中打开`http://localhost:3000`或`http://<SPARK_IP_OR_HOSTNAME>:3000`以访问代理界面。

## 步骤 8. 测试视频处理工作流程

运行基本测试以验证视频分析管道是否根据您的部署正常运行。用户界面附带了一些预先填充的示例视频，用于上传和测试

**对于标准 VSS 部署**

按照步骤 [here](https://docs.nvidia.com/vss/latest/quickstart.html#deploy) 导航 VSS 代理 UI。
- 访问 `http://localhost:3000` 处的 VSS 代理界面
- 从 NGC [here](https://docs.nvidia.com/vss/latest/quickstart.html#download-sample-data-from-ngc) 下载样本数据并上传视频和测试功能 [here](https://docs.nvidia.com/vss/latest/quickstart.html#download-sample-data-from-ngc)
  

## 步骤 9. 清理和回滚

要完全删除 VSS 部署并释放系统资源 [Follow](https://docs.nvidia.com/vss/latest/quickstart.html#step-5-teardown-the-agent)：

> [!WARNING]
> 这将破坏所有处理过的视频数据和分析结果。

```bash
## For Standard VSS deployment
scripts/dev-profile.sh down
```

## 步骤 10. 后续步骤

部署 VSS 后，您现在可以：

**标准VSS部署：**
- 在端口 3000 访问完整的 VSS 功能
- 测试视频摘要和问答功能
- 配置知识图和图数据库
- 与现有视频处理工作流程集成

## 故障排除

| 症状 | 原因 | 使固定 |
|---------|--------|-----|
| 容器无法启动，并显示“拉取访问被拒绝” | nvcr.io 凭据缺失或不正确 | 使用有效凭据重新运行 `docker login nvcr.io` |
| Web 界面无法访问 | 服务仍在启动或端口冲突 | 等待 2-3 分钟，检查 `docker ps` 容器状态 |

> [!NOTE]
> DGX Spark 使用统一内存架构 (UMA)，可实现 GPU 和 CPU 之间的动态内存共享。 
> 由于许多应用程序仍在更新以利用 UMA，因此即使在 
> DGX Spark 的内存容量。如果发生这种情况，请使用以下命令手动刷新缓冲区缓存：
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
