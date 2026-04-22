# DGX 仪表板

> 监控你的 DGX 系统并启动 JupyterLab

## 目录

- [概述](#overview)
- [操作步骤](#instructions)
- [故障排查](#troubleshooting)

---

<a id="overview"></a>
## 概述

## 基本思路

DGX Dashboard 是一个运行在 DGX Spark 设备本地的 Web 应用，提供系统更新、资源监控以及集成式 JupyterLab 环境的图形界面。用户既可以通过应用启动器在本地访问仪表板，也可以通过 NVIDIA Sync 或 SSH 隧道远程访问。对于远程使用场景来说，仪表板是更新系统包和固件最简单的方式。

## 你将完成的内容

你将学习如何在 DGX Spark 设备上访问并使用 DGX Dashboard。完成本指南后，你将能够启动带有预配置 Python 环境的 JupyterLab 实例、监控 GPU 性能、管理系统更新，并运行一个使用 Stable Diffusion 的示例 AI 工作负载。你还将了解多种访问方式，包括桌面快捷方式、NVIDIA Sync 和手动 SSH 隧道。

## 开始前需要了解

- 用于 SSH 连接和端口转发的基础终端操作
- 对 Python 环境和 Jupyter notebook 的基本理解

## 前置条件

**硬件要求：**
-  NVIDIA Grace Blackwell GB10 超级芯片系统

**软件要求：**
- NVIDIA DGX 操作系统
- 已安装 NVIDIA Sync（远程访问方式）或已配置 SSH 客户端

## 相关文件

- 用于 SDXL 的 Python 代码片段可在 [GitHub](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/dgx-dashboard/assets/jupyter-cell.py) 上获取


## 时间与风险

* **耗时：** 完整走完流程并运行示例 AI 工作负载约需 15-30 分钟
* **风险等级：** 低 - 主要是 Web 界面操作，对系统影响很小
* **回滚：** 可通过仪表板界面停止 JupyterLab 实例；正常使用过程中不会对系统造成永久更改。
* **最后更新：** 11/21/2025
  * 文案小幅修订

<a id="instructions"></a>
## 操作步骤

## 第 1 步：访问 DGX Dashboard

选择以下任一方式访问 DGX Dashboard Web 界面：

**选项 A：桌面快捷方式（本地访问）**

如果你可以直接访问 DGX Spark 设备本地桌面：

1. 登录 DGX Spark 设备上的 Ubuntu Desktop 环境
2. 点击屏幕左下角打开 Ubuntu 应用启动器
3. 在应用启动器中点击 DGX Dashboard 快捷方式
4. 仪表板会在默认浏览器中通过 `http://localhost:11000` 打开

**选项 B：NVIDIA Sync（推荐用于远程访问）**

如果你的本地机器已安装 NVIDIA Sync：

1. 点击系统托盘中的 NVIDIA Sync 图标
2. 在设备列表中选择你的 DGX Spark 设备
3. 点击 "Connect"
4. 点击 "DGX Dashboard" 启动仪表板
5. 仪表板会通过自动建立的 SSH 隧道，在默认浏览器中以 `http://localhost:11000` 打开

还没有 NVIDIA Sync？[在这里安装](/spark/connect-to-your-spark/sync)

**选项 C：手动 SSH 隧道**

如果希望在不使用 NVIDIA Sync 的情况下远程访问，你需要先[手动配置 SSH 隧道](/spark/connect-to-your-spark/manual-ssh)。

你必须为 Dashboard 服务（端口 11000）建立隧道；如果希望远程访问 JupyterLab，还需要为其建立隧道。每个用户账号分配到的 JupyterLab 端口号都不同。

1. 先 SSH 登录到 DGX Spark，并运行以下命令查看分配给你的 JupyterLab 端口：

```bash
cat /opt/nvidia/dgx-dashboard-service/jupyterlab_ports.yaml
```

2. 找到你的用户名，并记下对应的端口号。
3. 创建一个同时包含两个端口的新 SSH 隧道：

```bash
ssh -L 11000:localhost:11000 -L <ASSIGNED_PORT>:localhost:<ASSIGNED_PORT> <USERNAME>@<SPARK_DEVICE_IP>
```
将 `<USERNAME>` 替换为你的 DGX Spark 用户名，将 `<SPARK_DEVICE_IP>` 替换为设备 IP 地址。

再将 `<ASSIGNED_PORT>` 替换为 YAML 文件中的端口号。

然后打开浏览器并访问 `http://localhost:11000`。


## 第 2 步：登录 DGX Dashboard

浏览器中加载出仪表板后：

1. 在用户名输入框中输入你的 DGX Spark 系统用户名
2. 在密码输入框中输入系统密码
3. 点击 "Login" 进入仪表板界面

你应当会看到主仪表板，其中包含 JupyterLab 管理、系统监控和设置等面板。

## 第 3 步：启动 JupyterLab 实例

创建并启动一个 JupyterLab 环境：

1. 点击右侧面板中的 "Start" 按钮
2. 观察状态在以下阶段间切换：Starting → Preparing → Running
3. 等待状态显示为 "Running"（首次启动可能需要几分钟）
4. 如果状态变为 "Running" 后，JupyterLab 没有自动在浏览器中打开（例如弹窗被拦截），可以点击 "Open In Browser" 按钮

启动时，系统会自动创建默认工作目录（/home/<USERNAME>/jupyterlab），并自动设置虚拟环境。你可以通过查看工作目录中生成的 `requirements.txt` 文件来了解已安装的软件包。

后续如果你想更换工作目录并创建新的隔离环境，可以先点击 "Stop"，修改新的工作目录路径，再次点击 "Start"。

## 第 4 步：用示例 AI 工作负载进行测试

运行一个简单的 Stable Diffusion XL 图像生成示例，验证环境是否正常：

1. 在 JupyterLab 中创建新 notebook：File → New → Notebook
2. 点击 "Python 3 (ipykernel)" 创建 notebook
3. 添加一个新单元格并粘贴以下代码：

```python
import warnings
warnings.filterwarnings('ignore', message='.*cuda capability.*')
import tqdm.auto
tqdm.auto.tqdm = tqdm.std.tqdm

from diffusers import DiffusionPipeline
import torch
from PIL import Image
from datetime import datetime
from IPython.display import display

## --- Model setup ---
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    variant="fp16" if dtype==torch.float16 else None,
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

## --- Prompt setup ---
prompt = "a cozy modern reading nook with a big window, soft natural light, photorealistic"
negative_prompt = "low quality, blurry, distorted, text, watermark"

## --- Generation settings ---
height = 1024
width = 1024
steps = 30
guidance = 7.0

## --- Generate ---
result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=steps,
    guidance_scale=guidance,
    height=height,
    width=width,
)

## --- Save to file ---
image: Image.Image = result.images[0]
display(image)
image.save(f"sdxl_output.png")
print(f"Saved image as sdxl_output.png")
```

4. 运行该单元格（Shift+Enter 或点击 Run 按钮）
5. notebook 会下载模型并生成图像（首次运行可能需要几分钟）

## 第 5 步：监控 GPU 利用率

图像生成运行期间：

1. 切回浏览器中的 DGX Dashboard 标签页
2. 在监控面板中查看 GPU 遥测数据

## 第 6 步：停止 JupyterLab 实例

会话结束后：

1. 返回主 DGX Dashboard 标签页
2. 点击 JupyterLab 面板中的 "Stop" 按钮
3. 确认状态从 "Running" 变为 "Stopped"

## 第 6 步：管理系统更新

如果有系统更新可用，界面横幅或 Settings 页面中会有提示。

在 Settings 页面下的 "Updates" 选项卡中：

1. 点击 "Update" 打开确认对话框
2. 点击 "Update Now" 开始更新流程
3. 等待更新完成并让设备重启

> [！警告]
> 系统更新会升级软件包、固件（如有）并触发重启。继续前请先保存你的工作。

## 第 7 步：清理与回滚

若要清理资源并将系统恢复到原始状态：

1. 通过仪表板停止所有正在运行的 JupyterLab 实例
2. 删除 JupyterLab 工作目录

> [！警告]
> 如果你执行过系统更新，唯一的回滚方式是从系统备份或恢复介质进行恢复。

在正常使用仪表板的过程中，不会对系统造成永久性更改。

## 第 8 步：后续操作

完成 DGX Dashboard 配置后，你可以：

- 为不同项目创建额外的 JupyterLab 环境
- 使用仪表板管理系统维护和更新

<a id="troubleshooting"></a>
## 故障排查

| 现象 | 原因 | 解决方法 |
|---------|-------|-----|
| 用户无法运行更新 | 用户不在 sudo 组中 | 将用户加入 sudo 组：`sudo usermod -aG sudo <USERNAME>`；然后运行 `newgrp docker`|
| JupyterLab 无法启动 | 当前虚拟环境存在问题 | 在 JupyterLab 面板中更改工作目录并启动新实例 |
| SSH 隧道连接被拒绝 | IP 或端口不正确 | 验证 Spark 设备 IP，并确认 SSH 服务正在运行 |
| 监控中看不到 GPU | 驱动问题 | 使用 `nvidia-smi` 检查 GPU 状态 |


最新已知问题请参阅 [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)。
