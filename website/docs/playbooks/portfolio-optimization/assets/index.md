---
id: portfolio-optimization-assets
title: assets
sidebar_label: assets
---

# **DGX Spark 上的投资组合优化笔记本**
___

## **概述**
___
<br />

![arch_diagram](assets/arch_diagram.png)

**[`cvar_basic.ipynb`](cvar_basic.ipynb)** 是完整的投资组合优化演练 Jupyter 笔记本，演示了使用 NVIDIA DGX Spark 的 GPU 加速投资组合优化技术。  它主要使用新的专用库 **[cuFolio](https://www.nvidia.com/en-us/on-demand/session/gtc25-dlit71690/)**，该库基于 NVIDIA 的 **[cuOpt](https://github.com/NVIDIA/cuopt)** 以及 NVIDIA RAPIDS 的 **[cuML](https://github.com/rapidsai/cuml)** 和 **[cuGraph](https://github.com/rapidsai/cugraph)** 构建。

## **[CLICK HERE TO GET STARTED](cvar_basic.ipynb)**

本笔记本的分步演练涵盖：

- 数据准备和预处理
- 场景生成
- **[Mean-CVaR (Conditional Value-at-Risk)](https://www.youtube.com/shorts/9u-VrCyneM4)** 投资组合优化
- 实施现实世界的限制（集中度限制、杠杆、营业额）
- 投资组合构建和分析
- 绩效评估和回测

如果您想深入了解笔记本本身，**[check out the blog: Accelerating Real-Time Financial Decisions with Quantitative Portfolio Optimization](https://developer.nvidia.com/blog/accelerating-real-time-financial-decisions-with-quantitative-portfolio-optimization/)**

**请务必使用投资组合优化内核运行笔记本！** 说明将位于笔记本的开头。

下载的股票数据将存储在`data`中。  计算结果保存在 `results` 文件夹中。

![optimization](assets/cvar.png)
<br />
<br />

___
## **DIY安装**
___
<br />

安装 Portfolio Optimization 包的复杂程度适中，因此我们创建了一些脚本来轻松构建 Python 环境。

您将需要使用 `pip`/`uv` 或 `docker` 安装 RAPIDS 25.10 和 Jupyter。  请参阅 [RAPIDS Installation Selector](https://docs.rapids.ai/install/#selector) 了解更多详情。

示例：
```bash
pip install "cudf-cu13==25.10.*" "cuml-cu13==25.10.*" jupyterlab
```
或者 

```bash
docker run --gpus all --pull always --rm -it \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 8888:8888 -p 8787:8787 -p 8786:8786 \
    nvcr.io/nvidia/rapidsai/notebooks:25.10-cuda13-py3.13
```

安装 RAPIDS 后，请运行以下命令来安装 Portfolio Optimization Jupyter Kernel。  如果您使用 Docker，请在 Docker 环境中运行这些。

```bash
cd Stock_Portfolio_Optimization
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# To add $HOME/.local/bin to your PATH, either restart your shell or run:
source $HOME/.local/bin/env

# Install with CUDA-specific dependencies
uv sync --extra cuda13

# Optional: Install development tools
# uv sync --extra cuda13 --extra dev  

# Create a Jupyter kernel for this environment
uv run python -m ipykernel install --user --name=portfolio-opt --display-name "Portfolio Optimization"

# Launch Jupyter Lab (if necessary)
uv run jupyter lab --no-browser --NotebookApp.token=''
```
<br />
<br />

___
## **后续步骤**
___
<br />

### **NVIDIA AI 蓝图的高级工作流程**

一旦您熟悉了基本工作流程，就可以在 **[NVIDIA AI Blueprints](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/)** 以任意顺序探索这些高级主题：

#### [`efficient_frontier.ipynb`](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/tree/main/notebooks/efficient_frontier.ipynb) - 高效前沿分析

本笔记本演示了如何：
- 通过解决多个优化问题生成**[efficient frontier](https://www.youtube.com/shorts/apvVgwg06hw)**
- 可视化不同投资组合配置的风险回报权衡
- 沿有效边界比较投资组合
- 利用GPU加速快速计算多个最优投资组合

#### [`rebalancing_strategies.ipynb`](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/tree/main/notebooks/rebalancing_strategies.ipynb) - 动态投资组合再平衡

本笔记本介绍了动态投资组合管理技术：
- 时间序列回测框架
- 测试各种再平衡策略（定期、基于阈值等）
- 评估交易成本对投资组合绩效的影响
- 分析不同市场条件下的策略绩效
- 比较多种再平衡方法
<br />
<br />

___
## **额外资源**
___
<br />

如果您想进一步了解如何使用类似的风险回报框架来制定投资组合优化问题，请查看 **[DLI course: Accelerating Portfolio Optimization](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-DS-09+V1)**

如有疑问或问题，请访问：
- [GitHub Issues](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/issues)
- [GitHub Discussions](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/discussions)

