---
id: single-cell-assets
title: assets
sidebar_label: assets
---

# **从这里开始单细胞分析手册**
___
## **目录**
___
<br />`

- [Get Started Now](#Get-Started-Now!)
- [Playbook Structure](#Playbook-Structure)
- [Next Steps](#Next-Steps)
<br />`

___
## **立即开始！**
___
<br />`

**[scRNA Analysis Preprocessing Notebook](scRNA_analysis_preprocessing.ipynb)** 是使用 [RAPIDS-singlecell](https://rapids-singlecell.readthedocs.io/en/latest/)（由 [scverse®](https://github.com/scverse) 开发的 GPU 加速库）的端到端 GPU 加速单细胞分析工作流程。  在此笔记本中，我们了解单元格，对数据集运行 ETL，然后可视化并探索结果。  完成该工作流程应该需要不到 3 分钟的时间。

### [CLICK HERE TO BEGIN](scRNA_analysis_preprocessing.ipynb)

使用 DGX Spark 可以帮助您轻松使用 [RAPIDS Open Source ecosystem](https://rapids.ai) GPU 加速基于数据科学和机器学习的工作流程，以便您可以比以往更快地从数据到信息再到见解！  

![cells](assets/rsc.png)


# ``<img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>`  ``<img src="https://canada1.discourse-cdn.com/flex035/uploads/forum11/original/1X/dfb6d71c9b8deb73aa10aa9bc47a0f8948d5304b.png" width="90px"/>` 
<br />`

___
## **深入研究**
___
<br />`

本笔记本适合那些刚开始对单细胞数据进行基本分析的人，因为端到端分析是最好的起点，您将逐步完成数据预处理、质量控制 (QC) 和清理、可视化和调查的步骤。  让我们深入研究一下这个过程！

![layout architecture](assets/scdiagram.png)

1.  加载和预处理数据
    - 使用 Scanpy 加载 h5ad 格式的稀疏矩阵
    - 预处理数据，实施标准 QC 指标来评估每个细胞以及每个基因的细胞和基因质量

2. QC 细胞直观地了解数据
    - 用户将学习如何直观地检查 5 个不同的图，这些图有助于反映单细胞数据的质量控制指标：
        - 识别正在经历凋亡的应激或死亡细胞
        - 空滴或死细胞
        - 基因计数异常的细胞
        - 低质量或过度占优势的细胞

3. 过滤异常细胞
    - 用户将学习如何去除表达基因数量过多的细胞
    - 用户将过滤掉线粒体含量异常的细胞

4. 消除不需要的变异来源
    - 选择大多数可变基因以更好地为分析提供信息并提高计算效率
    - 回归我们在视觉图中观察到的其他技术变化（注意，这实际上可以删除生物学相关信息，并且需要使用更复杂的数据集仔细考虑）
    - 使用 z 分数转换进行标准化

5. 聚类和可视化数据
    - 实施 PCA 以降低计算复杂度。我们使用 cuML 的 GPU 加速 PCA 实现，与基于 CPU 的方法相比，它显着加快了计算速度。
    - 通过使用基于图形的聚类生成 UMAP 图来直观地识别批次效应

6. 批量校正和分析
    - 使用 Harmony 消除特定于测定的批次效应
    - 重新计算 k 最近邻图并使用 UMAP 进行可视化。
    - 执行基于图的聚类
    - 使用其他方法可视化 (tSNE)

7. 从数据中探索生物信息
    - 差异表达分析：识别细胞类型的标记基因
        - 实施逻辑回归
        - 对区分细胞类型的基因进行排序
    - 轨迹分析
        - 实施扩散图以了解细胞类型的进展

这些笔记本对于想要快速评估易用性以及探索 RAPIDS-单细胞结果的生物学可解释性的单细胞科学家来说非常有价值。其次，科学家会发现学习将这些方法应用于非常大的数据集的价值。该存储库对于任何想要利用 RAPIDS-singlecell 运行和评估单细胞方法的数据科学家或开发人员也广泛有用。本教程使用的数据集为 [publicly available by 10X](https://www.10xgenomics.com/datasets) 和 [CZ cellxgene](https://cellxgene.cziscience.com/)。 

如果您喜欢这款笔记本和 GPU 加速功能，请执行以下两件事：
1. 通过 [Single Cell Analysis AI Blueprint](https://github.com/NVIDIA-AI-Blueprints/single-cell-analysis-blueprint/tree/main) 探索其余的单单元笔记本 
1. 请[learn more about them here](https://scverse.org/about/) 和[consider joining their community](https://scverse.org/join/) 支持scverse 的努力。
<br />`
<br />`

___

## **目录结构**
___
<br />`

- **[scRNA_analysis_preprocessing.ipynb](scRNA_analysis_preprocessing.ipynb)** - 主要剧本笔记本 
- `START_HERE.md` - Playbook 环境快速入门指南。  它还可以在 Jupyter Lab 的主目录中找到。  请从那里开始！
- `cuDF, cuML, and cuGraph folders` - 更多示例笔记本，可继续您的 GPU 加速数据科学之旅。
<br />`

___
## **DIY安装**
___
<br />`
如果您喜欢所看到的内容并希望为基因组学、生物信息学或单细胞研究运行更多 GPU 加速，请执行以下操作

```bash
pip install -r ./setup/requirements.txt
```

在这个需求文件中，有一些所需的库的固定版本。  当`pinned`时，它只会下载该特定版本，这将确保您作为产品的稳定性。  当 `unpinned` 时，pip 或 uv 将下载一切正常的最新版本。  如果您计划升级到最新的技术堆栈，则应该取消固定库，但您的情况可能会有所不同。

<br />`
<br />`

___
## **支持**
___
<br />`

如果您对这些笔记本有任何疑问或需要支持，请在 [Single Cell Analysis AI Blueprint](https://github.com/NVIDIA-AI-Blueprints/single-cell-analysis-blueprint/tree/main) 存储库上提出问题，我们将在那里回复。


