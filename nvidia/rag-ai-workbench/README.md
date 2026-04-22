# RAG在AI Workbench中的应用

> 安装并使用 AI Workbench 克隆并运行可重现的 RAG 应用程序

## 目录

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## 概述

## 基本思路

本演练演示了如何设置和运行代理检索增强生成 (RAG)
使用 NVIDIA AI Workbench 的项目。您将使用 AI Workbench 克隆并运行预构建的代理 RAG
智能路由查询、评估响应的相关性和幻觉的应用程序，以及
迭代评估和生成周期。该项目使用Gradio Web界面并且可以工作
具有 NVIDIA 托管的 API 端点或自托管模型。

## 你将完成什么

您将拥有一个在 NVIDIA AI Workbench 中运行的功能齐全的代理 RAG 应用程序，并带有 Web
您可以在其中提交查询并接收智能响应的界面。系统将演示
高级 RAG 功能，包括查询路由、响应评估和迭代细化，
让您亲身体验 AI Workbench 的开发环境和复杂的 RAG
架构。

## 开始之前需要了解什么

- 基本熟悉检索增强生成 (RAG) 概念
- 了解 API 密钥以及如何生成它们
- 舒适地使用网络应用程序和浏览器界面
- 对容器化开发环境的基本了解

## 先决条件

**硬件要求：**
-  NVIDIA Grace Blackwell GB10 超级芯片系统

**软件要求：**
- NVIDIA AI Workbench 已安装或准备安装
- 免费 NVIDIA API 密钥：在 [NGC API Keys](https://org.ngc.nvidia.com/setup/api-keys) 生成
- 免费 Tavily API 密钥：在 [Tavily](https://tavily.com/) 生成
- 用于克隆存储库和访问 API 的互联网连接
- 用于访问 Gradio 界面的 Web 浏览器

## 验证命令

- 验证 DGX Spark 系统上是否存在 NVIDIA AI Workbench 应用程序
- 验证您的 API 密钥是否有效且是最新的


## 时间与风险

* **预计时间：** 30-45 分钟（如果需要，包括 AI Workbench 安装）
* **风险级别：** 低 - 使用预构建的容器和已建立的 API
* **回滚：** 只需从 AI Workbench 中删除克隆的项目即可删除所有组件。在 AI Workbench 环境之外不会进行任何系统更改。
* **最后更新：** 2025 年 10 月 28 日
  * 少量文案编辑

## 指示

## 步骤 1. 安装 NVIDIA AI Workbench

在 DGX Spark 系统上安装 AI Workbench 并完成初始设置向导。

在 DGX Spark 上，打开 **NVIDIA AI Workbench** 应用程序并单击“开始安装”。

1. 安装向导将提示进行身份验证
2. 等待自动安装完成（几分钟）
3. 安装完成后单击“让我们开始吧”

> [！笔记]
> 如果您遇到以下错误消息，请重新启动 DGX Spark，然后重新打开 NVIDIA AI Workbench：
> “发生错误...容器工具未能达到就绪状态。重试：docker 未运行”

## 步骤 2. 验证 API 密钥要求

接下来，您应该确保在继续项目设置之前拥有两个必需的 API 密钥。请妥善保管这些钥匙！

* Tavilly API 密钥：https://tavily.com/
* NVIDIA API 密钥：https://org.ngc.nvidia.com/setup/api-keys 
* 确保此密钥具有“`Public API Endpoints`”权限

保持这两个密钥可供下一步使用。

## 步骤 3. 克隆代理 RAG 项目

然后，您可以将预构建的代理 RAG 项目从 GitHub 克隆到您的 AI Workbench 环境中。

在 AI Workbench 登录页面中，选择 **本地** 位置（如果尚未这样做），然后单击右上角的“克隆项目”。

将此 Git 存储库 URL 粘贴到克隆对话框中：https://github.com/NVIDIA/workbench-example-agentic-rag

单击“克隆”开始克隆和构建过程。

## 步骤 4. 配置项目机密

然后，您可以配置代理 RAG 应用程序正常运行所需的 API 密钥。

项目构建时，使用出现的黄色警告横幅配置 API 密钥：

1. 单击黄色横幅中的“配置”
2. 输入您的“`NVIDIA_API_KEY`”
3. 输入您的“`TAVILY_API_KEY`”
4. 保存配置

等待项目构建完成后再继续。

## 步骤 5. 启动聊天应用程序

您现在可以启动基于 Web 的聊天界面，在其中可以与代理 RAG 系统进行交互。

导航到 **环境** > **项目容器** > **应用程序** > **聊天** 并启动 Web 应用程序。

浏览器窗口将自动打开并加载 Gradio 聊天界面。

## 步骤 6. 测试基本功能

通过提交示例查询来验证代理 RAG 系统是否正常工作。

在聊天应用程序中，单击或键入示例查询，例如：`How do I add an integration in the CLI?`

等待代理系统处理并响应。响应虽然一般，但应该展示智能路由和评估。 

## 步骤 7. 验证项目

通过测试核心功能来确认您的设置正常工作。

验证以下组件是否正常工作：

* Web 应用程序加载无错误
* 示例查询返回响应
* 没有出现 API 身份验证错误
* 代理推理过程在“Monitor”下的界面中可见

## 步骤 8. 完成可选的快速入门

您可以通过上传数据、检索上下文和测试自定义查询来评估高级功能。

**子步骤 A：上传样本数据集**
完成应用内快速入门说明，上传示例数据集并测试改进的基于 RAG 的响应。

**子步骤 B：测试自定义数据集（可选）**
上传自定义数据集，调整路由器提示，然后提交自定义查询以测试自定义。

## 步骤 10. 清理和回滚

如果需要，您可以删除该项目。

> [！警告]
> 这将永久删除该项目和所有关联数据。

要完全删除项目：

1. 在 AI Workbench 中，单击项目旁边的三个点
2. 选择“删除项目”
3. 出现提示时确认删除

> [！笔记]
> 所有更改都包含在 AI Workbench 中。在 AI Workbench 环境之外未进行任何系统级修改。

## 步骤 11. 后续步骤

您还可以使用 Agentic RAG 系统探索更多高级功能和开发选项：

* 修改项目代码中的组件提示
* 上传不同的文档来测试路由和定制
* 尝试不同的查询类型和复杂程度
* 查看“监控”选项卡中的代理推理日志以了解决策

考虑自定义 Gradio UI 或将代理 RAG 组件集成到您自己的项目中。

## 故障排除

| 症状 | 原因 | 使固定 |
|---------|-------|-----|
| 泰维利 API 错误 | 互联网连接或 DNS 问题 | 等待并重试查询 |
| 401 未经授权 | API 密钥错误或格式错误 | 替换 Project Secrets 中的密钥并重新启动 |
| 403 未经授权 | API密钥缺少权限 | 生成具有正确访问权限的新密钥 |
| 代理循环超时 | 复杂查询超出时间限制 | 尝试更简单的查询或重试 |


有关最新的已知问题，请查看 [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)。
