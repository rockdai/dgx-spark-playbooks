<p align="center">
  <img src="src/images/dgx-spark-banner.png" alt="NVIDIA DGX Spark"/>
</p>

# DGX Spark 手册中文版

这是一个面向中文读者整理的 **DGX Spark Playbooks 中文翻译项目**，旨在帮助开发者、研究者、工程团队与 AI 爱好者更高效地理解和使用基于 NVIDIA Blackwell 架构的 DGX Spark。

本仓库基于原始 DGX Spark Playbooks 进行中文化整理，尽量保留原项目的结构、命令、代码块和操作路径，同时将说明文字、本地部署步骤与实践背景翻译为更自然、可读的中文表述，方便中文社区直接参考、学习与二次传播。

> **Community Notice**
>
> This repository is a community-driven Chinese translation based on the official DGX Spark Playbooks. It is made by the community and love, and is not affiliated with, endorsed by, or maintained by NVIDIA.

## 关于本项目

本项目主要提供以下内容：
- 常用 AI 框架与工具链的安装与配置指南
- 面向 DGX Spark 的推理、微调与开发环境说明
- 多机互联、网络访问、系统连接等基础操作指引
- 常见工作负载与示例项目的中文化参考文档

每个 playbook 通常包含前置条件、分步操作、故障排查建议以及示例代码。为了尽量保持与上游内容同步，仓库中保留了原始项目的目录结构与技术术语，并仅在适合的地方做必要的中文翻译与表达优化。

## 可用 Playbooks

### NVIDIA

- [Comfy UI](nvidia/comfy-ui/)
- [以环形拓扑连接三台 DGX Spark](nvidia/connect-three-sparks/)
- [配置本地网络访问](nvidia/connect-to-your-spark/)
- [连接两台 Spark](nvidia/connect-two-sparks/)
- [CUDA-X Data Science](nvidia/cuda-x-data-science/)
- [DGX Dashboard](nvidia/dgx-dashboard/)
- [FLUX.1 Dreambooth LoRA 微调](nvidia/flux-finetuning/)
- [安装并使用 Isaac Sim 与 Isaac Lab](nvidia/isaac/)
- [优化版 JAX](nvidia/jax/)
- [Live VLM WebUI](nvidia/live-vlm-webui/)
- [在 DGX Spark 上使用 llama.cpp 运行模型](nvidia/llama-cpp/)
- [LLaMA Factory](nvidia/llama-factory/)
- [在 DGX Spark 上使用 LM Studio](nvidia/lm-studio/)
- [构建并部署多智能体聊天机器人](nvidia/multi-agent-chatbot/)
- [多模态推理](nvidia/multi-modal-inference/)
- [通过交换机连接多台 DGX Spark](nvidia/multi-sparks-through-switch/)
- [双 Spark 的 NCCL 配置](nvidia/nccl/)
- [使用 NeMo 微调](nvidia/nemo-fine-tune/)
- [在 DGX Spark 上使用 NemoClaw、Nemotron 3 Super 与 Telegram](nvidia/nemoclaw/)
- [使用 llama.cpp 运行 Nemotron-3-Nano](nvidia/nemotron/)
- [在 Spark 上运行 NIM](nvidia/nim-llm/)
- [NVFP4 量化](nvidia/nvfp4-quantization/)
- [Ollama](nvidia/ollama/)
- [结合 Ollama 使用 Open WebUI](nvidia/open-webui/)
- [OpenClaw 🦞](nvidia/openclaw/)
- [在 DGX Spark 上使用 OpenShell 保护长期运行的 AI 智能体](nvidia/openshell/)
- [投资组合优化](nvidia/portfolio-optimization/)
- [使用 PyTorch 微调](nvidia/pytorch-fine-tune/)
- [在 AI Workbench 中构建 RAG 应用](nvidia/rag-ai-workbench/)
- [使用 SGLang 进行推理](nvidia/sglang/)
- [单细胞 RNA 测序](nvidia/single-cell/)
- [Spark 与 Reachy 拍照亭](nvidia/spark-reachy-photo-booth/)
- [投机采样](nvidia/speculative-decoding/)
- [在 Spark 上配置 Tailscale](nvidia/tailscale/)
- [使用 TRT LLM 进行推理](nvidia/trt-llm/)
- [在 DGX Spark 上从文本构建知识图谱](nvidia/txt2kg/)
- [在 DGX Spark 上使用 Unsloth](nvidia/unsloth/)
- [在 VS Code 中进行 Vibe Coding](nvidia/vibe-coding/)
- [使用 vLLM 进行推理](nvidia/vllm/)
- [VS Code](nvidia/vscode/)
- [构建视频搜索与摘要（VSS）智能体](nvidia/vss/)

## 资源

- **原始项目仓库**：https://github.com/NVIDIA/dgx-spark-playbooks
- **官方文档**：https://www.nvidia.com/en-us/products/workstations/dgx-spark/
- **开发者论坛**：https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10
- **服务条款**：https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf

## 许可证

请参阅：
- [LICENSE](LICENSE)了解本项目许可信息
- [LICENSE-3rd-party](LICENSE-3rd-party) 了解第三方许可信息
