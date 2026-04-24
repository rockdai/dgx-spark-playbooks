export type PlaybookGroup = {
  id: string;
  label: string;
  items: { slug: string; title: string }[];
};

export const playbookGroups: PlaybookGroup[] = [
  {
    id: "onboarding",
    label: "Onboarding",
    items: [
      { slug: "connect-to-your-spark", title: "配置本地网络访问" },
      { slug: "open-webui", title: "结合 Ollama 使用 Open WebUI" },
    ],
  },
  {
    id: "data-science",
    label: "Data Science",
    items: [
      { slug: "single-cell", title: "单细胞 RNA 测序" },
      { slug: "portfolio-optimization", title: "投资组合优化" },
      { slug: "cuda-x-data-science", title: "CUDA-X Data Science" },
      { slug: "txt2kg", title: "在 DGX Spark 上从文本构建知识图谱" },
      { slug: "jax", title: "优化版 JAX" },
    ],
  },
  {
    id: "tools",
    label: "Tools",
    items: [
      { slug: "dgx-dashboard", title: "DGX Dashboard" },
      { slug: "comfy-ui", title: "Comfy UI" },
      { slug: "connect-three-sparks", title: "以环形拓扑连接三台 DGX Spark" },
      { slug: "multi-sparks-through-switch", title: "通过交换机连接多台 DGX Spark" },
      { slug: "rag-ai-workbench", title: "在 AI Workbench 中构建 RAG 应用" },
      { slug: "tailscale", title: "在 Spark 上配置 Tailscale" },
      { slug: "vscode", title: "VS Code" },
    ],
  },
  {
    id: "fine-tuning",
    label: "Fine Tuning",
    items: [
      { slug: "flux-finetuning", title: "FLUX.1 Dreambooth LoRA 微调" },
      { slug: "llama-factory", title: "LLaMA Factory" },
      { slug: "nemo-fine-tune", title: "使用 NeMo 微调" },
      { slug: "pytorch-fine-tune", title: "使用 PyTorch 微调" },
      { slug: "unsloth", title: "在 DGX Spark 上使用 Unsloth" },
    ],
  },
  {
    id: "use-case",
    label: "Use Case",
    items: [
      { slug: "nemoclaw", title: "在 DGX Spark 上使用 NemoClaw、Nemotron 3 Super 与 Telegram" },
      { slug: "openshell", title: "在 DGX Spark 上使用 OpenShell 保护长期运行的 AI 智能体" },
      { slug: "openclaw", title: "OpenClaw 🦞" },
      { slug: "live-vlm-webui", title: "Live VLM WebUI" },
      { slug: "isaac", title: "安装并使用 Isaac Sim 与 Isaac Lab" },
      { slug: "vibe-coding", title: "在 VS Code 中进行 Vibe Coding" },
      { slug: "multi-agent-chatbot", title: "构建并部署多智能体聊天机器人" },
      { slug: "connect-two-sparks", title: "连接两台 Spark" },
      { slug: "nccl", title: "双 Spark 的 NCCL 配置" },
      { slug: "vss", title: "构建视频搜索与摘要（VSS）智能体" },
      { slug: "spark-reachy-photo-booth", title: "Spark 与 Reachy 拍照亭" },
    ],
  },
  {
    id: "inference",
    label: "Inference",
    items: [
      { slug: "speculative-decoding", title: "投机采样" },
      { slug: "llama-cpp", title: "在 DGX Spark 上使用 llama.cpp 运行模型" },
      { slug: "vllm", title: "使用 vLLM 进行推理" },
      { slug: "nemotron", title: "使用 llama.cpp 运行 Nemotron-3-Nano" },
      { slug: "sglang", title: "使用 SGLang 进行推理" },
      { slug: "trt-llm", title: "使用 TRT LLM 进行推理" },
      { slug: "nvfp4-quantization", title: "NVFP4 量化" },
      { slug: "multi-modal-inference", title: "多模态推理" },
      { slug: "nim-llm", title: "在 Spark 上运行 NIM" },
      { slug: "lm-studio", title: "在 DGX Spark 上使用 LM Studio" },
    ],
  },
];
