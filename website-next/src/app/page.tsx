import { SiteShell } from "@/components/SiteShell";
import styles from "./page.module.css";

const featuredPlaybooks = [
  {
    title: "连接你的 DGX Spark",
    description: "快速完成本地网络接入、远程访问与基础连接配置。",
    href: "/playbooks/connect-to-your-spark",
  },
  {
    title: "Open WebUI",
    description: "围绕 Ollama 与 Open WebUI 构建可直接使用的本地 AI 交互体验。",
    href: "/playbooks/open-webui",
  },
  {
    title: "vLLM",
    description: "面向高性能推理场景的部署与使用指南，保留原始模型名称与技术术语。",
    href: "/playbooks/vllm",
  },
];

export default function Home() {
  return (
    <SiteShell>
      <section className={styles.hero}>
        <div className={styles.heroGlow} />
        <p className={styles.eyebrow}>DGX Spark 中文社区</p>
        <h1 className={styles.title}>DGX Spark 中文社区</h1>
        <p className={styles.subtitle}>
          不再依赖通用文档站壳子，而是把 DGX Spark 的中文内容、购买入口与高频实践路径整合成一套更像产品站的前端层。
        </p>
        <div className={styles.actions}>
          <a className={styles.primaryAction} href="/intro">
            开始阅读
          </a>
          <a className={styles.secondaryAction} href="#featured-playbooks">
            浏览核心 Playbooks
          </a>
        </div>
      </section>

      <section id="featured-playbooks" className={styles.section}>
        <div className={styles.sectionHeader}>
          <p className={styles.sectionEyebrow}>精选入口</p>
          <h2>从最常用的场景开始</h2>
        </div>
        <div className={styles.cardGrid}>
          {featuredPlaybooks.map((item) => (
            <a key={item.href} className={styles.card} href={item.href}>
              <span className={styles.cardLabel}>Playbook</span>
              <h3>{item.title}</h3>
              <p>{item.description}</p>
              <span className={styles.cardCta}>查看详情</span>
            </a>
          ))}
        </div>
      </section>
    </SiteShell>
  );
}
