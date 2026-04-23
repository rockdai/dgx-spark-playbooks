import styles from "./site-shell.module.css";
import { TopBar } from "./TopBar";

type Props = {
  children: React.ReactNode;
};

export function SiteShell({ children }: Props) {
  return (
    <div className={styles.page}>
      <TopBar />
      <main className={styles.main}>{children}</main>
      <footer className={styles.footer}>
        <div className={styles.footerLinks}>
          <a href="https://ai.datav.run" target="_blank" rel="noreferrer">
            DataV.AI
          </a>
          <a href="https://github.com/NVIDIA/dgx-spark-playbooks" target="_blank" rel="noreferrer">
            官方 DGX Spark Playbooks
          </a>
        </div>
        <p className={styles.notice}>
          Community Notice: This website is a community-driven Chinese translation based on the official DGX Spark Playbooks. It is made by the community and love, and is not affiliated with, endorsed by, or maintained by NVIDIA.
        </p>
      </footer>
    </div>
  );
}
