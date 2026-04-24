import { notFound } from "next/navigation";
import { MarkdownContent } from "@/components/MarkdownContent";
import { SiteShell } from "@/components/SiteShell";
import { getPlaybook, listPlaybooks } from "@/lib/content";
import { playbookGroups } from "@/lib/playbook-groups";
import styles from "./page.module.css";

type Props = {
  params: Promise<{ slug: string }>;
};

export async function generateStaticParams() {
  return listPlaybooks().map((item) => ({ slug: item.slug }));
}

export default async function PlaybookPage({ params }: Props) {
  const { slug } = await params;
  const playbook = getPlaybook(slug);
  if (!playbook) {
    notFound();
  }

  return (
    <SiteShell>
      <section className={styles.hero}>
        <div className={styles.heroPanel}>
          <p className={styles.eyebrow}>Playbook</p>
          <h1>{playbook.title}</h1>
          {playbook.description ? <p className={styles.description}>{playbook.description}</p> : null}
          <div className={styles.metaRow}>
            <span>统一模板</span>
            <span>Markdown 驱动</span>
            <span>/{slug}</span>
          </div>
        </div>
      </section>

      <section className={styles.layout}>
        <aside className={styles.sidebar}>
          <div className={styles.sidebarCard}>
            <a className={styles.viewAll} href="/intro">
              ← View All Playbooks
            </a>
            <div className={styles.sidebarGroups}>
              {playbookGroups.map((group) => (
                <section key={group.id} className={styles.groupSection}>
                  <p className={styles.groupTitle}>
                    <span className={styles.groupIcon}>{group.icon}</span>
                    {group.label}
                  </p>
                  <nav className={styles.groupNav}>
                    {group.items.map((item) => (
                      <a
                        key={item.slug}
                        href={`/playbooks/${item.slug}`}
                        className={item.slug === slug ? styles.activeLink : ""}
                      >
                        {item.title}
                      </a>
                    ))}
                  </nav>
                </section>
              ))}
            </div>
          </div>
        </aside>

        <article className={styles.article}>
          <MarkdownContent content={playbook.body} />
        </article>
      </section>
    </SiteShell>
  );
}
