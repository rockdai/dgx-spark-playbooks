import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link className="button button--secondary button--lg" to="/intro">
            开始阅读
          </Link>
          <Link
            className="button button--outline button--secondary button--lg"
            to="/playbooks/vllm/">
            查看 Playbooks
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="DGX Spark 中文手册在线文档站，基于官方 DGX Spark Playbooks 的社区中文翻译。">
      <HomepageHeader />
      <main className="container margin-vert--lg">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <Heading as="h2">关于这个网站</Heading>
            <p>
              这是基于官方 DGX Spark Playbooks 搭建的中文在线文档站，用来提供更适合中文社区阅读、检索和长期维护的访问方式。
            </p>
            <p>
              本项目是社区驱动的中文翻译与整理版本，与 NVIDIA 公司无隶属、无背书、无官方维护关系。
            </p>
            <Heading as="h2">当前骨架能力</Heading>
            <ul>
              <li>已接入 Docusaurus 基础站点结构</li>
              <li>已将仓库中的 README 导入 docs 路由</li>
              <li>已具备本地开发、构建和后续部署基础</li>
            </ul>
            <Heading as="h2">下一步建议</Heading>
            <ul>
              <li>优化文档标题和 sidebar 分类</li>
              <li>修正 README 内部锚点与跨文档链接</li>
              <li>补充首页、免责声明、贡献指南和部署配置</li>
            </ul>
          </div>
        </div>
      </main>
    </Layout>
  );
}
