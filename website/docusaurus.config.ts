import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'DGX Spark 中文社区',
  tagline: '面向中文社区的 DGX Spark 在线文档与实践入口',
  future: {
    v4: true,
  },
  url: 'https://dgx-spark.ai',
  baseUrl: '/',
  organizationName: 'rockdai',
  projectName: 'dgx-spark-playbooks-cn',
  onBrokenLinks: 'warn',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
      onBrokenMarkdownImages: 'warn',
    },
  },
  i18n: {
    defaultLocale: 'zh-Hans',
    locales: ['zh-Hans'],
  },
  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: '/',
          editUrl: 'https://github.com/rockdai/dgx-spark-playbooks-cn/tree/main/website/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],
  themeConfig: {
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
      disableSwitch: true,
      defaultMode: 'dark',
    },
    navbar: {
      title: 'DGX Spark 中文手册',
      hideOnScroll: true,
      items: [
        {
          to: '/intro',
          label: '开始',
          position: 'left',
        },
        {
          to: '/playbooks/connect-to-your-spark/',
          label: '连接',
          position: 'left',
        },
        {
          to: '/playbooks/vllm/',
          label: '推理',
          position: 'left',
        },
        {
          to: '/playbooks/pytorch-fine-tune/',
          label: '微调',
          position: 'left',
        },
        {
          to: '/playbooks/openclaw/',
          label: '应用',
          position: 'left',
        },
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: '全部文档',
        },
        {
          href: 'https://common-buy.aliyun.com/?commodityCode=datav_spark_public_cn',
          label: '立即购买',
          position: 'right',
          className: 'navbar__buyButton',
          target: '_blank',
          rel: 'noopener noreferrer',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: '文档',
          items: [
            {
              label: '首页',
              to: '/',
            },
          ],
        },
        {
          title: '项目',
          items: [
            {
              label: '立即购买',
              href: 'https://common-buy.aliyun.com/?commodityCode=datav_spark_public_cn',
            },
            {
              label: 'DataV.AI',
              href: 'https://ai.datav.run',
            },
            {
              label: '官方 DGX Spark Playbooks',
              href: 'https://github.com/NVIDIA/dgx-spark-playbooks',
            },
          ],
        },
        {
          title: '声明',
          items: [
            {
              label: 'Community Notice',
              href: 'https://github.com/rockdai/dgx-spark-playbooks-cn#dgx-spark-手册中文版',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} DGX Spark 中文社区。Community-maintained, built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.vsDark,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
