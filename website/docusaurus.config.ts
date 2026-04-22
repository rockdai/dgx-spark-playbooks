import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'DGX Spark 中文手册',
  tagline: '面向中文社区的 DGX Spark Playbooks 在线文档站',
  favicon: 'img/favicon.ico',
  future: {
    v4: true,
  },
  url: 'https://example.com',
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
    },
    navbar: {
      title: 'DGX Spark 中文手册',
      logo: {
        alt: 'DGX Spark 中文手册',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: '文档',
        },
        {
          href: 'https://github.com/rockdai/dgx-spark-playbooks-cn',
          label: 'GitHub',
          position: 'right',
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
              label: 'GitHub 仓库',
              href: 'https://github.com/rockdai/dgx-spark-playbooks-cn',
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
      copyright: `Copyright © ${new Date().getFullYear()} DGX Spark 中文手册社区项目。Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
