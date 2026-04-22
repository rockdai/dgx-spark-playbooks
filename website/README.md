# DGX Spark 中文手册网站说明

这个目录包含基于 **Docusaurus** 搭建的文档网站骨架，用于在线展示 `dgx-spark-playbooks-cn` 中文手册。

## 目录说明

- `website/`：Docusaurus 站点工程
- `website/docs/`：导入后的文档内容
- `website/build/`：构建后的静态站点输出目录

## 本地开发

在 `website/` 目录下运行：

```bash
npm install
npm start
```

默认会启动本地开发服务器，方便预览和调试。

## 生产构建

```bash
npm install
npm run build
```

构建完成后，静态文件会输出到：

```bash
website/build
```

## 阿里云 ESA Pages 部署建议

从当前项目结构来看，这个站点适合以 **静态站点** 的方式部署到阿里云 **边缘安全加速 ESA Pages**。

### 推荐参数

如果 ESA Pages 支持指定子目录项目，建议使用：

- **Root Directory**: `website`
- **Install Command**: `npm install`
- **Build Command**: `npm run build`
- **Output Directory**: `build`

## Docusaurus 配置建议

当前 `docusaurus.config.ts` 已调整为更适合通用 Pages 平台的形式：

```ts
url: 'https://example.com',
baseUrl: '/',
```

部署时请按实际访问域名修改：

### 如果绑定独立域名并部署在根路径

```ts
url: 'https://docs.example.com',
baseUrl: '/',
```

### 如果部署在某个子路径下

例如：

```ts
url: 'https://example.com',
baseUrl: '/dgx-spark-playbooks-cn/',
```

## 当前已知问题

当前站点已经可以成功构建，但仍有一些待优化项：

- 部分 Markdown 链接仍沿用原始 README 结构，可能产生 broken links
- 一些目录锚点仍然指向英文标题 slug，需要进一步修正
- 少量文档中仍存在可继续清理的 HTML/Markdown 历史遗留写法

这些问题不会阻止站点构建，但会影响部分页面间跳转体验。

## 项目说明

本网站基于官方 DGX Spark Playbooks 的中文翻译内容构建，属于社区驱动项目，与 NVIDIA 公司无隶属、无背书、无官方维护关系。
