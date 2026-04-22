# 前端

Next.js React 应用程序为聊天机器人演示提供用户界面。

## 概述

前端提供聊天界面，支持：
- 多模型对话
- 文档上传和 RAG（检索增强生成）
- 图像处理能力
- 通过 WebSocket 进行实时流响应
- 主题切换（亮/暗模式）
- 模型和数据源的侧边栏配置

## 关键部件

- **QuerySection**：主要聊天界面，具有消息显示和输入
- **侧边栏**：模型、来源和聊天历史记录的配置面板
- **DocumentIngestion**：RAG 功能的文件上传接口
- **WelcomeSection**：带有快速启动模板的登陆页面
- **ThemeToggle**：暗/亮模式切换器

## 建筑学

使用 Next.js 14、TypeScript 和 CSS 模块构建。通过 REST API 和 WebSocket 连接与后端进行通信，以实现实时聊天流。

## Docker 故障排除

### 容器问题
- **端口冲突**：确保端口 3000 未被其他应用程序使用
- **构建失败**：使用 `docker system prune -a` 清除 Docker 缓存
- **热重载不起作用**：重新启动容器或检查卷安装

### 常用命令
```bash
# View frontend logs
docker logs frontend

# Restart frontend container
docker restart frontend

# Rebuild frontend
docker-compose up --build -d frontend

# Access container shell
docker exec -it frontend /bin/sh
```

### 性能问题
- 检查可用内存：`docker stats`
- 在 Docker Desktop 设置中增加 Docker 内存分配
- 清除 localhost:3000 的浏览器缓存和 cookie
