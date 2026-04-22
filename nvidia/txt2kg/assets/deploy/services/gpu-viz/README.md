# GPU图形可视化服务

## 🚀 概述

该目录包含可选的 GPU 加速图形可视化服务，这些服务与主 txt2kg 应用程序分开运行。这些服务为大规模图形提供了高级可视化功能。

**注意**：这些服务是**可选的**，不包含在默认的 docker-compose 配置中。它们必须单独运行。

## 📦 可用服务

### 1.统一GPU服务（`unified_gpu_service.py`）
在单个 FastAPI 服务中提供**本地 GPU (cuGraph)** 处理和**本地 CPU** 回退。

**处理模式：**
| 模式 | 描述 | 要求 |
|------|-------------|--------------|
| **本地 GPU (cuGraph)** | 在您的硬件上进行完整的 GPU 处理 | NVIDIA GPU + cuGraph |
| **本地CPU** | NetworkX 回退处理 | 没有任何 |

### 2. 远程GPU渲染服务（`remote_gpu_rendering_service.py`）
通过可嵌入 iframe 的可视化提供 GPU 加速的图形布局和渲染。

### 3.本地GPU服务（`local_gpu_viz_service.py`）
本地 GPU 处理服务，支持 WebSocket 实时更新。

## 🛠️设置

### 先决条件
- 支持 CUDA 的 NVIDIA GPU（适用于 GPU 模式）
- RAPIDS cuGraph（用于本地 GPU 处理）

### 安装

```bash
# Install dependencies
pip install -r deploy/services/gpu-viz/requirements.txt

# For remote WebGPU service
pip install -r deploy/services/gpu-viz/requirements-remote-webgpu.txt
```

### 运行服务

#### 统一GPU服务
```bash
cd deploy/services/gpu-viz
python unified_gpu_service.py
```

服务运行于：http://localhost:8080

#### 远程GPU渲染服务
```bash
cd deploy/services/gpu-viz
python remote_gpu_rendering_service.py
```

服务运行于：http://localhost:8082

#### 使用启动脚本
```bash
cd deploy/services/gpu-viz
./start_remote_gpu_services.sh
```

## 📡 API 使用

### 带有模式选择的流程图

```bash
curl -X POST http://localhost:8080/api/visualize \
  -H "Content-Type: application/json" \
  -d '{
    "graph_data": {
      "nodes": [{"id": "1", "name": "Node 1"}, {"id": "2", "name": "Node 2"}],
      "links": [{"source": "1", "target": "2", "name": "edge_1_2"}]
    },
    "processing_mode": "local_gpu",
    "layout_algorithm": "force_atlas2",
    "clustering_algorithm": "leiden",
    "compute_centrality": true
  }'
```

### 检查可用功能

```bash
curl http://localhost:8080/api/capabilities
```

回复：
```json
{
  "processing_modes": {
    "local_gpu": {"available": true, "description": "..."},
    "local_cpu": {"available": true, "description": "..."}
  },
  "has_rapids": true,
  "gpu_available": true
}
```

## 🎯 前端集成

txt2kg 前端包含用于 GPU 可视化的内置组件：

- `UnifiedGPUViewer`：连接到统一GPU服务
- `ForceGraphWrapper`：Three.js WebGPU 可视化（默认）

### 在前端使用 GPU 服务

前端具有可以连接到这些服务的 API 路由：
- `/api/unified-gpu/*`：统一GPU服务集成

要使用这些服务，请确保它们单独运行并相应地配置前端环境变量。

### 特定于模式的处理

```javascript
// Local GPU mode
const response = await fetch('/api/unified-gpu/visualize', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    graph_data: { nodes, links },
    processing_mode: 'local_gpu',
    layout_algorithm: 'force_atlas2',
    clustering_algorithm: 'leiden',
    compute_centrality: true
  })
})
```

## 🔧 配置选项

### 本地GPU模式  
- `layout_algorithm`：“force_atlas2”、“光谱”、“fruchterman_reingold”
- `clustering_algorithm`：“莱顿”、“鲁汶”、“光谱”
- `compute_centrality`：真/假

### 本地CPU模式
- 使用 NetworkX 回退进行基本处理
- 无需额外配置

## 📊 回复格式

```json
{
  "processed_nodes": [...],
  "processed_edges": [...],
  "processing_mode": "local_gpu",
  "layout_positions": {...}, // Only for local GPU mode
  "clusters": {...},
  "centrality": {...},
  "stats": {
    "node_count": 1000,
    "edge_count": 5000,
    "gpu_accelerated": true,
    "layout_computed": true,
    "clusters_computed": true
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## 🚀 统一方法的好处

### ✅ 优点
- **单一服务** - 一个端口，一次部署
- **模式切换** - 选择每个图表的最佳处理
- **回退处理** - 如果 GPU 不可用则优雅降级  
- **一致的 API** - 所有模式均具有相同的界面
- **更好的测试** - 模式之间的轻松比较

### 🎯 使用案例
- **本地GPU**：私有数据、大规模处理、自定义算法
- **本地CPU**：开发、测试、小图

## 🐛 故障排除

### 未检测到 GPU
```bash
# Check GPU availability
nvidia-smi

# Check RAPIDS installation
python -c "import cudf, cugraph; print('RAPIDS OK')"
```

### 服务健康
```bash
curl http://localhost:8080/api/health
```

## 📈 性能技巧

1. **大型图（>100k 节点）**：使用 `local_gpu` 模式
2. **开发**：使用 `local_cpu` 模式提高速度
3. **混合工作负载**：根据图形大小动态切换模式 
