# 复杂道路场景多目标识别系统

本项目提供一个可直接集成与部署的**复杂道路场景多目标识别系统**，支持以下 10 类目标：

- person
- rider
- car
- bus
- truck
- bike
- motorcycle
- traffic light
- traffic sign
- train

系统包含：

- **后端推理服务**（FastAPI）：支持单图、批处理与 WebSocket 流式输入。
- **前端实时展示**（Web）：实时视频、目标框、类别、置信度、统计仪表盘。
- **日志与监控**：记录推理延迟、FPS、类别分布，便于监控与模型迭代。

---

## 1. 项目结构

```text
.
├── backend
│   ├── app.py          # FastAPI 服务入口
│   ├── detector.py     # 检测器封装（Ultralytics + Dummy fallback）
│   ├── monitor.py      # 推理监控与日志记录
│   └── schemas.py      # API 数据结构
├── frontend
│   └── index.html      # 实时展示页面
├── logs
│   └── inference.log   # 推理日志（运行时生成）
└── requirements.txt
```

---

## 2. 快速启动

### 2.1 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2.2 启动服务

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

访问：

- 前端页面：`http://127.0.0.1:8000/`
- API 文档：`http://127.0.0.1:8000/docs`

---

## 3. 模型接入方式

默认使用 `DummyRoadDetector` 进行端到端联调。

如果你已训练完成模型（推荐 Ultralytics YOLO 权重），可通过环境变量加载：

```bash
export MODEL_PATH=/path/to/your/best.pt
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

> `DetectorFactory` 会优先尝试加载真实模型；加载失败时自动回退到 Dummy 检测器，保证服务可用。

---

## 4. API 说明

### 4.1 单图推理

- `POST /infer/image`
- form-data：`file`（图片文件）

返回：检测框、类别、置信度、类别分布、延迟。

### 4.2 批处理推理

- `POST /infer/batch`
- form-data：`files`（多张图片）

返回：每张图检测结果 + 批次平均延迟。

### 4.3 流式推理

- `WS /ws/stream`
- 前端按帧发送 base64 JPEG，后端逐帧返回检测结果和监控指标。

### 4.4 指标查询

- `GET /metrics`
- 返回平均延迟、平均 FPS、累计帧数、检测类别分布。

---

## 5. 监控与日志

- 每帧日志写入 `logs/inference.log`（JSON Line 格式）。
- 记录字段：`timestamp`、`latency_ms`、`labels`、`frame_count`。
- 可直接接入 ELK / Loki / Prometheus exporter 进行生产化监控。

---

## 6. 前端展示能力

- 实时摄像头视频播放
- 目标 bounding box 绘制
- 类别 + 置信度叠加
- Latency / FPS / Total Frames / Detections 看板
- 类别分布实时刷新

---

## 7. 部署建议

- 用 **Gunicorn + UvicornWorker** 部署多进程。
- 将 WebSocket 与推理服务置于同域，避免跨域与代理升级问题。
- 建议将推理服务与前端通过 Nginx 统一入口。
- GPU 环境下，可将模型实例化放到应用启动阶段并做预热推理。

