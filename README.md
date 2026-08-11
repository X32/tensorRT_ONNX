# Jetson TensorRT 部署实战

在 NVIDIA Jetson Orin 上部署 ResNet18，通过 TensorRT 优化实现相比 ONNX Runtime **7~30 倍** 的推理加速。本项目覆盖从 **环境搭建 → 模型转换 → 引擎生成 → 推理实现 → 性能对比** 的完整流程。

---

## 🚀 核心成果

| 推理方式 | 平均延迟 | 吞吐量 | vs ONNX Runtime | 模型大小 |
|----------|----------|--------|-----------------|----------|
| ONNX Runtime | 39.6 ms | 25 QPS | 1.0x（基准） | 92 MB |
| **TensorRT FP32** | 5.4 ms | 187 QPS | **7.4x** | 45 MB |
| **TensorRT FP16** | 2.7 ms | 374 QPS | **15x** ⭐ | 23 MB |
| **TensorRT INT8** | 1.3 ms | 748 QPS | **30x** | 12 MB |

> 各精度版本预测结果完全一致，量化无精度损失。**生产环境推荐 FP16**，边缘极限场景用 INT8。

---

## 🛠️ 环境信息

| 项目 | 版本 |
|------|------|
| 设备 | NVIDIA Jetson Orin（ARM64） |
| 系统 | Ubuntu 22.04 + JetPack 6.x（L4T 36.5） |
| CUDA | 12.6 |
| TensorRT | 10.3.0 |
| PyTorch | 2.x（jetson-containers 镜像内置） |
| 部署方式 | Docker 容器（基于 `dustynv/jetson-containers`） |

---

## 📁 项目结构

```
tensorRT/
├── onnx_test.ipynb              # 主测试：多精度性能对比
├── video_speed_test.ipynb       # 视频推理速度对比（ONNX vs TRT）
├── src/                         # 源代码（按功能模块化）
│   ├── inference/               #   TensorRT 推理实现
│   ├── calibration/             #   INT8 校准脚本
│   ├── export/                  #   模型导出（PyTorch→ONNX）
│   ├── test/                    #   测试与验证脚本
│   └── utils/                   #   辅助工具
├── DOC/                         # 完整文档与技术博客
├── resnet18.onnx                # ONNX 模型（已提交）
└── *.engine                     # TensorRT 引擎（.gitignore 排除）
```

> 详细说明见 [src/README.md](src/README.md)

---

## ⚡ 快速开始

### 1. 启动 Docker 容器

```bash
docker run -d --name jupyter-tensorrt --runtime=nvidia \
    -v "$(pwd)":/workspace -w /workspace -p 8889:8888 \
    jupyter-tensorrt:complete
```

### 2. 模型导出（PyTorch → ONNX）

```bash
python src/export/export_resnet.py      # 输出 resnet18.onnx
```

### 3. 生成 TensorRT 引擎

```bash
# FP32 / FP16 / INT8
trtexec --onnx=resnet18.onnx --saveEngine=resnet18.engine
trtexec --onnx=resnet18.onnx --saveEngine=resnet18_fp16.engine --fp16
trtexec --onnx=resnet18.onnx --saveEngine=resnet18_int8.engine --int8
```

### 4. 执行推理

```python
from src.inference.tensorrt_inference_fixed import TensorRTInference
import numpy as np

inferencer = TensorRTInference("resnet18_fp16.engine")
result = inferencer.infer(np.random.randn(1, 3, 224, 224).astype(np.float32))
print(result.shape)   # (1, 1000)
```

或直接打开 notebook 运行完整对比实验：
- `onnx_test.ipynb` — 单图多精度性能对比
- `video_speed_test.ipynb` — 视频逐帧推理速度对比

---

## 📚 文档导航

| 文档 | 说明 |
|------|------|
| 📖 [技术博客（完整版）](DOC/TECH_BLOG_Jetson_TensorRT_Complete_Guide.md) | 环境搭建 → 模型转换 → 引擎生成 → 推理 → 部署全流程 |
| 📄 [技术博客（精简版）](DOC/TECH_BLOG_CONCISE.md) | 核心要点与性能数据，5 分钟速读 |
| 📊 [性能测试报告](DOC/PERFORMANCE_REPORT.md) | 多精度详细测试数据与分析 |
| 🐳 [镜像环境配置报告](DOC/TensorRT_镜像环境配置完整报告.md) | Docker 镜像构建与故障排除 |

---

## 🔑 关键技术点

- **Docker 容器化**：基于 `jetson-containers`，环境可重复、易部署
- **TensorRT 优化**：图融合、内核自动调优、Tensor Core 加速
- **多精度推理**：FP32/FP16/INT8，按场景选择
- **TensorRT 10.3 API 兼容**：使用 `num_io_tensors` / `get_tensor_name` / `execute_v2` 新 API

---

**测试设备**：NVIDIA Jetson Orin ｜ **TensorRT**：10.3.0 ｜ **更新日期**：2026-08-11
