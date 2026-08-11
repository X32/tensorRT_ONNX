# TensorRT 引擎使用指南

## 🎯 快速开始

### 1. INT8 引擎（最高性能）

```bash
# 使用 trtexec 进行推理
trtexec --loadEngine=resnet18_int8.engine

# 性能测试
trtexec --loadEngine=resnet18_int8.engine --duration=60

# 批处理推理
trtexec --loadEngine=resnet18_int8.engine --batch=4
```

**性能指标**：
- 吞吐量：2,112 QPS
- 延迟：0.51ms
- 文件大小：12MB

### 2. FP16 引擎（推荐生产环境）

```bash
# 使用 trtexec 进行推理
trtexec --loadEngine=resnet18_fp16.engine

# 性能测试
trtexec --loadEngine=resnet18_fp16.engine --duration=60

# 批处理推理
trtexec --loadEngine=resnet18_fp16.engine --batch=4
```

**性能指标**：
- 吞吐量：1,294 QPS
- 延迟：0.82ms
- 文件大小：23MB

---

## 📊 性能对比

| 引擎 | 精度 | 吞吐量 | 延迟 | 大小 | 推荐用途 |
|------|------|--------|------|------|----------|
| FP16 | 半精度 | 1,294 QPS | 0.82ms | 23MB | 生产环境 ⭐ |
| INT8 | 量化 | 2,112 QPS | 0.51ms | 12MB | 极限性能 🚀 |

---

## 🔧 常用命令

### 基础推理
```bash
trtexec --loadEngine=<engine_file>
```

### 性能测试
```bash
trtexec --loadEngine=<engine_file> --duration=60
```

### 批处理
```bash
trtexec --loadEngine=<engine_file> --batch=4
```

### 详细性能分析
```bash
trtexec --loadEngine=<engine_file> --verbose --duration=30
```

### 导出性能报告
```bash
trtexec --loadEngine=<engine_file> --exportTimes=performance.json
```

---

## 💡 实用技巧

### 1. 提高稳定性
```bash
trtexec --loadEngine=resnet18_int8.engine --useSpinWait
```

### 2. 设置运行时长
```bash
# 运行 10 秒
trtexec --loadEngine=resnet18_int8.engine --duration=10

# 运行 5 分钟
trtexec --loadEngine=resnet18_int8.engine --duration=300
```

### 3. 设置迭代次数
```bash
trtexec --loadEngine=resnet18_int8.engine --iterations=100
```

### 4. 查看详细信息
```bash
trtexec --loadEngine=resnet18_int8.engine --verbose
```

---

## 🚨 常见问题

### Q: 为什么不推荐使用 calibrate_resnet.py？

**A**: 因为：
1. ✅ trtexec 已自动完成 INT8 校准
2. ❌ pycuda 安装复杂且容易失败
3. ✅ trtexec 性能更好更稳定
4. ✅ 引擎已构建成功，可直接使用

### Q: 如何选择 FP16 还是 INT8？

**A**:
- **FP16**: 精度要求高，推荐生产环境
- **INT8**: 性能要求高，内存受限场景

### Q: 引擎可以在不同设备上使用吗？

**A**: 不可以。引擎是设备特定的，需要在同一架构设备上重新构建。

---

## 📁 文件说明

### 引擎文件
- `resnet18_int8.engine` - INT8 量化引擎
- `resnet18_fp16.engine` - FP16 半精度引擎

### 验证脚本
- `test_tensorrt.py` - 环境测试
- `verify_onnx.py` - ONNX 验证
- `int8_inference.py` - Python 推理示例

### 文档
- `PERFORMANCE_REPORT.md` - 性能对比报告
- `TensorRT_环境配置报告.md` - 完整技术文档

---

## 🎉 总结

你现在拥有**两个优化好的 TensorRT 引擎**：

1. **INT8 引擎** (2,112 QPS) - 极限性能
2. **FP16 引擎** (1,294 QPS) - 生产推荐

**推荐命令**：
```bash
# 极限性能
trtexec --loadEngine=resnet18_int8.engine

# 生产推荐
trtexec --loadEngine=resnet18_fp16.engine
```

**🚀 开始使用 TensorRT 高性能推理！**
