# TensorRT 项目源代码结构

## 📁 文件组织说明

本项目源代码已按功能模块化整理，便于维护和扩展。

### 🗂️ 目录结构

```
src/
├── inference/          # 推理相关脚本
├── calibration/        # INT8 校准脚本
├── export/            # 模型导出脚本
├── test/              # 测试和验证脚本
└── utils/             # 工具和辅助脚本
```

## 📂 各文件夹详细说明

### 🔧 inference/ - 推理引擎
**功能**: TensorRT 模型推理的核心实现

**文件**:
- `tensorrt_inference_fixed.py` - TensorRT 10.3.0 兼容的推理类
  - 支持 FP32/FP16/INT8 多精度推理
  - 自动 API 兼容性检测
  - GPU 内存管理优化
  
- `int8_inference.py` - INT8 专用推理实现
  - INT8 量化推理
  - 性能优化配置

### 🎯 calibration/ - 模型校准
**功能**: INT8 量化校准数据生成

**文件**:
- `calibrate_resnet.py` - 完整的 INT8 校准流程
  - 使用 ImageNet 数据集进行校准
  - 生成校准缓存文件
  
- `calibrate_resnet_simple.py` - 简化版校准脚本
  - 快速校准流程
  - 适合测试和开发

### 📤 export/ - 模型导出
**功能**: PyTorch/ONNX 模型导出为 TensorRT Engine

**文件**:
- `export_resnet.py` - ResNet18 模型导出
  - PyTorch → ONNX → TensorRT Engine
  - 支持多精度 (FP32/FP16/INT8)
  - 包含完整的预处理流程

### 🧪 test/ - 测试验证
**功能**: 模型测试和验证脚本

**文件**:
- `test_engine.py` - TensorRT Engine API 诊断
  - 检测 TensorRT 版本兼容性
  - 验证引擎加载和执行
  
- `get_tensor_names.py` - 获取引擎 tensor 信息
  - 输入输出 tensor 名称和形状
  - 数据类型和模式检查
  
- `verify_onnx.py` - ONNX 模型验证
  - 检查 ONNX 模型完整性
  - 验证输入输出规格
  
- `test_tensorrt.py` - TensorRT 基础测试
  - 简单的推理验证
  
- `test_trtpy.py` - TensorRT Python API 测试
  - API 功能验证
  
- `dev_test.py` - 开发调试测试
  - 快速功能验证
  
- `teset_ONNX.py` - ONNX 测试脚本

### 🛠️ utils/ - 工具脚本
**功能**: 辅助工具和实用脚本

**文件**:
- `convert_to_doc.py` - 文档格式转换
  - 将 Markdown 转换为其他格式
  
- `simple_network.py` - 简单网络示例
  - 基础网络结构示例
  
- `trtpy_guide.py` - TensorRT Python 指南
  - API 使用指南和示例

## 🚀 使用示例

### 推理使用
```python
from src.inference.tensorrt_inference_fixed import TensorRTInference

# 创建推理器
inferencer = TensorRTInference("resnet18_fp16.engine")

# 执行推理
result = inferencer.infer(input_data)
```

### 模型导出
```bash
python src/export/export_resnet.py
```

### INT8 校准
```bash
python src/calibration/calibrate_resnet_simple.py
```

### 测试验证
```bash
# 检查引擎兼容性
python src/test/test_engine.py

# 获取 tensor 信息
python src/test/get_tensor_names.py
```

## 📋 开发流程

1. **模型准备**: 使用 `export/` 中的脚本导出模型
2. **校准** (可选): 使用 `calibration/` 进行 INT8 校准
3. **测试**: 使用 `test/` 中的脚本验证模型
4. **推理**: 使用 `inference/` 中的类进行推理

## 🔧 维护说明

- **添加新推理功能**: 在 `inference/` 中创建新文件
- **添加测试**: 在 `test/` 中添加对应的测试脚本
- **工具函数**: 放在 `utils/` 中
- **保持一致性**: 遵循现有的代码风格和命名规范

## 📝 注意事项

1. **路径依赖**: 部分脚本假设在项目根目录运行
2. **环境要求**: 需要 TensorRT、PyCUDA、PyTorch 等依赖
3. **文件引用**: 脚本之间的相对路径需要相应调整

## 🔄 迁移记录

- **2026-08-11**: 将所有 Python 脚本从根目录迁移到 src/ 文件夹
- **组织原则**: 按功能模块分类，便于维护和扩展