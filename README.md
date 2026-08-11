# TensorRT 环境安装与使用报告

## 环境信息

- **设备**: NVIDIA Jetson (JetPack 6.0 R36.5)
- **系统**: Ubuntu aarch64
- **TensorRT 版本**: 10.3.0
- **Python 版本**: 3.10.12
- **虚拟环境路径**: `/home/zy/tensorrt_env`

## 虚拟环境路径

```
/home/zy/tensorrt_env
├── bin/           # 可执行文件
├── lib/           # 库文件
│   └── python3.10/
│       └── site-packages/
│           ├── tensorrt -> /usr/lib/python3.10/dist-packages/tensorrt
│           ├── torch/          # PyTorch 2.13.0
│           ├── torchvision/    # TorchVision 0.28.0
│           ├── onnx/           # ONNX 1.22.0
│           └── onnxscript/     # ONNX Script 0.7.1
└── pyvenv.cfg     # 虚拟环境配置
```

## 激活虚拟环境

```bash
source ~/tensorrt_env/bin/activate
```

激活后，命令提示符将显示 `(tensorrt_env)` 前缀。

## 安装步骤总结

### 1. 安装 TensorRT 核心组件

```bash
# 更新包列表
sudo apt update

# 安装 TensorRT 开发包和 Python 绑定
sudo apt install -y tensorrt-dev python3-libnvinfer python3-libnvinfer-dev libnvinfer-samples

# 安装 DLA 编译器（解决 libnvdla_compiler.so 依赖）
sudo apt install -y nvidia-l4t-dla-compiler

# 安装 CUDLA 库
sudo apt install -y libcudla-12-6

# 更新库缓存
sudo ldconfig
```

### 2. 使用 uv 创建虚拟环境

```bash
# 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建 Python 3.10 虚拟环境
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.10 ~/tensorrt_env
```

### 3. 配置 TensorRT 符号链接

在虚拟环境 `site-packages` 中创建指向系统 TensorRT 包的符号链接：

```bash
cd ~/tensorrt_env/lib/python3.10/site-packages/
ln -sf /usr/lib/python3.10/dist-packages/tensorrt tensorrt
ln -sf /usr/lib/python3.10/dist-packages/tensorrt_lean tensorrt_lean
ln -sf /usr/lib/python3.10/dist-packages/tensorrt_dispatch tensor_dispatch
```

### 4. 安装 Python 包

```bash
source ~/tensorrt_env/bin/activate

# 设置 UV 环境变量（解决网络超时问题）
export UV_HTTP_TIMEOUT=300

# 安装 PyTorch、TorchVision、ONNX 和相关依赖
uv pip install torch torchvision onnx numpy onnxscript
```

### 5. 验证安装

```bash
source ~/tensorrt_env/bin/activate
python -c "import tensorrt, torch, torchvision, onnx; print('所有包导入成功')"
```

## 使用方法

### 激活环境

```bash
source ~/tensorrt_env/bin/activate
```

### 退出环境

```bash
deactivate
```

### 导出 PyTorch 模型到 ONNX

```bash
source ~/tensorrt_env/bin/activate
python src/export/export_resnet.py
```

输出：`resnet18.onnx`

### 测试 TensorRT 基本功能

```bash
# 检查引擎兼容性
source ~/tensorrt_env/bin/activate
python src/test/test_engine.py

# 获取 tensor 信息
python src/test/get_tensor_names.py

# 基础推理测试
python src/test/test_tensorrt.py
```

### 使用 TensorRT 工具

```bash
# trtexec 工具路径
/usr/src/tensorrt/bin/trtexec --help

# 使用 trtexec 运行 ONNX 模型
/usr/src/tensorrt/bin/trtexec --onnx=resnet18.onnx --saveEngine=resnet18.engine
```

## 已安装的包版本

```
tensorrt: 10.3.0
torch: 2.13.0+cu130
torchvision: 0.28.0+cu130
onnx: 1.22.0
onnxscript: 0.7.1
numpy: 2.2.6
pillow: 12.3.0
protobuf: 7.35.1
sympy: 1.14.0
```

## 常见问题

### Q1: 如何检查 TensorRT 是否正常工作？

```bash
source ~/tensorrt_env/bin/activate
python -c "import tensorrt as trt; print(f'TensorRT {trt.__version__} 工作正常')"
```

### Q2: CUDA 驱动版本警告

如遇 CUDA 驱动版本警告，可以忽略（JetPack 6.0 的驱动版本与 PyTorch 2.13 的期望版本不完全匹配）。

### Q3: 如何重新安装虚拟环境？

```bash
# 删除现有环境
rm -rf ~/tensorrt_env

# 重新创建
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.10 ~/tensorrt_env

# 重新安装包
source ~/tensorrt_env/bin/activate
uv pip install torch torchvision onnx numpy onnxscript
```

## 项目文件结构

### 📁 根目录文件
```
tensorRT/
├── src/                        # 源代码文件夹 (按功能分类)
│   ├── inference/             # 推理相关脚本
│   ├── calibration/           # INT8 校准脚本
│   ├── export/               # 模型导出脚本
│   ├── test/                 # 测试和验证脚本
│   └── utils/                # 工具和辅助脚本
├── DOC/                       # 文档文件夹
│   ├── PERFORMANCE_REPORT.md # 详细性能报告
│   └── QUICK_SUMMARY.md      # 快速总结
├── CUDA/                      # CUDA 相关文件
├── onnx_test.ipynb           # 主测试 notebook
├── PERFORMANCE_REPORT.md      # 性能测试报告 (根目录副本)
├── QUICK_SUMMARY.md          # 快速总结 (根目录副本)
├── resnet18.onnx             # ONNX 模型
├── resnet18.engine           # TensorRT FP32 引擎
├── resnet18_fp16.engine      # TensorRT FP16 引擎
├── resnet18_int8.engine      # TensorRT INT8 引擎
├── banana1.jpg               # 测试图片
└── imagenet_class_index.csv # ImageNet 类别标签
```
```
tensorRT/
├── src/                        # 源代码文件夹 (按功能分类)
│   ├── inference/             # 推理相关脚本
│   ├── calibration/           # INT8 校准脚本
│   ├── export/               # 模型导出脚本
│   ├── test/                 # 测试和验证脚本
│   └── utils/                # 工具和辅助脚本
├── DOC/                       # 文档文件夹
├── CUDA/                      # CUDA 相关文件
├── onnx_test.ipynb           # 主测试 notebook
├── resnet18.onnx             # ONNX 模型
├── resnet18.engine           # TensorRT FP32 引擎
├── resnet18_fp16.engine      # TensorRT FP16 引擎
├── resnet18_int8.engine      # TensorRT INT8 引擎
├── banana1.jpg               # 测试图片
└── imagenet_class_index.csv # ImageNet 类别标签
```

### 🗂️ 源代码结构 (`src/`)
详细说明请查看 [src/README.md](src/README.md)

- **inference/**: TensorRT 推理实现
- **calibration/**: INT8 量化校准
- **export/**: 模型导出工具
- **test/**: 测试验证脚本
- **utils/**: 辅助工具

## 文件位置

- **TensorRT 库**: `/usr/lib/aarch64-linux-gnu/`
- **TensorRT Python 包**: `/usr/lib/python3.10/dist-packages/`
- **虚拟环境**: `/home/zy/tensorrt_env`
- **trtexec 工具**: `/usr/src/tensorrt/bin/trtexec`
- **当前工作目录**: `/home/zy/Desktop/workplace/tensorRT`

## 快速开始指南

### 1. 推理测试 (推荐使用 Jupyter Notebook)
```bash
# 在 Docker 容器中打开 notebook
jupyter lab
```
然后在 `onnx_test.ipynb` 中运行完整的多精度性能对比测试。

### 2. 命令行推理
```python
from src.inference.tensorrt_inference_fixed import TensorRTInference
import numpy as np

# 创建推理器
inferencer = TensorRTInference("resnet18_fp16.engine")

# 准备输入数据
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# 执行推理
result = inferencer.infer(input_data)
print(f"推理结果: {result.shape}")
```

### 3. 生成不同精度的 Engine
```bash
# FP16 Engine
trtexec --onnx=resnet18.onnx --saveEngine=resnet18_fp16.engine --fp16

# INT8 Engine (需要校准)
trtexec --onnx=resnet18.onnx --saveEngine=resnet18_int8.engine --int8
```

## 📝 技术博客

基于本项目实践，我们编写了详实的技术博客：

### 🔗 博客文章
📖 **[Jetson Edge AI 部署完整指南](TECH_BLOG_CONCISE.md)** - 精简版，适合快速阅读
- 核心技术要点
- 实测性能数据
- 生产部署建议

📖 **[完整技术指南](TECH_BLOG_Jetson_TensorRT_Complete_Guide.md)** - 详尽版，适合深入学习
- 环境搭建详解
- 模型转换流程
- 推理实现细节
- 故障排除指南

### 🎯 核心亮点
- **7-30倍性能提升** vs ONNX Runtime
- **完整部署流程** 从环境到生产
- **实测数据验证** 真实性能表现
- **生产级代码** 可直接使用的实现

## 📊 性能测试报告

我们已经完成了详细的性能测试，包含以下报告：

### 快速查看
📖 **[快速总结](QUICK_SUMMARY.md)** - 一分钟了解核心结果
- 关键数据对比
- 推荐选择建议
- 实际应用指南

### 深入分析  
📊 **[详细报告](PERFORMANCE_REPORT.md)** - 完整的技术分析
- 测试方法论
- 详细性能数据
- 技术原理解析
- 应用场景推荐

### 核心结论
```
TensorRT 在 Jetson 设备上实现了:
- FP32:  7.4倍 性能提升 vs ONNX Runtime
- FP16: 15倍  性能提升 (推荐⭐⭐⭐⭐⭐)  
- INT8: 30倍  性能提升 (极限性能)
```

**推荐**: 生产环境使用 FP16，边缘计算使用 INT8

## 下一步

现在你可以：

1. ✅ 导出更多 PyTorch 模型到 ONNX
2. ✅ 使用 TensorRT 优化 ONNX 模型
3. ✅ 使用 TensorRT 进行高性能推理
4. ✅ 开发深度学习应用
5. ✅ 探索 `src/` 文件夹中的模块化脚本
6. ✅ 阅读 `DOC/` 文件夹中的详细文档

---

**生成日期**: 2026-08-07
**TensorRT 版本**: 10.3.0
**虚拟环境**: `/home/zy/tensorrt_env`
