# TensorRT-Edge-LLM C++ 运行时编译指南

## 文档信息

- **版本**: v0.6.0
- **编译日期**: 2026-08-13
- **环境**: jupyter-tensorrt 容器
- **目标平台**: Jetson Orin Nano (SM_87)
- **文档状态**: ✅ 编译成功验证

---

## 1. 编译背景与目标

### 1.1 为什么需要编译 C++ 运行时

TensorRT-Edge-LLM 提供 Python 工具链（量化、ONNX 导出）和 C++ 运行时（引擎构建、推理运行）。用户已完成 Python 工具链的使用，但缺少 C++ 运行时的核心工具：

- **llm_build**: 从 ONNX 构建 TensorRT 引擎
- **llm_inference**: 运行 LLM 模型推理
- **llm_bench**: 性能基准测试

### 1.2 编译环境信息

| 组件 | 版本/信息 | 状态 |
|------|----------|------|
| **容器** | jupyter-tensorrt:complete_v2 | ✅ 运行中 |
| **平台** | Jetson Orin Nano (SM_87) | ✅ 原生编译 |
| **GCC/G++** | 11.4.0 | ✅ |
| **CMake** | 3.31.10 | ✅ (要求 ≥3.20) |
| **CUDA** | 12.6.85 | ✅ |
| **TensorRT** | 10.3.0 | ✅ |
| **cuDNN** | 9.3.0 | ✅ |

---

## 2. 编译前准备

### 2.1 项目位置

- **宿主机**: `/home/zy/Desktop/workplace/tensorRT/TensorRT-Edge-LLM`
- **容器内**: `/workspace/TensorRT-Edge-LLM`
- **映射**: 通过 Docker 卷映射共享

### 2.2 关键依赖检查

编译前必须确保以下依赖完整：

```bash
# 检查 CUDA 版本
nvcc --version  # 应显示 CUDA 12.6

# 检查 TensorRT 头文件
ls /usr/include/NvInfer.h  # 必须存在

# 检查 TensorRT 库文件
ls /usr/local/cuda/targets/aarch64-linux/lib/libnvinfer.so  # 必须存在
```

---

## 3. 详细编译步骤

### 3.1 步骤 1: 初始化 Git 子模块 ⭐ 关键步骤

**问题**: Git 子模块未初始化导致编译失败

**解决方案**:
```bash
# 进入容器
docker exec -it jupyter-tensorrt bash

# 配置 git 安全目录（解决权限问题）
git config --global --add safe.directory /workspace/TensorRT-Edge-LLM
git config --global --add safe.directory /workspace/TensorRT-Edge-LLM/3rdParty/NVTX
git config --global --add safe.directory /workspace/TensorRT-Edge-LLM/3rdParty/googletest
git config --global --add safe.directory /workspace/TensorRT-Edge-LLM/3rdParty/nlohmannJson

# 初始化子模块
cd /workspace/TensorRT-Edge-LLM
git submodule update --init --recursive
```

**验证**:
```bash
ls 3rdParty/nlohmannJson/include/nlohmann/json.hpp  # 必须存在
```

### 3.2 步骤 2: CMake 配置

**问题**: TensorRT 库路径在容器环境中特殊布局

**解决方案**:
```bash
cd /workspace/TensorRT-Edge-LLM
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCUDA_CTK_VERSION=12.6 \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DNV_ONNX_PARSER_LIB=/usr/local/cuda/targets/aarch64-linux/lib/libnvonnxparser.so \
    -DNVINFER_LIB=/usr/local/cuda/targets/aarch64-linux/lib/libnvinfer.so
```

**参数说明**:
- `-DCMAKE_BUILD_TYPE=Release`: 优化编译，生成高性能可执行文件
- `-DTRT_PACKAGE_DIR=/usr`: TensorRT 安装根目录
- `-DCUDA_CTK_VERSION=12.6`: 匹配容器环境的 CUDA 版本
- `-DCMAKE_CUDA_ARCHITECTURES=87`: Jetson Orin 架构
- `-DNV_ONNX_PARSER_LIB`: 显式指定 ONNX 解析器库路径
- `-DNVINFER_LIB`: 显式指定 TensorRT 推理库路径

**验证**:
```bash
# 检查 CMakeCache.txt 确认配置正确
grep CMAKE_CUDA_ARCHITECTURES CMakeCache.txt
```

### 3.3 步骤 3: 编译核心工具

**问题**: 容器内存限制，并行编译可能 OOM

**解决方案**:
```bash
cd /workspace/TensorRT-Edge-LLM/build

# 编译所有 LLM 工具
make -j2 llm_build llm_inference llm_bench

# 编译多模态工具（如需要）
make -j2 visual_build audio_build qwen3_tts_inference

# 或者一次性编译所有工具
make -j2
```

**编译进度**:
- **0-30%**: 编译 edgellmBuilder 静态库
- **30-60%**: 编译 LLM 核心工具
- **60-90%**: 编译多模态工具
- **90-100%**: 链接和生成最终可执行文件

**预计时间**: 15-30 分钟（取决于 Jetson 性能和编译目标）

### 3.4 步骤 4: 验证编译结果

**验证命令**:
```bash
# 检查所有生成的可执行文件
find /workspace/TensorRT-Edge-LLM/build/examples -name '*build' -o -name '*inference' -o -name '*bench' -type f -executable

# 检查 LLM 工具
ls -lh /workspace/TensorRT-Edge-LLM/build/examples/llm/

# 检查多模态工具
ls -lh /workspace/TensorRT-Edge-LLM/build/examples/multimodal/
ls -lh /workspace/TensorRT-Edge-LLM/build/examples/omni/

# 测试工具帮助信息
/workspace/TensorRT-Edge-LLM/build/examples/llm/llm_build --help
/workspace/TensorRT-Edge-LLM/build/examples/llm/llm_inference --help
/workspace/TensorRT-Edge-LLM/build/examples/multimodal/visual_build --help
/workspace/TensorRT-Edge-LLM/build/examples/multimodal/audio_build --help
```

---

## 4. 遇到的问题与解决方案

### 4.1 Git 子模块权限问题

**错误**:
```
fatal: detected dubious ownership in repository at '/workspace/TensorRT-Edge-LLM'
```

**解决**: 添加安全目录例外
```bash
git config --global --add safe.directory /workspace/TensorRT-Edge-LLM
```

### 4.2 CMake 缓存冲突

**错误**:
```
CMake Error: The current CMakeCache.txt directory is different than the directory where CMakeCache.txt was created
```

**解决**: 清理 build 目录
```bash
rm -rf /workspace/TensorRT-Edge-LLM/build && mkdir -p build
```

### 4.3 TensorRT 库查找失败

**错误**:
```
CMake Error: NV_ONNX_PARSER_LIB set to NOTFOUND
```

**解决**: 显式指定库路径
```bash
-DNV_ONNX_PARSER_LIB=/usr/local/cuda/targets/aarch64-linux/lib/libnvonnxparser.so
```

### 4.4 链接时找不到库

**错误**:
```
/usr/bin/ld: cannot find -lNVINFER_LIB-NOTFOUND
```

**解决**: 同时指定 NVINFER_LIB 路径
```bash
-DNVINFER_LIB=/usr/local/cuda/targets/aarch64-linux/lib/libnvinfer.so
```

---

## 5. 编译结果验证

### 5.1 生成的工具列表

| 类别 | 工具名称 | 大小 | 位置 | 功能 | 状态 |
|------|----------|------|------|------|------|
| **LLM 核心** | `llm_build` | 287K | `build/examples/llm/` | 构建 LLM TensorRT 引擎 | ✅ |
| **LLM 核心** | `llm_inference` | 43M | `build/examples/llm/` | 运行 LLM 模型推理 | ✅ |
| **LLM 核心** | `llm_bench` | 42M | `build/examples/llm/` | LLM 性能基准测试 | ✅ |
| **多模态** | `visual_build` | 252K | `build/examples/multimodal/` | 构建视觉编码器引擎 | ✅ |
| **多模态** | `audio_build` | 259K | `build/examples/multimodal/` | 构建音频编码器引擎 | ✅ |
| **多模态** | `qwen3_tts_inference` | 42M | `build/examples/omni/` | Qwen3 TTS 推理 | ✅ |

### 5.2 工具功能验证

#### LLM 核心工具

**llm_build 帮助信息**:
```
Usage: llm_build [--help] --onnxDir <dir> --engineDir <dir>
Options:
  --onnxDir              输入 ONNX 目录路径 (必需)
  --engineDir            输出 TensorRT 引擎目录路径 (必需)
  --maxInputLen          模型最大输入长度 (默认: 1024)
  --maxKVCacheCapacity   KV 缓存最大容量 (默认: 4096)
  --maxBatchSize         构建器最大批次大小 (默认: 4)
```

**llm_inference 帮助信息**:
```
Usage: llm_inference [--help] --engineDir=<path> --inputFile=<path>
Options:
  --engineDir            引擎目录路径
  --inputFile            输入 JSON 文件路径
  --outputFile           输出 JSON 文件路径 (可选)
  --dumpProfile          转储性能摘要到控制台
  --debug                启用调试日志
```

#### 多模态工具

**visual_build 帮助信息**:
```
Usage: visual_build [--help] --onnxDir <dir> --engineDir <dir>
Options:
  --onnxDir              视觉编码器 ONNX 目录 (必需)
  --engineDir            引擎输出目录 (必需)
  --minImageTokens       最小图像 token 数 (默认: 4)
  --maxImageTokens       最大图像 token 数 (默认: 1024)
  --maxImageTokensPerImage 单张图像最大 token (默认: 512)
```

**audio_build 帮助信息**:
```
Usage: audio_build [--help] --onnxDir <dir> --engineDir <dir>
Options:
  --onnxDir              音频编码器 ONNX 目录 (必需)
  --engineDir            引擎输出目录 (必需)
  --minTimeSteps         最小音频时间步 (默认: 100)
  --maxTimeSteps         最大音频时间步 (默认: 6000)
  --minCodeLen           最小编码长度 (Code2Wav 模式)
  --optCodeLen           最佳编码长度 (Code2Wav 模式)
  --maxCodeLen           最大编码长度 (Code2Wav 模式)
```

---

## 6. 工具使用指南

### 6.1 构建 Qwen2.5-1.5B TensorRT 引擎

```bash
/workspace/TensorRT-Edge-LLM/build/examples/llm/llm_build \
  --onnxDir /workspace/qwen25_1.5b_trt/Qwen2.5-1.5B-Instruct/onnx \
  --engineDir /workspace/qwen25_1.5b_trt/Qwen2.5-1.5B-Instruct/engine \
  --maxInputLen 2048 \
  --maxBatchSize 1
```

### 6.2 运行模型推理

```bash
# 创建输入 JSON 文件
cat > request.json << EOF
{
  "requests": [
    {
      "text_input": "你好，介绍一下你自己"
    }
  ]
}
EOF

# 运行推理
/workspace/TensorRT-Edge-LLM/build/examples/llm/llm_inference \
  --engineDir /workspace/qwen25_1.5b_trt/Qwen2.5-1.5B-Instruct/engine \
  --inputFile request.json \
  --outputFile response.json
```

### 6.3 性能基准测试

```bash
/workspace/TensorRT-Edge-LLM/build/examples/llm/llm_bench \
  --engineDir /workspace/qwen25_1.5b_trt/Qwen2.5-1.5B-Instruct/engine \
  --inputFile benchmark_requests.json
```

### 6.4 构建视觉编码器引擎

```bash
/workspace/TensorRT-Edge-LLM/build/examples/multimodal/visual_build \
  --onnxDir /workspace/visual_encoder/onnx \
  --engineDir /workspace/visual_encoder/engine \
  --maxImageTokens 1024 \
  --maxImageTokensPerImage 512
```

**用途**: 构建图像处理和视觉理解的 TensorRT 引擎

### 6.5 构建音频编码器引擎

```bash
# 音频编码器模式
/workspace/TensorRT-Edge-LLM/build/examples/multimodal/audio_build \
  --onnxDir /workspace/audio_encoder/onnx \
  --engineDir /workspace/audio_encoder/engine \
  --maxTimeSteps 6000

# Code2Wav 模式
/workspace/TensorRT-Edge-LLM/build/examples/multimodal/audio_build \
  --onnxDir /workspace/code2wav/onnx \
  --engineDir /workspace/code2wav/engine \
  --minCodeLen 1 \
  --optCodeLen 300 \
  --maxCodeLen 2000
```

**用途**: 构建语音处理和音频编码的 TensorRT 引擎

### 6.6 运行 Qwen3 TTS 推理

```bash
/workspace/TensorRT-Edge-LLM/build/examples/omni/qwen3_tts_inference \
  --engineDir /workspace/qwen3_tts/engine \
  --inputFile text_input.txt \
  --outputFile audio_output.wav
```

**用途**: 运行 Qwen3 模型的文本转语音推理

---

## 7. 环境变量配置

### 7.1 可选环境变量

```bash
# LLM 工具路径环境变量（可选）
export LLM_BUILD_TOOL=/workspace/TensorRT-Edge-LLM/build/examples/llm/llm_build
export LLM_INFERENCE_TOOL=/workspace/TensorRT-Edge-LLM/build/examples/llm/llm_inference
export LLM_BENCH_TOOL=/workspace/TensorRT-Edge-LLM/build/examples/llm/llm_bench

# 多模态工具路径环境变量（可选）
export VISUAL_BUILD_TOOL=/workspace/TensorRT-Edge-LLM/build/examples/multimodal/visual_build
export AUDIO_BUILD_TOOL=/workspace/TensorRT-Edge-LLM/build/examples/multimodal/audio_build
export TTS_INFERENCE_TOOL=/workspace/TensorRT-Edge-LLM/build/examples/omni/qwen3_tts_inference

# 创建快捷方式（可选）
alias llm-build='$LLM_BUILD_TOOL'
alias llm-inference='$LLM_INFERENCE_TOOL'
alias llm-bench='$LLM_BENCH_TOOL'
alias visual-build='$VISUAL_BUILD_TOOL'
alias audio-build='$AUDIO_BUILD_TOOL'
alias tts-inference='$TTS_INFERENCE_TOOL'
```

---

## 8. 常见问题 FAQ

### Q1: 编译时出现内存不足怎么办？

**A**: 减少并行编译数量：
```bash
make -j1  # 单线程编译
# 或
make -j2  # 双线程编译
```

### Q2: 如何确认编译是否成功？

**A**: 检查三个条件：
1. 可执行文件存在：`ls build/examples/llm/llm_*`
2. 工具可执行：`llm_build --help`
3. 无编译错误退出

### Q3: 编译后的工具可以复制到其他机器吗？

**A**: 不建议。编译产物针对当前 Jetson 平台优化，复制到其他架构可能无法运行。

### Q4: 如何重新编译？

**A**: 清理 build 目录后重新配置：
```bash
rm -rf build
mkdir build && cd build
# 重新运行 CMake 和 make 命令
```

---

## 9. 性能优化建议

### 9.1 编译优化

- **Release 模式**: 使用 `-DCMAKE_BUILD_TYPE=Release` 启用优化
- **架构特定**: `-DCMAKE_CUDA_ARCHITECTURES=87` 针对 Orin 优化
- **并行编译**: 根据内存情况调整 `-j` 参数

### 9.2 运行时优化

- **GPU 频率**: 使用 `jetson_clocks` 设置最大性能
- **内存锁定**: 预分配 GPU 内存避免碎片化
- **批处理**: 调整 `--maxBatchSize` 平衡吞吐和延迟

---

## 10. 附录

### 10.1 完整编译脚本

```bash
#!/bin/bash
# TensorRT-Edge-LLM C++ 运行时完整编译脚本

set -e  # 遇到错误立即退出

echo "=== TensorRT-Edge-LLM C++ 运行时编译 ==="
echo "日期: $(date)"
echo "平台: $(uname -m)"

# 配置路径
PROJECT_DIR="/workspace/TensorRT-Edge-LLM"
BUILD_DIR="$PROJECT_DIR/build"
TOOLS_DIR="$BUILD_DIR/examples/llm"

# 步骤 1: 初始化子模块
echo "步骤 1: 初始化 Git 子模块..."
cd $PROJECT_DIR
git submodule update --init --recursive

# 步骤 2: 配置 CMake
echo "步骤 2: 配置 CMake..."
mkdir -p $BUILD_DIR && cd $BUILD_DIR
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCUDA_CTK_VERSION=12.6 \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DNV_ONNX_PARSER_LIB=/usr/local/cuda/targets/aarch64-linux/lib/libnvonnxparser.so \
    -DNVINFER_LIB=/usr/local/cuda/targets/aarch64-linux/lib/libnvinfer.so

# 步骤 3: 编译工具
echo "步骤 3: 编译核心工具..."
make -j2 llm_build llm_inference llm_bench

# 步骤 4: 验证结果
echo "步骤 4: 验证编译结果..."
ls -lh $TOOLS_DIR/llm_*

echo "=== 编译完成 ==="
echo "工具位置: $TOOLS_DIR"
echo "测试命令: $TOOLS_DIR/llm_build --help"
```

### 10.2 相关文档链接

- [TensorRT-Edge-LLM 官方文档](https://nvidia.github.io/TensorRT-Edge-LLM/)
- [Jetson Orin 开发者指南](https://developer.nvidia.com/embedded/jetson-orin)
- [CUDA 12.6 文档](https://docs.nvidia.com/cuda/archive/12.6/)

### 10.3 技术支持

如有问题，请参考：
- TensorRT-Edge-LLM GitHub Issues
- NVIDIA 开发者论坛
- Jetson 社区资源

---

## 文档版本历史

| 版本 | 日期 | 作者 | 变更说明 |
|------|------|------|----------|
| v1.0 | 2026-08-13 | Claude Code | 初始版本，基于实际编译经验 |

---

**编译状态**: ✅ 成功验证
**文档状态**: ✅ 完整可用
**最后更新**: 2026-08-13