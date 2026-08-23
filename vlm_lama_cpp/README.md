# Qwen2.5-VL-3B llama.cpp 部署项目

基于 llama.cpp 在 NVIDIA Jetson Orin 8GB 上部署 Qwen2.5-VL-3B-Instruct 多模态大语言模型的完整解决方案。

## 项目概述

本项目展示了如何在边缘设备上部署具有视觉理解能力的多模态大语言模型，支持文本生成、图像理解、OCR 等功能。

### 主要特性

- 🎯 **多模态能力**: 支持文本生成和视觉理解
- ⚡ **边缘部署**: 专为 Jetson Orin 8GB 优化
- 🔧 **完整工具链**: 从下载到测试的自动化脚本
- 📊 **性能监控**: 详细的性能统计和基准测试
- 🚀 **即用脚本**: 开箱即用的启动和测试脚本

## 项目结构

```
vlm_lama_cpp/
├── DOC/                                    # 文档目录
│   ├── qwen2.5-vl-3B_deploy_log.md        # 部署日志
│   └── qwen2.5-vl-3b_llama_experiment_report.md  # 实验报告
├── test_vlm.py                             # VLM 测试脚本
├── test_ocr.py                             # OCR 专项测试
├── test.jpg                                # 测试图片
├── test_resized.jpg                        # 缩放后测试图片
└── document.jpg                            # OCR 测试文档
```

## 快速开始

### 1. 环境准备

**系统要求**:
- NVIDIA Jetson Orin 8GB RAM
- JetPack 6.2 (CUDA 12.6)
- Ubuntu Linux

**安装依赖**:
```bash
sudo apt update
sudo apt install -y build-essential cmake git curl
pip install Pillow  # 用于图片处理
```

### 2. 编译 llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)
```

### 3. 下载模型

```bash
# 下载主模型
huggingface-cli download Taoufik/Qwen2.5-VL-3B-Instruct-Q4_K_M-GGUF \
  --local-dir ./qwen_vl_gguf/

# 下载视觉投影组件
wget https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf \
  -P /path/to/models/
```

## 使用指南

### 纯文本模式

**启动服务**:
```bash
cd llama.cpp
./build/bin/llama-server \
  -m /path/to/qwen2.5-vl-3b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 2048
```

**测试文本生成**:
```bash
python test_vlm.py --prompt "你好，请介绍一下自己"
```

### 多模态模式

**启动完整服务**:
```bash
cd llama.cpp
./build/bin/llama-server \
  -m /path/to/qwen2.5-vl-3b-instruct-q4_k_m.gguf \
  --mmproj /path/to/Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf \
  --ctx-size 3072 \
  -ngl 99 \
  --host 127.0.0.1 \
  --port 8080
```

**测试视觉理解**:
```bash
# 基础图像理解
python test_vlm.py --image ./test.jpg --prompt "描述这张图片的内容"

# 自动缩放图片（避免超上下文）
python test_vlm.py --image ./test.jpg --resize 768

# OCR 文字识别
python test_ocr.py --image ./document.jpg --type document
```

## 测试脚本功能

### test_vlm.py - VLM 综合测试

**功能**:
- 纯文本对话测试
- 图像理解测试
- 自动图片缩放优化
- 性能统计（tok/s、耗时）
- 健康检查

**用法**:
```bash
# 纯文本
python test_vlm.py --prompt "解释量子计算"

# 视觉理解
python test_vlm.py --image ./test.jpg --resize 768 --max-tokens 512

# 自定义服务器
python test_vlm.py --url http://192.168.1.100:8080 --image ./test.jpg
```

### test_ocr.py - OCR 专项测试

**功能**:
- 普通文字识别
- 表格结构识别
- 文档内容提取

**用法**:
```bash
# 普通文字
python test_ocr.py --image ./document.jpg

# 表格识别
python test_ocr.py --image ./table.jpg --type table

# 文档识别
python test_ocr.py --image ./doc.jpg --type document
```

## 性能基准

### Jetson Orin 8GB 性能数据

| 任务类型 | Prompt速度 | 生成速度 | 内存占用 | 延迟 |
|---------|-----------|---------|---------|------|
| 纯文本生成 | 157.5 t/s | 19.4 t/s | <3GB | ~1.1s |
| 图像理解 | 61.5 t/s | 16.7 t/s | ~4.5GB | ~7.0s |
| CLI文本 | 147.6 t/s | 23.1 t/s | <3GB | ~2.1s |
| CLI图像 | 120.1 t/s | 9.0 t/s | ~4GB | ~8.5s |

### 模型规格

- **模型**: Qwen2.5-VL-3B-Instruct
- **量化**: Q4_K_M (4-bit)
- **大小**: ~1.8GB (主模型) + mmproj
- **上下文**: 最大 32K (推荐 2048-4096)

## 常见问题

### Q1: 图片处理超出上下文限制？

**解决方案**:
```bash
# 方案1: 缩小图片
python test_vlm.py --image ./large.jpg --resize 768

# 方案2: 增加上下文
./llama-server --ctx-size 4096 ...
```

### Q2: 视觉功能报错 500？

**原因**: 缺少 mmproj 组件

**解决**: 确保启动时包含 `--mmproj` 参数

### Q3: 内存不足？

**优化方案**:
- 减小上下文大小: `--ctx-size 2048`
- 使用更小的图片: `--resize 512`
- 检查其他进程占用

## API 接口

### OpenAI 兼容格式

**端点**: `http://localhost:8080/v1/chat/completions`

**示例请求**:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-vl-3b",
    "messages": [
      {"role": "user", "content": "你好"}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

**图像理解请求**:
```bash
IMG_BASE64=$(base64 -w 0 image.jpg)

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMG_BASE64"'"}},
        {"type": "text", "text": "描述这张图片"}
      ]
    }]
  }'
```

## 技术要点

### 模型组件
- **主模型 GGUF**: 量化的语言模型参数
- **mmproj GGUF**: 视觉编码器和投影层

### 性能优化
- **GPU 加速**: `-ngl 99` 全部层卸载到 GPU
- **上下文管理**: 根据任务调整 `--ctx-size`
- **图片预处理**: 缩小图片减少 token 消耗
- **量化选择**: Q4_K_M 平衡精度和速度

### 部署建议
1. **纯文本应用**: 只需主模型，内存占用低
2. **多模态应用**: 必须加载 mmproj 组件
3. **边缘部署**: 使用 4-bit 量化，控制上下文大小
4. **实时应用**: 优先考虑 GPU 加速和批处理

## 应用场景

- **🏭 工业检测**: 产品质量检查，缺陷识别
- **📱 智能监控**: 场景理解，异常检测  
- **📄 文档处理**: OCR，表格提取，内容理解
- **🤖 离线助手**: 本地知识问答，图像理解
- **🚗 边缘计算**: 车载系统，机器人视觉

## 相关文档

- [部署日志](DOC/qwen2.5-vl-3B_deploy_log.md) - 详细的部署过程记录
- [实验报告](DOC/qwen2.5-vl-3b_llama_experiment_report.md) - 完整的实验分析报告
- [llama.cpp 文档](https://github.com/ggerganov/llama.cpp) - 官方文档
- [Qwen2.5-VL 模型卡](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) - 模型详情

## 贡献与支持

### 问题反馈
- 提交 Issue 描述问题
- 提供系统环境和错误日志
- 包含复现步骤

### 功能建议
- 欢迎提出新功能建议
- 贡献测试脚本和用例
- 分享部署经验和优化

## 许可证

本项目遵循相关开源许可：
- llama.cpp: MIT License
- Qwen2.5-VL 模型: 按官方许可使用
- 本项目代码: MIT License

## 致谢

感谢以下项目和社区：
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - 推理框架
- [Qwen](https://github.com/QwenLM/Qwen) - 模型提供
- [NVIDIA Jetson](https://developer.nvidia.com/embedded/jetson) - 硬件平台

---

**最后更新**: 2024年8月23日
**版本**: v1.0
**维护者**: Edge AI Lab
