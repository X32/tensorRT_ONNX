# llama.cpp 部署 Qwen2.5-VL-3B 实验报告

## 实验概述

本实验旨在在 NVIDIA Jetson Orin 8GB 平台上使用 llama.cpp 部署 Qwen2.5-VL-3B-Instruct 多模态大语言模型，验证其在边缘设备上的文本生成和视觉理解能力。

### 实验目标
- 在 Jetson Orin 上成功编译和部署 llama.cpp
- 部署 Qwen2.5-VL-3B-Instruct Q4_K_M 量化模型
- 验证模型的文本生成能力
- 验证模型的多模态视觉理解能力
- 评估在边缘设备上的性能表现

---

## 实验环境

### 硬件配置
- **设备**: NVIDIA Jetson Orin 8GB RAM
- **架构**: ARM64, sm_87
- **CUDA**: 12.6 (JetPack 6.2)

### 软件环境
- **操作系统**: Linux (Ubuntu based)
- **编译工具**: build-essential, cmake, git, curl
- **模型格式**: GGUF (Quantized 4-bit)

---

## 实验步骤

### 1. 环境准备与 llama.cpp 编译

#### 1.1 系统依赖安装
```bash
sudo apt update
sudo apt install -y build-essential cmake git curl
```

#### 1.2 llama.cpp 源码获取与编译
```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
# 启用 CUDA backend（JetPack 6.2 自带 CUDA 12.6，Orin = sm_87 会自动识别）
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)
```

**编译要点**：
- 启用 CUDA backend 以利用 Jetson Orin 的 GPU 加速
- 多线程编译加快构建速度
- Release 优化版本提升运行性能

---

### 2. 模型下载

#### 2.1 主模型下载
```bash
huggingface-cli download Taoufik/Qwen2.5-VL-3B-Instruct-Q4_K_M-GGUF \
  --local-dir ./qwen_vl_gguf/
```

#### 2.2 视觉投影组件下载
```bash
wget https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf \
  -P /home/zy/Desktop/workplace/models/qwen_vl_gguf/
```

**模型组成**：
- 主模型: `qwen2.5-vl-3b-instruct-q4_k_m.gguf` (~1.8GB)
- 视觉投影: `Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf` (~必要组件)

---

### 3. 文本能力测试

#### 3.1 命令行模式测试
```bash
./build/bin/llama-cli \
  -m /home/zy/Desktop/workplace/models/qwen_vl_gguf/qwen2.5-vl-3b-instruct-q4_k_m.gguf \
  -p "解释量子计算的基本原理" \
  -n 512 \
  --ctx-size 2048
```

**测试结果**：
- 模型成功生成关于量子计算的详细解释
- **推理性能**: Prompt 147.6 t/s, Generation 23.1 t/s
- 输出质量：逻辑清晰，内容准确，符合预期

#### 3.2 服务器模式测试

**启动服务器**：
```bash
./build/bin/llama-server \
  -m /home/zy/Desktop/workplace/models/qwen_vl_gguf/qwen2.5-vl-3b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 2048
```

**API 测试**：
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "你好，请用一句话介绍一下你自己"}
    ],
    "max_tokens": 128
  }'
```

**服务器性能数据**：
- **Prompt处理**: 157.47 tokens/s
- **文本生成**: 19.37 tokens/s
- **响应时间**: 158.76ms (prompt), 981.08ms (generation)
- **内存占用**: 稳定运行在 8GB 限制内

![VLM 服务器运行状态](qwen2.5-vl-3B_server.jpg)

---

### 4. 视觉理解能力测试

#### 4.1 CLI 模式图像识别

```bash
./build/bin/llama-cli \
  -m /home/zy/Desktop/workplace/models/qwen_vl_gguf/qwen2.5-vl-3b-instruct-q4_k_m.gguf \
  --mmproj /home/zy/Desktop/workplace/models/qwen_vl_gguf/Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf \
  --image /home/zy/Desktop/workplace/tensorRT/vlm_lama_cpp/test_resized.jpg \
  -p "这张图片里有什么？" \
  -n 256 \
  --ctx-size 2048
```

**图像识别结果**：
- 准确识别冰箱内容物：饮料、果汁、运动饮料等
- 细节描述：注意到冰箱门上的小动物剪影纸张
- 环境观察：描述了冰箱的整洁格局
- **性能**: Prompt 120.1 t/s, Generation 9.0 t/s

#### 4.2 服务器模式视觉测试

**完整服务启动**：
```bash
./build/bin/llama-server \
  -m /home/zy/Desktop/workplace/models/qwen_vl_gguf/qwen2.5-vl-3b-instruct-q4_k_m.gguf \
  --mmproj /home/zy/Desktop/workplace/models/qwen_vl_gguf/Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf \
  -ngl 99 \
  --ctx-size 3072 \
  --host 127.0.0.1 \
  --port 8080
```

**Python 自动化测试**：
```bash
python test_vlm.py --url http://localhost:8080 --image ./test.jpg \
  --max-tokens 512 --resize 768
```

**视觉测试结果**：
- **图片处理**: 自动缩放至 768px 长边 (432x768)
- **内容识别**: 详细描述冰箱内物品布局和种类
- **性能指标**:
  - 总耗时: 6.99s
  - 生成速度: 16.7 tok/s
  - Token使用: 117 completion + 430 prompt = 547 total

![图文转换测试结果](img2text.jpg)

---

## 实验结果分析

### 性能表现总结

| 测试场景 | Prompt速度 | 生成速度 | 总耗时 | 内存占用 |
|---------|-----------|---------|--------|---------|
| 纯文本CLI | 147.6 t/s | 23.1 t/s | ~2.1s | <3GB |
| 纯文本Server | 157.5 t/s | 19.4 t/s | ~1.1s | <3GB |
| 图像识别CLI | 120.1 t/s | 9.0 t/s | ~8.5s | ~4GB |
| 多模态Server | 61.5 t/s | 16.7 t/s | ~7.0s | ~4.5GB |

### 功能验证结果

✅ **文本生成能力**: 完全正常，逻辑清晰，内容准确
✅ **视觉理解能力**: 准确识别图像内容，细节描述到位
✅ **多模态融合**: 成功处理图文结合的复杂任务
✅ **API 兼容性**: OpenAI 格式接口，易于集成

### 边缘设备适应性

**优势**：
- 模型量化后体积适中 (~1.8GB)，适合边缘部署
- 内存占用可控，8GB RAM 足够运行
- GPU 加速有效，推理速度实用
- 支持灵活的上下文大小调整

**限制**：
- 图片处理需要额外的 mmproj 组件
- 高分辨率图片会显著增加 token 消耗
- 视觉任务相比纯文本速度较慢

---

## 技术要点与经验总结

### 1. 模型组件理解
- **主模型 GGUF**: 包含量化的语言模型参数
- **mmproj GGUF**: 视觉编码器和投影层，处理图像输入
- **两者配合**: 完整的多模态能力需要同时加载

### 2. 性能优化策略
- **GPU 加速**: `-ngl 99` 将所有层卸载到 GPU
- **上下文管理**: 根据任务调整 `--ctx-size`
- **图片预处理**: 缩小图片尺寸减少 token 消耗
- **量化选择**: Q4_K_M 平衡了精度和速度

### 3. 常见问题解决

**问题**: 图片处理超出上下文限制
**解决**: 调整 `--ctx-size 4096` 或使用图片缩放

**问题**: 视觉功能不可用 (500错误)
**解决**: 确保加载 mmproj 组件

**问题**: 端口占用
**解决**: 检查并释放端口或更换端口号

---

## 结论与展望

### 实验结论

1. **部署成功**: 在 Jetson Orin 8GB 上成功部署了完整的 Qwen2.5-VL-3B 多模态模型
2. **功能验证**: 文本生成和视觉理解功能均正常工作
3. **性能可行**: 推理速度满足实际应用需求
4. **内存可控**: 总体内存占用在 8GB 限制内

### 应用前景

该部署方案适用于：
- **智能边缘设备**: 工业检测、智能监控
- **离线AI助手**: 本地知识问答、文档理解
- **多模态应用**: 图文理解、OCR、场景分析

### 未来改进方向

1. **性能优化**: 进一步优化内存使用和推理速度
2. **功能扩展**: 添加更多视觉任务支持
3. **部署简化**: 开发一键部署脚本
4. **应用集成**: 构建完整的应用解决方案

---

## 附录

### 关键命令参考

**启动完整服务**:
```bash
./build/bin/llama-server \
  -m /path/to/qwen2.5-vl-3b-instruct-q4_k_m.gguf \
  --mmproj /path/to/mmproj.gguf \
  --ctx-size 3072 \
  -ngl 99 \
  --host 0.0.0.0 \
  --port 8000
```

**测试脚本使用**:
```bash
# 纯文本测试
python test_vlm.py --prompt "解释量子计算"

# 视觉理解测试
python test_vlm.py --image ./test.jpg --resize 768

# OCR 能力测试
python test_ocr.py --image ./document.jpg --type document
```

![OCR 文字识别测试](ocrtest.jpg)

### 实验时间
- **开始**: 2024年8月23日
- **完成**: 2024年8月23日
- **总时长**: 约4小时

### 实验环境版本信息
- **llama.cpp**: 最新稳定版
- **CUDA**: 12.6
- **JetPack**: 6.2
- **Python**: 3.10+

---

**实验人员**: AI 研究团队
**实验地点**: 边缘AI实验室
**文档版本**: v1.0
**最后更新**: 2024年8月23日
