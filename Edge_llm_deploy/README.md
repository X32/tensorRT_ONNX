# TensorRT-Edge-LLM 部署指南

基于 TensorRT-Edge-LLM v0.6.0 在 Jetson Orin 8GB 设备上部署 Qwen2.5-0.5B 大语言模型，提供 OpenAI 兼容的 HTTP 推理服务。

## 快速开始

```bash
# 环境检查
./scripts/deploy.sh check

# 完整部署（从零到可用）
./scripts/deploy.sh all

# 启动服务
./scripts/deploy.sh start-server

# 验证部署
./scripts/deploy.sh test

# 性能测试
./scripts/deploy.sh benchmark
```

## 环境要求

| 组件 | 版本要求 | 说明 |
|------|----------|------|
| 硬件 | Jetson Orin 8GB | sm_87 架构，统一内存 |
| 系统 | Ubuntu 22.04 / JetPack 6.2 | L4T R36.x |
| CUDA | 12.6 | 与 JetPack 6.2 绑定 |
| TensorRT | 10.3.0.30 | 决定导出模式选择 |
| Python | 3.10+ | 容器内预配置 |
| 模型 | Qwen2.5-0.5B-Instruct | 8GB 设备最大支持 |

**⚠️ 版本关键**：必须使用 TensorRT-Edge-LLM **v0.6.0**，默认分支 v0.9.x 面向 Thor/JetPack 7.1，不兼容 Orin/JP6.2。

## 性能参考

| 指标 | 数值 | 说明 |
|------|------|------|
| 生成速度 | 22~30 tok/s (平均 ~26.5) | 实测 4 次平均 |
| TTFT | ~1.9s | 含引擎加载开销 |
| 峰值内存 | ~4.23GB (55.7%) | 推理阶段 |
| GPU 功耗 | ~15.2W | MAXN 模式 |
| GPU 利用率 | 99% | 算力打满 |

## 完整部署流程

### 阶段一：环境准备

```bash
# 启动容器（基于 JP6.2 的 aarch64 容器）
docker run -d --name jupyter-tensorrt --runtime=nvidia \
    -v "$(pwd)":/workspace -w /workspace \
    --network=host jupyter-tensorrt:latest

# 进入容器
docker exec -it jupyter-tensorrt bash
```

### 阶段二：安装 Python 工具链

```bash
# 克隆源码（锁定 v0.6.0）
git clone --depth 1 --branch v0.6.0 \
    https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM

# 安装主包（必须 --no-deps）
pip install . --no-deps

# 安装量化工具
pip install nvidia-modelopt==0.39.0 --no-deps
pip install omegaconf pulp pydantic nvidia-ml-py ninja

# 安装核心依赖
pip install transformers==4.57.6 datasets==4.4.2 peft==0.18.1 \
            pillow==12.1.1 backoff==2.2.1 soundfile==0.13.1 \
            librosa==0.11.0 einops==0.8.2

# 安装隐式依赖
pip install scipy --only-binary :all:
pip install torchprofile
pip install onnx-graphsurgeon

# 验证安装
python -c "import tensorrt_edgellm; print(tensorrt_edgellm.__version__)"  # 期望 0.6.0
tensorrt-edgellm-export-llm --help
```

### 阶段三：编译 C++ Runtime

```bash
cd TensorRT-Edge-LLM  # 如果不在目录中
rm -rf build && mkdir -p build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-orin \
  -DNV_ONNX_PARSER_LIB=/usr/local/cuda/targets/aarch64-linux/lib/libnvonnxparser.so

make -j4  # 编译约 3-5 分钟
```

**产物**：`llm_build`、`llm_inference`、`llm_bench`、`libNvInfer_edgellm_plugin.so`

### 阶段四：导出 ONNX

```bash
# 下载模型
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
    --local-dir /workspace/qwen25_0.5b/Qwen2.5-0.5B-Instruct

# 导出 ONNX（plugin 模式，不要加 --trt_native_ops）
tensorrt-edgellm-export-llm \
    --model_dir /workspace/qwen25_0.5b/Qwen2.5-0.5B-Instruct \
    --output_dir /workspace/qwen25_0.5b/onnx_new \
    --device cuda
```

**验证导出**：
```bash
python3 -c "
import onnx
from collections import Counter
m = onnx.load('/workspace/qwen25_0.5b/onnx_new/model.onnx')
print('total:', len(m.graph.node))
print(Counter(n.domain for n in m.graph.node))
"
# 期望输出：total: 1670, {'': 基础算子, 'trt': 28}
```

### 阶段五：构建 TRT 引擎

```bash
# 拷贝产物到宿主机
exit  # 退出容器
docker cp jupyter-tensorrt:/workspace/qwen25_0.5b/onnx_new \
    ./qwen25_0.5b/
docker cp jupyter-tensorrt:/workspace/TensorRT-Edge-LLM/build \
    ./TensorRT-Edge-LLM/
docker stop jupyter-tensorrt  # 释放内存

# 切换无桌面模式 + MAXN（关键！）
sudo systemctl isolate multi-user.target
sudo nvpmodel -m 0  # MAXN 25W 模式

# 检查内存（确保 lfb ≥ 400MB）
tegrastats --interval 1000 --stop 3

# 构建引擎
export EDGELLM_PLUGIN_PATH=$(pwd)/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
./TensorRT-Edge-LLM/build/examples/llm/llm_build \
    --onnxDir qwen25_0.5b/onnx_new \
    --engineDir qwen25_0.5b/engine_new \
    --maxBatchSize 1 --maxInputLen 512 --maxKVCacheCapacity 2048
```

**成功判据**：出现 `LLM engine built successfully`，产物包含 `llm.engine`、`config.json`、`tokenizer.json`

### 阶段六：推理部署

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 HTTP 服务
python3 llm_server.py \
    --engine-dir qwen25_0.5b/engine_new \
    --llm-inference TensorRT-Edge-LLM/build/examples/llm/llm_inference \
    --plugin TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so \
    --host 0.0.0.0 --port 8000

# 测试推理
python3 llm_client.py
```

## 脚本化部署

使用 `scripts/deploy.sh` 自动化部署：

```bash
# 查看所有命令
./scripts/deploy.sh help

# 环境检查
./scripts/deploy.sh check

# 分步部署
./scripts/deploy.sh install-python   # 安装 Python 工具链
./scripts/deploy.sh install-cpp      # 编译 C++ Runtime
./scripts/deploy.sh export-onnx      # 导出 ONNX
./scripts/deploy.sh build-engine     # 构建引擎
./scripts/deploy.sh start-server     # 启动服务

# 一键部署
./scripts/deploy.sh all
```

## 常见问题

### P0: 版本不兼容
**症状**：`jetson-orin` 目标不存在，编译失败  
**解决**：确保使用 `git checkout v0.6.0`，验证 `git describe --tags`

### P2: CMake 配置错误
**症状**：`NOTFOUND` 错误  
**解决**：显式指定 `NV_ONNX_PARSER_LIB=/usr/local/cuda/targets/aarch64-linux/lib/libnvonnxparser.so`

### P5: 导出模式错误
**症状**：构建时 `Cannot find plugin`  
**解决**：导出时不加 `--trt_native_ops`，使用默认 plugin 模式

### P7-P10: 构建内存不足
**症状**：`CUDA error 2: out of memory`  
**解决**：
1. 使用无桌面模式：`sudo systemctl isolate multi-user.target`
2. 设置 MAXN 功耗：`sudo nvpmodel -m 0`
3. 确保 lfb ≥ 400MB：`tegrastats --interval 1000 --stop 3`
4. 8GB 设备仅支持 0.5B 模型

### 验证检查点
```bash
# 版本验证
python -c "import tensorrt_edgellm; print(tensorrt_edgellm.__version__)"  # 0.6.0

# 导出验证
python3 -c "import onnx; from collections import Counter; m=onnx.load('model.onnx'); print(Counter(n.domain for n in m.graph.node))"  # {'': ..., 'trt': 28}

# 内存验证
tegrastats --interval 1000 --stop 3  # lfb ≥ 400MB

# 引擎验证
ls qwen25_0.5b/engine_new/  # llm.engine, config.json, tokenizer.json
```

## 服务管理

```bash
# 启动服务
./scripts/deploy.sh start-server

# 停止服务
sudo pkill -f llm_server.py

# 查看日志
sudo journalctl -u llm-server -f  # 如果使用 systemd

# 测试服务
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen2.5-0.5b","messages":[{"role":"user","content":"你好"}]}'
```

## 性能测试

```bash
# 运行性能测试
./scripts/benchmark.sh

# 自定义测试轮数
ROUNDS=10 ./scripts/benchmark.sh

# 指定服务器地址
./scripts/benchmark.sh 192.168.1.100
```

**输出指标**：
- TTFT（首 token 延迟）
- 生成速度（tok/s）
- 峰值内存
- GPU 功耗

## 故障排除

### 引擎加载失败
```bash
# 检查插件路径
export EDGELLM_PLUGIN_PATH=$(pwd)/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
echo $EDGELLM_PLUGIN_PATH

# 检查引擎文件
ls -lh qwen25_0.5b/engine_new/llm.engine
```

### 推理速度慢
```bash
# 检查功耗模式
sudo nvpmodel -q  # 期望 MAXN 25W

# 检查 GPU 频率
sudo jetson_clocks

# 检查内存使用
tegrastats --interval 1000
```

### HTTP 服务问题
```bash
# 检查端口占用
sudo netstat -tulpn | grep 8000

# 检查服务状态
ps aux | grep llm_server.py

# 查看错误日志
sudo journalctl -xe | grep llm
```

## OpenAI SDK 对接

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<jetson-ip>:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="qwen2.5-0.5b",
    messages=[{"role": "user", "content": "你好，请介绍一下自己"}],
    max_tokens=128
)

print(response.choices[0].message.content)
```

## 版本锚点

- **TensorRT-Edge-LLM**: v0.6.0 (commit `d2118b3c`)
- **JetPack**: 6.2 (L4T R36.x)
- **TensorRT**: 10.3.0.30
- **CUDA**: 12.6
- **目标架构**: sm_87 (Jetson Orin)
- **模型**: Qwen2.5-0.5B-Instruct

## 技术架构

**部署策略**：混合部署，容器内编译导出，宿主机构建推理

| 环境 | 用途 | 原因 |
|------|------|------|
| 容器 | 编译 C++ Runtime + 导出 ONNX | 环境隔离、工具链齐全 |
| 宿主机 | 构建引擎 + 推理 + 服务 | 避开容器内存开销 |

**已知限制**：
- 8GB 设备构建上限 0.5B 模型
- v0.6.0 无流式输出，`stream=True` 降级为一次性返回
- 并发请求串行处理（全局锁）

## 参考资料

- [完整实战报告](../DOC/TensorRT-Edge-LLM 从安装到引擎可用全流程实战报告.md)
- [容器安装报告](../DOC/TensorRT-Edge-LLM_容器安装报告_v0.6.0.md)
- [C++ 编译指南](../DOC/TensorRT-Edge-LLM_C++_编译指南.md)
- [官方文档](https://github.com/NVIDIA/TensorRT-Edge-LLM)