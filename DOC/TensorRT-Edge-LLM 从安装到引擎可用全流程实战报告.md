# TensorRT-Edge-LLM 从安装到引擎可用全流程实战报告

> **适用平台**：Jetson Orin Nano Super DevKit 8GB / JetPack 6.2 / CUDA 12.6 / TensorRT 10.3.0.30
> **软件版本**：TensorRT-Edge-LLM **v0.6.0**（Python 静态工具链 pip 安装 + C++ Runtime 源码编译）+ Qwen2.5 系列
> **最终成果**：Qwen2.5-0.5B-Instruct 引擎构建成功 → 推理验证通过 → OpenAI 兼容 HTTP 服务部署可用
> **日期**：2026-08-13

---

## 0. 结论先行（先看这一段）

**完整链路一句话**：容器内导出 ONNX（plugin 模式）→ `docker cp` 拷到宿主机 → 宿主机无桌面模式窗口下构建引擎 → `llm_inference` 推理验证 → FastAPI 网关做成 OpenAI 兼容服务。

**⚠️ 安装分两部分，缺一不可**：① **Python 静态工具链**（`tensorrt-edgellm-export-llm` / `quantize-llm` 等 **11 个命令行工具**，pip 安装，负责导出 ONNX 与量化）② **C++ Runtime 工具**（`llm_build` / `llm_inference` / `llm_bench`，源码编译，负责构建与推理）。两者必须同源同版本（v0.6.0），分别详见 §4 / §5。

**8GB Orin 的构建能力红线**：本地最多构建 **0.5B** 模型（embedding 表 260MB）；1.5B 实测五次全败（embedding 表 445MB 连续块分配不出）。16GB 上限 3B；7B 需要 64GB AGX 构建后拷回。

**0.5B 引擎性能实测**（§8.3）：生成速率 **22~30 tok/s**（平均 ~26.5）、TTFT **~1.9s**（含引擎加载开销）、峰值内存 **~4.23GB（55.7%）**、功耗 **~15.2W**、GPU 利用率 99%（算力打满，热余量充足）。

**全程共踩 11 个坑，无一例外都是"版本匹配 / 环境配置 / 内存碎片"三类工程问题**——没有一个是模型或代码本身的问题。**第一个坑就是装错版本**：按默认分支装了 v0.9.0，但 JP6.2（TRT 10.3.x）必须用 v0.6.0（详见 P0）——官方文档没提这层版本对应关系，是后续踩坑才发现。

---

## 1. 环境与版本锚点

| 项       | 值                                                            | 说明                                        |
| -------- | ------------------------------------------------------------- | ------------------------------------------- |
| 硬件     | Jetson Orin Nano Super DevKit**8GB**                    | 统一内存架构（CPU/GPU 共享物理内存），sm_87 |
| 系统     | Ubuntu 22.04.5 LTS / glibc 2.35                               | 宿主机与容器一致                            |
| JetPack  | **6.2**（L4T 36.x）                                     | meta 包 nvidia-tensorrt 6.2.3               |
| TensorRT | **10.3.0.30**（CUDA 12.5/12.6）                         | 决定导出模式选择的硬约束                    |
| 容器     | 基于 JP6.2 的 aarch64 容器（jupyter-tensorrt）                | 编译 + 导出用                               |
| 模型     | Qwen2.5-1.5B-Instruct（失败）→ Qwen2.5-0.5B-Instruct（成功） | vocab=151936                                |

**版本锚点（全流程最重要的一条规则）**：`quantize / export`（Python 环境）与 C++ Runtime 编译必须使用**同一个 git tag**（v0.6.0）。export 产出的 ONNX 节点格式与 build 端插件注册名一一对应，跨版本必挂。验证：`git describe --tags` 应显示 `v0.6.0`。

**⚠️ 软件版本与 JetPack 强绑定（本项目第一个坑，详见 P0）**：**不要 clone 默认分支**——默认分支是最新 v0.9.x，面向 Thor 时代（JetPack 7.1 / CUDA 13 / TRT 10.15+，官方文档默认流程全是 `jetson-thor`）。Orin + JetPack 6.2（TRT 10.3.x）必须用 **v0.6.x** 版本线（toolchain 显式支持 `jetson-orin`，CUDA 12.6 / SM_87 / TRT 10.3 锚点）。官方 README 没有"版本 ↔ 平台"对照矩阵，按默认分支安装必踩坑。一开始我安装错了版本导致后面无法编译runtime工具，所以特别提醒。

---

## 2. 全流程总览

```
阶段一          阶段二                阶段三            阶段四            阶段五             阶段六
环境准备   →   安装 Python        →  编译 C++       →  导出 ONNX    →  构建 TRT 引擎  →  推理 + 部署
JP6.2 容器      静态工具链           Runtime          plugin 模式       llm_build         llm_inference
               pip --no-deps        cmake 源码编译                     (无桌面窗口)      + FastAPI 网关
   │              │                   │                │                  │                  │
   │              ▼                   ▼                ▼                  ▼                  ▼
   │        tensorrt-edgellm-    llm_build /        model.onnx        llm.engine        output.json
   │        export-llm 等        llm_inference /    (含 28 个          + config.json    + /v1/chat/
   │        11 个 CLI            libNvInfer_edgellm AttentionPlugin)  + tokenizer.json  completions
   │                            _plugin.so
```

各阶段产物清单：

| 阶段 | 产物                                                                                         | 位置                                         |
| ---- | -------------------------------------------------------------------------------------------- | -------------------------------------------- |
| 安装 | `tensorrt-edgellm-*` 11 个 CLI 工具（export-llm / quantize-llm / insert-lora / ...）       | 容器 pip 环境（`/opt/venv`）               |
| 编译 | `llm_build` / `llm_inference` / `llm_bench` + `libNvInfer_edgellm_plugin.so`（37MB） | 容器`build/`，拷至宿主机 `~/trt_build/`  |
| 导出 | `model.onnx`（1670 节点，含 28 个 `AttentionPlugin`）                                    | 容器`/workspace/.../onnx_new/`，拷至宿主机 |
| 构建 | `llm.engine` + `config.json` + `tokenizer.json`                                        | 宿主机`~/engineMake_0_5b/engine_new/`      |
| 推理 | `output.json`                                                                              | 宿主机                                       |
| 性能 | `perf_report_20260813_163619.md`（4 次实测汇总，19:38 更新版，见 §8.3）                   | 宿主机                                       |

---

## 3. 阶段一：环境准备

### 3.1 硬件与系统要求

- Jetson Orin 系列（Nano / NX / AGX），**sm_87 GPU 架构**
- JetPack 6.2（TRT 10.3.x）。**注意**：官方文档 System Requirements 只写了 Thor/JP7.1，但 v0.6.0 源码 `cmake/aarch64_linux_toolchain.cmake` 明确支持 `jetson-orin` 目标（CUDA 12.6 / SM_87 / aarch64），与 JP6.2 完全匹配——文档偏向不代表不支持
- **版本选择**：clone 后必须 `git checkout v0.6.0`（默认分支 v0.9.x 无 `jetson-orin` 目标，面向 Thor，见 P0）。判断某个 tag 是否支持 Orin：看它的 toolchain 文件里有没有 `jetson-orin`
- 精度能力：Orin 仅支持 FP16 / INT8 / INT4（无 FP8 / FP4）

### 3.2 部署策略：容器 vs 宿主机

本方案采用**混合部署**，各取所长：

| 环境                     | 用途                         | 原因                                     |
| ------------------------ | ---------------------------- | ---------------------------------------- |
| 容器（jupyter-tensorrt） | 编译 C++ Runtime + 导出 ONNX | 环境隔离、Python 导出工具链齐全          |
| 宿主机（无桌面模式）     | 构建引擎 + 推理 + 服务       | 避开容器层内存开销，直接使用整机可用内存 |

> 结论来自实测：同样环境下容器内构建失败 → 宿主机直跑仍失败 → 最终确认瓶颈是 8GB 设备物理内存上限，但**宿主机无桌面模式确实能多挤出 1GB+ 内存**，是构建成败的关键变量之一（详见问题 P9/P10）。

---

## 4. 阶段二：安装 Python 静态工具链（tensorrt-edgellm v0.6.0）

> ⚠️ **安装分两部分，缺一不可**：① **Python 静态工具链**——`tensorrt-edgellm-export-llm` / `quantize-llm` 等 11 个命令行工具，pip 安装，负责**导出 ONNX 与量化**（本节）；② **C++ Runtime 工具**——`llm_build` / `llm_inference` / `llm_bench`，源码编译，负责**构建与推理**（见 §5）。详细安装过程见姊妹文档《TensorRT-Edge-LLM_容器安装报告_v0.6.0.md》。

### 4.1 核心安装命令（分层安装策略）

```bash
# ① 克隆源码（--depth 1 --branch 直接锁定 v0.6.0，天然规避 P0 的默认分支坑）
git clone --depth 1 --branch v0.6.0 https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM

# ② 安装主包（必须 --no-deps！原因见 4.2）
pip install . --no-deps

# ③ 安装量化工具 nvidia-modelopt（同样 --no-deps + 手动补运行时依赖）
pip install nvidia-modelopt==0.39.0 --no-deps
pip install omegaconf pulp pydantic nvidia-ml-py ninja

# ④ 安装模型/数据集等核心依赖（分批装，避免超时）
pip install transformers==4.57.6 datasets==4.4.2 peft==0.18.1 \
            pillow==12.1.1 backoff==2.2.1 soundfile==0.13.1 \
            librosa==0.11.0 einops==0.8.2

# ⑤ 隐式依赖（"import 报错 → 缺啥装啥"，见 4.3）
pip install scipy --only-binary :all:
pip install torchprofile
pip install onnx-graphsurgeon
```

### 4.2 为什么必须 `--no-deps`（版本冲突）

容器环境已有 **aarch64 平台优化过的** `torch 2.11.0` / `onnx 1.22.0`，而项目声明要求 `torch~=2.10.0` / `onnx==1.19.0`——直接 `pip install .` 会因 `onnx==1.19.0` 无匹配发行版直接失败：

```
ERROR: No matching distribution found for onnx==1.19.0
```

Jetson 容器里的包是平台优化过的（强制降级会丢失平台优化），所以**运行时兼容性比版本号精确匹配更重要**：用 `--no-deps` 跳过依赖解析、保持环境现有版本，实测完全兼容。

### 4.3 隐式依赖（迭代式发现）

`--no-deps` 的代价是依赖要手动补齐，且**有些依赖连声明里都没写**，只能靠迭代发现：`import tensorrt_edgellm` 报错 → 装缺的包 → 再试，循环到通过。本环境共发现 3 个：

| 隐式依赖              | 用途                                       | 安装注意                                                                                  |
| --------------------- | ------------------------------------------ | ----------------------------------------------------------------------------------------- |
| `scipy`             | modelopt NAS 模块（`scipy.interpolate`） | ⚠️**必须 `--only-binary :all:`**——aarch64 源码编译要 30+ 分钟，预编译轮子秒装 |
| `torchprofile`      | modelopt 模型性能分析                      | 普通安装                                                                                  |
| `onnx-graphsurgeon` | ONNX 图操作（INT4 量化必需）               | 普通安装                                                                                  |

### 4.4 安装验证（11 个命令全部可用）

```bash
python -c "import tensorrt_edgellm; print(tensorrt_edgellm.__version__)"   # 期望 0.6.0
python -c "import modelopt; print(modelopt.__version__)"                    # 期望 0.39.0
tensorrt-edgellm-export-llm --help                                          # 有 usage 输出即可用
```

**产物清单**：11 个 `tensorrt-edgellm-*` 命令行工具（本方案核心用到 `export-llm`；`quantize-llm` 本方案未用——原因见 P9 量化救不了构建 OOM）：

| 命令                                                                 | 功能                           | 本方案  |
| -------------------------------------------------------------------- | ------------------------------ | ------- |
| `tensorrt-edgellm-export-llm`                                      | 导出 LLM → ONNX               | ✅ 核心 |
| `tensorrt-edgellm-quantize-llm`                                    | LLM 量化（int8_sq / int4_awq） | ❌ 未用 |
| `tensorrt-edgellm-export-draft` / `quantize-draft`               | Draft 模型导出 / 量化          | ❌      |
| `tensorrt-edgellm-export-audio` / `preprocess-audio`             | 音频模型导出 / 预处理          | ❌      |
| `tensorrt-edgellm-export-visual`                                   | 视觉模型导出                   | ❌      |
| `tensorrt-edgellm-insert-lora` / `process-lora` / `merge-lora` | LoRA 适配器插入 / 处理 / 合并  | ❌      |
| `tensorrt-edgellm-reduce-vocab`                                    | 词汇表缩减                     | ❌      |

> **环境注意**：不要新建 Python 虚拟环境——容器 `/opt/venv` 已配置 aarch64 优化的 PyPI 源，新建 venv 会丢失平台预编译包。

---

## 5. 阶段三：编译 C++ Runtime（v0.6.0）

### 5.1 获取源码并配置

```bash
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM
cd TensorRT-Edge-LLM
git checkout v0.6.0          # 版本锚点，必须与导出端一致

rm -rf build && mkdir -p build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-orin \
  -DNV_ONNX_PARSER_LIB=/usr/local/cuda/targets/aarch64-linux/lib/libnvonnxparser.so
make -j4
```

**正确配置的判据**（configure 输出必须同时满足）：

1. `CUDA_CTK_VERSION set to 12.6`
2. FMHA 日志的排除列表**不再含 87**（详见 P1）
3. 无 `NOTFOUND`
4. `Build files written`

> 编译约 3-5 分钟（Orin Nano）。内存吃紧时用 `-j4` 而不是 `-j$(nproc)`。

### 5.2 编译产物清单

```
build/libNvInfer_edgellm_plugin.so.1.0   (37MB, 运行时插件库, 必须)
build/examples/llm/llm_build             (引擎构建工具)
build/examples/llm/llm_inference         (推理工具)
build/examples/llm/llm_bench             (性能测试工具)
build/examples/multimodal/visual_build, audio_build  (多模态, 本方案未用)
```

---

## 6. 阶段四：导出 ONNX（plugin 模式）

### 6.1 下载模型 + 导出

```bash
# 容器内
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir qwen25_0.5b_trt/Qwen2.5-0.5B-Instruct

tensorrt-edgellm-export-llm \
  --model_dir  /workspace/qwen25_0.5b_trt/Qwen2.5-0.5B-Instruct \
  --output_dir /workspace/qwen25_0.5b_trt/onnx_new \
  --device cuda
```

**⚠️ 关键：不要加 `--trt_native_ops`**（原因见 P5）。

### 6.2 验证导出模式（构建前的必做检查）

```bash
python3 -c "
import onnx
from collections import Counter
m = onnx.load('.../onnx_new/model.onnx')
print('total:', len(m.graph.node))
print(Counter(n.domain for n in m.graph.node))
"
```

**期望结果**（Qwen2.5-0.5B，28 层）：

```
total: 节点数
domain 分布: {'': 基础算子, 'trt': 28}     ← 28 个插件节点
```

出现 `('Attention','')` 或 `('RotaryEmbedding','')`（空 domain）= 错用了 TRT Native 模式，必须重导。

---

## 7. 阶段五：构建 TRT 引擎（llm_build）

### 7.1 标准构建命令（宿主机）

```bash
# ① 拷贝产物（容器 → 宿主机）
mkdir -p /home/zy/Desktop/workplace/engineMake_0_5b
docker cp jupyter-tensorrt:/workspace/qwen25_0.5b_trt/onnx_new \
  /home/zy/Desktop/workplace/engineMake_0_5b/
docker stop jupyter-tensorrt      # 释放内存

# ② 切无桌面模式 + MAXN（关键！释放 GPU 内存，见 P10）
sudo systemctl isolate multi-user.target
sudo nvpmodel -m 0
tegrastats --interval 1000 --stop 3   # 确认 lfb ≥ 400MB（0.5B 需 260MB）

# ③ 构建（0.5B，512/2048 参数）
cd /home/zy/Desktop/workplace/engineMake_0_5b
export EDGELLM_PLUGIN_PATH=/home/zy/Desktop/workplace/engineMake/trt_build/libNvInfer_edgellm_plugin.so
/home/zy/Desktop/workplace/engineMake/trt_build/examples/llm/llm_build \
  --onnxDir onnx_new --engineDir engine_new \
  --maxBatchSize 1 --maxInputLen 512 --maxKVCacheCapacity 2048
```

**参数含义**：`maxInputLen` = 单次输入 token 上限；`maxKVCacheCapacity` = 总 KV 容量（输入+输出）。序列总长 = 输入 + 输出，`maxKVCacheCapacity` 决定总容量。

**成功判据**：

1. 日志无 `Cannot open plugin library` / `Cannot find plugin`
2. autotuner 阶段无 `NvMapMemAllocInternalTagged error 12` / `CUDA error 2`
3. 出现 `LLM engine built successfully`
4. 产物三件套：`engine_new/llm.engine` + `config.json` + `tokenizer.json`

> `llm_build` 与插件库是**模型无关的通用工具**：同一份 `trt_build/` 可以构建任意模型（1.5B、0.5B 都是它）。引擎绑定的是 sm_87 架构 + TRT 版本 + 插件版本，不绑定具体模型。

---

## 8. 阶段六：推理验证 + 部署

### 8.1 推理验证（llm_inference）

```bash
cat > input.json << 'EOF'
{
  "batch_size": 1,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "max_generate_length": 128,
  "requests": [
    {"messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "用一句话介绍你自己"}
    ]}
  ]
}
EOF

export EDGELLM_PLUGIN_PATH=/home/zy/Desktop/workplace/engineMake/trt_build/libNvInfer_edgellm_plugin.so
/home/zy/Desktop/workplace/engineMake/trt_build/examples/llm/llm_inference \
  --engineDir engine_new --inputFile input.json --outputFile output.json
cat output.json
```

**实测输出**（验证通过）：

```json
"output_text": "我是来自阿里云的大规模语言模型，我叫通义千问。"
```

`formatted_complete_request` 含完整的 `<|im_start|>system/user/assistant` chat template 三段，说明模板拼装正确。

### 8.2 部署为 OpenAI 兼容服务

**事实**：v0.6.0 **没有内置 HTTP server**（官方 OpenAI 兼容 server 是后续版本 + Thor 平台功能）。用 FastAPI 网关包装 `llm_inference`（输入输出本身就是 OpenAI 兼容 JSON 格式），实现 `/v1/chat/completions` + `/v1/models`：

```bash
pip install fastapi uvicorn
python3 llm_server.py \
  --engine-dir    engineMake_0_5b/engine_new \
  --llm-inference engineMake/trt_build/examples/llm/llm_inference \
  --plugin        engineMake/trt_build/libNvInfer_edgellm_plugin.so \
  --port 8000
```

测试：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-0.5b","messages":[{"role":"user","content":"你好"}]}'
```

任意 OpenAI SDK 对接：

```python
from openai import OpenAI
client = OpenAI(base_url="http://<jetson_ip>:8000/v1", api_key="not-needed")
```

可选 systemd 开机自启（见附录 A）。推理阶段内存需求远小于构建（实测系统峰值 ~4.23GB / 55.7%，见 §8.3），**不需要无桌面模式**，桌面可正常使用。

### 8.3 性能验证（llm_bench 实测）

引擎构建 + 推理验证通过后，用 `llm_bench` 对 Qwen2.5-0.5B-Instruct 引擎做了 4 次实测（2026-08-13 19:38 更新生成，完整明细见 `perf_report_20260813_163619.md`）：

| 测试   | Prompt                       | TTFT (ms) | 生成速率 (tok/s) | 峰值内存 (MB) | 峰值功耗 (mW) | GPU 利用率 | GPU 温度 |
| ------ | ---------------------------- | --------- | ---------------- | ------------- | ------------- | ---------- | -------- |
| test_1 | 你好，请介绍一下自己         | 1595.52   | 21.85            | 4229（55.6%） | 15050         | 99.0%      | 57.1°C  |
| test_2 | 写一首关于人工智能的诗       | 1981.66   | 25.16            | 4237（55.7%） | 15246         | 99.0%      | 58.1°C  |
| test_3 | 解释量子计算的基本原理       | 1974.50   | 30.46            | 4236（55.7%） | 15246         | 99.0%      | 58.4°C  |
| test_4 | 如何优化 TensorRT 模型性能？ | 1979.08   | 28.44            | 4236（55.7%） | 15246         | 99.0%      | 58.6°C  |

**性能画像（4 次平均）**：

| 指标                  | 实测值                                    | 说明                                                                |
| --------------------- | ----------------------------------------- | ------------------------------------------------------------------- |
| TTFT（首 token 延迟） | **~1.9s**                           | 含引擎加载开销（每次运行重新加载 ~1-2s），纯推理首 token 应显著更低 |
| 生成速率              | **21.9 ~ 30.5 tok/s**（平均 ~26.5） | 随生成内容/长度波动，属正常                                         |
| Prefill 速率          | ~3.6-4.1 tok/s                            | 短 prompt 下 prefill 占比小                                         |
| 峰值内存              | **~4.23GB（55.7%）**                | 0.5B fp16 权重 + KV + 激活 + runtime                                |
| 峰值功耗              | **~15.2W**                          | 边缘设备可长时间稳定运行                                            |
| GPU 利用率            | **99%**                             | 计算已打满                                                          |
| GPU 温度              | ~58°C                                    | 低于降频线（90°C），热余量充足                                     |

**四点结论**：

1. **内存实测修正**：§8.1 曾估计推理峰值 ~1.5GB，实测系统峰值 **~4.23GB（55.7%）**——0.5B fp16 权重（~1GB）+ KV cache + 激活 + runtime 实际占用更高。但结论不变：仍远低于 8GB 上限，**桌面模式可正常使用**，服务部署不受影响
2. **GPU 是当前瓶颈，但热/功耗余量充足**：利用率 99% + 温度 ~58°C + 功耗 ~15.2W → 算力已打满，距离散热（90°C 降频线）与功耗上限仍有相当余量。性能提升方向：换更大模型（提高显存带宽利用率）或量化（降低计算量）
3. **TTFT 的主因是引擎加载**：`llm_inference` 每次运行重新加载引擎（~1-2s）占 TTFT ~1.9s 的大头——印证了 §10 的部署取舍：生产环境若写 C++ 常驻进程（引擎只加载一次），TTFT 可降到几百 ms 量级
4. **实用定位**：~26 tok/s 生成速度 + 15.2W 功耗，非常适合语音交互、轻量问答等边缘实时场景

---

## 9. 遇到的问题与解决方案（全记录，按流程阶段）

### P0（安装）版本误装：默认分支 v0.9.0 → 应装 v0.6.0 ⭐ 时间上最早的坑

- **现象**：`git clone https://github.com/NVIDIA/TensorRT-Edge-LLM` 后直接按**默认分支**（最新 release v0.9.0）安装，后续编译/构建阶段出现与版本相关的不兼容问题，排查之后才意识到版本装错了——JP6.2（TRT 10.3.x）需要的是 **v0.6.0**
- **根因**：**TensorRT-Edge-LLM 的版本与 JetPack/TRT 强绑定**，且官方 README 只展示最新版流程、没有"版本 ↔ 平台"对照矩阵：| 版本线                                                                                                                                                                              | 面向平台 | `EMBEDDED_TARGET` | 锚点                                 |
  | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------------- | ------------------------------------ |
  | **v0.9.x（默认分支）**                                                                                                                                                        | Thor     | `jetson-thor`     | JetPack 7.1 / CUDA 13.x / TRT 10.15+ |
  | **v0.6.x（本方案）**                                                                                                                                                          | Orin     | `jetson-orin`     | JetPack 6.2 / CUDA 12.6 / TRT 10.3.x |
  | 默认分支即 v0.9.0 → 与 JP6.2 的 TRT 10.3.x 错配。官方文档 System Requirements 只写了 Thor/JP7.1，没有提示 Orin 用户需切到 v0.6.x——**这个坑文档里完全没有，是踩了才发现的** |          |                     |                                      |
- **后果**：版本错配影响编译目标名（v0.9.x 无 `jetson-orin`）、export 的 ONNX 节点格式、插件注册名、依赖版本（v0.9.x 系面向 CUDA 13 时代），跨版本必挂
- **修复**：

```bash
git checkout v0.6.0
git describe --tags      # 验证：应显示 v0.6.0
```

然后重新走编译/导出流程

- **验证**：`git describe --tags` = `v0.6.0` + 编译 configure 输出 `CUDA_CTK_VERSION set to 12.6` + FMHA 排除列表不含 87
- **预防**：动手前先确认目标 tag 的 `cmake/aarch64_linux_toolchain.cmake` 是否含 `jetson-orin` 目标——含 Orin 的版本线（v0.6.x）才可用于 JP6.2

**P0 补充（安装）Python 静态工具链的依赖坑 — 版本冲突 + 隐式依赖**

- **现象**：`pip install .` 直接失败 `ERROR: No matching distribution found for onnx==1.19.0`；改用 `--no-deps` 装完主包后，`import tensorrt_edgellm` 连环报 `ModuleNotFoundError`（scipy → torchprofile → onnx_graphsurgeon，装一个报下一个）
- **根因**：① 项目声明 `onnx==1.19.0`，而容器已有 aarch64 平台优化的 `onnx 1.22.0`——pip 依赖解析器强制精确匹配失败；② `--no-deps` 跳过依赖解析后，主包运行时依赖（**部分甚至没写进声明**）必须手动补齐，只能靠导入报错迭代发现
- **修复**：分层安装 + 迭代式发现（完整命令见 §4.1-4.3）：主包 `--no-deps` → modelopt `--no-deps` + 手动补 omegaconf/pydantic/pulp/nvidia-ml-py/ninja → 补 transformers/datasets 等核心依赖 → scipy（`--only-binary :all:`）/torchprofile/onnx-graphsurgeon 三个隐式依赖
- **关键原则**：Jetson 容器环境是平台优化过的，**运行时兼容性 > 版本号精确匹配**；不要新建 venv（会丢 aarch64 预编译包）；scipy 在 aarch64 上源码编译 30+ 分钟，必须用预编译轮子

### P1（编译）FMHA 日志 `EXCLUDE_SM_87` — 架构配置错误

- **现象**：编译输出 `FMHA Kernels: Excluding SM architectures: EXCLUDE_SM_87;EXCLUDE_SM_100;...`
- **根因**：FMHA 的架构排除规则 = "不在 `CMAKE_CUDA_ARCHITECTURES` 中的架构"。输出排除 87 说明实际架构集 = {80,86,89}，即顶层 CMakeLists 的默认值生效 → `AARCH64_BUILD` 未定义 → **漏传 `CMAKE_TOOLCHAIN_FILE` 或 `EMBEDDED_TARGET=jetson-orin`**
- **后果**：FMHA cubin 编成 80/86/89 的 SASS，SM87 上不能最优运行（FMHA = Fused Multi-Head Attention，LLM 推理性能核心）
- **修复**：`rm -rf build` 重新配置，补全 4 个关键参数（见 §5.1）
- **验证**：排除列表不再含 87 + `CUDA_CTK_VERSION set to 12.6`

> 小坑：手动 `-DCMAKE_CUDA_ARCHITECTURES=87` 无效——顶层 CMakeLists 的普通变量会遮蔽命令行 cache 值。必须通过 toolchain 定义。

### P2（编译）`NV_ONNX_PARSER_LIB NOTFOUND` — 双因叠加

- **现象**：架构配置正确后，报 `NV_ONNX_PARSER_LIB NOTFOUND`
- **根因 ①**：命令某行末尾漏 `\`，后续 `-D` 参数被 bash 当独立命令执行（`bash: -D... No such file or directory` 即证据）
- **根因 ②**：**v0.6.0 源码 bug**——顶层 CMakeLists 的 `find_library(NV_ONNX_PARSER_LIB)` 的 `PATH_SUFFIXES` **硬编码 `x86_64-linux-gnu`**（没有 `${CMAKE_SYSTEM_PROCESSOR}-linux-gnu`），aarch64 上自动搜索必然失败（`NVINFER_LIB` 无此问题）
- **修复**：显式指定 aarch64 路径 `-DNV_ONNX_PARSER_LIB=/usr/local/cuda/targets/aarch64-linux/lib/libnvonnxparser.so`
- **验证**：configure 通过、无 NOTFOUND

> 顺带：configure 输出中 "Configurable variable ... set to 12.6" 出现两次是 toolchain 的 `set_ifndef` 宏被顶层与 toolchain 各调用一次的正常现象，不是错误。

### P3（编译）`math.h: No such file or directory` — 容器缺 libc6-dev

- **现象**：编译到 89%（utilKernels.cu / embeddingKernels.cu）报 `/usr/include/c++/11/cmath:45: #include_next <math.h>: No such file or directory`
- **根因**：容器精简镜像缺 glibc 开发头文件 **libc6-dev**（math.h 在 `/usr/include/aarch64-linux-gnu/math.h`）。CUDA 的 `math_functions.h` 用 `#include_next` 回落系统 math.h，缺头即炸；前面 89% 文件没走到这条头文件链所以能过
- **修复**：`apt-get update && apt-get install -y libc6-dev`，之后 `make` 增量编译直接继续，无需删 build

### P4（编译）装完 libc6-dev 仍找不到 math.h — 平铺头文件布局

- **现象**：装完 libc6-dev 后 `/usr/include/aarch64-linux-gnu/math.h` 仍不存在，但 `dpkg -L libc6-dev` 显示头在 `/usr/include/math.h`
- **根因**：该容器的 glibc 头**全部平铺在 `/usr/include/`**（非标准 multiarch 布局），`aarch64-linux-gnu/` 目录几乎为空。`#include_next` 搜索链为 `c++/11 → aarch64-linux-gnu → /usr/include`，在第二步**扑空就直接报错**，不会继续搜到 `/usr/include`
- **修复**（软链接补齐 multiarch 目录）：

```bash
# 最小修复：单头
ln -sf /usr/include/math.h /usr/include/aarch64-linux-gnu/math.h
# 标准做法：整套软链（multiarch 目录整体为空时）
cd /usr/include
for f in *; do ln -sf "/usr/include/$f" "/usr/include/aarch64-linux-gnu/$f"; done
```

- **为什么用 `ln -sf` 不用 `cp`**：不占空间 + 随 glibc 包升级自动同步
- **验证**：`ls -la /usr/include/aarch64-linux-gnu/math.h` 显示 `-> /usr/include/math.h`

> 经验：Jetson 自定义容器常见非标准头布局；只有 nvcc 编 `.cu` 才触发 include_next 问题，纯 C 直接搜 `/usr/include` 不会暴露。

### P5（导出/构建）`Cannot find plugin` — ONNX 导出模式错配 ⭐ 最深的坑

- **现象**：`dlopen` 成功（无 Cannot open），但 3 个自定义算子仍报 `Cannot find plugin`，parse 失败
- **根因**：v0.6.0 导出有**两种模式**：
  - **默认 plugin 模式**：导出 `trt::AttentionPlugin`（domain=`trt`），依赖 Edge-LLM 插件库
  - **`--trt_native_ops` TRT Native 模式**：导出 `Attention` / `RotaryEmbedding` / `TensorScatter`（空 domain + `TRT_decomposable` 属性），**要求 TensorRT ≥ 10.15**（源码注释原文）
- 用户 ONNX 含空 domain 的 `Attention` + TRT_decomposable → 是 Native 模式产物 → JP6.2 的 TRT 10.3.x 不满足 → 插件名（`AttentionPlugin`）也匹配不上 → parse 失败
- **修复**：去掉 `--trt_native_ops`，用默认 plugin 模式重新导出
- **验证**：ONNX 节点检查，domain 分布 `{'': 1642, 'trt': 28}`，28 个 `AttentionPlugin`

**JP6.2 必读的导出模式取舍表**：

| 模式                      | 导出算子            | domain  | 要求                            | JP6.2 (TRT 10.3.x)     |
| ------------------------- | ------------------- | ------- | ------------------------------- | ---------------------- |
| 默认**plugin 模式** | `AttentionPlugin` | `trt` | 依赖插件库                      | ✅**必须用这个** |
| `--trt_native_ops`      | `Attention` 等    | 空      | **TRT ≥ 10.15**（JP7.x） | ❌                     |

### P6（构建）`Cannot open plugin library` — 相对路径加载失败

- **现象**：`trtUtils.h:73 Cannot open plugin library: build/libNvInfer_edgellm_plugin.so`，随后 3 个算子报 Cannot find plugin → 构建卡在解析阶段（构建根本没开始，后续 2480 行全是连锁报错）
- **根因**：llm_build 默认用**相对路径** `build/libNvInfer_edgellm_plugin.so` 找插件库（相对**当前工作目录**解析，非相对可执行文件）。绝对路径调用 + 工作目录不在仓库根 → 找不到
- **修复**（推荐 ①）：

```bash
export EDGELLM_PLUGIN_PATH=/home/zy/Desktop/workplace/engineMake/trt_build/libNvInfer_edgellm_plugin.so
# 备选：cd 仓库根再跑相对路径；或 LD_PRELOAD=.../libNvInfer_edgellm_plugin.so
```

### P7（构建）autotuner OOM 445MB — 内存碎片化（lfb 不足）

- **现象**：28 个 AttentionPlugin 全部 "Successfully created"（插件匹配闭环 ✅），进入 autotuner 后约 1 分钟 OOM：`NvMapMemAllocInternalTagged: error 12` + `CUDA error 2 for 466747648-byte allocation` + `autotuner CHECK_EQ(status,myelinSuccess) LHS:21`
- **诊断三连**（实测）：| 指标                 | 值              | 结论                               |
  | -------------------- | --------------- | ---------------------------------- |
  | cgroup`memory.max` | `max`         | ✅ 容器无内存限制                  |
  | `free -h`          | free 5.2GB      | ✅ 总量充足                        |
  | `tegrastats`       | `lfb 108x4MB` | ❌**最大连续空闲块 = 432MB** |
- **根因**：Jetson 统一内存架构，autotuner 要 445MB **连续块**，而 lfb 只有 432MB——**差 13MB** → ENOMEM。free 总量 5.2GB 但全是碎片，拼不出连续大块
- **两个关键认知**：
  1. `ForeignNode[/Unsqueeze.../Cast]` 不是节点问题，是 autotuner 没内存跑 tactic 的**连带报错**，别去改 ONNX
  2. **swap 在此无效**：碎片是物理连续块问题，swap 换出不解决连续性
- **第一轮尝试（无效）**：`sync && echo 3 > /proc/sys/vm/drop_caches && echo 1 > /proc/sys/vm/compact_memory` → lfb 纹丝不动（108×4MB）。**结论：碎片在 NvMap heap 层（Jetson 统一内存 carveout），Linux buddy 的内存规整碰不到它**

### P8（构建）降参仍 OOM — 445MB 是 embedding 权重表（固定分配）

- **现象**：降参（`maxInputLen 1024→512`、`maxKVCacheCapacity 4096→2048`）重跑仍失败，分配数字 **466747392B** 与上次 **466747648B** 只差 256B（对齐误差）→ **同一笔固定分配**
- **数学解码**：

```
466,747,392 B = 151,936 (vocab_size) × 1,536 (hidden_size) × 2 (fp16 字节)
              = Qwen2.5-1.5B 的 embedding 权重表 ≈ 445 MB
```

- **根因**：构建器要把整张 embedding 表拷成**一块连续内存**，大小只由模型结构决定，与 `maxInputLen / maxKVCacheCapacity / maxBatchSize` **毫无关系**。降参路线彻底无效
- **通用公式（构建前预判）**：`embedding 表 = vocab_size × hidden_size × dtype_bytes`。1.5B = 445MB，0.5B = 151936×896×2 ≈ **260MB**
- **解法**：`sudo reboot` 重启 Jetson 整机（NvMap heap 碎片只有重启能清零，开机 lfb 恢复数 GB），重启后立即构建

### P9（构建）重启后仍失败 — 8GB 设备硬性内存天花板 ⭐ 决定性发现

- **现象链**：重启后 lfb 恢复 684MB → 容器内重试失败 → `docker cp` 到宿主机直跑仍失败 → 最干净窗口（重启 + 无桌面 + 无容器 + lfb 708MB + RAM 仅 928MB）仍失败。**五个环境全部死在同一个 445MB 分配上**
- **内存账**（构建中段的真实状态，与空闲时不同）：

```
builder Init kernel library   ≈ 986MB   ← 固定吃，与模型无关
autotuner workspace           ≈ 数百MB  ← 交错分配
embedding 表（1.5B）           445MB
```

空闲 lfb 708MB ≠ 构建中段状态。构建时这些分配**交错进行、反复试探**，8GB（RAM 7607MB）设备的 NvMap 峰值连续块必然不足

- **结论**：1.5B 在 8GB Orin 上构建**不可行**——不是碎片、不是容器、不是环境，是硬性峰值天花板
- **为什么量化救不了**：SQ/AWQ 量化的都是 Linear 层（矩阵乘），**embedding 是查表操作默认保持 fp16**——445MB 原样还在；且校准本身要加载全量 fp16 模型，先 OOM。量化是"构建成功后的优化手段"，不是解决构建 OOM 的钥匙
- **唯一出路**：换 **Qwen2.5-0.5B**（embedding 260MB < 历史最低 lfb 432MB，任何状态必过）

### P10（构建）最终成功方案 — 无桌面模式 + 重启窗口 + 0.5B

**成功组合拳**（全部要素缺一不可）：

1. **无桌面模式**：`sudo systemctl isolate multi-user.target`——桌面环境（Xorg/compositor）占 1GB+ GPU 内存，切掉后释放
2. **MAXN 功耗模式**：`sudo nvpmodel -m 0`——内存分配最充足
3. **趁重启窗口**：整机重启后 NvMap heap 清零，**立即**构建（别开桌面、别启容器，每多一分钟 lfb 都在掉）
4. **模型换 0.5B**：embedding 260MB，在 lfb 708MB 下余量充足

**lfb 判读速查**（tegrastats 输出 `RAM xxxx/7607MB (lfb Nx4MB)`）：

- `lfb Nx4MB` → 实际大小 = N × 4MB
- 0.5B 需要 ≥ 65×4MB（260MB）；1.5B 需要 ≥ 112×4MB（445MB）
- 参考：首次失败 108×4MB=432MB → 重启后 171×4MB=684MB → 两次失败构建后 5×4MB=20MB（NvMap 耗尽）

---

## 10. 推理部署补充说明

- **v0.6.0 无内置 HTTP server**：`experimental/server` + Python bindings + 流式输出是后续版本 + Thor 平台功能，JP6.2 用不了
- **部署取舍**：FastAPI + subprocess 调 llm_inference（已交付方案）每次请求重新加载引擎 ~1-2s（0.5B 很小，可接受）；生产级可写 C++ 常驻进程（`rt::LLMInferenceRuntime` 构造函数加载引擎一次，`handleRequest()` 反复调用）
- **流式输出**：v0.6.0 runtime 无 token 级流式回调，`stream=True` 会降级为一次性返回，客户端仍能拿到完整内容

---

## 11. 关键技术结论（沉淀）

1. **版本锚点（双重绑定）**：① 软件版本 ↔ 平台：Orin/JP6.2（TRT 10.3.x）必须用 v0.6.x，Thor/JP7.x 才用最新 v0.9.x——**不要 clone 默认分支**；② quantize/export 与 C++ runtime 编译必须同一 git tag；构建/推理必须同一 plugin 库
2. **JP6.2 必用 plugin 模式导出**：`--trt_native_ops` 需 TRT ≥ 10.15（JP7.x）
3. **构建可行性红线 = embedding 表**：`vocab × hidden × 2B(fp16)`。8GB Orin 上限 ~445MB，实际只有 0.5B（260MB）稳过；1B/3B 的 embedding（hidden=2048 → 593MB）比 1.5B 还大
4. **lfb（最大连续空闲块）比 free 总量关键**：NvMap heap 碎片只有整机重启能清零；drop_caches/compact/swap/降参全无效
5. **构建期内存 ≫ 推理期内存**：builder Init 固定 ~1GB + autotuner workspace + 全量 fp16 embedding；推理只吃 权重+KV+激活
6. **构建峰值估算**：≈ 权重 × 2.5~3 + builder ~1GB（由 8GB 实测标定）。16GB 上限 3B（int8 稳，fp16 边缘）；7B 需 64GB AGX 构建
7. **引擎可跨 Orin 机型迁移**：同 sm_87 + 同 TRT 版本 + 同插件，16GB 构建的引擎可直接拷回 8GB 推理；x86 云端无法代构建（架构绑定）
8. **连带报错 vs 真根因**：ForeignNode 报错、连续 2480 行 plugin 报错都是表象；用"两次分配数字是否一致"验证是否为同一固定分配
9. **0.5B 推理性能实测**（§8.3）：生成 22~30 tok/s（平均 ~26.5）、TTFT ~1.9s（**引擎每次重新加载占大头**，常驻进程可降到几百 ms）、峰值内存 ~4.23GB（55.7%）、功耗 ~15.2W、GPU 99% 利用率但温度仅 ~58°C——算力是瓶颈，热/功耗余量充足

---

## 附录 A：命令速查（从零到可用完整版）

```bash
# ═══ 阶段零：安装 Python 静态工具链（容器内，详见 §4）═══
git clone --depth 1 --branch v0.6.0 https://github.com/NVIDIA/TensorRT-Edge-LLM.git && cd TensorRT-Edge-LLM
pip install . --no-deps                              # 主包（--no-deps 避开 onnx==1.19.0 冲突，P0补充）
pip install nvidia-modelopt==0.39.0 --no-deps        # 量化工具（量化才需要，导出可不装但建议装齐）
pip install omegaconf pulp pydantic nvidia-ml-py ninja   # modelopt 运行时依赖
pip install transformers==4.57.6 datasets==4.4.2 peft==0.18.1 \
            pillow==12.1.1 backoff==2.2.1 soundfile==0.13.1 \
            librosa==0.11.0 einops==0.8.2
pip install scipy --only-binary :all:                # 隐式依赖（aarch64 必须预编译轮子！）
pip install torchprofile && pip install onnx-graphsurgeon   # 隐式依赖
tensorrt-edgellm-export-llm --help                   # 验证：11 个 CLI 工具可用

# ═══ 阶段一：编译 C++ Runtime（容器内，源码已在阶段零 clone）═══
git checkout v0.6.0   # ⚠️ 版本锚点；阶段零已 clone 到 v0.6.0（P0）
apt-get update && apt-get install -y libc6-dev          # 缺头文件时报错再装
rm -rf build && mkdir -p build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-orin \
  -DNV_ONNX_PARSER_LIB=/usr/local/cuda/targets/aarch64-linux/lib/libnvonnxparser.so
make -j4
# 编译失败缺 math.h 时（平铺头布局容器）：
ln -sf /usr/include/math.h /usr/include/aarch64-linux-gnu/math.h

# ═══ 阶段二：导出 ONNX（容器内）═══
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir qwen25_0.5b_trt/Qwen2.5-0.5B-Instruct
tensorrt-edgellm-export-llm \
  --model_dir qwen25_0.5b_trt/Qwen2.5-0.5B-Instruct \
  --output_dir qwen25_0.5b_trt/onnx_new --device cuda
# 验证导出模式（期望 {'':基础, 'trt':28}）：
python3 -c "import onnx;from collections import Counter;m=onnx.load('qwen25_0.5b_trt/onnx_new/model.onnx');print(Counter(n.domain for n in m.graph.node))"

# ═══ 阶段三：构建引擎（宿主机无桌面）═══
docker cp jupyter-tensorrt:/workspace/qwen25_0.5b_trt/onnx_new /home/zy/Desktop/workplace/engineMake_0_5b/
docker stop jupyter-tensorrt
sudo systemctl isolate multi-user.target && sudo nvpmodel -m 0
tegrastats --interval 1000 --stop 3        # 确认 lfb ≥ 65x4MB
cd /home/zy/Desktop/workplace/engineMake_0_5b
export EDGELLM_PLUGIN_PATH=/home/zy/Desktop/workplace/engineMake/trt_build/libNvInfer_edgellm_plugin.so
/home/zy/Desktop/workplace/engineMake/trt_build/examples/llm/llm_build \
  --onnxDir onnx_new --engineDir engine_new \
  --maxBatchSize 1 --maxInputLen 512 --maxKVCacheCapacity 2048
# 构建完恢复桌面：
sudo systemctl set-default graphical.target && sudo systemctl isolate graphical.target

# ═══ 阶段四：推理验证 ═══
# 写 input.json（OpenAI 兼容格式，见 §8.1）
/home/zy/Desktop/workplace/engineMake/trt_build/examples/llm/llm_inference \
  --engineDir engine_new --inputFile input.json --outputFile output.json
cat output.json

# ═══ 阶段四补充：性能测试（可选，实测汇总见 §8.3）═══
/home/zy/Desktop/workplace/engineMake/trt_build/examples/llm/llm_bench \
  --engineDir engine_new --mode prefill --inputLen 128 --outputLen 64
# 搭配 tegrastats 采样系统资源：tegrastats --interval 1000

# ═══ 阶段五：部署（可选 systemd 自启）═══
sudo tee /etc/systemd/system/edgellm-server.service > /dev/null << 'EOF'
[Unit]
Description=TensorRT-Edge-LLM OpenAI Server
After=network.target
[Service]
User=zy
WorkingDirectory=/home/zy/Desktop/workplace
ExecStart=/usr/bin/python3 /home/zy/Desktop/workplace/llm_server.py \
  --engine-dir engineMake_0_5b/engine_new \
  --llm-inference engineMake/trt_build/examples/llm/llm_inference \
  --plugin engineMake/trt_build/libNvInfer_edgellm_plugin.so
Restart=always
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable --now edgellm-server
```

## 附录 B：OOM 诊断法（三步定位）

```
① 容器限制？    cat /sys/fs/cgroup/memory.max          （max = 无限制）
② 总量够吗？    free -h                                  （够但失败 → 碎片问题）
③ 连续块够吗？  tegrastats  看 (lfb Nx4MB)               （关键指标！）
    └─ lfb ≥ embedding 表大小 → 碎片不是主因，查构建峰值公式（权重×2.5+1GB）
    └─ lfb < embedding 表大小 → NvMap 碎片，重启整机，开机立即构建
```

## 附录 C：参考资料

- TensorRT-Edge-LLM v0.6.0 官方 Installation Guide / Quick Start / Engine Builder Developer Guide
- v0.6.0 源码：`cmake/aarch64_linux_toolchain.cmake`、`cpp/CMakeLists.txt`（FMHA 逻辑）、`examples/export/attention_trt.py`（TRT Native + TRT≥10.15 注释）、`cpp/plugins/attentionPlugin.cpp`（注册名）、`cpp/trtUtils.h`（loadEdgellmPluginLib）
- jetson-ai-lab TensorRT-Edge-LLM 教程（EDGELLM_PLUGIN_PATH / LD_PRELOAD 实操）
- 姊妹文档：《TensorRT-Edge-LLM_容器安装报告_v0.6.0.md》（Python 静态工具链安装全过程，含依赖树与版本冲突细节）
- 姊妹文档：《TensorRT-Edge-LLM编译与安装排障报告.md》（问题 0-8 的深度展开版）