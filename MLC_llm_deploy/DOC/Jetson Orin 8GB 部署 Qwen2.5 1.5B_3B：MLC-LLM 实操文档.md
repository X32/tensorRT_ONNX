# Jetson Orin 8GB 部署 Qwen2.5 1.5B/3B：MLC-LLM 实操文档

> 适用机型：Jetson Orin Nano / Orin Nano Super 8GB（sm_87，Ampere 1024 CUDA 核，128-bit LPDDR5）
> 配套软件：JetPack 6.2（R36.4，CUDA 12.6.x）
> 模型目标：Qwen2.5-1.5B-Instruct / Qwen2.5-3B-Instruct，q4f16_1 量化
> 文档日期：2026-08-14 ｜ 状态：调研完成，待实操验证（已补 §9 EAGLE 推测解码第二阶段优化）

---

## 0. TL;DR（结论速览）

| 问题                   | 结论                                                                                                                         |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| MLC-LLM 是什么         | 基于 TVM 的**编译器路线**：把模型编译成 Jetson 专属 CUDA 库（.so），不是运行时解释执行                                 |
| 为什么能绕开 8GB 红线  | **编译在 PC 上做**（交叉编译），产物拷进 Jetson 即跑，构建期峰值内存根本不出现在 Jetson 上                             |
| 8GB 上跑得动吗         | 能。1.5B q4f16_1 权重仅 ~1.2GB，推理峰值 ~2.5-3GB；3B ~2.2GB 权重，峰值 ~3.5-4GB，留一半余量                                 |
| 预期速度               | 社区实测 1.5B：20-40 tok/s；NVIDIA 官方基准（Orin Nano Super，INT4）：3B 级 38-43 tok/s、7B 达 21.75 tok/s                   |
| 推荐路径               | **PC 交叉编译 → scp 拷回 Jetson**（性能与省心兼顾）；备选 dustynv 预构建容器（最快落地）                              |
| 和 TRT-Edge-LLM 的关系 | 不冲突。0.5B 继续用已跑通的 TRT 引擎；1.5B/3B 这档交给 MLC-LLM                                                               |
| 还能再快吗（第二阶段） | 能。serve 支持 EAGLE 推测解码（**无损**，预期 1.5-1.8x）：挂 ~136MB 轻量 head，1.5B 从 ~30 提到 ~45-50 tok/s，详见 §9 |

**三个必须记住的关键词**：`--host aarch64-unknown-linux-gnu`（交叉编译目标）、`"arch":"sm_87"`（Orin GPU 架构）、`q4f16_1`（默认量化格式）。

---

## 1. 为什么是 MLC-LLM：原理回顾

### 1.1 三条路线的本质区别

| 路线                          | 本质                                                        | 8GB 的痛点                                                                                   |
| ----------------------------- | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| TRT-Edge-LLM（你已跑通 0.5B） | NVIDIA 运行时**解释执行** plugin ONNX，引擎带设备指纹 | 必须在 Jetson 上构建，构建峰值内存 = 权重×2.5~3 + builder，1.5B 的 445MB embedding 直接 OOM |
| 标准 ONNX + ORT CUDA EP       | 通用运行时执行标准 ONNX                                     | 无 fused attention，只有 10-20 tok/s；ORT TensorRT EP 会重演 OOM                             |
| **MLC-LLM（本文）**     | **编译器**把模型变成专属 CUDA 程序                    | 编译在 PC 完成（无设备交叉编译），8GB 只承担运行期负载                                       |

### 1.2 三个设计决策，正好打中你的三个痛点

1. **构建/运行解耦**：`mlc_llm compile` 的产物是普通 CUDA 共享库（.so）+ MLC 格式参数目录。PC 上（或任意 CUDA 环境）编译生成，scp 过去就能跑——**构建内存压力不存在于 Jetson 上**。
2. **静态量化是编译期的事**：q4f16_1 在编译期把权重矩阵压成 int4（每 4 bit 一个权重 + 每组一个 fp16 scale），激活保持 fp16。**注意一个细节**：MLC 的 q4f16_1 默认对 embedding 层和 lm_head 保留 fp16（"q4f16" 的 f16 即指此），较新版本可用 `--quantize-embedding` 把 embedding 也压成 int4，再省 0.3-0.5GB。这和你问过的 TRT int8_sq 不同——TRT 的 SQ 量化不碰 embedding（所以 1.5B 那 445MB 在 TRT 里是死的），MLC 有路可走。
3. **kernel 为硬件定制**：TVM 的 TensorIR 针对 sm_87 的形状做融合，生成贴近手写 CUDA 的 kernel，配合 paged KV cache，比 ORT 通用 kernel 快一档。

### 1.3 为什么 PC 能给 Jetson 交叉编译（原理）

TVM 的 codegen（把计算图变成 CUDA C 源码）是 **host 侧动作**，不依赖目标 GPU 存在。编译分两步：

```
TVM 生成 CUDA C 源码（含 sm_87 的 cubin 目标）
        ↓ 交给 nvcc（CUDA 官方支持 x86 主机 → aarch64 目标交叉编译）
链接为 aarch64 Linux 的 .so（用 aarch64 交叉 GCC）
        ↓
产物：Jetson 原生 .so + MLC 权重目录
```

所以 PC 端只需要：CUDA Toolkit（12.6）+ 带 LLVM 的 TVM（能生成任意架构代码）+ aarch64 交叉 GCC。这就是"无设备交叉编译"，TVM 官方文档明确支持。

---

## 2. 方案选型：三条路径怎么选

| 路径                             | 你要做什么                                            | 成本                                       | 适合                                 |
| -------------------------------- | ----------------------------------------------------- | ------------------------------------------ | ------------------------------------ |
| **A. PC 交叉编译**（推荐） | PC 上 Docker 容器转换权重 + 编译 .so，scp 拷回 Jetson | 一次配置 Docker 镜像，编译 10-30 分钟/模型 | 追求性能、要长期部署、以后可能换模型 |
| **B. dustynv 预构建容器**  | Jetson 上拉 `dustynv/mlc` 容器，自带 TVM+MLC 编译链 | 容器镜像几个 GB，首次 JIT 编译等一会儿     | 最快看效果、不想配 PC 环境           |
| **C. Jetson 本地源码构建** | Jetson 上从源码编译 TVM + MLC-LLM                     | 数小时，8GB 上编译链可行但慢               | 只有 Jetson 没有 PC、且愿意等        |

> **建议组合**：Jetson 端 runtime 用路径 B（dustynv 容器）或路径 C（本地构建）准备好；**引擎库（.so）始终用路径 A 在 PC 上交叉编译**——这是性能和内存的甜点位。三者不冲突，可以并存。

---

## 3. 前置条件清单

### 3.1 PC 端（编译机）

| 项     | 要求                                                                               |
| ------ | ---------------------------------------------------------------------------------- |
| 系统   | x86_64 Linux（Ubuntu 22.04/24.04 已验证），或 Windows WSL2 / macOS 走 Docker       |
| 显卡   | 不需要 Jetson；推荐 NVIDIA GPU（RTX 30/40 系）加速编译期算子自动调优，没有也能编译 |
| Docker | Docker + NVIDIA Container Toolkit（`--gpus all`）                                |
| 磁盘   | ≥ 30GB（模型原始权重 + MLC 格式权重 + 镜像）                                      |
| 网络   | 能访问 HuggingFace（或配置 HF_ENDPOINT 镜像）                                      |

### 3.2 Jetson 端（运行机）

| 项   | 要求                                                                              |
| ---- | --------------------------------------------------------------------------------- |
| 系统 | JetPack 6.2 / R36.4（CUDA 12.6.x）；`cat /etc/nv_tegra_release` 确认 R36 或 R35 |
| 架构 | aarch64（Jetson 全是），GPU sm_87                                                 |
| 磁盘 | ≥ 10GB（模型权重 ~2GB + runtime/容器）                                           |
| 内存 | 8GB 统一内存（OS 占用后可用约 7GB）——本文所有数字按这个算                       |

### 3.3 版本锚点（最重要的一节）

| 组件    | 版本                                                    | 说明                                                              |
| ------- | ------------------------------------------------------- | ----------------------------------------------------------------- |
| MLC-LLM | 固定 release（如 v0.12.0）或 nightly                    | **别混用**：convert_weight / compile / run 三段必须同一版本 |
| TVM     | MLC-LLM 仓库 `3rdparty/tvm` 同源                      | 不要单独 pip 装别的 tvm，版本会错位                               |
| CUDA    | PC：12.6.x（cross 目标 sm_87）；Jetson：JP6.2 自带 12.6 | 与 TVM `USE_CUDA=ON` 匹配                                       |
| LLVM    | 20（TVM codegen 用）                                    | 博客实操验证版本                                                  |

---

## 4. 路径 A：PC 交叉编译（推荐）

> 以下流程综合自 mshr-h.com 的 AGX Orin 交叉编译实操（2026-03-19，PC=Ubuntu 24.04 + RTX 4090，目标=AGX Orin sm_87）。已按 Qwen2.5 适配。

### 4.1 第 0 步：准备交叉编译镜像

博客验证的镜像依赖清单（基于 `nvidia/cuda:12.6.3-devel-ubuntu22.04`）：

- 系统：build-essential、git、git-lfs、curl、ccache、zlib/libedit/libxml2/zstd dev 包
- **LLVM 20**（`apt.llvm.org/llvm.sh` 安装）
- **Rust**（MLC 部分组件需要）
- **uv** + Python 3.13 虚拟环境
- **Bootlin aarch64 交叉工具链**（`aarch64--glibc--stable-2022.08-1`，解压到 `/opt/toolchains/`）
- 源码构建 TVM（`USE_LLVM=ON`、`USE_CUDA=ON`、`-DCMAKE_CUDA_ARCHITECTURES=87`）——参考 §6.2 的同款命令，只是跑在 PC 上
- 源码构建 MLC-LLM（editable 安装）

一个关键的容器内环境脚本 `jetson-aarch64-env.sh`（示意）：

```bash
# /usr/local/bin/jetson-aarch64-env.sh
export PATH="/opt/toolchains/aarch64--glibc--stable-2022.08-1/bin:$PATH"
export CC=aarch64-linux-gcc
export CXX=aarch64-linux-g++
export HOST_CUDA_ARCH=89          # PC 显卡架构（示例 RTX 4090）
export TVM_CUDA_ARCHS="87;89"     # 编译 TVM 时同时启用两档
```

> ⚠️ 提示：博客原文未贴 Dockerfile 全文，上面的依赖清单是按其描述整理的"已知良好配置"骨架。首次搭镜像若踩坑，优先查 TVM 是否带 LLVM（`python -c "import tvm; print(tvm.support.libinfo()['USE_LLVM'])"` 应为 `ON`）。

### 4.2 第 1 步：下载模型（PC，宿主机）

```bash
mkdir -p dist/models
export TARGET_MODEL_NAME="Qwen2.5-1.5B-Instruct"   # 或 Qwen2.5-3B-Instruct
uvx hf download Qwen/${TARGET_MODEL_NAME} --local-dir dist/models/${TARGET_MODEL_NAME}
```

### 4.3 第 2 步：转换权重 + 生成配置（目标无关，任何机器）

```bash
export QUANTIZATION_TYPE="q4f16_1"
export CONV_TEMPLATE="qwen2"      # Qwen2.5 用 qwen2 对话模板

# 权重转 MLC 格式（int4 量化在此步完成）
mlc_llm convert_weight ./dist/models/${TARGET_MODEL_NAME}/ \
  --quantization ${QUANTIZATION_TYPE} \
  -o dist/${TARGET_MODEL_NAME}-${QUANTIZATION_TYPE}-MLC

# 生成 mlc-chat-config.json（含量化、模板、上下文长度等）
mlc_llm gen_config ./dist/models/${TARGET_MODEL_NAME}/ \
  --quantization ${QUANTIZATION_TYPE} \
  --conv-template ${CONV_TEMPLATE} \
  -o dist/${TARGET_MODEL_NAME}-${QUANTIZATION_TYPE}-MLC/
```

> 想再省 embedding 内存，可给 `convert_weight` 加 `--quantize-embedding`（新版支持），1.5B 再省 ~330MB、3B 再省 ~440MB。

### 4.4 第 3 步：交叉编译引擎库（核心步骤）

进入容器（挂载工作目录）：

```bash
docker run --rm -it \
  --gpus all --runtime=nvidia \
  -e LOCAL_UID="$(id -u)" \
  -e LOCAL_GID="$(id -g)" \
  --mount type=bind,src="$PWD",dst=/workspace \
  -w /workspace \
  mlc-llm-jetson:cu126 \
  bash
```

容器内执行（**关键参数已加粗**）：

```bash
source /usr/local/bin/jetson-aarch64-env.sh

export TARGET_MODEL_NAME="Qwen2.5-1.5B-Instruct"   # 或 Qwen2.5-3B-Instruct
export QUANTIZATION_TYPE="q4f16_1"
mkdir -p ./dist/libs

mlc_llm compile ./dist/${TARGET_MODEL_NAME}-${QUANTIZATION_TYPE}-MLC/mlc-chat-config.json \
  --device '{"kind":"cuda","tag":"","keys":["cuda","gpu"],"max_num_threads":1024,"thread_warp_size":32,"arch":"sm_87","max_threads_per_block":1024,"max_shared_memory_per_block":49152}' \
  --host aarch64-unknown-linux-gnu \
  --opt "flashinfer=0;cublas_gemm=0;faster_transformer=0;cudagraph=1;cutlass=1;ipc_allreduce_strategy=NONE" \
  -o ./dist/libs/${TARGET_MODEL_NAME}-${QUANTIZATION_TYPE}-cuda-jetson_orin_sm87.so
```

参数解读：

| 参数                                 | 含义          | 为什么                                                                                                                             |
| ------------------------------------ | ------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `--device '...arch":"sm_87"...'`   | 目标 GPU 架构 | Orin 全系都是 Ampere sm_87，**必须显式指定**，否则默认编译给 PC 显卡                                                         |
| `--host aarch64-unknown-linux-gnu` | 目标运行平台  | 让 nvcc + 交叉 GCC 产出 aarch64 的 .so                                                                                             |
| `--opt` 字符串                     | 算子开关      | 博客作者的"已知良好配置"（关 flashinfer/cublas_gemm/faster_transformer，开 cudagraph/cutlass），非通用规则，先照抄，性能不满意再调 |

编译耗时：1.5B 约 10-20 分钟、3B 约 20-40 分钟（PC 有 NVIDIA GPU 时）。

### 4.5 第 4 步：拷贝到 Jetson

**要拷两份**（缺一不可）：

```bash
# ① MLC 格式权重目录（目标无关）
scp -r ./dist/${TARGET_MODEL_NAME}-${QUANTIZATION_TYPE}-MLC jetson:/home/$USER/work/dist/

# ② 交叉编译的引擎库（Jetson 专属 .so）
scp ./dist/libs/${TARGET_MODEL_NAME}-${QUANTIZATION_TYPE}-cuda-jetson_orin_sm87.so \
    jetson:/home/$USER/work/dist/libs/
```

> 💡 `.so` 文件是 sm_87 架构的，**同架构跨机型通用**：在 AGX Orin 上编译的引擎库，Nano/Orin 8GB 一样能跑（这正是 sm_87 架构迁移的原理，和 TRT 引擎同理）。

---

## 5. 路径 B（最快落地）：dustynv 预构建容器

NVIDIA Jetson 生态的知名维护者（dusty-nv）维护了含 TVM + MLC-LLM 编译链的预构建容器，tag 与 JetPack 版本对应：

```bash
# JP6.2 对应 r36.4.0 系 tag（以 dustynv/mlc:0.20.0-r36.4.0 为例）
docker run --runtime nvidia -it --rm \
  -v ~/work:/workspace \
  -p 8000:8000 \
  dustynv/mlc:0.20.0-r36.4.0
```

容器内自带 `mlc_llm` CLI。直接把 §4.4 交叉编译的 `.so` + 权重目录挂进去跑（§7 的命令），或让它在 Jetson 上 JIT 编译（首次慢，之后缓存）。适合：不想碰 PC、先看效果。

> 注意：容器 tag 里的版本号会随上游更新，用 `docker pull dustynv/mlc` 前先到 GitHub（dusty-nv/jetson-containers）确认与 JP 版本匹配。

---

## 6. Jetson 端：runtime 准备（两种来源二选一）

### 6.1 方式一：直接用 dustynv 容器（推荐）

见 §5，`mlc_llm` 已就绪，跳到 §7。

### 6.2 方式二：Jetson 本地源码构建（备选）

> 完整命令来自 mshr-h.com 的 AGX Orin 32GB 源码构建实操（2026-03-12）。8GB 上编译链本身可行（编译的是 C++ 代码，不加载模型权重），但耗时长、建议开 swap 或空闲时段跑。

```bash
# ① 系统依赖
sudo apt update
sudo apt install -y \
  build-essential git git-lfs curl ca-certificates pkg-config wget \
  ccache libtinfo-dev zlib1g-dev libedit-dev libxml2-dev libzstd-dev \
  llvm-20-dev libpolly-20-dev
git lfs install

# LLVM 20
wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && sudo ./llvm.sh 20

# ② uv + Python 虚拟环境
curl -LsSf https://astral.sh/uv/install.sh | sh
cd ~/work && uv venv -p 3.13 .venv && source .venv/bin/activate
uv pip install cmake ninja setuptools

# ③ 克隆 MLC-LLM（带子模块）
git clone --recursive https://github.com/mlc-ai/mlc-llm
cd mlc-llm

# ④ 构建 tvm-ffi
uv pip install --editable 3rdparty/tvm-ffi --verbose --config-setting editable=compat \
  --config-setting cmake.args="-G Ninja" \
  --config-setting cmake.args="-DCMAKE_BUILD_TYPE=RelWithDebInfo" \
  --config-setting cmake.args="-DTVM_FFI_ATTACH_DEBUG_SYMBOLS=ON" \
  --config-setting cmake.args="-DTVM_FFI_BUILD_TESTS=OFF" \
  --config-setting cmake.args="-DTVM_FFI_BUILD_PYTHON_MODULE=ON" \
  --config-setting cmake.args="-DCMAKE_C_COMPILER_LAUNCHER=ccache" \
  --config-setting cmake.args="-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"

# ⑤ 构建 TVM（Jetson 关键：CUDA 架构锁死 87）
cd 3rdparty/tvm
uv pip install --editable . --verbose --config-setting editable=compat \
  --config-setting cmake.args="-G Ninja" \
  --config-setting cmake.args="-DCMAKE_BUILD_TYPE=RelWithDebInfo" \
  --config-setting cmake.args="-DUSE_LLVM=llvm-config-20 --link-static" \
  --config-setting cmake.args="-DHIDE_PRIVATE_SYMBOLS=ON" \
  --config-setting cmake.args="-DUSE_CUDA=ON" \
  --config-setting cmake.args="-DCMAKE_CUDA_ARCHITECTURES=87" \
  --config-setting cmake.args="-DUSE_CUBLAS=ON" \
  --config-setting cmake.args="-DUSE_CUTLASS=ON" \
  --config-setting cmake.args="-DUSE_THRUST=ON" \
  --config-setting cmake.args="-DUSE_NVTX=ON"

# ⑥ 验证 TVM
cd ../..
python -c "import tvm; print('USE_CUDA:', tvm.support.libinfo().get('USE_CUDA')); print('tvm.cuda().exist:', tvm.cuda().exist)"
# 期望输出：USE_CUDA: ON / tvm.cuda().exist: True

# ⑦ 构建 MLC-LLM
uv pip install --editable . --verbose --config-setting editable=compat \
  --config-setting cmake.args="-G Ninja" \
  --config-setting cmake.args="-DCMAKE_BUILD_TYPE=RelWithDebInfo" \
  --config-setting cmake.args="-DTVM_SOURCE_DIR='3rdparty/tvm'" \
  --config-setting cmake.args="-DUSE_CUDA=ON" \
  --config-setting cmake.args="-DUSE_CUTLASS=ON" \
  --config-setting cmake.args="-DUSE_CUBLAS=ON" \
  --config-setting cmake.args="-DUSE_VULKAN=OFF" \
  --config-setting cmake.args="-DUSE_METAL=OFF" \
  --config-setting cmake.args="-DUSE_OPENCL=OFF" \
  --config-setting cmake.args="-DUSE_OPENCL_ENABLE_HOST_PTR=OFF" \
  --config-setting cmake.args="-DUSE_THRUST=ON" \
  --config-setting cmake.args="-DCMAKE_CUDA_ARCHITECTURES=87" \
  --config-setting cmake.args="-DFLASHINFER_CUDA_ARCHITECTURES=87"
uv pip install --editable python --verbose --config-setting editable=compat

# ⑧ 验证
mlc_llm chat -h
python -c "import mlc_llm; print(mlc_llm)"
```

> ⏱️ AGX 32GB 上全流程约 1-2 小时；8GB 上更慢，但可行。若 `tvm.cuda().exist` 为 False，多半是 CUDA 环境变量没导出，先 `export PATH=/usr/local/cuda/bin:$PATH` 再查。

---

## 7. Jetson 端：运行与部署

### 7.1 交互验证（chat）

```bash
cd ~/work && source .venv/bin/activate   # 或进入 dustynv 容器

export TARGET_MODEL_NAME="Qwen2.5-1.5B-Instruct"
export QUANTIZATION_TYPE="q4f16_1"

mlc_llm chat --device "cuda:0" \
  --model-lib "./dist/libs/${TARGET_MODEL_NAME}-${QUANTIZATION_TYPE}-cuda-jetson_orin_sm87.so" \
  "./dist/${TARGET_MODEL_NAME}-${QUANTIZATION_TYPE}-MLC"
```

能进入交互对话 = 全链路打通。

### 7.2 OpenAI 兼容服务（serve，带流式）

```bash
mlc_llm serve --device cuda:0 \
  --model-lib "./dist/libs/${TARGET_MODEL_NAME}-${QUANTIZATION_TYPE}-cuda-jetson_orin_sm87.so" \
  "./dist/${TARGET_MODEL_NAME}-${QUANTIZATION_TYPE}-MLC" \
  --host 0.0.0.0 --port 8000 \
  --max-batch-size 1 \
  --max-total-sequence-length 4096
```

验证：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen2.5-1.5B-Instruct","messages":[{"role":"user","content":"你好"}],"stream":true,"max_tokens":128}'
```

### 7.3 Python API（MLCEngine，集成到 FastAPI/ROS2）

```python
from mlc_llm import MLCEngine

engine = MLCEngine(
    model="./dist/Qwen2.5-1.5B-Instruct-q4f16_1-MLC",
    model_lib="./dist/libs/Qwen2.5-1.5B-Instruct-q4f16_1-cuda-jetson_orin_sm87.so",
    device="cuda:0",
)

for response in engine.chat.completions.create(
    messages=[{"role": "user", "content": "你好"}],
    model="Qwen2.5-1.5B-Instruct",
    stream=True,
    max_tokens=128,
):
    for choice in response.choices:
        print(choice.delta.content, end="", flush=True)

engine.terminate()
```

### 7.4 与现有部署的衔接

- **流式**：MLC-LLM 原生支持流式（chat/serve/MLCEngine 都是真流式），对比你 TRT-Edge-LLM v0.6.0 要改源码重编译才能流式——这是 MLC 的明显加分项，ROS2 集成直接接 `/v1/chat/completions?stream=true` 即可。
- **网关**：你已有的 FastAPI 网关（`llm_server.py`）逻辑可以原样复用，只需把后端 subprocess 调用从 `llm_inference` 换成 `mlc_llm serve` 的 HTTP 接口（或者更简单——直接去掉网关层，让 ROS2 节点直连 MLC serve）。
- **并存**：0.5B TRT 引擎继续服务低延迟小任务；1.5B/3B MLC serve 处理需要智能的任务。两者互不冲突。

---

## 8. 性能预期与内存账本

### 8.1 内存账本（修正版，q4f16_1）

Qwen2.5-1.5B 和 3B 的 embedding 与 lm_head **tie 权重**（共享一张表），账本如下：

| 项                                          | 1.5B                          | 3B                 |
| ------------------------------------------- | ----------------------------- | ------------------ |
| 总参数                                      | 1.54B                         | 3.09B              |
| embedding 表（fp16，q4f16_1 默认保留）      | 467MB                         | 622MB              |
| 其余线性层（int4 + fp16 scale）             | ~0.74GB                       | ~1.56GB            |
| **权重合计（q4f16_1）**               | **~1.2GB**              | **~2.2GB**   |
| 加 `--quantize-embedding` 后              | ~0.9GB                        | ~1.7GB             |
| **推理峰值（含 KV cache + runtime）** | **~2.5-3GB**            | **~3.5-4GB** |
| 8GB 机器可用内存                            | ~7GB                          | ~7GB               |
| 余量                                        | ~4GB（可加长上下文/多 batch） | ~3GB               |

**对照你的实测**：0.5B 的 TRT 引擎（fp16，无量化）推理峰值 4.23GB。MLC 的 1.5B q4f16_1 峰值反而更低——这就是静态量化连注意力/FFN 一起压的效果。

> 想跑更大上下文：1.5B 在 8GB 上开 32K 上下文也基本可行（KV cache 1.5B 每 token 很小）；3B 建议先 8K-16K。

### 8.2 速度预期（社区区间，需实测）

| 模型                  | 来源                               | tok/s     |
| --------------------- | ---------------------------------- | --------- |
| Qwen2.5-7B INT4       | NVIDIA 官方博客（Orin Nano Super） | 21.75     |
| 3B 级 INT4            | NVIDIA 官方博客（Orin Nano Super） | 38-43     |
| 1.5B q4f16_1          | 社区实测范围（8GB 级设备）         | 20-40     |
| 3B q4f16_1            | 按 7B/3B 官方数据推算（保守）      | 20-30     |
| 你的 0.5B TRT（对照） | 实测                               | 21.9-30.5 |

> ⚠️ 官方 3B 级 38-43 tok/s 是在 Nano Super（25W 模式）测的。你的机器若跑 15W 模式，预期打七折左右（3B ≈ 25-30 tok/s）。最终以 `mlc_llm serve` 下自己的实测为准。

### 8.3 预期指标汇总（1.5B，8GB）

| 指标             | 预期                                                       |
| ---------------- | ---------------------------------------------------------- |
| TTFT（首 token） | < 1s（远优于 TRT 0.5B 的 1.9s，因为权重小 + fused kernel） |
| 生成速度         | 25-40 tok/s                                                |
| 峰值内存         | ~2.5-3GB                                                   |
| 功耗             | 略低于 TRT 0.5B 的 15.2W（权重搬运少）                     |

---

## 9. 第二阶段优化：EAGLE 推测解码（可选加速）

> **定位：不是第一步。** 先把 §7 的 chat/serve 跑出基线（20-40 tok/s），再上 EAGLE。收益明确、内存无压力、**无损**叠加，且性价比在模型越大时越明显。

### 9.1 原理：推测解码 + 特征级草稿

**推测解码（Speculative Decoding）**：LLM 生成是 memory-bound——每出一个 token 都要把全部权重从内存读一遍，GPU 算力大量闲置。思路是用一个"草稿"先猜 γ 个候选 token，目标模型做**一次** forward pass 同时验证这 γ 个——接受则净赚 γ-1 次前向。数学上拒绝采样（`min(1, p/q)`）保证**输出分布与普通解码逐 token 一致（无损）**，因此可以和量化叠加。

**EAGLE（Extrapolation Algorithm for Greater Language-model Efficiency）**：推测解码家族里的"改良草稿"——不训练完整小模型，而是给目标模型挂一个**极轻量回归头**，与独立小模型草稿对比：

| 维度      | 独立小模型草稿（small_draft）      | EAGLE 头                                                                                                |
| --------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------- |
| 预测对象  | 直接猜 token（离散，15 万选 1）    | 预测目标模型倒数第二层的**特征向量**（压扁前保留了全部判断依据），再借目标模型 LM head 反推 token |
| 接受率 α | 40-60%                             | 60-80%                                                                                                  |
| 额外内存  | 第二个完整模型（0.5B fp16 ≈ 1GB） | 目标模型 2-5% 参数（EAGLE-3~68M，fp16 **~136MB**）                                                     |
| 版本演进  | —                                 | EAGLE-1（2.7-3.5x）→ EAGLE-2（动态草稿树，3.05-4.26x）→ EAGLE-3（多层级特征融合，3.0-6.5x）           |

> ⚠️ **3-6x 是 70B 级模型的数据**。小模型（1.5B/3B）上 decode 的 memory-bound 程度低、闲置算力少，预期收益收敛到 **1.5-1.8x**，别被论文标题的数字误导。

### 9.2 MLC-LLM 支持状态（官方）

`mlc_llm serve` 原生支持三种推测解码模式（草稿模型通过 `--additional-models` 传入；具体参数以你装的版本 `mlc_llm serve --help` 为准）：

| 模式            | 草稿形态   | 备注                                           |
| --------------- | ---------- | ---------------------------------------------- |
| `small_draft` | 独立小模型 | 内存贵（~1GB），接受率一般                     |
| `eagle`       | EAGLE 头   | **本文推荐**：内存省（~136MB）、接受率高 |
| `medusa`      | 多解码头   | 另一类轻量头方案                               |

### 9.3 head 从哪来：社区有现成的，不用自己训练

- **腾讯 AngelSlim** 在 HuggingFace 发布了 Qwen2.5 系列（1.5B-32B）的**预训练 EAGLE3 head**；
- **SpecJAX** 也发布了 Qwen2.5 的 heads 可作备选。

### 9.4 落地四步（复用你现有的 mlc-cross 三件套）

1. **下载 head**：从 HF 拉取 `Qwen2.5-1.5B` 对应的 EAGLE3 head 权重；
2. **转换权重**：`mlc_llm convert_weight <head 目录> --quantization q4f16_1 -o ...`（head 参数极小，秒级完成）；
3. **交叉编译**：`mlc_llm compile`，参数与 §4.4 完全相同（`--host aarch64-unknown-linux-gnu` + `"arch":"sm_87"`），产出 head 的 `.so`；
4. **scp 拷回 Jetson**，与主模型 .so 放同一目录。

### 9.5 serve 配置（带 EAGLE）

```bash
mlc_llm serve --device cuda:0 \
  --model-lib ./dist/libs/Qwen2.5-1.5B-Instruct-q4f16_1-cuda-jetson_orin_sm87.so \
  --additional-models ./dist/libs/Qwen2.5-1.5B-Instruct-EAGLE3-q4f16_1-cuda-jetson_orin_sm87.so \
  --speculative-mode eagle \
  ./dist/Qwen2.5-1.5B-Instruct-q4f16_1-MLC \
  --host 0.0.0.0 --port 8000 \
  --max-batch-size 1 --max-total-sequence-length 4096
```

> 参数名随 MLC 版本演进（`--speculative-mode` / `--additional-models` 等），跑之前先 `mlc_llm serve --help` 确认。

### 9.6 内存账本与预期收益

| 项                                        | 数值                                                                                                |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------- |
| 1.5B q4f16_1 基线                         | ~2.5-3GB / 20-40 tok/s                                                                              |
| + EAGLE3 head（fp16）                     | +136MB                                                                                              |
| **合计**                            | **~3.2GB，8GB 依然宽裕**（对比 small_draft 模式要 +1GB）                                      |
| 预期收益（batch=1，AngelSlim 实测同口径） | Qwen3-1.7B**1.69x** / 4B 1.66x / 8B 1.70x → 1.5B 合理预期 **1.5-1.8x（~45-50 tok/s）** |

### 9.7 三个必须知道的边界（实操会踩的坑）

1. **head 是模型专属的**：Qwen2.5-1.5B 的 head 只能配 Qwen2.5-1.5B，不能拿 3B 的 head 混用；
2. **head 也要交叉编译**：同样要转 MLC 格式 + 编出 sm_87 的 `.so`，§4 流程直接复用；
3. **端到端实操记录少**："MLC serve + eagle + Qwen2.5-1.5B"这个组合目前还没有完整踩坑帖——机制上支持、组件都有，但需要你自己跑一遍验证（对比：TRT-Edge-LLM 在 JP7.1/Thor 上已集成 EAGLE，但你的 JP6.2/Orin 暂未集成，这正是 MLC 这条路的加分点）。

---

## 10. 常见坑（按优先级）

| #  | 坑                         | 现象                                                   | 解法                                                                                           |
| -- | -------------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| P0 | 版本混用                   | convert_weight/compile/serve 用不同版本，报 API 不兼容 | 三段全锁同一 tag/nightly 日期                                                                  |
| P1 | TVM 没带 LLVM              | `mlc_llm compile` 报找不到 LLVM                      | 验证 `tvm.support.libinfo()['USE_LLVM'] == 'ON'`；用 llvm-config-20 重编                     |
| P2 | 忘指定 sm_87               | .so 编译给 x86/别的架构，Jetson 上段错误               | compile 时 `--device` 的 `arch` 必须 `sm_87`；`--host` 必须 aarch64                    |
| P3 | 只拷 .so 没拷权重目录      | serve 报找不到模型权重                                 | §4.5 两份都要 scp                                                                             |
| P4 | CUDA 版本错位              | Jetson 端 `tvm.cuda().exist: False`                  | JP6.2 自带 CUDA 12.6，确保 PATH 里有 nvcc                                                      |
| P5 | 8GB 上本地 JIT 编译 OOM    | 路径 B 首次跑 HF 模型时编译卡死                        | 优先用 PC 交叉编译产物；JIT 时挂 swap（`sudo fallocate -l 8G /swapfile`）                    |
| P6 | `--opt` 参数照抄后性能差 | 某些模型对算子开关敏感                                 | 逐个试开 flashinfer/cublas_gemm 看差异（Orin 上 cutlass 通常最优）                             |
| P7 | EAGLE head 与模型不匹配    | serve 启动报 head 结构/词表不一致                      | head 严格对应同一模型规格（1.5B 的 head 只能配 1.5B）；head 也要走 §4 交叉编译出 sm_87 的 .so |

---

## 11. 决策总结与后续步骤

**给你的最终建议**：

1. **1.5B 要速度** → MLC-LLM 交叉编译路线（本文 §4 + §6.1 + §7.2），预计 25-40 tok/s、峰值 ~3GB。
2. **3B 也能上** → 同路线换 `Qwen2.5-3B-Instruct`，编译参数不变，预计 20-30 tok/s、峰值 ~4GB。
3. **0.5B 继续用 TRT** → 已跑通，别浪费；MLC 只负责 1.5B/3B 档。
4. **落地顺序建议**：先走 §5 的 dustynv 容器快速验证 MLC 在你的机器上的真实速度 → 满意后再搭 PC 交叉编译做正式部署 → 跑通基线后按 §9 上 EAGLE 推测解码冲 45-50 tok/s → 最后接 ROS2（MLC serve 原生流式 + OpenAI 兼容 API）。

**待你确认后我可以继续做的**：

- ① 写一份 PC 交叉编译的完整 Dockerfile + jetson-aarch64-env.sh（可直接 build 的版本）；
- ② 把本文 §7.2 的 serve 命令封装成 systemd 服务 / 启动脚本，对接你现有的 ROS2 网关；
- ③ 1.5B/3B 跑通后，出一份《MLC-LLM vs TRT-Edge-LLM 实测对比报告》（tok/s / 内存 / 功耗 / TTFT）。

---

## 附录 A：命令速查表

```bash
# PC：转换权重（目标无关）
mlc_llm convert_weight ./model/ --quantization q4f16_1 -o ./model-MLC
mlc_llm gen_config ./model/ --quantization q4f16_1 --conv-template qwen2 -o ./model-MLC/

# PC：交叉编译（Jetson sm_87）
mlc_llm compile ./model-MLC/mlc-chat-config.json \
  --device '{"kind":"cuda","arch":"sm_87","max_num_threads":1024,"thread_warp_size":32,"max_threads_per_block":1024,"max_shared_memory_per_block":49152}' \
  --host aarch64-unknown-linux-gnu \
  --opt "flashinfer=0;cublas_gemm=0;faster_transformer=0;cudagraph=1;cutlass=1" \
  -o ./lib-sm87.so

# Jetson：运行
mlc_llm serve --device cuda:0 --model-lib ./lib-sm87.so ./model-MLC --host 0.0.0.0 --port 8000

# Jetson：EAGLE 推测解码（第二阶段优化，见 §9）
mlc_llm serve --device cuda:0 \
  --model-lib ./lib-sm87.so \
  --additional-models ./lib-eagle3-sm87.so \
  --speculative-mode eagle \
  ./model-MLC --host 0.0.0.0 --port 8000
```

## 附录 B：版本锚点表

| 组件     | 推荐版本                | 备注        |
| -------- | ----------------------- | ----------- |
| JetPack  | 6.2（R36.4）            | 你当前环境  |
| MLC-LLM  | v0.12.0 或对应 nightly  | 三段统一    |
| CUDA     | PC 12.6.x / Jetson 12.6 | JP6.2 自带  |
| LLVM     | 20                      | TVM codegen |
| 目标架构 | sm_87                   | Orin 全系   |

## 附录 C：参考链接

- MLC-LLM 官方仓库：github.com/mlc-ai/mlc-llm
- 官方安装文档：llm.mlc.ai/docs/install/tvm.html
- HF 官方 MLC 模型库：huggingface.co/mlc-ai（Qwen2.5 系列 q4f16_1 多档可选）
- mshr-h.com：Jetson AGX Orin 源码构建（2026-03-12）与交叉编译（2026-03-19）两篇实操
- dusty-nv/jetson-containers：`dustynv/mlc` 预构建容器
- NVIDIA Jetson AI Lab：Orin Nano Super 上 MLC INT4 官方基准（7B 21.75 tok/s、3B 级 38-43 tok/s）
- 腾讯 AngelSlim（HF）：Qwen2.5 系列 1.5B-32B 的预训练 EAGLE3 head（§9 用）；SpecJAX 的 Qwen2.5 heads 可作备选
- MLC-LLM 推测解码文档：`mlc_llm serve --help` 中的 `--speculative-mode` / `--additional-models`（三种模式：small_draft / eagle / medusa）
