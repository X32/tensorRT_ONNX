# ============================================================================
# PC 交叉编译镜像：x86_64 PC → Jetson Orin (aarch64, sm_87)
# 基础: nvidia/cuda:12.6.3-devel-ubuntu22.04（与 Jetson JP6.2 CUDA 12.6 对齐）
# 产物: mlc-llm-jetson:cu126
#
# 构建（PC 上，需 Docker + NVIDIA Container Toolkit）:
#   docker build -t mlc-llm-jetson:cu126 .
#
# 运行（挂载工作目录）:
#   docker run --rm -it --gpus all --runtime=nvidia \
#     -e LOCAL_UID="$(id -u)" -e LOCAL_GID="$(id -g)" \
#     --mount type=bind,src="$PWD",dst=/workspace -w /workspace \
#     mlc-llm-jetson:cu126 bash
#
# 版本锚点: MLC-LLM v0.20.0 / TVM 同源 3rdparty/tvm / LLVM 20 / CUDA 12.6
# ============================================================================
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ARG MLC_VERSION=v0.20.0

# ---- 系统依赖 ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git git-lfs curl ca-certificates pkg-config wget ccache \
        zlib1g-dev libedit-dev libxml2-dev libzstd-dev libtinfo-dev \
        python3 python3-pip sudo \
    && rm -rf /var/lib/apt/lists/*

# ---- LLVM 20（TVM codegen 依赖）----
RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh 20 \
    && apt-get install -y --no-install-recommends llvm-20-dev libpolly-20-dev clang-20 \
    && rm -rf /var/lib/apt/lists/*

# ---- Rust（MLC 部分组件需要）----
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# ---- uv + Python 3.13 虚拟环境 ----
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN uv venv -p 3.13 /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv

# ---- Bootlin aarch64 交叉工具链 ----
RUN mkdir -p /opt/toolchains \
    && wget -q -O toolchain.tar.xz \
        https://toolchains.bootlin.com/downloads/releases/toolchains/aarch64/tarballs/aarch64--glibc--stable-2022.08-1.tar.xz \
    && tar -xJf toolchain.tar.xz -C /opt/toolchains \
    && rm toolchain.tar.xz

# ---- 交叉编译环境脚本 ----
COPY jetson-aarch64-env.sh /usr/local/bin/jetson-aarch64-env.sh
RUN chmod +x /usr/local/bin/jetson-aarch64-env.sh

# ---- 源码构建 MLC-LLM v0.20.0（含同源 TVM）----
RUN git clone --recursive --branch ${MLC_VERSION} --depth 1 \
        https://github.com/mlc-ai/mlc-llm /opt/mlc-llm

# tvm-ffi（TVM 前置依赖）
RUN cd /opt/mlc-llm \
    && uv pip install --editable 3rdparty/tvm-ffi --verbose --config-setting editable=compat \
        --config-setting cmake.args="-G Ninja" \
        --config-setting cmake.args="-DCMAKE_BUILD_TYPE=RelWithDebInfo" \
        --config-setting cmake.args="-DTVM_FFI_ATTACH_DEBUG_SYMBOLS=ON" \
        --config-setting cmake.args="-DTVM_FFI_BUILD_TESTS=OFF" \
        --config-setting cmake.args="-DTVM_FFI_BUILD_PYTHON_MODULE=ON" \
        --config-setting cmake.args="-DCMAKE_C_COMPILER_LAUNCHER=ccache" \
        --config-setting cmake.args="-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"

# TVM：同时启用 PC 架构(61=1080Ti)与 Jetson 目标架构(87=Orin)
RUN cd /opt/mlc-llm/3rdparty/tvm \
    && uv pip install --editable . --verbose --config-setting editable=compat \
        --config-setting cmake.args="-G Ninja" \
        --config-setting cmake.args="-DCMAKE_BUILD_TYPE=RelWithDebInfo" \
        --config-setting cmake.args="-DUSE_LLVM=llvm-config-20 --link-static" \
        --config-setting cmake.args="-DHIDE_PRIVATE_SYMBOLS=ON" \
        --config-setting cmake.args="-DUSE_CUDA=ON" \
        --config-setting cmake.args="-DCMAKE_CUDA_ARCHITECTURES=61;87" \
        --config-setting cmake.args="-DUSE_CUBLAS=ON" \
        --config-setting cmake.args="-DUSE_CUTLASS=ON" \
        --config-setting cmake.args="-DUSE_THRUST=ON" \
        --config-setting cmake.args="-DUSE_NVTX=ON"

# MLC-LLM 本体
RUN cd /opt/mlc-llm \
    && uv pip install --editable . --verbose --config-setting editable=compat \
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
        --config-setting cmake.args="-DCMAKE_CUDA_ARCHITECTURES=61;87" \
        --config-setting cmake.args="-DFLASHINFER_CUDA_ARCHITECTURES=87" \
    && uv pip install --editable python --verbose --config-setting editable=compat

# ---- 构建验证（fail-fast）----
RUN python -c "import tvm; assert tvm.support.libinfo().get('USE_LLVM') == 'ON', 'LLVM 未启用'; print('TVM USE_LLVM: ON')" \
    && mlc_llm chat -h >/dev/null && echo "mlc_llm CLI OK"

WORKDIR /workspace
CMD ["bash"]
