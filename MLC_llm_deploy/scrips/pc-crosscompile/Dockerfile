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
# 版本锚点: 与 Jetson 侧运行时 dustynv/mlc:0.20.0-r36.4.0 严格同源（见 MLC_COMMIT）
# ============================================================================
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# 钉 dustynv/mlc:0.20.0-r36.4.0 容器内的实际源码提交（mlc-llm main @ 2025-05-01），
# 其子模块指针即旧代 FFI 的 TVM（mlc-ai/relax @ 9c894f78, 2024-02-22）。
# 不能用 v0.20.0 release tag：它钉的是新一代 tvm-ffi TVM（b628d91f），
# 编出的 .so 引用 TVMFFIErrorSetRaisedFromCStrParts 等符号，dustynv 运行时没有，
# dlopen 报 undefined symbol。
ARG MLC_COMMIT=d2118b3c9d56da6d1e66dfe2667f650020417010

# ---- 系统依赖 ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git git-lfs curl ca-certificates pkg-config wget ccache \
        lsb-release gnupg software-properties-common \
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

# ---- uv + Python 3.10 虚拟环境（旧代 TVM 的 Cython 不支持 3.13；与容器运行时一致）----
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN uv venv -p 3.10 /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv

# ---- Bootlin aarch64 交叉工具链 ----
RUN mkdir -p /opt/toolchains \
    && wget -q -O toolchain.tar.xz \
        https://toolchains.bootlin.com/downloads/releases/toolchains/aarch64/tarballs/aarch64--glibc--stable-2024.05-1.tar.xz \
    && tar -xJf toolchain.tar.xz -C /opt/toolchains \
    && rm toolchain.tar.xz

# ---- 交叉编译环境脚本 ----
COPY jetson-aarch64-env.sh /usr/local/bin/jetson-aarch64-env.sh
RUN chmod +x /usr/local/bin/jetson-aarch64-env.sh

# ---- 源码构建 MLC-LLM（钉 commit，与 dustynv 运行时同源）----
RUN git clone https://github.com/mlc-ai/mlc-llm /opt/mlc-llm \
    && cd /opt/mlc-llm \
    && git checkout ${MLC_COMMIT} \
    && git submodule update --init --recursive --depth 1

# cmake/ninja：直接用 cmake 构建（老一代流程）需要显式安装；
# 之前 scikit-build-core 在隔离环境里自带，镜像 PATH 里没有
RUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# setuptools/cython：TVM 的 CMake 会裸调 /opt/venv/bin/python setup.py build_ext
# 编 Cython 扩展，不经过 pip 的构建隔离；uv venv 默认极简不含 setuptools
RUN uv pip install setuptools cython

# Python.h：uv venv -p 3.10 直接复用系统 /usr/bin/python3.10（uv 只在系统没有
# 对应版本时才下载托管版），头文件需 python3.10-dev 提供，落在
# /usr/include/python3.10——正是 distutils 编 Cython 扩展时查找的路径
RUN apt-get update && apt-get install -y --no-install-recommends python3.10-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ls /usr/include/python3.10/Python.h

# ---- 老式两步构建（9c894f78/d2118b3 这一代的官方流程）----
# 注意: 这一代 TVM/mlc-llm 的打包是 setup.py 时代（python/ 子目录、setuptools 后端），
# 不支持 scikit-build-core 的 --config-setting cmake.args 传参——C++ 必须用 cmake 直接编。
# ① TVM: 编 libtvm.so（CUDA+CUTLASS 会连带编 fpA_intB_gemm/flash_attn 并链入 libtvm）
RUN cd /opt/mlc-llm/3rdparty/tvm \
    && mkdir -p build && cd build \
    && cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DUSE_LLVM="llvm-config-20 --link-static" \
        -DHIDE_PRIVATE_SYMBOLS=ON \
        -DUSE_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="61;87" \
        -DUSE_CUBLAS=ON \
        -DUSE_CUTLASS=ON \
        -DUSE_NVTX=ON \
    && ninja

# ② MLC-LLM: 编 libmlc_llm_module.so（add_subdirectory 带 BUILD_DUMMY_LIBTVM，
#    动态链接 ① 的 libtvm.so，由下方 ldconfig 层保证可寻址）
# tokenizers-cpp 的 CMakeLists 会用 TOKENIZERS_CPP_RUST_FLAGS(无条件置空)覆盖
# RUSTFLAGS 环境变量，env 路线封死；新版 rustc 的 dangerous_implicit_autorefs
# deny-lint 对这份老代码会硬报错——按 rustc 建议把两处隐式引用改成显式，
# 语义等价，编译产物不变
RUN cd /opt/mlc-llm \
    && sed -i 's/(\*handle)\.decode_str\.len()/(\&(*handle).decode_str).len()/; s/(\*handle)\.id_to_token_result\.len()/(\&(*handle).id_to_token_result).len()/' \
        3rdparty/tokenizers-cpp/rust/src/lib.rs \
    && [ "$(grep -c '(&(\*handle)' 3rdparty/tokenizers-cpp/rust/src/lib.rs)" -ge 2 ] \
    && mkdir -p build && cd build \
    && cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    && ninja

# ③ Python 包: 从各自 python/ 目录装（真正的 setup.py 在这里，仓库根目录的
#    pyproject.toml 只是代码风格配置，从根目录装会得到 UNKNOWN-0.0.0 空包）
RUN cd /opt/mlc-llm/3rdparty/tvm/python \
    && uv pip install --editable . \
    && cd /opt/mlc-llm/python \
    && uv pip install --editable .

# pytest：新一代 TVM 的顶层导入链依赖 pytest；当前钉的旧代 TVM 不需要，
# 但保留此层无害（独立层，不扰动编译缓存，将来升级版本可少踩一次坑）
RUN uv pip install pytest

# ---- 运行时库路径固化 ----
# libtvm.so 依赖 fpA_intB_gemm/flash_attn 等 MLC 插件库，它们散落在构建目录里。
# 之前只在验证层用临时 LD_LIBRARY_PATH 解决，但 docker run 起的容器不继承该变量，
# 交叉编译脚本里的容器会报 "libfpA_intB_gemm.so: cannot open shared object file"。
# 将全部产物目录写入 ld.so.conf 并刷新缓存，任何容器无需手动设置即可加载。
# （不含 CUDA stubs 目录——运行时 --gpus 会注入真驱动，stub 进缓存会顶掉真驱动）
RUN find /opt/mlc-llm /opt/venv -name '*.so' -printf '%h\n' 2>/dev/null | sort -u \
        > /etc/ld.so.conf.d/mlc-llm.conf \
    && ldconfig

# ---- 构建验证（fail-fast）----
# 注意: docker build 阶段无 GPU 驱动注入，用 CUDA stub 库满足 import tvm 的
# libcuda.so.1 依赖（仅此 RUN 层内生效，运行时由 --gpus 注入真驱动）
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && export LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs" \
    && python -c "import tvm; tvm.get_global_func('target.build.llvm'); print('TVM LLVM codegen: OK')" \
    && mlc_llm chat -h >/dev/null && echo "mlc_llm CLI OK"

WORKDIR /workspace
CMD ["bash"]
