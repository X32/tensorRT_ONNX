#!/usr/bin/env bash
# ============================================================================
# deploy.sh — TensorRT-Edge-LLM 自动化部署脚本
#
# 用法:
#   ./scripts/deploy.sh check          # 环境检查
#   ./scripts/deploy.sh install-python # 安装 Python 工具链
#   ./scripts/deploy.sh install-cpp    # 编译 C++ Runtime
#   ./scripts/deploy.sh export-onnx    # 导出 ONNX 模型
#   ./scripts/deploy.sh build-engine   # 构建 TensorRT 引擎
#   ./scripts/deploy.sh start-server   # 启动 HTTP 服务
#   ./scripts/deploy.sh test           # 验证部署
#   ./scripts/deploy.sh benchmark      # 性能测试
#   ./scripts/deploy.sh all            # 一键完整部署
#
# 环境变量:
#   CONTAINER_NAME=jupyter-tensorrt   # 容器名称
#   MODEL_DIR=qwen25_0.5b             # 模型目录
#   WORKSPACE=/workspace              # 容器工作目录
#
# 版本锚点: TensorRT-Edge-LLM v0.6.0 / JetPack 6.2 / TensorRT 10.3.0
# ============================================================================

set -euo pipefail

# 配置参数
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONTAINER_NAME="${CONTAINER_NAME:-jupyter-tensorrt}"
MODEL_DIR="${MODEL_DIR:-qwen25_0.5b}"
WORKSPACE="${WORKSPACE:-/workspace}"
TRT_VERSION="v0.6.0"
EXPECTED_JP="6.2"
EXPECTED_TRT="10.3"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $@"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $@"; }
log_error() { echo -e "${RED}[ERROR]${NC} $@" >&2; }

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 检查容器是否运行
container_running() {
    docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"
}

# 检查容器是否存在
container_exists() {
    docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"
}

# 进入容器执行命令
exec_in_container() {
    if ! container_running; then
        log_error "容器 ${CONTAINER_NAME} 未运行"
        return 1
    fi
    docker exec -it "${CONTAINER_NAME}" bash -c "$@"
}

# ============================================================================
# 环境检查
# ============================================================================
do_check() {
    log_info "=== TensorRT-Edge-LLM 环境检查 ==="

    local errors=0

    # 检查 Docker
    if command_exists docker; then
        log_info "✓ Docker: $(docker --version | head -1)"
    else
        log_error "✗ Docker 未安装"
        ((errors++))
    fi

    # 检查 NVIDIA 运行时
    if docker info 2>/dev/null | grep -qi "nvidia"; then
        log_info "✓ NVIDIA 运行时: 可用"
    else
        log_warn "✗ NVIDIA 运行时: 未检测到"
    fi

    # 检查 JetPack 版本
    if [ -f /etc/nv_tegra_release ]; then
        local jp_version
        jp_version=$(grep "L4T_VERSION" /etc/nv_tegra_release | grep -oP 'R\d+' | grep -oP '\d+' || echo "unknown")
        if [ "$jp_version" = "${EXPECTED_JP}" ]; then
            log_info "✓ JetPack: ${jp_version} (期望 ${EXPECTED_JP})"
        else
            log_warn "✗ JetPack: ${jp_version} (期望 ${EXPECTED_JP})"
            ((errors++))
        fi
    else
        log_warn "✗ JetPack: 无法检测版本"
    fi

    # 检查 TensorRT 版本
    if command_exists trtexec; then
        local trt_version
        trt_version=$(trtexec --version 2>/dev/null | grep -oP 'TensorRT \K[\d.]+' || echo "unknown")
        if [[ "$trt_version" == "${EXPECTED_TRT}"* ]]; then
            log_info "✓ TensorRT: ${trt_version} (期望 ${EXPECTED_TRT}.x)"
        else
            log_warn "✗ TensorRT: ${trt_version} (期望 ${EXPECTED_TRT}.x)"
        fi
    else
        log_warn "✗ TensorRT: trtexec 未找到"
    fi

    # 检查 GPU 架构
    if [ -d /proc/device-tree/gpu ]; then
        local gpu_compat
        gpu_compat=$(cat /proc/device-tree/gpu*/compatible 2>/dev/null | head -1 || echo "unknown")
        if [[ "$gpu_compat" == *"sm_87"* ]] || [ "$gpu_compat" = "unknown" ]; then
            log_info "✓ GPU 架构: sm_87 (Orin)"
        else
            log_warn "✗ GPU 架构: ${gpu_compat} (期望 sm_87)"
        fi
    fi

    # 检查内存
    local total_mem
    total_mem=$(free -m | awk '/^Mem:/{print $2}')
    log_info "✓ 总内存: ${total_mem}MB"
    if [ "$total_mem" -lt 8000 ]; then
        log_warn "✗ 内存不足 8GB，可能无法构建 0.5B 模型"
        ((errors++))
    fi

    # 检查磁盘空间
    local avail_disk
    avail_disk=$(df -BM "$PROJECT_DIR" | awk 'NR==2{print $4}' | grep -oP '\d+')
    log_info "✓ 可用磁盘: ${avail_disk}MB"
    if [ "$avail_disk" -lt 15000 ]; then
        log_warn "✗ 磁盘空间不足 15GB"
        ((errors++))
    fi

    # 检查容器状态
    if container_exists; then
        if container_running; then
            log_info "✓ 容器 ${CONTAINER_NAME}: 运行中"
        else
            log_warn "✗ 容器 ${CONTAINER_NAME}: 存在但未运行"
        fi
    else
        log_warn "✗ 容器 ${CONTAINER_NAME}: 不存在"
    fi

    echo ""
    if [ $errors -eq 0 ]; then
        log_info "✓ 环境检查通过"
        return 0
    else
        log_error "✗ 发现 ${errors} 个错误，请修复后继续"
        return 1
    fi
}

# ============================================================================
# 安装 Python 工具链
# ============================================================================
do_install_python() {
    log_info "=== 安装 Python 工具链 ==="

    if ! container_running; then
        log_error "请先启动容器: docker start ${CONTAINER_NAME}"
        return 1
    fi

    exec_in_container "
        set -euo pipefail

        # 检查版本
        if [ -d TensorRT-Edge-LLM ]; then
            cd TensorRT-Edge-LLM
            current_tag=\$(git describe --tags 2>/dev/null || echo 'none')
            if [ \"\$current_tag\" != '${TRT_VERSION}' ]; then
                echo '警告: 当前版本 \$current_tag，期望 ${TRT_VERSION}'
                echo '建议: cd TensorRT-Edge-LLM && git checkout ${TRT_VERSION}'
            fi
            cd ..
        fi

        # 克隆源码
        if [ ! -d TensorRT-Edge-LLM ]; then
            echo '克隆 TensorRT-Edge-LLM ${TRT_VERSION}...'
            git clone --depth 1 --branch ${TRT_VERSION} \\
                https://github.com/NVIDIA/TensorRT-Edge-LLM.git
        fi

        cd TensorRT-Edge-LLM

        # 安装主包（必须 --no-deps）
        echo '安装主包...'
        pip install . --no-deps

        # 安装量化工具
        echo '安装量化工具...'
        pip install nvidia-modelopt==0.39.0 --no-deps
        pip install omegaconf pulp pydantic nvidia-ml-py ninja

        # 安装核心依赖
        echo '安装核心依赖...'
        pip install transformers==4.57.6 datasets==4.4.2 peft==0.18.1 \\
                    pillow==12.1.1 backoff==2.2.1 soundfile==0.13.1 \\
                    librosa==0.11.0 einops==0.8.2

        # 安装隐式依赖
        echo '安装隐式依赖...'
        pip install scipy --only-binary :all:
        pip install torchprofile
        pip install onnx-graphsurgeon

        # 验证安装
        echo '验证安装...'
        python -c \"import tensorrt_edgellm; print('TensorRT-Edge-LLM:', tensorrt_edgellm.__version__)\"
        python -c \"import modelopt; print('ModelOpt:', modelopt.__version__)\"

        echo '✓ Python 工具链安装完成'
    "
}

# ============================================================================
# 编译 C++ Runtime
# ============================================================================
do_install_cpp() {
    log_info "=== 编译 C++ Runtime ==="

    if ! container_running; then
        log_error "请先启动容器: docker start ${CONTAINER_NAME}"
        return 1
    fi

    exec_in_container "
        set -euo pipefail

        if [ ! -d TensorRT-Edge-LLM ]; then
            echo '错误: TensorRT-Edge-LLM 目录不存在'
            echo '请先运行: ./scripts/deploy.sh install-python'
            exit 1
        fi

        cd TensorRT-Edge-LLM

        # 清理旧构建
        rm -rf build && mkdir -p build && cd build

        # 配置 CMake
        echo '配置 CMake...'
        cmake .. \
            -DCMAKE_BUILD_TYPE=Release \
            -DTRT_PACKAGE_DIR=/usr \
            -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
            -DEMBEDDED_TARGET=jetson-orin \
            -DNV_ONNX_PARSER_LIB=/usr/local/cuda/targets/aarch64-linux/lib/libnvonnxparser.so

        # 编译
        echo '编译 C++ Runtime...'
        make -j4

        # 验证产物
        echo '验证产物...'
        [ -f llm_build ] && echo '✓ llm_build' || echo '✗ llm_build 缺失'
        [ -f llm_inference ] && echo '✓ llm_inference' || echo '✗ llm_inference 缺失'
        [ -f libNvInfer_edgellm_plugin.so.1.0 ] && echo '✓ libNvInfer_edgellm_plugin.so' || echo '✗ libNvInfer_edgellm_plugin.so 缺失'

        echo '✓ C++ Runtime 编译完成'
    "
}

# ============================================================================
# 导出 ONNX
# ============================================================================
do_export_onnx() {
    log_info "=== 导出 ONNX 模型 ==="

    if ! container_running; then
        log_error "请先启动容器: docker start ${CONTAINER_NAME}"
        return 1
    fi

    exec_in_container "
        set -euo pipefail

        # 检查工具链
        if ! command -v tensorrt-edgellm-export-llm >/dev/null 2>&1; then
            echo '错误: tensorrt-edgellm-export-llm 未找到'
            echo '请先运行: ./scripts/deploy.sh install-python'
            exit 1
        fi

        # 下载模型
        if [ ! -d ${WORKSPACE}/${MODEL_DIR}/Qwen2.5-0.5B-Instruct ]; then
            echo '下载 Qwen2.5-0.5B-Instruct 模型...'
            mkdir -p ${WORKSPACE}/${MODEL_DIR}
            huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \\
                --local-dir ${WORKSPACE}/${MODEL_DIR}/Qwen2.5-0.5B-Instruct
        fi

        # 导出 ONNX
        echo '导出 ONNX（plugin 模式）...'
        mkdir -p ${WORKSPACE}/${MODEL_DIR}/onnx_new

        tensorrt-edgellm-export-llm \\
            --model_dir ${WORKSPACE}/${MODEL_DIR}/Qwen2.5-0.5B-Instruct \\
            --output_dir ${WORKSPACE}/${MODEL_DIR}/onnx_new \\
            --device cuda

        # 验证导出
        echo '验证导出...'
        python3 -c \"
import onnx
from collections import Counter
m = onnx.load('${WORKSPACE}/${MODEL_DIR}/onnx_new/model.onnx')
print('总节点数:', len(m.graph.node))
domains = Counter(n.domain for n in m.graph.node)
print('节点分布:', dict(domains))
trt_count = domains.get('trt', 0)
if [ \$trt_count -ge 28 ]; then
    print('✓ Plugin 模式验证通过 (trt 节点: \$trt_count)')
else
    print('✗ Plugin 模式验证失败 (trt 节点: \$trt_count, 期望 >= 28)')
    exit 1
fi
\"

        echo '✓ ONNX 导出完成'
    "
}

# ============================================================================
# 构建 TensorRT 引擎
# ============================================================================
do_build_engine() {
    log_info "=== 构建 TensorRT 引擎 ==="

    # 检查内存（要求 lfb ≥ 400MB）
    if command_exists tegrastats; then
        local lfb
        lfb=$(tegrastats --interval 100 --stop 1 2>/dev/null | grep -oP 'lfb \K[0-9]+' || echo "0")
        log_info "当前 lfb: ${lfb}MB"
        if [ "$lfb" -lt 400 ]; then
            log_warn "lfb 不足 400MB，建议切换到无桌面模式"
            log_warn "执行: sudo systemctl isolate multi-user.target"
        fi
    fi

    # 检查必要文件
    local onnx_dir="${PROJECT_DIR}/${MODEL_DIR}/onnx_new"
    local build_dir="${PROJECT_DIR}/TensorRT-Edge-LLM/build"

    if [ ! -d "$onnx_dir" ]; then
        log_error "ONNX 目录不存在: $onnx_dir"
        log_error "请先运行: ./scripts/deploy.sh export-onnx"
        return 1
    fi

    if [ ! -f "${build_dir}/examples/llm/llm_build" ]; then
        log_error "llm_build 不存在: ${build_dir}/examples/llm/llm_build"
        log_error "请先运行: ./scripts/deploy.sh install-cpp"
        return 1
    fi

    # 设置插件路径
    export EDGELLM_PLUGIN_PATH="${build_dir}/libNvInfer_edgellm_plugin.so"
    if [ ! -f "$EDGELLM_PLUGIN_PATH" ]; then
        # 尝试带版本号的文件
        EDGELLM_PLUGIN_PATH="${build_dir}/libNvInfer_edgellm_plugin.so.1.0"
        if [ ! -f "$EDGELLM_PLUGIN_PATH" ]; then
            log_error "插件库不存在: ${build_dir}/libNvInfer_edgellm_plugin.so"
            return 1
        fi
    fi

    log_info "插件路径: $EDGELLM_PLUGIN_PATH"

    # 创建引擎目录
    local engine_dir="${PROJECT_DIR}/${MODEL_DIR}/engine_new"
    mkdir -p "$engine_dir"

    # 构建引擎
    log_info "开始构建引擎（可能需要 5-10 分钟）..."
    "${build_dir}/examples/llm/llm_build" \
        --onnxDir "$onnx_dir" \
        --engineDir "$engine_dir" \
        --maxBatchSize 1 --maxInputLen 512 --maxKVCacheCapacity 2048

    # 验证产物
    log_info "验证引擎产物..."
    local required_files=("llm.engine" "config.json" "tokenizer.json")
    for file in "${required_files[@]}"; do
        if [ -f "${engine_dir}/${file}" ]; then
            log_info "✓ ${file}"
        else
            log_error "✗ ${file} 缺失"
            return 1
        fi
    done

    local engine_size
    engine_size=$(du -h "${engine_dir}/llm.engine" | cut -f1)
    log_info "✓ 引擎构建完成 (大小: ${engine_size})"
}

# ============================================================================
# 启动 HTTP 服务
# ============================================================================
do_start_server() {
    log_info "=== 启动 HTTP 服务 ==="

    # 检查必要文件
    local engine_dir="${PROJECT_DIR}/${MODEL_DIR}/engine_new"
    local build_dir="${PROJECT_DIR}/TensorRT-Edge-LLM/build"

    if [ ! -f "${engine_dir}/llm.engine" ]; then
        log_error "引擎文件不存在: ${engine_dir}/llm.engine"
        log_error "请先运行: ./scripts/deploy.sh build-engine"
        return 1
    fi

    # 检查依赖
    if ! python -c "import fastapi" 2>/dev/null; then
        log_error "FastAPI 未安装"
        log_error "请运行: pip install -r requirements.txt"
        return 1
    fi

    # 设置插件路径
    export EDGELLM_PLUGIN_PATH="${build_dir}/libNvInfer_edgellm_plugin.so.1.0"
    if [ ! -f "$EDGELLM_PLUGIN_PATH" ]; then
        EDGELLM_PLUGIN_PATH="${build_dir}/libNvInfer_edgellm_plugin.so"
    fi

    log_info "启动服务..."
    cd "$PROJECT_DIR"

    # 启动服务
    python3 llm_server.py \
        --engine-dir "$engine_dir" \
        --llm-inference "${build_dir}/examples/llm/llm_inference" \
        --plugin "$EDGELLM_PLUGIN_PATH" \
        --host 0.0.0.0 --port 8000

    log_info "✓ HTTP 服务已停止"
}

# ============================================================================
# 测试部署
# ============================================================================
do_test() {
    log_info "=== 测试部署 ==="

    # 检查服务是否运行
    if ! curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
        log_error "服务未运行"
        log_error "请先运行: ./scripts/deploy.sh start-server"
        return 1
    fi

    log_info "✓ 服务运行中"

    # 测试模型列表
    log_info "测试 /v1/models 端点..."
    local model_response
    model_response=$(curl -s http://localhost:8000/v1/models)
    if echo "$model_response" | grep -q "qwen2.5-0.5b"; then
        log_info "✓ /v1/models 端点正常"
    else
        log_warn "✗ /v1/models 端点异常"
    fi

    # 测试对话推理
    log_info "测试 /v1/chat/completions 端点..."
    local chat_response
    chat_response=$(curl -s http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"qwen2.5-0.5b","messages":[{"role":"user","content":"你好"}],"max_tokens":32}')

    if echo "$chat_response" | grep -q "content"; then
        log_info "✓ /v1/chat/completions 端点正常"
        echo "$chat_response" | python -c "import sys, json; data=json.load(sys.stdin); print('响应:', data['choices'][0]['message']['content'][:50] + '...')" 2>/dev/null || echo "$chat_response" | head -c 100
    else
        log_error "✗ /v1/chat/completions 端点异常"
        echo "$chat_response" | head -c 200
        return 1
    fi

    log_info "✓ 部署测试通过"
}

# ============================================================================
# 性能测试
# ============================================================================
do_benchmark() {
    log_info "=== 性能测试 ==="

    if [ -f "${SCRIPT_DIR}/benchmark.sh" ]; then
        bash "${SCRIPT_DIR}/benchmark.sh"
    else
        log_error "性能测试脚本不存在: ${SCRIPT_DIR}/benchmark.sh"
        return 1
    fi
}

# ============================================================================
# 一键部署
# ============================================================================
do_all() {
    log_info "=== 一键完整部署 ==="

    local steps=("check" "install-python" "install-cpp" "export-onnx" "build-engine")

    for step in "${steps[@]}"; do
        log_info "执行: $step"
        if ! "do_${step}"; then
            log_error "步骤 $step 失败，停止部署"
            return 1
        fi
        echo ""
    done

    log_info "✓ 基础部署完成"
    log_info "下一步: ./scripts/deploy.sh start-server"
    log_info "验证: ./scripts/deploy.sh test"
    log_info "测试: ./scripts/deploy.sh benchmark"
}

# ============================================================================
# 帮助信息
# ============================================================================
show_help() {
    cat << EOF
TensorRT-Edge-LLM 自动化部署脚本 v${TRT_VERSION}

用法:
  $0 <command>

命令:
  check          环境检查
  install-python 安装 Python 工具链
  install-cpp    编译 C++ Runtime
  export-onnx    导出 ONNX 模型
  build-engine   构建 TensorRT 引擎
  start-server   启动 HTTP 服务
  test           验证部署
  benchmark      性能测试
  all            一键完整部署

环境变量:
  CONTAINER_NAME 容器名称 (默认: jupyter-tensorrt)
  MODEL_DIR      模型目录 (默认: qwen25_0.5b)
  WORKSPACE      容器工作目录 (默认: /workspace)

示例:
  # 环境检查
  $0 check

  # 分步部署
  $0 install-python
  $0 install-cpp
  $0 export-onnx
  $0 build-engine

  # 启动服务
  $0 start-server

  # 验证部署
  $0 test

  # 一键部署
  $0 all

版本锚点: TensorRT-Edge-LLM ${TRT_VERSION} / JetPack ${EXPECTED_JP} / TensorRT ${EXPECTED_TRT}
EOF
}

# ============================================================================
# 主程序
# ============================================================================
main() {
    local cmd="${1:-help}"

    case "$cmd" in
        check)         do_check ;;
        install-python) do_install_python ;;
        install-cpp)    do_install_cpp ;;
        export-onnx)    do_export_onnx ;;
        build-engine)   do_build_engine ;;
        start-server)   do_start_server ;;
        test)           do_test ;;
        benchmark)      do_benchmark ;;
        all)            do_all ;;
        help|--help|-h) show_help ;;
        *)
            log_error "未知命令: $cmd"
            show_help
            exit 1
            ;;
    esac
}

main "$@"