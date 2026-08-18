#!/usr/bin/env bash
# ============================================================================
# build_engine.sh — TensorRT 引擎构建脚本
#
# 专门处理内存敏感的引擎构建过程，自动管理无桌面模式和功耗设置
#
# 用法:
#   ./scripts/build_engine.sh [onnx_dir] [engine_dir]
#
# 环境变量:
#   EDGELLM_PLUGIN_PATH    # 插件库路径
#   MAX_INPUT_LEN          # 最大输入长度 (默认 512)
#   MAX_KV_CAPACITY        # 最大 KV 容量 (默认 2048)
#   SKIP_MODE_SWITCH       # 跳过无桌面模式切换 (默认 false)
#
# 版本要求: TensorRT-Edge-LLM v0.6.0 + JetPack 6.2
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $@"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $@"; }
log_error() { echo -e "${RED}[ERROR]${NC} $@" >&2; }
log_step() { echo -e "${BLUE}[STEP]${NC} $@"; }

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 检查内存状态
check_memory_status() {
    if ! command_exists tegrastats; then
        log_warn "tegrastats 不可用，跳过内存检查"
        return 0
    fi

    log_step "检查内存状态..."
    local mem_info
    mem_info=$(tegrastats --interval 1000 --stop 3 2>/dev/null)

    # 提取内存信息
    local mem_usage
    mem_usage=$(echo "$mem_info" | grep -oP 'RAM [0-9]+/[0-9]+MiB' | head -1)
    log_info "内存使用: $mem_usage"

    # 提取 lfb (最大连续空闲块)
    local lfb
    lfb=$(echo "$mem_info" | grep -oP 'lfb \K[0-9]+' || echo "0")
    log_info "最大连续空闲块 (lfb): ${lfb}MB"

    # 检查 lfb 是否足够 (0.5B 模型需要 ~260MB，留出余量)
    if [ "$lfb" -lt 400 ]; then
        log_warn "lfb 不足 400MB (当前 ${lfb}MB)"
        log_warn "可能导致引擎构建失败"
        return 1
    fi

    log_info "✓ 内存状态良好"
    return 0
}

# 切换到无桌面模式
switch_to_headless_mode() {
    if [ "${SKIP_MODE_SWITCH:-false}" = "true" ]; then
        log_warn "跳过无桌面模式切换 (SKIP_MODE_SWITCH=true)"
        return 0
    fi

    log_step "切换到无桌面模式..."

    # 检查当前运行级别
    local current_target
    current_target=$(systemctl get-default)

    if [ "$current_target" = "multi-user.target" ]; then
        log_info "已经在无桌面模式"
    else
        log_info "当前模式: $current_target"

        # 切换到无桌面模式
        log_info "执行: sudo systemctl isolate multi-user.target"
        sudo systemctl isolate multi-user.target

        # 等待切换完成
        sleep 3
        log_info "✓ 已切换到无桌面模式"
    fi

    # 设置 MAXN 功耗模式
    log_step "设置 MAXN 功耗模式..."
    if command_exists nvpmodel; then
        sudo nvpmodel -m 0
        log_info "✓ MAXN 模式已启用"
    else
        log_warn "nvpmodel 不可用，跳过功耗模式设置"
    fi
}

# 恢复桌面模式
restore_graphical_mode() {
    if [ "${SKIP_MODE_SWITCH:-false}" = "true" ]; then
        return 0
    fi

    log_step "恢复图形模式..."
    sudo systemctl isolate graphical.target
    log_info "✓ 已恢复图形模式"
}

# 设置插件路径
setup_plugin_path() {
    local build_dir="$1"

    # 检查插件库
    if [ -n "${EDGELLM_PLUGIN_PATH:-}" ]; then
        log_info "使用环境变量插件路径: $EDGELLM_PLUGIN_PATH"
        if [ ! -f "$EDGELLM_PLUGIN_PATH" ]; then
            log_error "插件库不存在: $EDGELLM_PLUGIN_PATH"
            return 1
        fi
        export EDGELLM_PLUGIN_PATH
        return 0
    fi

    # 自动查找插件库
    local plugin_paths=(
        "$build_dir/libNvInfer_edgellm_plugin.so"
        "$build_dir/libNvInfer_edgellm_plugin.so.1.0"
    )

    for plugin_path in "${plugin_paths[@]}"; do
        if [ -f "$plugin_path" ]; then
            export EDGELLM_PLUGIN_PATH="$plugin_path"
            log_info "找到插件库: $plugin_path"
            return 0
        fi
    done

    log_error "未找到插件库"
    return 1
}

# 构建引擎
build_engine() {
    local onnx_dir="$1"
    local engine_dir="$2"
    local build_dir="$3"

    local max_input_len="${MAX_INPUT_LEN:-512}"
    local max_kv_capacity="${MAX_KV_CAPACITY:-2048}"

    log_step "配置参数:"
    log_info "ONNX 目录: $onnx_dir"
    log_info "引擎目录: $engine_dir"
    log_info "最大输入长度: $max_input_len"
    log_info "最大 KV 容量: $max_kv_capacity"
    log_info "插件路径: $EDGELLM_PLUGIN_PATH"

    # 创建引擎目录
    mkdir -p "$engine_dir"

    # 检查 llm_build 工具
    local llm_build="$build_dir/examples/llm/llm_build"
    if [ ! -f "$llm_build" ]; then
        log_error "llm_build 不存在: $llm_build"
        return 1
    fi

    log_info "llm_build 工具: $llm_build"

    # 再次检查内存
    if ! check_memory_status; then
        log_warn "内存检查警告，但仍尝试构建..."
    fi

    # 开始构建
    log_step "开始构建 TensorRT 引擎..."
    log_info "这可能需要 5-10 分钟，请耐心等待..."

    local build_start=$(date +%s)

    if "$llm_build" \
        --onnxDir "$onnx_dir" \
        --engineDir "$engine_dir" \
        --maxBatchSize 1 \
        --maxInputLen "$max_input_len" \
        --maxKVCacheCapacity "$max_kv_capacity"; then
        local build_end=$(date +%s)
        local build_time=$((build_end - build_start))
        log_info "✓ 构建完成 (耗时: ${build_time}秒)"
    else
        log_error "✗ 构建失败"
        return 1
    fi
}

# 验证引擎产物
verify_engine() {
    local engine_dir="$1"

    log_step "验证引擎产物..."

    local required_files=("llm.engine" "config.json" "tokenizer.json")
    local all_ok=true

    for file in "${required_files[@]}"; do
        local file_path="$engine_dir/$file"
        if [ -f "$file_path" ]; then
            local size
            size=$(du -h "$file_path" | cut -f1)
            log_info "✓ $file ($size)"
        else
            log_error "✗ $file 缺失"
            all_ok=false
        fi
    done

    if [ "$all_ok" = true ]; then
        # 显示引擎大小
        local engine_size
        engine_size=$(du -h "$engine_dir/llm.engine" | cut -f1)
        log_info "✓ 引擎验证通过 (大小: $engine_size)"
        return 0
    else
        log_error "✗ 引擎验证失败"
        return 1
    fi
}

# 清理临时文件
cleanup_temp_files() {
    log_step "清理临时文件..."
    # 清理可能的临时文件
    find /tmp -name "llm_build_*" -mmin +60 -delete 2>/dev/null || true
    log_info "✓ 清理完成"
}

# 主函数
main() {
    local onnx_dir="${1:-}"
    local engine_dir="${2:-}"

    # 参数解析
    if [ -z "$onnx_dir" ] || [ -z "$engine_dir" ]; then
        # 使用默认路径
        onnx_dir="${PROJECT_DIR}/../qwen25_0.5b_trt/onnx_new"
        engine_dir="${PROJECT_DIR}/../qwen25_0.5b_trt/engine_new"
        log_info "使用默认路径"
        log_info "ONNX 目录: $onnx_dir"
        log_info "引擎目录: $engine_dir"

        # 如果默认路径不存在，尝试相对路径
        if [ ! -d "$onnx_dir" ]; then
            onnx_dir="${PROJECT_DIR}/qwen25_0.5b/onnx_new"
            engine_dir="${PROJECT_DIR}/qwen25_0.5b/engine_new"
            log_info "默认路径不存在，尝试备选路径"
        fi
    fi

    # 查找构建目录
    local build_dir
    for possible_dir in \
        "${PROJECT_DIR}/../TensorRT-Edge-LLM/build" \
        "${PROJECT_DIR}/TensorRT-Edge-LLM/build" \
        "$(dirname "$PROJECT_DIR")/TensorRT-Edge-LLM/build"
    do
        if [ -d "$possible_dir" ]; then
            build_dir="$possible_dir"
            break
        fi
    done

    if [ -z "$build_dir" ]; then
        log_error "未找到 TensorRT-Edge-LLM 构建目录"
        log_error "请确保已经编译 C++ Runtime"
        return 1
    fi

    log_info "构建目录: $build_dir"

    # 检查 ONNX 目录
    if [ ! -d "$onnx_dir" ]; then
        log_error "ONNX 目录不存在: $onnx_dir"
        return 1
    fi

    if [ ! -f "$onnx_dir/model.onnx" ]; then
        log_error "ONNX 模型不存在: $onnx_dir/model.onnx"
        return 1
    fi

    # 设置插件路径
    if ! setup_plugin_path "$build_dir"; then
        return 1
    fi

    # 切换到无桌面模式
    if ! switch_to_headless_mode; then
        log_warn "模式切换失败，但继续构建"
    fi

    # 构建引擎
    if ! build_engine "$onnx_dir" "$engine_dir" "$build_dir"; then
        log_error "引擎构建失败"
        # 尝试恢复桌面模式
        restore_graphical_mode
        return 1
    fi

    # 验证引擎
    if ! verify_engine "$engine_dir"; then
        log_error "引擎验证失败"
        restore_graphical_mode
        return 1
    fi

    # 清理临时文件
    cleanup_temp_files

    # 恢复桌面模式
    restore_graphical_mode

    log_info "=== 引擎构建成功 ==="
    log_info "引擎位置: $engine_dir"
    log_info "下一步: python3 llm_server.py --engine-dir $engine_dir --llm-inference $build_dir/examples/llm/llm_inference --plugin $EDGELLM_PLUGIN_PATH"
}

# 显示帮助
show_help() {
    cat << EOF
TensorRT 引擎构建脚本

用法:
  $0 [onnx_dir] [engine_dir]

参数:
  onnx_dir      ONNX 模型目录 (包含 model.onnx)
  engine_dir    引擎输出目录

环境变量:
  EDGELLM_PLUGIN_PATH   插件库路径 (自动检测)
  MAX_INPUT_LEN        最大输入长度 (默认 512)
  MAX_KV_CAPACITY      最大 KV 容量 (默认 2048)
  SKIP_MODE_SWITCH     跳过无桌面模式切换 (默认 false)

示例:
  # 使用默认路径
  $0

  # 指定路径
  $0 /path/to/onnx /path/to/engine

  # 自定义参数
  MAX_INPUT_LEN=1024 MAX_KV_CAPACITY=4096 $0

  # 跳过模式切换
  SKIP_MODE_SWITCH=true $0

注意:
  - 需要 8GB+ 内存 (lfb ≥ 400MB)
  - 自动切换到无桌面模式释放内存
  - 构建时间 5-10 分钟
EOF
}

# 解析命令行参数
case "${1:-}" in
    help|--help|-h) show_help; exit 0 ;;
esac

main "$@"