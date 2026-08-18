#!/usr/bin/env bash
# ============================================================================
# eval_baseline.sh — MLC-LLM 基准模型准备脚本
#
# 处理 FP16 模型的下载、部署和服务切换，支持 q4f16_1 <-> q0f16 互换
#
# 用法:
#   ./eval_baseline.sh [command]
#
# 命令:
#   download  - 下载 FP16 模型 (q0f16)
#   switch    - 切换模型 (q4f16_1 <-> q0f16)
#   verify    - 验证当前服务状态
#   help      - 显示帮助信息
#
# 环境变量:
#   TARGET_MODEL    - 目标模型 (q4f16_1 or q0f16)
#   SKIP_MEMORY_CHECK - 跳过内存检查 (默认: false)
#
# 参考: jetson-run.sh (部署模式)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$EVAL_DIR")"
MLC_SCRIPTS_DIR="$PROJECT_DIR/scrips/JIT_deploy_scrips"

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

# 模型配置
MODEL_Q4F16="mlc-ai/Qwen2.5-1.5B-Instruct-q4f16_1-MLC"
MODEL_Q0F16="mlc-ai/Qwen2.5-1.5B-Instruct-q0f16-MLC"

DEFAULT_MODEL_REPO="${MODEL_Q4F16}"

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 检查内存状态（仿照 build_engine.sh）
check_memory_status() {
    if [ "${SKIP_MEMORY_CHECK:-false}" = "true" ]; then
        log_warn "跳过内存检查 (SKIP_MEMORY_CHECK=true)"
        return 0
    fi

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

    # FP16 模型约 3.2GB，需要更多预留空间
    if [ "$lfb" -lt 4000 ]; then
        log_error "内存不足: lfb=${lfb}MB (需要 ≥4000MB)"
        log_error "建议操作:"
        log_error "  1. 切换到无桌面模式: sudo systemctl isolate multi-user.target"
        log_error "  2. 停止不必要的服务"
        log_error "  3. 或者跳过检查: SKIP_MEMORY_CHECK=true $0"
        return 1
    fi

    log_info "✓ 内存状态良好 (lfb: ${lfb}MB)"
    return 0
}

# 获取当前模型
get_current_model() {
    if ! command_exists docker; then
        echo "unknown"
        return
    fi

    # 检查容器状态
    if sudo docker ps | grep -q "mlc-serve"; then
        # 从容器环境变量获取模型信息
        local model_repo
        model_repo=$(sudo docker exec mlc-serve printenv MODEL_REPO 2>/dev/null || echo "")
        if [ -n "$model_repo" ]; then
            echo "$model_repo"
            return
        fi
    fi

    echo "unknown"
}

# 切换模型
switch_model() {
    local target_model="${TARGET_MODEL:-}"

    if [ -z "$target_model" ]; then
        log_error "请指定目标模型: TARGET_MODEL=q4f16_1 或 TARGET_MODEL=q0f16"
        return 1
    fi

    log_step "切换模型到: $target_model"

    # 获取当前模型
    local current_model
    current_model=$(get_current_model)

    log_info "当前模型: $current_model"

    # 检查是否已经是目标模型
    if [ "$current_model" = "$target_model" ]; then
        log_info "✓ 已经是目标模型，无需切换"
        return 0
    fi

    # 停止当前服务
    log_step "停止当前服务..."
    if sudo docker ps | grep -q "mlc-serve"; then
        sudo docker stop mlc-serve
        log_info "✓ 服务已停止"
    else
        log_info "✓ 服务未运行"
    fi

    # 设置目标模型
    case "$target_model" in
        q4f16_1)
            export MODEL_REPO="$MODEL_Q4F16"
            log_info "目标模型: Qwen2.5-1.5B-Instruct-q4f16_1-MLC"
            ;;
        q0f16)
            export MODEL_REPO="$MODEL_Q0F16"
            log_info "目标模型: Qwen2.5-1.5B-Instruct-q0f16-MLC"

            # 检查内存状态（FP16 模型需要更多内存）
            if ! check_memory_status; then
                log_error "内存检查失败，无法切换到 FP16 模型"
                return 1
            fi
            ;;
        *)
            log_error "未知模型: $target_model"
            log_info "支持的模型: q4f16_1, q0f16"
            return 1
            ;;
    esac

    # 重启服务
    log_step "重启服务..."
    log_info "使用模型: $MODEL_REPO"

    # 调用 jetson-run.sh 启动服务
    if [ -f "$MLC_SCRIPTS_DIR/jetson-run.sh" ]; then
        # 使用 MODEL_REPO 环境变量启动
        MODEL_REPO="$MODEL_REPO" "$MLC_SCRIPTS_DIR/jetson-run.sh" serve
    else
        log_error "找不到 jetson-run.sh 脚本 (路径: $MLC_SCRIPTS_DIR/jetson-run.sh)"
        return 1
    fi

    if [ $? -eq 0 ]; then
        log_info "✓ 模型切换完成"
        log_info "新模型: $MODEL_REPO"

        # 等待服务启动
        log_info "等待服务启动..."
        sleep 10

        # 验证服务状态
        verify_service
    else
        log_error "模型切换失败"
        return 1
    fi
}

# 下载 FP16 模型
download_fp16_model() {
    log_step "准备下载 FP16 模型..."
    log_info "目标模型: $MODEL_Q0F16"
    log_info "预计大小: ~3.2GB"

    # 检查内存状态
    if ! check_memory_status; then
        log_error "内存检查失败，无法下载 FP16 模型"
        return 1
    fi

    log_info "FP16 模型下载说明:"
    log_info "1. 模型将使用 MLC-LM 自动下载"
    log_info "2. 下载位置: ~/.cache/mlc_llm/"
    log_info "3. 首次启动服务时自动下载"

    log_step "切换到 FP16 模型（首次启动时会自动下载）..."

    # 切换到 FP16 模型
    TARGET_MODEL=q0f16 switch_model

    log_info "✓ FP16 模型准备完成"
}

# 验证服务状态
verify_service() {
    log_step "验证服务状态..."

    # 检查服务是否运行
    if ! sudo docker ps | grep -q "mlc-serve"; then
        log_error "服务未运行"
        return 1
    fi

    log_info "✓ 容器运行中"

    # 检查服务端点
    local max_attempts=5
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -s -m 5 "http://localhost:8000/v1/models" >/dev/null 2>&1; then
            log_info "✓ 服务端点正常"

            # 获取模型信息
            local model_info
            model_info=$(curl -s "http://localhost:8000/v1/models")
            local model
            model=$(echo "$model_info" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "unknown")

            log_info "当前模型: $model"
            return 0
        fi

        log_info "等待服务启动... ($attempt/$max_attempts)"
        sleep 5
        ((attempt++))
    done

    log_error "服务端点不可用"
    return 1
}

# 显示帮助
show_help() {
    cat << EOF
MLC-LLM 基准模型准备脚本

用法:
  $0 [command]

命令:
  download  - 下载 FP16 模型 (q0f16)
  switch    - 切换模型 (q4f16_1 <-> q0f16)
  verify    - 验证当前服务状态
  help      - 显示帮助信息

环境变量:
  TARGET_MODEL        目标模型 (q4f16_1 or q0f16)
  SKIP_MEMORY_CHECK   跳过内存检查 (默认: false)

示例:
  # 下载 FP16 模型
  $0 download

  # 切换到 FP16 模型
  TARGET_MODEL=q0f16 $0 switch

  # 切换回 q4f16_1 模型
  TARGET_MODEL=q4f16_1 $0 switch

  # 验证服务状态
  $0 verify

  # 跳过内存检查切换
  SKIP_MEMORY_CHECK=true TARGET_MODEL=q0f16 $0 switch

模型说明:
  q4f16_1  - 4-bit 量化模型 (约 840MB)
  q0f16    - FP16 基线模型 (约 3.2GB)

内存要求:
  q4f16_1: 需要 ~2GB 内存
  q0f16:   需要 ~4GB 内存

注意:
  - FP16 模型首次启动时会自动下载
  - 下载过程需要稳定的网络连接
  - 建议在切换到 FP16 模型前检查可用内存

EOF
}

# 主函数
main() {
    local command="${1:-}"

    if [ -z "$command" ]; then
        show_help
        exit 0
    fi

    case "$command" in
        download) download_fp16_model ;;
        switch) switch_model ;;
        verify) verify_service ;;
        help|--help|-h) show_help ;;
        *)
            log_error "未知命令: $command"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"