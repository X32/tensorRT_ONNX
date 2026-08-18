#!/usr/bin/env bash
# ============================================================================
# benchmark.sh — TensorRT-Edge-LLM 性能测试脚本
#
# 测量 TensorRT-Edge-LLM HTTP 服务性能，包括 TTFT、吞吐量、内存、功耗
#
# 用法:
#   ./scripts/benchmark.sh [host] [port]
#
# 环境变量:
#   ROUNDS          # 测试轮数 (默认 5)
#   WARMUP          # 预热轮数 (默认 2)
#   MAX_TOKENS      # 最大生成长度 (默认 256)
#   PROMPT          # 测试提示词
#   TIMEOUT         # 请求超时秒 (默认 120)
#
# 输出: 终端报告 + benchmark-result.txt
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

# 默认参数
HOST="${1:-localhost}"
PORT="${2:-8000}"
ROUNDS="${ROUNDS:-5}"
WARMUP="${WARMUP:-2}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TIMEOUT="${TIMEOUT:-120}"
PROMPT="${PROMPT:-请用中文介绍一下你自己，包括你的能力和你擅长的任务。}"
OUT_FILE="benchmark-result.txt"

# 检查服务可用性
check_service() {
    log_step "检查服务可用性..."

    if ! curl -s -m 5 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
        log_error "服务不可用: http://${HOST}:${PORT}"
        return 1
    fi

    # 获取模型信息
    local model_info
    model_info=$(curl -s "http://${HOST}:${PORT}/v1/models")
    MODEL=$(echo "$model_info" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "unknown")

    if [ "$MODEL" = "unknown" ]; then
        log_warn "无法获取模型信息"
        MODEL="qwen2.5-0.5b"
    fi

    log_info "✓ 服务运行正常"
    log_info "模型: $MODEL"
}

# 启动 tegrastats 监控
start_tegrastats() {
    if ! command -v tegrastats >/dev/null 2>&1; then
        log_warn "tegrastats 不可用，跳过硬件监控"
        return 1
    fi

    log_step "启动硬件监控..."
    sudo tegrastats --interval 1000 --logfile "$(pwd)/tegrastats.log" &
    TEGRA_PID=$!
    trap 'sudo kill ${TEGRA_PID} 2>/dev/null || true' EXIT
    log_info "tegrastats 监控中 (PID: ${TEGRA_PID})"
}

# 停止 tegrastats 并分析结果
analyze_tegrastats() {
    if [ ! -f "$(pwd)/tegrastats.log" ]; then
        return 0
    fi

    log_step "分析硬件监控数据..."

    sudo kill ${TEGRA_PID} 2>/dev/null || true
    sleep 1

    local log_file="$(pwd)/tegrastats.log"

    # 内存峰值
    local peak_mem
    peak_mem=$(grep -oP 'RAM [0-9]+/[0-9]+MiB' "$log_file" \
        | awk '{print $2}' | cut -d/ -f1 | sort -n | tail -1)

    if [ -n "$peak_mem" ]; then
        log_info "峰值内存: ${peak_mem} MiB"
        echo "峰值内存: ${peak_mem} MiB" >> "$OUT_FILE"
    fi

    # GPU 功耗峰值
    local peak_power
    peak_power=$(grep -oP 'POM_5V_GPU_PWR [0-9]+' "$log_file" \
        | awk '{print $2}' | sort -n | tail -1)

    if [ -n "$peak_power" ]; then
        log_info "峰值 GPU 功耗: ${peak_power} mW"
        echo "峰值 GPU 功耗: ${peak_power} mW" >> "$OUT_FILE"
    fi

    # GPU 频率
    local avg_freq
    avg_freq=$(grep -oP 'GPU @[0-9]+' "$log_file" \
        | awk -F@ '{sum+=$2; count++} END {if(count>0) print int(sum/count); else print 0}')

    if [ "$avg_freq" -gt 0 ]; then
        log_info "平均 GPU 频率: ${avg_freq} MHz"
        echo "平均 GPU 频率: ${avg_freq} MHz" >> "$OUT_FILE"
    fi
}

# 单次推理测试
single_inference() {
    local url="http://${HOST}:${PORT}/v1/chat/completions"

    local payload
    payload=$(cat << EOF
{
  "model": "$MODEL",
  "messages": [{"role": "user", "content": "$PROMPT"}],
  "max_tokens": $MAX_TOKENS,
  "temperature": 0.7,
  "top_p": 0.9
}
EOF
)

    local start_time=$(date +%s.%N)
    local response
    response=$(curl -s -m "$TIMEOUT" -X POST "$url" \
        -H "Content-Type: application/json" \
        -d "$payload")
    local end_time=$(date +%s.%N)

    local total_time=$(echo "$end_time - $start_time" | bc)

    # 检查响应
    if ! echo "$response" | grep -q "content"; then
        log_error "推理失败"
        echo "$response" | head -c 200
        return 1
    fi

    # 提取生成内容
    local content
    content=$(echo "$response" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])")

    # 计算 token 数 (简单估计: 中文约 2-3 字符/token)
    local char_count=${#content}
    local token_count=$((char_count / 2))

    echo "$total_time|$token_count|$content"
}

# 运行性能测试
run_benchmark() {
    log_step "开始性能测试..."
    log_info "测试轮数: ${WARMUP} 预热 + ${ROUNDS} 测量"
    log_info "最大生成长度: $MAX_TOKENS tokens"

    # 输出文件头
    {
        echo "== TensorRT-Edge-LLM 性能测试 =="
        echo "时间: $(date '+%F %T')"
        echo "服务: http://${HOST}:${PORT}"
        echo "模型: $MODEL"
        echo "测试轮数: ${WARMUP} 预热 + ${ROUNDS} 测量"
        echo "最大生成长度: $MAX_TOKENS tokens"
        echo ""
    } | tee "$OUT_FILE"

    # 功耗模式信息
    if command_exists nvpmodel; then
        local power_mode
        power_mode=$(nvpmodel -q 2>/dev/null || echo "unknown")
        echo "功耗模式: $power_mode" | tee -a "$OUT_FILE"
    fi

    echo "" | tee -a "$OUT_FILE"

    # 预热轮
    log_step "预热 (${WARMUP} 轮)..."
    for i in $(seq 1 $WARMUP); do
        log_info "预热 $i/$WARMUP"
        if ! result=$(single_inference); then
            log_error "预热失败"
            return 1
        fi
        sleep 1
    done

    log_info "✓ 预热完成"

    # 测试轮
    log_step "正式测试 (${ROUNDS} 轮)..."

    local total_times=()
    local token_counts=()
    local ttfts=()
    local contents=()

    for i in $(seq 1 $ROUNDS); do
        log_info "测试 $i/$ROUNDS"

        local start=$(date +%s.%N)
        if ! result=$(single_inference); then
            log_error "测试 $i 失败"
            continue
        fi
        local end=$(date +%s.%N)

        # 解析结果
        local total_time=$(echo "$result" | cut -d'|' -f1)
        local token_count=$(echo "$result" | cut -d'|' -f2)
        local content=$(echo "$result" | cut -d'|' -f3-)

        # 估计 TTFT (假设首 token 时间约占总时间的 20%)
        local ttft=$(echo "$total_time * 0.2" | bc)

        total_times+=("$total_time")
        token_counts+=("$token_count")
        ttfts+=("$ttft")
        contents+=("$content")

        local throughput=$(echo "$token_count / $total_time" | bc)
        log_info "  时延: ${total_time}s | Token: ${token_count} | 吞吐: ${throughput} tok/s"

        sleep 1
    done

    # 统计分析
    log_step "统计分析..."

    # 计算平均值
    local avg_total=0
    local avg_tokens=0
    local avg_ttft=0
    local avg_throughput=0

    for i in $(seq 0 $((${#total_times[@]} - 1))); do
        avg_total=$(echo "$avg_total + ${total_times[$i]}" | bc)
        avg_tokens=$(echo "$avg_tokens + ${token_counts[$i]}" | bc)
        avg_ttft=$(echo "$avg_ttft + ${ttfts[$i]}" | bc)
    done

    avg_total=$(echo "$avg_total / ${#total_times[@]}" | bc)
    avg_tokens=$(echo "$avg_tokens / ${#token_counts[@]}" | bc)
    avg_ttft=$(echo "$avg_ttft / ${#ttfts[@]}" | bc)
    avg_throughput=$(echo "$avg_tokens / $avg_total" | bc)

    # 输出结果
    {
        echo "== 测试结果 =="
        echo "测试轮数: ${#total_times[@]}"
        echo ""
        echo "== 平均值 =="
        echo "总时延: $(printf '%.3f' $avg_total) s"
        echo "TTFT:   $(printf '%.3f' $avg_ttft) s"
        echo "生成 Token: $(printf '%.0f' $avg_tokens)"
        echo "吞吐量:   $(printf '%.2f' $avg_throughput) tok/s"
        echo ""
        echo "== 详细结果 =="
        for i in $(seq 0 $((${#total_times[@]} - 1))); do
            local throughput=$(echo "${token_counts[$i]} / ${total_times[$i]}" | bc)
            echo "轮 $((i+1)): 时延 $(printf '%.3f' ${total_times[$i]})s | Token ${token_counts[$i]} | 吞吐 $(printf '%.2f' $throughput) tok/s"
        done
        echo ""
        echo "== 生成示例 =="
        echo "${contents[0]:0:200}..."
    } | tee -a "$OUT_FILE"

    log_info "✓ 测试完成"
    log_info "平均时延: $(printf '%.3f' $avg_total) s"
    log_info "平均吞吐: $(printf '%.2f' $avg_throughput) tok/s"
}

# 主函数
main() {
    log_info "=== TensorRT-Edge-LLM 性能测试 ==="

    # 检查服务
    if ! check_service; then
        log_error "请先启动服务: python3 llm_server.py ..."
        return 1
    fi

    # 启动硬件监控
    start_tegrastats

    # 运行测试
    if ! run_benchmark; then
        log_error "性能测试失败"
        return 1
    fi

    # 分析硬件数据
    analyze_tegrastats

    log_info "=== 测试完成 ==="
    log_info "详细报告: $OUT_FILE"
}

# 显示帮助
show_help() {
    cat << EOF
TensorRT-Edge-LLM 性能测试脚本

用法:
  $0 [host] [port]

参数:
  host  服务主机 (默认: localhost)
  port  服务端口 (默认: 8000)

环境变量:
  ROUNDS     测试轮数 (默认: 5)
  WARMUP     预热轮数 (默认: 2)
  MAX_TOKENS 最大生成长度 (默认: 256)
  PROMPT     测试提示词
  TIMEOUT    请求超时秒 (默认: 120)

示例:
  # 本地测试
  $0

  # 远程测试
  $0 192.168.1.100 8000

  # 自定义参数
  ROUNDS=10 MAX_TOKENS=512 $0

输出:
  - 终端实时显示
  - benchmark-result.txt 详细报告
  - tegrastats.log 硬件监控数据

注意:
  - 需要 tegrastats 进行硬件监控
  - 预热轮结果不计入统计
  - TTFT 为估计值 (20% 总时延)
EOF
}

# 解析命令行参数
case "${1:-}" in
    help|--help|-h) show_help; exit 0 ;;
esac

main "$@"