#!/usr/bin/env bash
# ============================================================================
# accuracy_test.sh — MLC-LLM 量化精度评估脚本
#
# 测量 q4f16_1 量化模型与 FP16 基线模型的精度差异
#
# 用法:
#   ./accuracy_test.sh [command]
#
# 命令:
#   prepare  - 准备测试数据（下载 FP16 模型、准备测试用例）
#   baseline - 建立 FP16 基线
#   test     - 执行精度测试（q4f16_1 vs FP16）
#   compare  - 对比分析结果
#   all      - 完整流程（prepare + baseline + test + compare）
#
# 环境变量:
#   HOST            - 服务主机 (默认: localhost)
#   PORT            - 服务端口 (默认: 8000)
#   TEST_CASES     - 测试用例文件 (默认: quick_check.json)
#   TEMPERATURE    - 温度参数 (默认: 0.0，贪心解码)
#   MAX_TOKENS     - 最大生成长度 (默认: 256)
#   TIMEOUT        - 请求超时秒 (默认: 120)
#
# 输出: 终端报告 + reports/accuracy_comparison.txt
#
# 参考: Edge_llm_deploy/benchmark.sh (框架模式)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 颜色输出（复用现有模式）
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
HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
TEST_CASES="${TEST_CASES:-$PROJECT_DIR/testcases/quick_check.json}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TIMEOUT="${TIMEOUT:-120}"
REPORTS_DIR="$PROJECT_DIR/reports"

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 检查服务可用性（仿照 benchmark.sh）
check_service() {
    log_step "检查服务可用性..."

    if ! curl -s -m 5 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
        log_error "服务不可用: http://${HOST}:${PORT}"
        log_error "请先启动服务: ./jetson-run.sh serve"
        return 1
    fi

    # 获取模型信息
    local model_info
    model_info=$(curl -s "http://${HOST}:${PORT}/v1/models")
    MODEL=$(echo "$model_info" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "unknown")

    if [ "$MODEL" = "unknown" ]; then
        log_warn "无法获取模型信息"
        MODEL="qwen2.5-1.5b"
    fi

    log_info "✓ 服务运行正常"
    log_info "模型: $MODEL"
}

# 检查内存状态（仿照 build_engine.sh）
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

    # FP16 模型约 3.2GB，需要更多预留空间
    if [ "$lfb" -lt 4000 ]; then
        log_warn "lfb 不足 4000MB (当前 ${lfb}MB)"
        log_warn "建议切换到无桌面模式: sudo systemctl isolate multi-user.target"
        return 1
    fi

    log_info "✓ 内存状态良好"
    return 0
}

# 准备测试数据
prepare_test_data() {
    log_step "准备测试数据..."

    # 创建报告目录
    mkdir -p "$REPORTS_DIR"
    log_info "✓ 报告目录: $REPORTS_DIR"

    # 检查测试用例文件
    if [ ! -f "$TEST_CASES" ]; then
        log_error "测试用例文件不存在: $TEST_CASES"
        log_info "将创建默认测试用例..."

        # 创建默认 quick_check.json
        cat > "$PROJECT_DIR/testcases/quick_check.json" << 'EOF'
{
  "test_cases": [
    {"type": "math", "question": "计算 25 * 34 + 17 = ?", "answer": "857"},
    {"type": "knowledge", "question": "中国的首都是哪里？", "answer": "北京"},
    {"type": "reasoning", "question": "如果今天是星期三，后天是星期几？", "answer": "星期五"},
    {"type": "calculation", "question": "100 - 37 + 15 = ?", "answer": "78"},
    {"type": "knowledge", "question": "一年有多少个月？", "answer": "12"},
    {"type": "math", "question": "12 * 12 = ?", "answer": "144"},
    {"type": "reasoning", "question": "如果昨天是星期一，今天是星期几？", "answer": "星期二"},
    {"type": "knowledge", "question": "水的化学式是什么？", "answer": "H2O"},
    {"type": "calculation", "question": "45 + 67 = ?", "answer": "112"},
    {"type": "reasoning", "question": "一个苹果加两个苹果等于几个苹果？", "answer": "三个"}
  ]
}
EOF
        log_info "✓ 默认测试用例已创建: $PROJECT_DIR/testcases/quick_check.json"
    else
        log_info "✓ 测试用例文件: $TEST_CASES"
    fi

    # 检查内存状态
    check_memory_status || log_warn "内存检查警告，但继续准备测试数据"

    log_info "✓ 测试数据准备完成"
}

# 建立基线（FP16 模型测试）
establish_baseline() {
    log_step "建立 FP16 基线..."

    if ! check_service; then
        return 1
    fi

    log_info "当前模型: $MODEL"
    log_info "注意: 请确保当前运行的是 FP16 模型 (q0f16)"
    log_warn "如果运行的是 q4f16_1 模型，请先切换到 FP16 模型"

    # 运行精度测试
    run_accuracy_test_internal "baseline"

    log_info "✓ 基线数据已保存到: $REPORTS_DIR/accuracy_baseline.txt"
}

# 执行精度测试（q4f16_1 模型）
run_accuracy_test() {
    log_step "执行精度测试..."

    if ! check_service; then
        return 1
    fi

    log_info "当前模型: $MODEL"
    log_info "注意: 请确保当前运行的是 q4f16_1 模型"

    # 运行精度测试
    run_accuracy_test_internal "q4f16_1"

    log_info "✓ 测试结果已保存到: $REPORTS_DIR/accuracy_q4f16_1.txt"
}

# 内部精度测试函数（Python 实现）
run_accuracy_test_internal() {
    local test_type="$1"  # "baseline" or "q4f16_1"
    local output_file="$REPORTS_DIR/accuracy_${test_type}.txt"

    log_step "开始精度测试 ($test_type)..."
    log_info "测试用例: $TEST_CASES"
    log_info "温度参数: $TEMPERATURE (贪心解码)"
    log_info "最大长度: $MAX_TOKENS tokens"

    # Python 精度测试脚本（仿照 benchmark.sh 的内嵌模式）
    python3 - "$HOST" "$PORT" "$MODEL" "$TEST_CASES" "$TEMPERATURE" "$MAX_TOKENS" "$output_file" "$TIMEOUT" <<'EOF'
import sys, json, time, os
from urllib.request import urlopen, Request

host, port, model, test_cases_path = sys.argv[1:5]
temperature, max_tokens, output_file, timeout = (float(sys.argv[5]), int(sys.argv[6]), sys.argv[7], int(sys.argv[8]))
url = f"http://{host}:{port}/v1/chat/completions"

# 读取测试用例
with open(test_cases_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

test_cases = test_data.get('test_cases', [])
total = len(test_cases)

print(f"测试用例数量: {total}")
print(f"开始测试...")

results = []
correct = 0

for i, case in enumerate(test_cases, 1):
    question = case['question']
    expected = case['answer']

    try:
        # 构造请求（固定 temperature=0.0 保证可复现）
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": question}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }).encode('utf-8')

        req = Request(url, data=payload, headers={'Content-Type': 'application/json'})

        start_time = time.time()
        with urlopen(req, timeout=timeout) as resp:
            response = json.load(resp)
            end_time = time.time()

        actual = response['choices'][0]['message']['content'].strip()
        elapsed = end_time - start_time

        # 简单匹配评估（检查答案是否包含在回复中）
        match = expected.lower() in actual.lower()
        if match:
            correct += 1

        result = {
            'question': question,
            'expected': expected,
            'actual': actual,
            'match': match,
            'time': elapsed
        }
        results.append(result)

        # 显示进度
        status = "✓" if match else "✗"
        print(f"{status} {i}/{total}: {question[:30]}... (匹配: {match})")

    except Exception as e:
        print(f"✗ {i}/{total}: 失败 - {str(e)}")
        result = {
            'question': question,
            'expected': expected,
            'actual': f"ERROR: {str(e)}",
            'match': False,
            'time': 0.0
        }
        results.append(result)

# 计算统计
accuracy = correct / total if total > 0 else 0.0
avg_time = sum(r['time'] for r in results if r['time'] > 0) / len(results)

# 保存结果
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(f"=== 精度测试结果 ({output_file.split('/')[-1].replace('.txt', '')}) ===\n")
    f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"模型: {model}\n")
    f.write(f"测试用例: {test_cases_path}\n")
    f.write(f"温度参数: {temperature}\n")
    f.write(f"准确率: {accuracy:.1%} ({correct}/{total})\n")
    f.write(f"平均时延: {avg_time:.3f}s\n")
    f.write("\n=== 详细结果 ===\n")

    for i, r in enumerate(results, 1):
        status = "✓" if r['match'] else "✗"
        f.write(f"{status} 问题 {i}: {r['question']}\n")
        f.write(f"  期望: {r['expected']}\n")
        f.write(f"  实际: {r['actual'][:100]}...\n")
        f.write(f"  匹配: {r['match']}\n\n")

# 输出汇总
print(f"\n=== 汇总 ===")
print(f"准确率: {accuracy:.1%} ({correct}/{total})")
print(f"平均时延: {avg_time:.3f}s")
print(f"详细报告: {output_file}")

EOF

    if [ $? -eq 0 ]; then
        log_info "✓ 精度测试完成"
    else
        log_error "精度测试失败"
        return 1
    fi
}

# 对比分析结果
compare_results() {
    log_step "对比分析结果..."

    local baseline_file="$REPORTS_DIR/accuracy_baseline.txt"
    local q4f16_file="$REPORTS_DIR/accuracy_q4f16_1.txt"
    local comparison_file="$REPORTS_DIR/accuracy_comparison.txt"

    if [ ! -f "$baseline_file" ]; then
        log_error "基线文件不存在: $baseline_file"
        log_info "请先运行: $0 baseline"
        return 1
    fi

    if [ ! -f "$q4f16_file" ]; then
        log_error "测试文件不存在: $q4f16_file"
        log_info "请先运行: $0 test"
        return 1
    fi

    # 使用 Python 脚本进行详细对比分析
    python3 "$SCRIPT_DIR/accuracy_report.py" "$baseline_file" "$q4f16_file" "$comparison_file"

    if [ $? -eq 0 ]; then
        log_info "✓ 对比分析完成"
    else
        log_error "对比分析失败"
        return 1
    fi
}

# 运行完整流程
run_all_phases() {
    log_step "运行完整精度评估流程..."

    prepare_test_data
    establish_baseline
    run_accuracy_test
    compare_results

    log_info "✓ 完整流程执行完成"
}

# 显示帮助
show_help() {
    cat << EOF
MLC-LLM 量化精度评估脚本

用法:
  $0 [command]

命令:
  prepare  - 准备测试数据（下载 FP16 模型、准备测试用例）
  baseline - 建立 FP16 基线（确保当前运行 FP16 模型）
  test     - 执行精度测试（确保当前运行 q4f16_1 模型）
  compare  - 对比分析结果（需要 baseline 和 test 结果）
  all      - 运行完整流程

环境变量:
  HOST         服务主机 (默认: localhost)
  PORT         服务端口 (默认: 8000)
  TEST_CASES   测试用例文件 (默认: testcases/quick_check.json)
  TEMPERATURE  温度参数 (默认: 0.0，贪心解码)
  MAX_TOKENS   最大生成长度 (默认: 256)
  TIMEOUT      请求超时秒 (默认: 120)

示例:
  # 完整流程
  $0 all

  # 分步执行
  $0 prepare
  $0 baseline
  $0 test
  $0 compare

  # 自定义参数
  TEST_CASES=testcases/ceval_subset.json $0 test

注意:
  - 确保服务已启动: ./jetson-run.sh serve
  - baseline 阶段需要 FP16 模型
  - test 阶段需要 q4f16_1 模型
  - 温度参数固定为 0.0 保证可复现性

输出:
  - 终端实时显示
  - reports/accuracy_baseline.txt (FP16 基线)
  - reports/accuracy_q4f16_1.txt (q4f16_1 结果)
  - reports/accuracy_comparison.txt (对比分析)

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
        prepare) prepare_test_data ;;
        baseline) establish_baseline ;;
        test) run_accuracy_test ;;
        compare) compare_results ;;
        all) run_all_phases ;;
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