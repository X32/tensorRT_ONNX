#!/usr/bin/env bash
# ============================================================================
# accuracy_demo.sh — MLC-LLM 精度评估功能演示
#
# 快速演示如何使用新的精度评估功能
#
# 用法: ./accuracy_demo.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_section() { echo -e "${BLUE}=== $1 ===${NC}"; }
log_info() { echo -e "${GREEN}[INFO]${NC} $@"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $@"; }

clear
echo -e "${BLUE}"
cat << "EOF"
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║       MLC-LLM 量化精度评估系统                              ║
║                                                              ║
║       q4f16_1 vs FP16 精度对比测试                           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

log_section "功能概述"

echo ""
echo "本系统提供以下功能："
echo ""
echo "1. accuracy_test.sh  - 主评估脚本"
echo "   • prepare  - 准备测试数据"
echo "   • baseline - 建立 FP16 基线"
echo "   • test     - 执行精度测试"
echo "   • compare  - 对比分析结果"
echo "   • all      - 完整流程"
echo ""
echo "2. eval_baseline.sh - 模型管理脚本"
echo "   • download - 下载 FP16 模型"
echo "   • switch   - 切换模型"
echo "   • verify   - 验证服务状态"
echo ""
echo "3. accuracy_report.py - 详细分析脚本"
echo "   • 生成详细的精度对比报告"
echo ""

log_section "可用测试集"

echo ""
echo "测试用例文件："
echo "• quick_check.json       - 快速验证集 (10题)"
echo "• ceval_subset.json      - C-Eval 抽样 (30题)"
echo "• gsm8k_cn_100.json      - GSM8K 中文版 (50题)"
echo "• business_prompts.jsonl - 业务回归集 (20题)"
echo ""

log_section "快速演示"

echo ""
echo "演示模式选择："
echo ""
echo "1. 完整演示 - 需要运行的服务和模型切换"
echo "2. 功能演示 - 只展示脚本功能（无需服务）"
echo "3. 帮助信息 - 显示详细使用说明"
echo ""
read -p "请选择 (1/2/3): " choice

case "$choice" in
    1)
        log_section "完整演示"
        echo ""
        log_warn "注意：此演示需要："
        log_warn "1. MLC-LLM 服务已启动"
        log_warn "2. 需要切换模型（q4f16_1 <-> FP16）"
        log_warn "3. 足够的内存（FP16 需要 ~4GB）"
        echo ""
        read -p "继续吗？ (y/N): " confirm

        if [[ "$confirm" =~ ^[Yy]$ ]]; then
            log_info "开始完整演示..."

            echo ""
            log_section "步骤 1: 准备测试数据"
            "$SCRIPT_DIR/accuracy_test.sh" prepare

            echo ""
            log_section "步骤 2: 检查服务状态"
            "$SCRIPT_DIR/eval_baseline.sh" verify || {
                log_warn "服务未启动，请先启动服务："
                log_warn "./jetson-run.sh serve"
                exit 1
            }

            echo ""
            log_section "当前演示完成"
            echo "完整演示需要手动执行以下步骤："
            echo ""
            echo "1. 切换到 FP16 模型:"
            echo "   TARGET_MODEL=q0f16 ./eval_baseline.sh switch"
            echo ""
            echo "2. 建立 FP16 基线:"
            echo "   ./accuracy_test.sh baseline"
            echo ""
            echo "3. 切换回 q4f16_1 模型:"
            echo "   TARGET_MODEL=q4f16_1 ./eval_baseline.sh switch"
            echo ""
            echo "4. 执行精度测试:"
            echo "   ./accuracy_test.sh test"
            echo ""
            echo "5. 分析结果:"
            echo "   ./accuracy_test.sh compare"
            echo ""
        else
            log_info "演示已取消"
        fi
        ;;

    2)
        log_section "功能演示"
        echo ""

        log_info "演示 1: 准备测试数据"
        "$SCRIPT_DIR/accuracy_test.sh" prepare
        echo ""

        log_info "演示 2: 显示帮助信息"
        "$SCRIPT_DIR/accuracy_test.sh" help
        echo ""

        log_info "演示 3: 显示模型管理帮助"
        "$SCRIPT_DIR/eval_baseline.sh" help
        echo ""

        log_info "功能演示完成！"
        echo ""
        echo "下一步："
        echo "1. 启动 MLC-LLM 服务：./jetson-run.sh serve"
        echo "2. 运行完整测试：./accuracy_test.sh all"
        ;;

    3)
        log_section "使用说明"
        echo ""
        cat << EOF
📋 详细使用说明

🔧 基本用法：
   # 快速开始
   ./accuracy_test.sh all

   # 使用不同测试集
   TEST_CASES=../testcases/ceval_subset.json ./accuracy_test.sh all

🔄 模型切换：
   # 切换到 FP16 模型
   ./eval_baseline.sh download  # 首次下载 FP16
   TARGET_MODEL=q0f16 ./eval_baseline.sh switch

   # 切换回 q4f16_1 模型
   TARGET_MODEL=q4f16_1 ./eval_baseline.sh switch

📊 结果分析：
   # 自动生成对比报告
   ./accuracy_test.sh compare

   # 手动生成详细报告
   python3 accuracy_report.py \\
     ../../reports/accuracy_baseline.txt \\
     ../../reports/accuracy_q4f16_1.txt \\
     ../../reports/accuracy_detailed.txt

📁 文件位置：
   脚本: scrips/JIT_deploy_scrips/
   测试: testcases/
   报告: reports/
   文档: DOC/accuracy_eval_guide.md

🎯 精度标准：
   • 优秀: Δacc ≤ 2%
   • 良好: Δacc ≤ 5%
   • 可接受: Δacc ≤ 10%
   • 需检查: Δacc > 10%

📖 更多信息:
   查看 DOC/accuracy_eval_guide.md 获取完整文档

EOF
        ;;

    *)
        log_warn "无效选择"
        exit 1
        ;;
esac

echo ""
log_section "演示结束"
echo ""
log_info "更多信息请查看: DOC/accuracy_eval_guide.md"