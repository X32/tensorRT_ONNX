#!/usr/bin/env python3
# ============================================================================
# accuracy_report.py — MLC-LLM 精度评估报告生成脚本
#
# 分析 q4f16_1 vs FP16 的精度差异，生成详细报告
#
# 用法:
#   python3 accuracy_report.py [baseline_result] [q4f16_result] [output_file]
#
# 参考: benchmark.sh 的统计分析方法 (P50/P90)
# ============================================================================

import sys
import json
import re
import os
from typing import Tuple, List, Dict, Any
from difflib import SequenceMatcher
import time

def extract_accuracy_data(file_path: str) -> Dict[str, Any]:
    """从结果文件中提取准确率数据"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"结果文件不存在: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取准确率信息
    acc_match = re.search(r'准确率:\s*([\d.]+%)\s*\((\d+)/(\d+)\)', content)
    if not acc_match:
        raise ValueError(f"无法从文件中提取准确率: {file_path}")

    percentage = acc_match.group(1)
    correct = int(acc_match.group(2))
    total = int(acc_match.group(3))

    # 提取详细结果
    results = []
    result_pattern = re.compile(
        r'[✓✗]\s*问题\s*(\d+):\s*(.+?)\n\s*期望:\s*(.+?)\n\s*实际:\s*(.+?)\n\s*匹配:\s*(True|False)',
        re.MULTILINE
    )

    for match in result_pattern.finditer(content):
        question_num = int(match.group(1))
        question = match.group(2).strip()
        expected = match.group(3).strip()
        actual = match.group(4).strip()
        match_result = match.group(5) == 'True'

        results.append({
            'question_num': question_num,
            'question': question,
            'expected': expected,
            'actual': actual,
            'match': match_result
        })

    return {
        'percentage': percentage,
        'correct': correct,
        'total': total,
        'accuracy_value': float(percentage.rstrip('%')),
        'results': results
    }

def calculate_text_similarity(text1: str, text2: str) -> float:
    """计算文本相似度"""
    return SequenceMatcher(None, text1, text2).ratio()

def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """计算统计数据 (P50, P90, min, max)"""
    if not values:
        return {'median': 0.0, 'p90': 0.0, 'min': 0.0, 'max': 0.0}

    sorted_values = sorted(values)
    n = len(sorted_values)

    # 中位数 P50
    median = sorted_values[n // 2] if n % 2 == 1 else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2

    # P90
    p90_index = min(int(0.9 * (n - 1)), n - 1)
    p90 = sorted_values[p90_index]

    return {
        'median': median,
        'p90': p90,
        'min': min(values),
        'max': max(values)
    }

def analyze_accuracy_comparison(baseline_file: str, q4f16_file: str) -> Dict[str, Any]:
    """分析精度对比"""

    # 提取数据
    baseline_data = extract_accuracy_data(baseline_file)
    q4f16_data = extract_accuracy_data(q4f16_file)

    # 计算准确率差异
    acc_delta = baseline_data['accuracy_value'] - q4f16_data['accuracy_value']

    # 计算文本相似度
    similarities = []
    baseline_results = baseline_data['results']
    q4f16_results = q4f16_data['results']

    for base_res, q4_res in zip(baseline_results, q4f16_results):
        similarity = calculate_text_similarity(base_res['actual'], q4_res['actual'])
        similarities.append(similarity)

    # 计算相似度统计
    sim_stats = calculate_statistics(similarities)

    # 分析失败案例
    baseline_failures = [r for r in baseline_results if not r['match']]
    q4f16_failures = [r for r in q4f16_results if not r['match']]

    # 共同失败案例
    common_failures = []
    for base_res, q4_res in zip(baseline_results, q4f16_results):
        if not base_res['match'] and not q4_res['match']:
            common_failures.append({
                'question': base_res['question'],
                'expected': base_res['expected'],
                'baseline_actual': base_res['actual'],
                'q4f16_actual': q4_res['actual']
            })

    # FP16 正确但 q4f16_1 错误的案例 (精度损失案例)
    accuracy_loss_cases = []
    for base_res, q4_res in zip(baseline_results, q4f16_results):
        if base_res['match'] and not q4_res['match']:
            accuracy_loss_cases.append({
                'question': base_res['question'],
                'expected': base_res['expected'],
                'baseline_actual': base_res['actual'],
                'q4f16_actual': q4_res['actual']
            })

    return {
        'baseline': baseline_data,
        'q4f16': q4f16_data,
        'accuracy_delta': acc_delta,
        'similarities': similarities,
        'similarity_stats': sim_stats,
        'baseline_failures': baseline_failures,
        'q4f16_failures': q4f16_failures,
        'common_failures': common_failures,
        'accuracy_loss_cases': accuracy_loss_cases
    }

def generate_conclusion(analysis: Dict[str, Any]) -> str:
    """生成结论"""

    acc_delta = analysis['accuracy_delta']
    sim_median = analysis['similarity_stats']['median']
    loss_cases = len(analysis['accuracy_loss_cases'])

    if acc_delta <= 2:
        conclusion = "✓ 精度损失极小，量化效果优秀"
        level = "优秀"
    elif acc_delta <= 5:
        conclusion = "✓ 精度损失可接受，量化效果良好"
        level = "良好"
    elif acc_delta <= 10:
        conclusion = "⚠ 精度损失较大，建议检查量化配置"
        level = "一般"
    else:
        conclusion = "✗ 精度损失严重，需要重新评估量化策略"
        level = "较差"

    return conclusion, level

def generate_report(analysis: Dict[str, Any]) -> str:
    """生成详细报告"""

    baseline = analysis['baseline']
    q4f16 = analysis['q4f16']
    acc_delta = analysis['accuracy_delta']
    sim_stats = analysis['similarity_stats']

    # 生成结论
    conclusion, level = generate_conclusion(analysis)

    report = f"""
{'=' * 60}
MLC-LLM 量化精度评估报告
{'=' * 60}
生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
评估等级: {level}

{'=' * 60}
一、准确率对比
{'=' * 60}

FP16 基线 (q0f16):
  - 准确率: {baseline['percentage']} ({baseline['correct']}/{baseline['total']})
  - 错误数: {baseline['total'] - baseline['correct']}

q4f16_1 量化模型:
  - 准确率: {q4f16['percentage']} ({q4f16['correct']}/{q4f16['total']})
  - 错误数: {q4f16['total'] - q4f16['correct']}

精度差异:
  - 准确率损失: {acc_delta:+.1f}%
  - 精度保留率: {100 - acc_delta:.1f}%

{'=' * 60}
二、文本相似度分析
{'=' * 60}

文本相似度统计:
  - 中位数 (P50): {sim_stats['median']:.1%}
  - P90: {sim_stats['p90']:.1%}
  - 最小值: {sim_stats['min']:.1%}
  - 最大值: {sim_stats['max']:.1%}

{'=' * 60}
三、误差分析
{'=' * 60}

精度损失案例 (FP16正确，q4f16_1错误):
  - 数量: {len(analysis['accuracy_loss_cases'])} 个
"""

    # 显示部分精度损失案例
    if analysis['accuracy_loss_cases']:
        report += "\n示例:\n"
        for i, case in enumerate(analysis['accuracy_loss_cases'][:3], 1):
            report += f"  {i}. 问题: {case['question'][:50]}...\n"
            report += f"     期望: {case['expected']}\n"
            report += f"     FP16: {case['baseline_actual'][:80]}...\n"
            report += f"     q4f16_1: {case['q4f16_actual'][:80]}...\n"

    # 共同失败案例
    common_failures = len(analysis['common_failures'])
    report += f"""
共同失败案例 (两个模型都错误):
  - 数量: {common_failures} 个
"""

    if common_failures > 0:
        report += "\n注意: 这些可能是测试用例或模型固有的困难问题\n"

    # 结论和建议
    report += f"""
{'=' * 60}
四、评估结论
{'=' * 60}

{conclusion}

{'=' * 60}
五、建议
{'=' * 60}
"""

    if acc_delta <= 5:
        report += """
1. 量化配置良好，可以继续使用 q4f16_1 模型
2. 定期监控关键业务场景的精度表现
3. 如需进一步提升精度，可考虑调整量化参数
"""
    elif acc_delta <= 10:
        report += """
1. 检查量化参数配置是否合理
2. 分析精度损失案例，优化相关测试用例
3. 考虑在关键场景使用更高精度的模型
4. 监控生产环境中的模型表现
"""
    else:
        report += """
1. 严重警告：精度损失过大
2. 重新评估量化策略和配置
3. 考虑使用 FP16 或更高精度模型
4. 检查模型训练和量化流程
"""

    report += f"\n{'=' * 60}\n报告结束\n{'=' * 60}\n"

    return report

def save_report(report: str, output_file: str):
    """保存报告到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"详细报告已保存: {output_file}")

def main():
    if len(sys.argv) < 3:
        print("用法: python3 accuracy_report.py <baseline_result> <q4f16_result> [output_file]")
        print("示例: python3 accuracy_report.py reports/accuracy_baseline.txt reports/accuracy_q4f16_1.txt reports/accuracy_comparison.txt")
        sys.exit(1)

    baseline_file = sys.argv[1]
    q4f16_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "reports/accuracy_detailed_report.txt"

    try:
        # 分析数据
        print("开始分析精度数据...")
        analysis = analyze_accuracy_comparison(baseline_file, q4f16_file)

        # 生成报告
        print("生成评估报告...")
        report = generate_report(analysis)

        # 输出到终端
        print(report)

        # 保存报告
        save_report(report, output_file)

        print("✓ 分析完成")

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()