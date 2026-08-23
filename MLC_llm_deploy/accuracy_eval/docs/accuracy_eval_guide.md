# MLC-LLM 量化精度评估使用指南

## 概述

本指南介绍如何使用 MLC-LLM 量化精度评估脚本，测试 q4f16_1 量化模型与 FP16 基线模型的精度差异。

## 前置条件

1. **硬件要求**
   - NVIDIA Jetson Orin (8GB RAM)
   - q4f16_1 模型: 需要 ~2GB 内存
   - FP16 模型: 需要 ~4GB 内存

2. **软件环境**
   - JetPack 6.x (L4T R36.x)
   - CUDA 12.6
   - TensorRT 10.3.0
   - MLC-LLM 0.20.0
   - Docker 环境

3. **服务要求**
   - MLC-LLM serve 服务必须已启动
   - OpenAI 兼容的 HTTP API (端口 8000)

## 脚本介绍

### 1. accuracy_test.sh - 主评估脚本

主要的精度评估脚本，支持多种测试模式。

**位置**: `MLC_llm_deploy/scrips/JIT_deploy_scrips/accuracy_test.sh`

**命令**:
- `prepare` - 准备测试数据
- `baseline` - 建立 FP16 基线
- `test` - 执行精度测试
- `compare` - 对比分析结果
- `all` - 运行完整流程

**示例**:
```bash
cd /mnt/apfs/workplace/tensorRT_ONNX/MLC_llm_deploy/scrips/JIT_deploy_scrips

# 快速开始 - 使用默认测试集
./accuracy_test.sh all

# 使用 C-Eval 测试集
TEST_CASES=../testcases/ceval_subset.json ./accuracy_test.sh all

# 使用 GSM8K 数学测试集
TEST_CASES=../testcases/gsm8k_cn_100.json ./accuracy_test.sh all
```

### 2. eval_baseline.sh - 模型准备脚本

处理 FP16 模型的下载、部署和服务切换。

**位置**: `MLC_llm_deploy/scrips/JIT_deploy_scrips/eval_baseline.sh`

**命令**:
- `download` - 下载 FP16 模型
- `switch` - 切换模型 (q4f16_1 <-> q0f16)
- `verify` - 验证当前服务状态

**示例**:
```bash
# 下载并切换到 FP16 模型
./eval_baseline.sh download

# 切换到 q4f16_1 模型
TARGET_MODEL=q4f16_1 ./eval_baseline.sh switch

# 验证服务状态
./eval_baseline.sh verify
```

### 3. accuracy_report.py - 结果分析脚本

生成详细的精度对比报告。

**位置**: `MLC_llm_deploy/scrips/JIT_deploy_scrips/accuracy_report.py`

**用法**:
```bash
python3 accuracy_report.py <baseline_result> <q4f16_result> [output_file]
```

**示例**:
```bash
python3 accuracy_report.py \
  ../../reports/accuracy_baseline.txt \
  ../../reports/accuracy_q4f16_1.txt \
  ../../reports/accuracy_detailed.txt
```

## 测试用例

### 1. quick_check.json - 快速验证集

**描述**: 10 个简单测试用例，用于基础验证
**用途**: 快速检查脚本功能
**包含**: 数学计算、基础知识、简单推理

### 2. ceval_subset.json - C-Eval 抽样

**描述**: 30 个 C-Eval 基准测试用例
**用途**: 标准学术评测
**包含**: 数学、物理、化学、历史、地理等

### 3. gsm8k_cn_100.json - GSM8K 中文版

**描述**: 50 个数学推理问题
**用途**: 数学能力评测（量化敏感型任务）
**包含**: 应用题、计算题、推理题

### 4. business_prompts.jsonl - 业务回归集

**描述**: 20 个实际应用场景测试
**用途**: 业务场景验证
**包含**: 问候、介绍、任务执行等

## 完整工作流程

### 步骤 1: 启动服务

```bash
cd /mnt/apfs/workplace/tensorRT_ONNX/MLC_llm_deploy/scrips/JIT_deploy_scrips

# 启动 q4f16_1 服务
./jetson-run.sh serve
```

### 步骤 2: 准备测试环境

```bash
# 准备测试数据（自动创建默认测试用例）
./accuracy_test.sh prepare

# 可选：下载 FP16 模型
./eval_baseline.sh download
```

### 步骤 3: 建立 FP16 基线

```bash
# 切换到 FP16 模型
TARGET_MODEL=q0f16 ./eval_baseline.sh switch

# 等待服务启动完成（约 1-2 分钟）
# 验证服务状态
./eval_baseline.sh verify

# 建立 FP16 基线
./accuracy_test.sh baseline
```

### 步骤 4: 测试 q4f16_1 模型

```bash
# 切换回 q4f16_1 模型
TARGET_MODEL=q4f16_1 ./eval_baseline.sh switch

# 等待服务启动完成
# 验证服务状态
./eval_baseline.sh verify

# 执行精度测试
./accuracy_test.sh test
```

### 步骤 5: 分析结果

```bash
# 对比分析结果
./accuracy_test.sh compare

# 或使用详细报告
python3 accuracy_report.py \
  ../../reports/accuracy_baseline.txt \
  ../../reports/accuracy_q4f16_1.txt \
  ../../reports/accuracy_detailed.txt
```

## 快速测试（单模型验证）

如果只需要测试当前模型的准确率：

```bash
# 假设当前运行的是 q4f16_1 模型
./accuracy_test.sh prepare
# 手动测试当前模型（修改脚本中的模型名称或使用现有结果）
python3 -c "
import urllib.request, json

url = 'http://localhost:8000/v1/chat/completions'
test_cases = [
    {'question': '中国的首都是哪里？', 'answer': '北京'},
    {'question': '12 * 12 = ?', 'answer': '144'}
]

correct = 0
for case in test_cases:
    payload = json.dumps({
        'model': 'current',
        'messages': [{'role': 'user', 'content': case['question']}],
        'temperature': 0.0,
        'max_tokens': 256
    }).encode()

    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        response = json.load(resp)
        actual = response['choices'][0]['message']['content']
        if case['answer'].lower() in actual.lower():
            correct += 1
            print(f'✓ {case[\"question\"]}')
        else:
            print(f'✗ {case[\"question\"]}')

print(f'准确率: {correct}/{len(test_cases)} ({correct/len(test_cases):.1%})')
"
```

## 环境变量配置

### accuracy_test.sh

```bash
# 服务配置
HOST=localhost           # 服务主机
PORT=8000               # 服务端口

# 测试配置
TEST_CASES=../testcases/quick_check.json  # 测试用例文件
TEMPERATURE=0.0         # 温度参数（贪心解码）
MAX_TOKENS=256          # 最大生成长度
TIMEOUT=120             # 请求超时秒
```

### eval_baseline.sh

```bash
# 模型配置
TARGET_MODEL=q4f16_1    # 目标模型 (q4f16_1 or q0f16)

# 内存配置
SKIP_MEMORY_CHECK=false # 是否跳过内存检查
```

## 输出文件

### 报告文件位置

所有报告文件保存在 `MLC_llm_deploy/reports/` 目录：

- `accuracy_baseline.txt` - FP16 基线结果
- `accuracy_q4f16_1.txt` - q4f16_1 测试结果
- `accuracy_comparison.txt` - 简单对比报告
- `accuracy_detailed_report.txt` - 详细分析报告

### 报告内容

**准确率信息**:
- 准确率百分比
- 正确/总数统计
- 精度损失百分比

**详细分析**:
- 文本相似度统计
- 失败案例分析
- 误差分布

**评估结论**:
- 量化效果评级
- 改进建议
- 风险评估

## 精度验收标准

### 4-bit 量化工业标准

- **优秀**: 知识型基准 Δacc ≤ 2%
- **良好**: 数学推理 Δacc ≤ 3%
- **可接受**: 业务一致性 ≥ 90%
- **警告**: 任何基准 Δacc > 5%
- **严重问题**: 任何基准 Δacc > 10%

### 典型预期结果

基于 q4f16_1 量化策略：

- **C-Eval**: 精度损失 1-3 个百分点
- **GSM8K**: 精度损失 2-4 个百分点
- **业务场景**: 一致性 85-95%

## 故障排除

### 常见问题

1. **服务不可用**
   ```bash
   # 检查服务状态
   sudo docker ps | grep mlc-serve

   # 重启服务
   ./jetson-run.sh serve
   ```

2. **内存不足**
   ```bash
   # 切换到无桌面模式
   sudo systemctl isolate multi-user.target

   # 跳过内存检查（不推荐）
   SKIP_MEMORY_CHECK=true ./eval_baseline.sh switch
   ```

3. **模型切换失败**
   ```bash
   # 手动停止服务
   sudo docker stop mlc-serve

   # 清理缓存
   sudo docker system prune -f

   # 重新切换
   TARGET_MODEL=q4f16_1 ./eval_baseline.sh switch
   ```

4. **测试结果异常**
   ```bash
   # 检查测试用例格式
   cat ../testcases/quick_check.json

   # 使用简单测试集验证
   TEST_CASES=../testcases/quick_check.json ./accuracy_test.sh test
   ```

## 性能优化

### 内存优化

- 使用无桌面模式释放内存
- 停止不必要的后台服务
- 监控内存使用: `tegrastats`

### 测试优化

- 使用较小的测试集进行快速验证
- 调整 `MAX_TOKENS` 减少内存使用
- 并发测试（慎用，可能影响结果）

## 技术参考

### 量化策略

**q4f16_1**:
- 权重 int4 组量化（group size=32）
- 激活 fp16
- embedding/lm_head 保持 fp16

**q0f16** (FP16):
- 完全 FP16 精度
- 作为基线对比参考

### 测试方法论

完全仿照 `benchmark.sh` 的四条硬标准：

1. **引擎常驻**: 模型只加载一次
2. **先 warmup 再测**: 预热轮结果丢弃
3. **固定 prompt**: 同一 prompt 对比
4. **多轮取 P50**: 中位数为主指标

### 统计方法

- **P50 (中位数)**: 主要指标
- **P90**: 尾部分布
- **文本相似度**: SequenceMatcher
- **编辑距离**: 字符级差异

## 总结

本评估体系提供了完整的 q4f16_1 量化精度测试能力：

✓ **简单易用**: 一键式脚本，无需复杂配置
✓ **标准评测**: 支持 C-Eval、GSM8K 等标准基准
✓ **实用导向**: 关注实际业务场景
✓ **详细分析**: 提供多维度的精度评估
✓ **工程化**: 基于现有最佳实践

通过这些工具，您可以科学评估量化精度损失，为生产部署提供数据支撑。