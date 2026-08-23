# MLC-LLM 量化精度评估 - 详细步骤指南

## 🎯 目标说明

**我们要做什么？**
比较两个模型的准确率差异：
- **FP16 模型**（高精度基线） vs **q4f16_1 模型**（4-bit 量化版本）

**为什么要做？**
量化会节省内存和加速推理，但可能损失精度。这个测试就是要**量化**这个精度损失到底是多少。

---

## 📋 前置条件检查

### 步骤 0：确保环境准备好

**这步做什么：** 检查你的 Jetson 设备是否准备好运行测试

```bash
# 进入脚本目录
cd /mnt/apfs/workplace/tensorRT_ONNX/MLC_llm_deploy/scrips/JIT_deploy_scrips

# 检查脚本是否都在
ls -la accuracy_* eval_baseline.sh
```

**应该看到：**
- `accuracy_test.sh` ✅
- `eval_baseline.sh` ✅
- `accuracy_report.py` ✅
- `accuracy_demo.sh` ✅

**如果缺少文件：**
```bash
# 从 git 拉取最新代码
cd /mnt/apfs/workplace/tensorRT_ONNX
git pull
```

---

## 🚀 完整测试流程（推荐新手）

### 方法一：一键自动化（最简单）

**这步做什么：** 让脚本自动完成所有步骤，适合第一次使用

```bash
# 运行完整测试
./accuracy_test.sh all
```

**脚本会自动做什么：**
1. ✅ 检查服务是否运行
2. ✅ 准备测试数据和目录
3. ✅ 对当前模型进行测试
4. ✅ 生成对比报告

**优点：** 简单，一条命令搞定
**缺点：** 需要手动切换模型，不理解具体过程

---

## 🔧 方法二：分步详细流程（推荐）

### 步骤 1：启动 MLC-LLM 服务

**这步做什么：** 确保 LLM 服务正在运行，这样才能进行测试

```bash
# 启动服务（如果还没启动）
./jetson-run.sh serve
```

**验证服务是否启动：**
```bash
# 检查服务状态
curl http://localhost:8000/v1/models
```

**应该看到类似输出：**
```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen2.5-1.5B-Instruct-q4f16_1-MLC",
      ...
    }
  ]
}
```

**如果失败：**
- 检查 docker 是否运行：`sudo docker ps`
- 重新启动服务：`./jetson-run.sh serve`

---

### 步骤 2：准备测试环境

**这步做什么：** 创建必要的目录和测试用例文件

```bash
# 准备测试数据
./accuracy_test.sh prepare
```

**脚本具体做什么：**
1. 📁 创建 `reports/` 目录（存放测试结果）
2. 📄 检查测试用例文件是否存在
3. 💾 如果测试用例不存在，自动创建默认的 `quick_check.json`
4. 🔍 检查内存是否足够（FP16 模型需要更多内存）

**输出示例：**
```
[STEP] 准备测试数据...
[INFO] ✓ 报告目录: /mnt/apfs/workplace/tensorRT_ONNX/MLC_llm_deploy/reports
[INFO] ✓ 测试用例文件: /mnt/apfs/workplace/tensorRT_ONNX/MLC_llm_deploy/testcases/quick_check.json
[INFO] ✓ 测试数据准备完成
```

---

### 步骤 3：建立 FP16 基线数据

**这步做什么：** 测试 FP16 高精度模型的表现，作为对比基准

#### 3.1 切换到 FP16 模型

**为什么要切换：** 我们需要先用高精度的 FP16 模型测试，得到"标准答案"

```bash
# 切换到 FP16 模型
./eval_baseline.sh download  # 首次需要下载 FP16 模型
TARGET_MODEL=q0f16 ./eval_baseline.sh switch
```

**脚本具体做什么：**
1. 🛑 停止当前运行的 q4f16_1 服务
2. 📥 下载 FP16 模型（如果首次运行，约 3.2GB）
3. 💾 检查内存是否足够（FP16 需要 ~4GB）
4. 🚀 启动 FP16 模型服务
5. ⏳ 等待服务启动完成（1-2 分钟）

**输出示例：**
```
[STEP] 切换模型到: q0f16
[INFO] 当前模型: mlc-ai/Qwen2.5-1.5B-Instruct-q4f16_1-MLC
[STEP] 停止当前服务...
[INFO] ✓ 服务已停止
[INFO] 目标模型: Qwen2.5-1.5B-Instruct-q0f16-MLC
[STEP] 启动服务...
[INFO] ✓ 模型切换完成
```

#### 3.2 验证 FP16 服务

**这步做什么：** 确保 FP16 模型服务正常运行

```bash
# 验证服务状态
./eval_baseline.sh verify
```

**应该看到：**
```
[STEP] 验证服务状态...
[INFO] ✓ 容器运行中
[INFO] ✓ 服务端点正常
[INFO] 当前模型: Qwen2.5-1.5B-Instruct-q0f16-MLC
```

#### 3.3 建立 FP16 基线数据

**这步做什么：** 让 FP16 模型回答测试题，记录准确率作为"标准答案"

```bash
# 建立 FP16 基线
./accuracy_test.sh baseline
```

**脚本具体做什么：**
1. 📖 读取测试用例文件（默认 10 个问题）
2. 🤖 逐个发送给 FP16 模型
3. ✅ 检查每个回答是否正确（答案是否包含在回复中）
4. 📊 计算准确率（正确数/总题数）
5. 💾 保存结果到 `reports/accuracy_baseline.txt`

**输出示例：**
```
[STEP] 建立 FP16 基线...
[INFO] ✓ 服务运行正常
[INFO] 模型: Qwen2.5-1.5B-Instruct-q0f16-MLC
[STEP] 开始精度测试 (baseline)...

测试用例数量: 10
开始测试...
✓ 1/10: 计算 25 * 34 + 17 = ?... (匹配: True)
✓ 2/10: 中国的首都是哪里？... (匹配: True)
✗ 3/10: 如果今天是星期三，后天是星期几？... (匹配: False)
...

=== 汇总 ===
准确率: 90.0% (9/10)
平均时延: 0.156s
详细报告: .../reports/accuracy_baseline.txt

[INFO] ✓ 精度测试完成
[INFO] ✓ 基线数据已保存到: .../reports/accuracy_baseline.txt
```

**结果解读：**
- FP16 模型准确率：90% （9/10 题正确）
- 这个 90% 就是我们的"基线标准"

---

### 步骤 4：测试 q4f16_1 量化模型

**这步做什么：** 现在测试 q4f16_1 量化模型，看它的准确率如何

#### 4.1 切换回 q4f16_1 模型

**为什么要切换：** 我们要测试量化模型的实际表现

```bash
# 切换回 q4f16_1 模型
TARGET_MODEL=q4f16_1 ./eval_baseline.sh switch
```

**脚本具体做什么：**
1. 🛑 停止 FP16 服务
2. 🚀 启动 q4f16_1 服务（更快启动，更小内存）
3. ⏳ 等待服务启动完成

#### 4.2 验证 q4f16_1 服务

```bash
# 验证服务状态
./eval_baseline.sh verify
```

#### 4.3 测试 q4f16_1 模型

**这步做什么：** 让 q4f16_1 模型回答同样的测试题，看准确率如何

```bash
# 执行精度测试
./accuracy_test.sh test
```

**脚本具体做什么：**
1. 📖 读取**相同的**测试用例文件
2. 🤖 逐个发送给 q4f16_1 模型
3. ✅ 检查每个回答是否正确
4. 📊 计算准确率
5. 💾 保存结果到 `reports/accuracy_q4f16_1.txt`

**输出示例：**
```
[STEP] 执行精度测试...
[INFO] ✓ 服务运行正常
[INFO] 模型: Qwen2.5-1.5B-Instruct-q4f16_1-MLC
[STEP] 开始精度测试 (q4f16_1)...

测试用例数量: 10
开始测试...
✓ 1/10: 计算 25 * 34 + 17 = ?... (匹配: True)
✓ 2/10: 中国的首都是哪里？... (匹配: True)
✗ 3/10: 如果今天是星期三，后天是星期几？... (匹配: False)
...

=== 汇总 ===
准确率: 80.0% (8/10)
平均时延: 0.089s
详细报告: .../reports/accuracy_q4f16_1.txt

[INFO] ✓ 精度测试完成
[INFO] ✓ 测试结果已保存到: .../reports/accuracy_q4f16_1.txt
```

**结果解读：**
- q4f16_1 模型准确率：80% （8/10 题正确）
- 比 FP16 慢一点（推理速度）

---

### 步骤 5：对比分析结果

**这步做什么：** 比较两个模型的结果，计算量化带来的精度损失

```bash
# 对比分析结果
./accuracy_test.sh compare
```

**脚本具体做什么：**
1. 📖 读取两个结果文件：
   - `reports/accuracy_baseline.txt` (FP16)
   - `reports/accuracy_q4f16_1.txt` (q4f16_1)
2. 📊 提取准确率数据
3. 🧮 计算精度损失：Δacc = FP16准确率 - q4f16_1准确率
4. 📝 生成对比报告
5. 💾 保存到 `reports/accuracy_comparison.txt`

**输出示例：**
```
[STEP] 对比分析结果...
=== MLC-LLM 量化精度对比分析 ===
生成时间: 2026-08-18 14:30:15
对比文件: accuracy_baseline.txt vs accuracy_q4f16_1.txt

=== 准确率对比 ===
FP16 基线:    90.0% (9/10)
q4f16_1:     80.0% (8/10)
精度损失:    +10.0%

=== 分析结论 ===
⚠ 精度损失较大，建议检查量化配置

详细数据请查看原文件: .../accuracy_baseline.txt 和 .../accuracy_q4f16_1.txt

对比报告已保存: .../reports/accuracy_comparison.txt

[INFO] ✓ 对比分析完成
```

**结果解读：**
- FP16 准确率：90%
- q4f16_1 准确率：80%
- **精度损失：10%** （这是量化带来的代价）

---

## 🎯 结果解读

### 精度损失标准

| 精度损失 | 评级 | 说明 | 建议 |
|---------|------|------|------|
| Δacc ≤ 2% | ⭐⭐⭐⭐⭐ 优秀 | 量化效果很好 | 继续使用 q4f16_1 |
| 2% < Δacc ≤ 5% | ⭐⭐⭐⭐ 良好 | 精度损失可接受 | 正常使用 |
| 5% < Δacc ≤ 10% | ⭐⭐⭐ 一般 | 需要关注 | 监控关键任务 |
| Δacc > 10% | ⭐⭐ 较差 | 精度损失大 | 考虑调整策略 |

### 实际结果示例

**好的量化效果（期望结果）：**
```
FP16 基线:    90.0% (9/10)
q4f16_1:     88.0% (8.8/10)
精度损失:    +2.0%
✓ 量化效果优秀
```

**需要关注的结果：**
```
FP16 基线:    90.0% (9/10)
q4f16_1:     80.0% (8/10)
精度损失:    +10.0%
⚠ 精度损失较大，建议检查配置
```

---

## 🔧 高级用法

### 使用更多测试题

**这步做什么：** 使用更大的测试集，得到更可靠的评估结果

```bash
# 使用 C-Eval 标准测试集（30题）
TEST_CASES=../testcases/ceval_subset.json ./accuracy_test.sh baseline
TEST_CASES=../testcases/ceval_subset.json ./accuracy_test.sh test
TEST_CASES=../testcases/ceval_subset.json ./accuracy_test.sh compare
```

**为什么这样做：**
- 10 题可能不够代表性
- 30 题标准测试集更可靠
- 包含多种类型：数学、物理、历史、地理等

### 使用数学推理测试

**这步做什么：** 专门测试数学能力，量化对数学推理影响更大

```bash
# 使用 GSM8K 数学测试集（47题）
TEST_CASES=../testcases/gsm8k_cn_100.json ./accuracy_test.sh baseline
TEST_CASES=../testcases/gsm8k_cn_100.json ./accuracy_test.sh test
TEST_CASES=../testcases/gsm8k_cn_100.json ./accuracy_test.sh compare
```

---

## 📊 查看详细报告

**这步做什么：** 查看更详细的分析报告，了解具体哪些题答错了

```bash
# 生成详细报告
python3 accuracy_report.py \
  ../../reports/accuracy_baseline.txt \
  ../../reports/accuracy_q4f16_1.txt \
  ../../reports/accuracy_detailed_report.txt
```

**详细报告包含：**
- 📈 准确率详细对比
- 📝 文本相似度分析
- 🔍 具体失败案例分析
- 💡 改进建议

---

## ⚠️ 常见问题解决

### 问题1：服务不可用

**现象：**
```
[ERROR] 服务不可用: http://localhost:8000
```

**解决：**
```bash
# 检查服务是否运行
sudo docker ps | grep mlc-serve

# 如果没有运行，启动服务
./jetson-run.sh serve

# 等待1-2分钟后重试
```

### 问题2：内存不足

**现象：**
```
[ERROR] 内存不足: lfb=3500MB (需要 ≥4000MB)
```

**解决：**
```bash
# 切换到无桌面模式释放内存
sudo systemctl isolate multi-user.target

# 或者跳过内存检查（不推荐）
SKIP_MEMORY_CHECK=true ./eval_baseline.sh switch
```

### 问题3：模型切换失败

**现象：**
```
[ERROR] 模型切换失败
```

**解决：**
```bash
# 手动停止服务
sudo docker stop mlc-serve

# 清理缓存
sudo docker system prune -f

# 重新切换
TARGET_MODEL=q4f16_1 ./eval_baseline.sh switch
```

---

## 📝 总结

### 完整流程回顾

1. **启动服务** - 确保 LLM 服务运行
2. **准备环境** - 创建测试目录和文件
3. **建立基线** - 测试 FP16 高精度模型
4. **测试量化** - 测试 q4f16_1 量化模型
5. **对比分析** - 计算精度差异

### 核心要点

- 🎯 **目的**：量化 q4f16_1 的精度损失
- 📊 **方法**：对比 FP16 vs q4f16_1 的准确率
- 🔧 **工具**：accuracy_test.sh + eval_baseline.sh
- 📈 **结果**：得到 Δacc 精度损失百分比

### 预期结果

基于 q4f16_1 的量化策略：
- **数学推理**：精度损失 2-4%
- **知识问答**：精度损失 1-3%
- **综合测试**：精度损失 1-5%

---

**现在您已经完全理解了整个精度评估流程！** 🎉

有问题随时查看 `DOC/accuracy_eval_guide.md` 获取更多详情。