# Qwen2.5-3B 模型部署指南

## 实测性能对比

基于 1.5B 验证成功（60 tok/s, TTFT 0.077s, 峰值 4.44GB），现在扩展支持 3B 模型。

| 指标 | 1.5B | 3B (预估) | 差异 |
|---|---|---|---|
| 生成速度 | 60 tok/s | 25-35 tok/s | 慢约 50% |
| TTFT | 0.077s | ~0.15s | 慢约 2x |
| 峰值内存 | 4.44GB | ~5.5GB | +1.1GB |
| 权重大小 | 840MB | 1.9GB | +1.1GB |
| 智能能力 | 中等 | 更强 | 复杂任务处理 |

## 内存优化策略

**现状**：Jetson 8GB，1.5B 峰值 4.44GB，余量 3.1GB

**3B 挑战**：
- 权重增加 +0.8GB
- KV cache 增量
- 默认 `context_window_size: 32768` 是内存陷阱

**解决方案**（通过 `MAX_SEQ_LEN=2048` 压参数）：
```bash
--overrides "max_num_sequence=1;max_total_seq_length=2048;context_window_size=2048;prefill_chunk_size=1280"
```

**预期峰值**：5.3-5.8GB，余量 1.5-2GB，可接受但较紧

## 快速部署步骤

### 1. 更新脚本（PC 侧）

已更新的脚本支持 `MODEL_SIZE` 环境变量切换，拷贝到 Jetson：

```bash
scp -r /mnt/nvme01/workplace/MLC_LLM_deploy/scripts jetson:~/Desktop/workplace/tensorRT/MLC_llm_deploy/
```

### 2. 下载 3B 权重（Jetson 侧）

```bash
MODEL_SIZE=3B ./jetson-run.sh download
```

**预期输出**：
- 使用 HF_ENDPOINT=https://hf-mirror.com
- 下载 mlc-ai/Qwen2.5-3B-Instruct-q4f16_1-MLC
- 最终大小：约 1.9GB（62 个 shard）

### 3. 启动 3B 服务（Jetson 侧）

```bash
# 停止 1.5B 服务（如果在运行）
./jetson-run.sh stop

# 启动 3B 服务（压参数）
MODEL_SIZE=3B MAX_SEQ_LEN=2048 ./jetson-run.sh serve

# 监控日志（JIT 编译比 1.5B 长）
sudo docker logs -f mlc-serve
```

**日志关键点**：
1. 加载本地权重
2. JIT 编译（3-10 分钟，swap 已就位）
3. 监听 8000 端口

### 4. 验证 3B 服务

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen2.5-3B-Instruct-q4f16_1-MLC","messages":[{"role":"user","content":"请用中文介绍一下量子计算的基本原理"}],"stream":true,"max_tokens":128}'
```

### 5. 采集 3B 性能基线

```bash
MODEL_SIZE=3B ./benchmark.sh
```

**预期结果**：
- TTFT：~0.15s
- tok/s：25-35
- 峰值内存：5.3-5.8GB
- 整机功耗：~22W

## 模型切换

### 从 1.5B 切换到 3B

```bash
# 停止当前服务
./jetson-run.sh stop

# 切换到 3B
MODEL_SIZE=3B MAX_SEQ_LEN=2048 ./jetson-run.sh serve
```

### 从 3B 切回 1.5B

```bash
# 停止当前服务
./jetson-run.sh stop

# 切换回 1.5B（默认参数）
./jetson-run.sh serve
```

## 网关集成（支持模型切换）

### 环境变量控制

```bash
# 启动网关（默认 1.5B）
uvicorn integration.llm_gateway:app --host 0.0.0.0 --port 8080

# 切换到 3B 模型
ACTIVE_MODEL=3b uvicorn integration.llm_gateway:app --host 0.0.0.0 --port 8080
```

### 客户端调用

```bash
# 1.5B 模型
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-1.5b","messages":[{"role":"user","content":"你好"}],"stream":true}'

# 3B 模型（需要 ACTIVE_MODEL=3b 启动网关）
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-3b","messages":[{"role":"user","content":"你好"}],"stream":true}'
```

## 决策建议

### 选择 1.5B 的场景
- 需要快速响应（实时对话）
- 内存余量紧张
- 简单问答任务
- 多用户并发（需要更多内存余量）

### 选择 3B 的场景
- 需要更强智能能力
- 复杂推理任务
- 专业领域应用
- 单用户或低并发场景
- 可以接受略慢的响应速度

## 故障排除

### 3B 服务启动失败

1. **检查内存**：`free -h`，确保有足够余量
2. **检查参数**：确认 `MAX_SEQ_LEN=2048` 已设置
3. **增加 swap**：`./jetson-run.sh swap-on`
4. **查看详细日志**：`sudo docker logs mlc-serve --tail 100`

### 性能不如预期

1. **确认功耗模式**：`nvpmodel -q`，应在 MAXN_SUPER 模式
2. **检查 GPU 利用率**：`tegrastats`，应接近 100%
3. **对比基线**：确保使用相同的测试 prompt
4. **JIT 缓存**：首次运行慢是正常的，第二次会快

## 下一步

3B 验证成功后：
1. 对比 1.5B vs 3B 的实测数据
2. 根据应用场景选择主力模型
3. 配置 systemd 开机自启
4. 集成到 ROS2 系统
