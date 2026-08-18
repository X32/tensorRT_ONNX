# MLC-LLM JIT 部署脚本

基于 MLC-LLM v0.20.0 的即时编译（JIT）部署方案，在 Jetson Orin 8GB 上运行 Qwen2.5-1.5B/3B 模型。

## 快速开始

### 1.5B 模型（推荐）

```bash
# 环境检查
./jetson-run.sh check

# 挂载 8GB swap（防止首次 JIT OOM）
./jetson-run.sh swap-on

# 下载模型权重（~840MB）
./jetson-run.sh download

# 启动服务（后台运行，端口 8000）
./jetson-run.sh serve

# 验证推理
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mlc-ai/Qwen2.5-1.5B-Instruct-q4f16_1-MLC","messages":[{"role":"user","content":"你好"}],"stream":true,"max_tokens":64}'

# 性能测试
./benchmark.sh
```

### 3B 模型（更强智能）

```bash
# 切换到 3B 模型（压上下文到 2048）
MODEL_SIZE=3B MAX_SEQ_LEN=2048 ./jetson-run.sh download
MODEL_SIZE=3B MAX_SEQ_LEN=2048 ./jetson-run.sh serve
MODEL_SIZE=3B ./benchmark.sh
```

## 脚本命令

### jetson-run.sh

| 命令 | 说明 |
|------|------|
| `check` | 检查 JetPack 版本、磁盘、Docker、功耗模式 |
| `shell` | 进入 dustynv/mlc 容器交互式 shell |
| `chat` | 容器内交互对话（测试用） |
| `download` | 通过镜像站下载权重（推荐） |
| `serve` | 启动 OpenAI 兼容服务（后台运行） |
| `stop` | 停止 serve 容器 |
| `swap-on` | 挂载 8GB swap（首次运行必需） |

### benchmark.sh

```bash
./benchmark.sh [jetson-ip]  # 默认 localhost
```

采集 TTFT、生成速度、峰值内存、GPU 功耗，输出到终端和 `benchmark-result.txt`。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_SIZE` | `1.5B` | 模型大小（1.5B 或 3B） |
| `MAX_SEQ_LEN` | `4096` | 最大序列长度（3B 建议设为 2048） |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace 镜像站 |

## 性能参考

| 模型 | 生成速度 | TTFT | 峰值内存 | 适用场景 |
|------|----------|------|----------|----------|
| 1.5B | 60 tok/s | 0.077s | 4.44GB | 日常对话、实时响应 |
| 3B | 25-35 tok/s | ~0.15s | ~5.5GB | 复杂推理、专业任务 |

## 环境要求

- **设备**: Jetson Orin (sm_87 架构)
- **JetPack**: 6.x (L4T R36.x)
- **CUDA**: 12.6
- **Docker**: 支持 nvidia runtime
- **磁盘**: ≥15GB 可用空间
- **内存**: 8GB（推荐先挂载 swap）

## 常见问题

**Q: 首次运行内存不足？**
```bash
./jetson-run.sh swap-on  # 必须先挂载 8GB swap
```

**Q: 权重下载失败？**
```bash
# 使用国内镜像站
export HF_ENDPOINT=https://hf-mirror.com
./jetson-run.sh download
```

**Q: 3B 模型 OOM？**
```bash
# 压低上下文长度
MAX_SEQ_LEN=2048 ./jetson-run.sh serve
```

**Q: 查看服务日志？**
```bash
sudo docker logs -f mlc-serve
```

## 服务管理

```bash
# 查看运行状态
sudo docker ps | grep mlc-serve

# 停止服务
./jetson-run.sh stop

# 重启服务
./jetson-run.sh stop && ./jetson-run.sh serve
```

## 版本锚点

- **MLC-LLM**: v0.20.0
- **容器**: dustynv/mlc:0.20.0-r36.4.0
- **CUDA**: 12.6
- **目标架构**: sm_87 (Orin)
