# MLC-LLM 部署套件：Qwen2.5-1.5B/3B on Jetson Orin Nano Super 8GB

基于 MLC-LLM v0.20.0 编译器路线，PC（GTX 1080 Ti）交叉编译 / dustynv 容器快速验证，
目标机 Jetson Orin Nano Super 8GB（JetPack 6.2 / R36.4 / CUDA 12.6 / sm_87）。

**实测性能**：
- **1.5B**：60 tok/s, TTFT 0.077s, 峰值 4.44GB
- **3B**：~25-35 tok/s, TTFT ~0.15s, 峰值 ~5.5GB（需要 `MAX_SEQ_LEN=2048` 压上下文）

## 快速开始

### 方式一：1.5B 模型（默认，推荐新手）

```bash
# 拷贝 scripts/ 到 Jetson 后
./jetson-run.sh check      # 环境检查
./jetson-run.sh swap-on    # 挂 8G swap（防首次 JIT OOM）
./jetson-run.sh download   # 下载 1.5B 权重（~840MB）
./jetson-run.sh serve      # 启动服务（:8000）

# 验证
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen2.5-1.5B-Instruct-q4f16_1-MLC","messages":[{"role":"user","content":"你好"}],"stream":true,"max_tokens":64}'

# 采基线
./benchmark.sh
```

### 方式二：3B 模型（更强智能，需压参数）

```bash
# 设置环境变量切换到 3B（压上下文到 2048 适配 8GB）
MODEL_SIZE=3B MAX_SEQ_LEN=2048 ./jetson-run.sh download   # 下载 3B 权重（~1.9GB）
MODEL_SIZE=3B MAX_SEQ_LEN=2048 ./jetson-run.sh serve      # 启动 3B 服务

# 采 3B 基线
MODEL_SIZE=3B ./benchmark.sh
```

### 模型对比

| 模型 | 速度 | TTFT | 峰值内存 | 智能能力 | 适用场景 |
|---|---|---|---|---|---|
| 1.5B | 60 tok/s | 0.077s | 4.44GB | 中等 | 日常对话、实时响应 |
| 3B | 25-35 tok/s | ~0.15s | ~5.5GB | 更强 | 复杂推理、专业任务 |

## 脚本参数说明

### jetson-run.sh 环境变量

- `MODEL_SIZE=3B`：切换到 3B 模型（默认 `1.5B`）
- `MAX_SEQ_LEN=2048`：调整最大序列长度（默认 `4096`，3B 建议压到 `2048`）

### benchmark.sh 环境变量

- `MODEL_SIZE=3B`：切换到 3B 模型进行性能测试（默认 `1.5B`）

## 目录结构

```
scripts/
  jetson-run.sh            # Jetson 快速验证（支持 1.5B/3B 切换）
  benchmark.sh             # 性能基线采集（支持模型切换）
  example-3b.sh            # 3B 模型部署示例
  pc-crosscompile/
    Dockerfile             # PC 交叉编译镜像
    crosscompile.sh        # 一键交叉编译
integration/
  llm_gateway.py           # FastAPI 网关（支持 1.5B/3B 路由）
  ros2/
    mlc_llm_client_node.py # ROS2 节点
  systemd/
    mlc-serve.service      # MLC serve 托管
    llm-gateway.service    # 网关托管
DOC/
  Jetson Orin 8GB 部署 Qwen2.5 1.5B/3B：MLC-LLM 实操文档.md
```

## 系统集成

### 网关启动（支持模型切换）

```bash
# 启动网关（默认 1.5B）
pip install fastapi uvicorn httpx
uvicorn integration.llm_gateway:app --host 0.0.0.0 --port 8080

# 切换到 3B 模型
ACTIVE_MODEL=3b uvicorn integration.llm_gateway:app --host 0.0.0.0 --port 8080

# 验证
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-1.5b","messages":[{"role":"user","content":"你好"}],"stream":true,"max_tokens":64}'
```

### ROS2 节点

```bash
ros2 run <pkg> mlc_llm_client_node --ros-args -p gateway_url:=http://localhost:8080
ros2 topic pub /llm/chat std_msgs/String '{data: "{\"prompt\":\"你好\",\"session_id\":\"t1\"}"}' --once
ros2 topic echo /llm/response/done
```

## 版本锚点（三段统一）

| 组件 | 版本 |
|---|---|
| MLC-LLM | v0.20.0 |
| 容器 | dustynv/mlc:0.20.0-r36.4.0 |
| CUDA | 12.6（PC 交叉编译镜像 / Jetson JP6.2 一致） |
| 目标架构 | sm_87（Orin 全系） |
| 1.5B 模型 | mlc-ai/Qwen2.5-1.5B-Instruct-q4f16_1-MLC |
| 3B 模型 | mlc-ai/Qwen2.5-3B-Instruct-q4f16_1-MLC |
