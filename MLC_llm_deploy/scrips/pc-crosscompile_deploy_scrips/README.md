# MLC-LLM 交叉编译部署

PC 交叉编译 → Jetson 运行，绕过 8GB 内存限制，无需设备端 JIT 编译。

## 快速开始

### 第一步：PC 端环境准备

```bash
cd pc-crosscompile_scrips

# 环境检查
./setup-pc.sh check

# 安装依赖 + 构建镜像（约 2-4 小时）
./setup-pc.sh all

# 或分步执行
./setup-pc.sh install   # 安装 uv、git-lfs 等工具
./setup-pc.sh build      # 构建 mlc-llm-jetson:cu126 镜像
```

### 第二步：PC 端交叉编译

```bash
# 编译 1.5B 模型（默认）
./crosscompile.sh

# 编译 3B 模型
./crosscompile.sh Qwen2.5-3B-Instruct
```

**产物位置**：
```
dist/Qwen2.5-<size>-Instruct-q4f16_1-MLC/          # MLC 权重
dist/libs/...-cuda-jetson_orin_sm87.so             # Jetson 引擎库
```

### 第三步：传输到 Jetson

```bash
# 方式一：scp 传输（推荐）
scp -r dist/ jetson@<jetson-ip>:~/work/mlc/

# 方式二：打包传输
tar czf mlc-dist.tar.gz dist/
scp mlc-dist.tar.gz jetson@<jetson-ip>:~/
# Jetson 上: tar xzf mlc-dist.tar.gz -C ~/work/mlc/
```

### 第四步：Jetson 端启动服务

```bash
# 拷贝 jetson-run.sh 到 Jetson 后
./jetson-run.sh check      # 环境检查
./jetson-run.sh serve      # 启动服务（自动使用交叉编译产物，免 JIT）

# 验证推理
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen2.5-1.5B-Instruct-q4f16_1-MLC","messages":[{"role":"user","content":"你好"}],"stream":true,"max_tokens":64}'
```

## 脚本说明

### setup-pc.sh

| 命令 | 说明 |
|------|------|
| `check` | 检查 NVIDIA GPU、Docker、权限、磁盘空间 |
| `install` | 安装 uv、git-lfs 等必要工具 |
| `build` | 构建 `mlc-llm-jetson:cu126` 交叉编译镜像 |
| `all` | check → install → build 全流程 |

### crosscompile.sh

```bash
./crosscompile.sh [模型名称]
```

**执行流程**：
1. 下载原始模型（`dist/models/`）
2. 转换为 MLC 格式（q4f16_1 量化）
3. 生成配置文件
4. 交叉编译 sm_87 引擎库

### jetson-run.sh

| 命令 | 说明 |
|------|------|
| `check` | Jetson 环境检查 |
| `serve` | 启动服务（优先使用交叉编译产物） |
| `chat` | 交互对话验证 |
| `stop` | 停止服务 |

## 环境变量

### crosscompile.sh

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_NAME` | `Qwen2.5-1.5B-Instruct` | 目标模型 |

### jetson-run.sh

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_SIZE` | `1.5B` | 模型大小 |
| `MAX_SEQ_LEN` | `4096` | 最大序列长度（3B 建议设为 2048） |
| `SPEC_MODE` | `disable` | 投机解码模式 |

## 环境要求

### PC 端（交叉编译）

- **GPU**: NVIDIA GPU（≥8GB VRAM 推荐）
- **系统**: Linux（Ubuntu 22.04 推荐）
- **Docker**: 支持 NVIDIA Container Toolkit
- **磁盘**: ≥50GB 可用空间

### Jetson 端（运行）

- **设备**: Jetson Orin (sm_87)
- **JetPack**: 6.x (L4T R36.x)
- **CUDA**: 12.6
- **内存**: 8GB
- **磁盘**: ≥15GB 可用空间

## 交叉编译优势

| 对比项 | JIT 方式 | 交叉编译方式 |
|--------|----------|--------------|
| 首次启动 | 需要 JIT 编译（5-10分钟） | 免 JIT，直接启动 |
| 内存需求 | 需要 8GB swap | 无需额外 swap |
| 启动时间 | 慢（编译时间长） | 快（秒级启动） |
| 部署复杂度 | 简单 | 需要交叉编译环境 |

## 版本锚点

- **MLC-LLM**: v0.20.0（commit `d2118b3c`）
- **TVM**: mlc-ai/relax @ `9c894f78`（2024-02-22）
- **CUDA**: 12.6（PC 和 Jetson 一致）
- **目标架构**: sm_87（Jetson Orin）
- **交叉编译镜像**: `mlc-llm-jetson:cu126`
- **Jetson 运行时**: `dustynv/mlc:0.20.0-r36.4.0`

## 常见问题

**Q: 交叉编译镜像构建失败？**
```bash
# 检查 NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```

**Q: 模型下载失败？**
```bash
# 配置镜像站
export HF_ENDPOINT=https://hf-mirror.com
./crosscompile.sh
```

**Q: Jetson 端找不到交叉编译产物？**
```bash
# 检查路径结构
ls ~/work/mlc/dist/Qwen2.5-1.5B-Instruct-q4f16_1-MLC/
ls ~/work/mlc/dist/libs/*sm87.so
```

**Q: 服务启动时仍进行 JIT？**
```bash
# 检查引擎库路径
find ~/work/mlc/dist -name "*sm87.so"
# 确保 jetson-run.sh 能找到该 .so 文件
```

## 性能参考

| 模型 | 生成速度 | TTFT | 峰值内存 | 启动方式 |
|------|----------|------|----------|----------|
| 1.5B | 60 tok/s | 0.077s | 4.44GB | 免 JIT |
| 3B | 25-35 tok/s | ~0.15s | ~5.5GB | 免 JIT |

## 技术原理

交叉编译在 PC 上完成，产物为：
- **权重转换**: PyTorch → MLC 格式（q4f16_1 量化）
- **引擎编译**: TVM → aarch64 + sm_87 CUDA 库

Jetson 端直接加载预编译库，跳过耗时且耗内存的 JIT 步骤。
