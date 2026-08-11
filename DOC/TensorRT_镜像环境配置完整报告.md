# TensorRT 镜像环境配置完整报告

## 📋 目录
1. [项目背景](#项目背景)
2. [环境准备](#环境准备)
3. [项目下载与安装](#项目下载与安装)
4. [基础镜像构建](#基础镜像构建)
5. [TensorRT 环境配置](#tensorrt-环境配置)
6. [容器保存为镜像](#容器保存为镜像)
7. [项目挂载与启动](#项目挂载与启动)
8. [验证测试](#验证测试)
9. [故障排除](#故障排除)
10. [最佳实践](#最佳实践)

---

## 项目背景

### 🎯 项目目标
在 NVIDIA Jetson Orin 设备上构建完整的深度学习模型部署环境，支持：
- **PyTorch 2.11.0** + **CUDA 12.6** 深度学习框架
- **TensorRT 10.3** 高性能推理引擎
- **PyCUDA** GPU 编程接口
- **Jupyter Lab** 交互式开发环境

### 📊 目标设备
- **硬件**: NVIDIA Jetson Orin
- **架构**: ARM64 (aarch64)
- **系统**: Ubuntu 22.04 (JetPack 6.5)
- **CUDA**: 12.6
- **GPU**: Orin (87 CUDA 架构)

---

## 环境准备

### 🔧 系统要求检查

#### 1. 确认 JetPack 版本
```bash
# 检查 JetPack 和 CUDA 版本
cat /etc/nv_tegra_release
```

**预期输出**:
```
# R36 (release), REVISION: 5.0, GCID: 43688277
# L4T: 36.5.0
# CUDA: 12.6
```

#### 2. 确认 Docker 安装
```bash
# 检查 Docker 版本
docker --version

# 检查 Docker 运行状态
docker ps
```

#### 3. 确认 CUDA 和 GPU
```bash
# 检查 CUDA 版本
nvcc --version

# 检查 GPU 设备
nvidia-smi

# 检查 TensorRT 宿主机安装
dpkg -l | grep tensorrt
```

### 📦 必要软件包
```bash
# 确保系统包管理器最新
sudo apt-get update

# 安装必要工具（如需要）
sudo apt-get install -y build-essential python3-dev
```

---

## 项目下载与安装

### 🌟 下载 jetson-containers 项目

#### 1. 克隆项目仓库
```bash
# 进入工作目录
cd ~/Desktop/software/

# 克隆 jetson-containers 项目
git clone https://github.com/dusty-nv/jetson-containers.git

# 进入项目目录
cd jetson-containers
```

#### 2. 安装项目依赖
```bash
# 运行安装脚本
bash install.sh

# 验证安装
jetson-containers build --list-packages | head -20
```

#### 3. 验证环境配置
```bash
# 检查 L4T 版本检测
python3 -c "from jetson_containers import l4t_version; print(l4t_version.L4T_VERSION)"

# 检查可用的 PyTorch 包
jetson-containers build --show-packages pytorch
```

---

## 基础镜像构建

### 🏗️ 方案选择分析

#### 方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **Dockerfile 构建** | 可重复、版本控制 | 需要代理、网络要求高 | 生产环境 |
| **容器提交方式** | 简单快速、无需网络 | 不可重复、难追踪 | 快速原型 |

#### 🎯 推荐方案：容器提交方式

**原因**：
- 网络环境限制（代理配置复杂）
- 快速迭代需求
- 调试和验证方便

### 📦 基础镜像获取

#### 1. 拉取官方 PyTorch 镜像
```bash
# 拉取 PyTorch 基础镜像
docker pull dustynv/jetson-containers:pytorch-r36.5.0-tegra-aarch64-cu126-22.04

# 验证镜像
docker images | grep pytorch
```

#### 2. 启动基础容器
```bash
# 启动临时容器安装基础组件
docker run -d --name jupyter-base --runtime=nvidia \
    -v "/home/zy/Desktop/workplace/Train_Custom_Dataset/":/workspace \
    -w /workspace \
    -p 8888:8888 \
    dustynv/jetson-containers:pytorch-r36.5.0-tegra-aarch64-cu126-22.04
```

### 🔧 基础环境安装

#### 1. 安装 Jupyter Lab
```bash
# 在容器中安装 Jupyter Lab
docker exec jupyter-base uv pip install jupyterlab notebook ipywidgets

# 设置 Jupyter 启动配置
docker exec jupyter-base mkdir -p /root/.jupyter
```

#### 2. 安装 ONNX 运行时
```bash
# 安装 ONNX 和 ONNX Runtime
docker exec jupyter-base uv pip install onnx onnxruntime

# 安装 ONNX Script（PyTorch ONNX 导出支持）
docker exec jupyter-base uv pip install onnxscript
```

#### 3. 安装 Torchvision
```bash
# 安装配套版本的 Torchvision
docker exec jupyter-base uv pip install torchvision==0.26.0
```

### 🏷️ 创建基础镜像
```bash
# 将配置好的容器保存为镜像
docker commit jupyter-base onnx-jupyter:latest

# 清理临时容器
docker stop jupyter-base
docker rm jupyter-base

# 验证镜像创建
docker images | grep onnx-jupyter
```

---

## TensorRT 环境配置

### 🎯 TensorRT 安装策略

#### 环境分析
```bash
# 检查容器内 TensorRT 文件
docker exec jupyter-project2 ls -la /usr/local/lib/python3.10/dist-packages/ | grep tensorrt
```

**发现结果**：
- ✅ 容器内已包含 TensorRT wheel 文件
- ✅ 文件位置：`/usr/local/lib/python3.10/dist-packages/`
- ✅ 版本：TensorRT 10.3.0
- ✅ 架构：aarch64 (ARM64)

### 📦 TensorRT 安装步骤

#### 1. 启动工作容器
```bash
# 从基础镜像启动新容器
docker run -d --name jupyter-project2 --runtime=nvidia \
    -v "/home/zy/Desktop/workplace/Train_Custom_Dataset/":/workspace \
    -w /workspace \
    -p 8889:8888 \
    onnx-jupyter:latest

# 等待容器启动
sleep 5
```

#### 2. 安装 TensorRT
```bash
# 使用 uv 安装 TensorRT（快速方式）
docker exec jupyter-project2 uv pip install \
    /usr/local/lib/python3.10/dist-packages/tensorrt-10.3.0-cp310-none-linux_aarch64.whl

# 验证 TensorRT 安装
docker exec jupyter-project2 python3 -c "
import tensorrt as trt
print(f'TensorRT版本: {trt.__version__}')
print('TensorRT安装成功！')
"
```

#### 3. 安装 PyCUDA（可选，用于高级 CUDA 操作）
```bash
# 安装 PyCUDA（需要编译，10-30分钟）
docker exec jupyter-project2 uv pip install pycuda

# 验证 PyCUDA 安装
docker exec jupyter-project2 python3 -c "
import pycuda
print(f'PyCUDA版本: {pycuda.VERSION}')
print('PyCUDA安装成功！')
"
```

**⚠️ 注意**：PyCUDA 安装需要从源码编译，在 ARM64 架构上可能需要 10-30 分钟。

---

## 容器保存为镜像

### 🎯 镜像保存时机

**推荐在以下时机保存镜像**：
- ✅ 完成 TensorRT 安装后
- ✅ 验证所有组件正常工作后
- ✅ 项目配置完成后

### 📋 容器保存命令

#### 1. 查看当前容器状态
```bash
# 检查运行中的容器
docker ps | grep jupyter-project2

# 检查容器内安装的包
docker exec jupyter-project2 pip list | grep -E "torch|tensor|cuda"
```

#### 2. 保存为镜像
```bash
# 将容器保存为新镜像
docker commit jupyter-project2 jupyter-tensorrt:complete

# 查看新创建的镜像
docker images | grep jupyter-tensorrt
```

#### 3. 镜像信息验证
```bash
# 检查镜像详细信息
docker inspect jupyter-tensorrt:complete | grep -A 5 "Created"

# 查看镜像大小
docker images jupyter-tensorrt:complete
```

---

## 项目挂载与启动

### 📂 项目目录结构

#### 推荐目录布局
```
~/Desktop/workplace/
├── Train_Custom_Dataset/          # 主项目目录
│   ├── notebooks/                  # Jupyter 笔记本
│   ├── models/                     # 模型文件
│   ├── data/                       # 数据集
│   └── scripts/                    # Python 脚本
└── TensorRT_Projects/              # TensorRT 专用项目
    ├── inference/                  # 推理代码
    ├── optimization/               # 模型优化
    └── deployment/                 # 部署脚本
```

### 🚀 启动脚本创建

#### 1. 主项目启动脚本
```bash
#!/bin/bash
# 🚀 启动包含TensorRT的Jupyter Lab - 主项目

PROJECT_DIR="/home/zy/Desktop/workplace/Train_Custom_Dataset/"

echo "🚀 启动 Jupyter Lab (TensorRT版)..."
echo "📂 项目目录: $PROJECT_DIR"
echo "🌐 访问地址: http://localhost:8889"

# 清理旧容器
docker stop jupyter-tensorrt 2>/dev/null || true
docker rm jupyter-tensorrt 2>/dev/null || true

# 启动新容器
docker run -d --name jupyter-tensorrt --runtime=nvidia \
    -v "$PROJECT_DIR":/workspace \
    -w /workspace \
    -p 8889:8888 \
    jupyter-tensorrt:complete

echo "✅ Jupyter Lab 已启动！访问: http://localhost:8889"
```

#### 2. TensorRT 专用项目脚本
```bash
#!/bin/bash
# 🚀 TensorRT 专用项目启动

PROJECT_DIR="/home/zy/Desktop/workplace/TensorRT_Projects/"

docker stop tensorrt-project 2>/dev/null || true
docker rm tensorrt-project 2>/dev/null || true

docker run -d --name tensorrt-project --runtime=nvidia \
    -v "$PROJECT_DIR":/workspace \
    -w /workspace \
    -p 8890:8888 \
    jupyter-tensorrt:complete

echo "✅ TensorRT项目已启动！访问: http://localhost:8890"
```

### 🔧 环境变量配置

#### GPU 权限配置
```bash
# 运行时添加 GPU 权限
docker run -d --name jupyter-tensorrt \
    --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=0 \
    -v "$PROJECT_DIR":/workspace \
    -w /workspace \
    -p 8889:8888 \
    jupyter-tensorrt:complete
```

#### 网络配置
```bash
# 指定网络模式
docker run -d --name jupyter-tensorrt \
    --runtime=nvidia \
    --network=bridge \
    --publish 8889:8888 \
    -v "$PROJECT_DIR":/workspace \
    jupyter-tensorrt:complete
```

---

## 验证测试

### 🧪 功能验证测试

#### 1. 基础环境测试
```bash
# Python 环境测试
docker exec jupyter-tensorrt python3 << 'EOF'
import sys
import torch
import tensorrt as trt

print("🎯 TensorRT 环境验证")
print("=" * 30)

# 基础环境信息
print(f"✅ Python版本: {sys.version.split()[0]}")
print(f"✅ PyTorch版本: {torch.__version__}")
print(f"✅ TensorRT版本: {trt.__version__}")
print(f"✅ CUDA可用: {torch.cuda.is_available()}")

# GPU 设备信息
if torch.cuda.is_available():
    print(f"✅ GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU数量: {torch.cuda.device_count()}")
    print(f"✅ CUDA版本: {torch.version.cuda}")
    print(f"✅ GPU架构: {torch.cuda.get_device_properties(0).arch}")

print("\n🎯 基础环境验证完成！")
EOF
```

#### 2. TensorRT 功能测试
```bash
# TensorRT 核心功能测试
docker exec jupyter-tensorrt python3 << 'EOF'
import tensorrt as trt

print("🔧 TensorRT 功能测试")
print("=" * 20)

# 创建 TensorRT Logger 和 Builder
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)

print(f"✅ TensorRT Logger创建成功")
print(f"✅ TensorRT Builder创建成功")

# 检查支持的特性
print(f"✅ 支持的ONNX版本: {trt.ONNX_PARSER_VERSION}")
print(f"✅ TensorRT平台: {trt.Builder.PLATFORM}")

# 检查插件支持
print(f"✅ TensorRT插件可用: {builder.is_network_supported()}")

print("\n🎯 TensorRT 功能验证完成！")
EOF
```

#### 3. GPU 计算测试
```bash
# GPU 推理性能测试
docker exec jupyter-tensorrt python3 << 'EOF'
import torch
import time

print("⚡ GPU 性能测试")
print("=" * 20)

# 创建测试模型
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.nn.Sequential(
    torch.nn.Linear(1000, 500),
    torch.nn.ReLU(),
    torch.nn.Linear(500, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 10)
).to(device)

# 性能测试
dummy_input = torch.randn(32, 1000).to(device)

# 预热
for _ in range(10):
    _ = model(dummy_input)

# 正式测试
start_time = time.time()
for _ in range(100):
    output = model(dummy_input)
end_time = time.time()

# 计算性能
total_time = end_time - start_time
avg_time = total_time / 100
throughput = 100 / total_time

print(f"✅ 100次推理总时间: {total_time:.3f}秒")
print(f"✅ 平均推理时间: {avg_time*1000:.2f}毫秒")
print(f"✅ 吞吐量: {throughput:.2f} 推理/秒")
print(f"✅ 计算设备: {device}")

print("\n🎯 GPU 性能测试完成！")
EOF
```

#### 4. Jupyter Lab 连接测试
```bash
# 检查 Jupyter Lab 进程
docker exec jupyter-tensorrt ps aux | grep jupyter

# 测试端口连接
curl -I http://localhost:8889

# 获取 Jupyter 访问令牌
docker exec jupyter-tensorrt jupyter server list
```

---

## 故障排除

### 🚨 常见问题与解决方案

#### 1. CUDA 不可用问题

**症状**：
```python
torch.cuda.is_available()  # 返回 False
```

**解决方案**：
```bash
# 检查容器 GPU 访问权限
docker run --runtime=nvidia --rm dustynv/jetson-containers:pytorch-r36.5.0-tegra-aarch64-cu126-22.04 nvidia-smi

# 确保容器启动时包含 --runtime=nvidia 参数
docker stop jupyter-tensorrt
docker rm jupyter-tensorrt
docker run -d --name jupyter-tensorrt --runtime=nvidia \
    -v "$PROJECT_DIR":/workspace \
    -p 8889:8888 \
    jupyter-tensorrt:complete
```

#### 2. TensorRT 导入错误

**症状**：
```
ImportError: libnvinfer.so: cannot open shared object file
```

**解决方案**：
```bash
# 检查 TensorRT 库链接
docker exec jupyter-tensorrt ldconfig -p | grep nvinfer

# 重新安装 TensorRT
docker exec jupyter-tensorrt pip uninstall tensorrt -y
docker exec jupyter-tensorrt uv pip install \
    /usr/local/lib/python3.10/dist-packages/tensorrt-10.3.0-cp310-none-linux_aarch64.whl
```

#### 3. 端口冲突问题

**症状**：
```
Error: Bind for 0.0.0.0:8889 failed: port is already allocated
```

**解决方案**：
```bash
# 查看占用端口的容器
docker ps | grep 8889

# 停止冲突容器
docker stop <冲突容器名>

# 或使用不同端口
docker run -d --name jupyter-tensorrt --runtime=nvidia \
    -v "$PROJECT_DIR":/workspace \
    -p 8891:8888 \
    jupyter-tensorrt:complete
```

#### 4. PyCUDA 编译超时

**症状**：
```
Building pycuda==2026.1... (卡在编译步骤)
```

**解决方案**：
```bash
# 取消当前安装 (Ctrl+C)
# 安装编译依赖
docker exec jupyter-tensorrt apt-get update
docker exec jupyter-tensorrt apt-get install -y build-essential python3-dev

# 重新安装（添加详细输出）
docker exec jupyter-tensorrt uv pip install pycuda --verbose

# 或跳过 PyCUDA（TensorRT 不强制依赖 PyCUDA）
```

#### 5. 磁盘空间不足

**症状**：
```
Error: No space left on device
```

**解决方案**：
```bash
# 清理 Docker 系统缓存
docker system prune -a

# 清理未使用的镜像
docker image prune -a

# 查看磁盘使用情况
docker system df
```

---

## 最佳实践

### 🎯 开发工作流

#### 1. 镜像管理策略
```bash
# 为不同环境创建专用镜像
jupyter-tensorrt:complete          # 完整版（含所有组件）
jupyter-tensorrt:base            # 基础版（仅核心组件）
jupyter-tensorrt:dev             # 开发版（含调试工具）
```

#### 2. 项目隔离策略
```bash
# 为不同项目使用独立容器
# 项目A: 主项目 (端口 8889)
# 项目B: TensorRT项目 (端口 8890)
# 项目C: 实验项目 (端口 8891)
```

#### 3. 数据持久化策略
```bash
# 使用 Docker Volume 持久化数据
docker volume create tensorrt_models
docker run -d --name jupyter-tensorrt --runtime=nvidia \
    -v tensorrt_models:/models \
    -v "$PROJECT_DIR":/workspace \
    -p 8889:8888 \
    jupyter-tensorrt:complete
```

### 📊 性能优化

#### 1. GPU 优化
```bash
# 设置 GPU 内存使用策略
docker run -d --name jupyter-tensorrt --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=0 \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v "$PROJECT_DIR":/workspace \
    -p 8889:8888 \
    jupyter-tensorrt:complete
```

#### 2. 内存优化
```bash
# 限制容器内存使用
docker run -d --name jupyter-tensorrt --runtime=nvidia \
    --memory="4g" \
    --memory-swap="8g" \
    -v "$PROJECT_DIR":/workspace \
    -p 8889:8888 \
    jupyter-tensorrt:complete
```

### 🔄 部署流程

#### 开发 → 测试 → 生产
```bash
# 1. 开发环境（本地）
./start_tensorrt_jupyter.sh

# 2. 测试环境（验证）
docker commit jupyter-tensorrt jupyter-tensorrt:test
docker run -d --name jupyter-test --runtime=nvidia jupyter-tensorrt:test

# 3. 生产环境（部署）
docker tag jupyter-tensorrt:complete registry/jupyter-tensorrt:prod
docker push registry/jupyter-tensorrt:prod
```

---

## 📚 参考资源

### 🔗 官方文档
- [NVIDIA TensorRT 文档](https://docs.nvidia.com/deeplearning/tensorrt/)
- [PyTorch Jetson 指南](https://github.com/dusty-nv/jetson-containers)
- [CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

### 🛠️ 工具参考
- [Docker 文档](https://docs.docker.com/)
- [Jupyter Lab 文档](https://jupyterlab.readthedocs.io/)
- [ONNX 规范](https://onnx.ai/onnx/intro/)

---

## 📊 附录

### A. 完整安装脚本

#### 一键安装脚本（完整版）
```bash
#!/bin/bash
# 🚀 TensorRT 环境完整安装脚本

set -e  # 遇到错误立即退出

PROJECT_DIR="/home/zy/Desktop/workplace/Train_Custom_Dataset/"
BASE_IMAGE="dustynv/jetson-containers:pytorch-r36.5.0-tegra-aarch64-cu126-22.04"
FINAL_IMAGE="jupyter-tensorrt:complete"
CONTAINER_NAME="jupyter-tensorrt-builder"

echo "🚀 开始 TensorRT 环境完整安装..."
echo "项目目录: $PROJECT_DIR"
echo "基础镜像: $BASE_IMAGE"
echo "最终镜像: $FINAL_IMAGE"
echo ""

# 步骤1: 清理旧容器
echo "🧹 清理旧容器..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# 步骤2: 启动构建容器
echo "📦 启动构建容器..."
docker run -d --name $CONTAINER_NAME --runtime=nvidia \
    -v "$PROJECT_DIR":/workspace \
    -w /workspace \
    -p 8888:8888 \
    $BASE_IMAGE

echo "⏳ 等待容器启动..."
sleep 5

# 步骤3: 安装 Jupyter Lab
echo "🔧 安装 Jupyter Lab..."
docker exec $CONTAINER_NAME uv pip install jupyterlab notebook ipywidgets

# 步骤4: 安装 ONNX 相关
echo "🔧 安装 ONNX 运行时..."
docker exec $CONTAINER_NAME uv pip install onnx onnxruntime onnxscript

# 步骤5: 安装 Torchvision
echo "🔧 安装 Torchvision..."
docker exec $CONTAINER_NAME uv pip install torchvision==0.26.0

# 步骤6: 安装 TensorRT
echo "🔧 安装 TensorRT..."
docker exec $CONTAINER_NAME uv pip install \
    /usr/local/lib/python3.10/dist-packages/tensorrt-10.3.0-cp310-none-linux_aarch64.whl

# 步骤7: 可选安装 PyCUDA
echo "🔧 安装 PyCUDA (可选，需要时间)..."
read -p "是否安装 PyCUDA? (y/n): " install_pycuda
if [ "$install_pycuda" = "y" ]; then
    echo "PyCUDA 安装中，需要 10-30 分钟..."
    docker exec $CONTAINER_NAME uv pip install pycuda
fi

# 步骤8: 验证安装
echo "🧪 验证安装..."
docker exec $CONTAINER_NAME python3 -c "
import torch
import tensorrt as trt
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ TensorRT: {trt.__version__}')
print(f'✅ CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU设备: {torch.cuda.get_device_name(0)}')
"

# 步骤9: 创建最终镜像
echo "🏷️ 创建最终镜像..."
docker commit $CONTAINER_NAME $FINAL_IMAGE

# 步骤10: 清理构建容器
echo "🧹 清理构建容器..."
docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME

echo ""
echo "✅ TensorRT 环境安装完成！"
echo "🖼️ 镜像名称: $FINAL_IMAGE"
echo ""
echo "🚀 启动命令:"
echo "docker run -d --name jupyter-tensorrt --runtime=nvidia \\"
echo "    -v \"$PROJECT_DIR\":/workspace \\"
echo "    -w /workspace \\"
echo "    -p 8889:8888 \\"
echo "    $FINAL_IMAGE"
```

### B. 环境配置文件

#### Docker Compose 配置
```yaml
version: '3.8'

services:
  jupyter-tensorrt:
    image: jupyter-tensorrt:complete
    container_name: jupyter-tensorrt
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./workspace:/workspace
      - tensorrt_models:/models
    ports:
      - "8889:8888"
    restart: unless-stopped

volumes:
  tensorrt_models:
```

### C. 快速参考命令

```bash
# 🚀 启动容器
docker start jupyter-tensorrt

# 🛑 停止容器
docker stop jupyter-tensorrt

# 🔄 重启容器
docker restart jupyter-tensorrt

# 📊 查看容器日志
docker logs jupyter-tensorrt

# 🔍 进入容器shell
docker exec -it jupyter-tensorrt bash

# 📋 查看容器进程
docker exec jupyter-tensorrt ps aux

# 💾 保存容器为新镜像
docker commit jupyter-tensorrt jupyter-tensorrt:backup

# 🗑️ 删除容器
docker rm jupyter-tensorrt

# 🖼️ 删除镜像
docker rmi jupyter-tensorrt:complete
```

---

## 🎯 总结

### ✅ 完成检查清单

- [ ] jetson-containers 项目下载完成
- [ ] 基础 PyTorch 镜像获取成功
- [ ] Jupyter Lab 安装完成
- [ ] ONNX 和 ONNX Runtime 安装完成
- [ ] Torchvision 安装完成
- [ ] TensorRT 安装成功
- [ ] (可选) PyCUDA 安装完成
- [ ] 容器保存为镜像完成
- [ ] 项目目录挂载配置完成
- [ ] 启动脚本创建完成
- [ ] 功能验证测试通过
- [ ] Jupyter Lab 可正常访问

### 🎯 最终成果

**构建完成的环境包含**：
- ✅ PyTorch 2.11.0 + CUDA 12.6
- ✅ TensorRT 10.3 高性能推理引擎
- ✅ Jupyter Lab 交互式开发环境
- ✅ GPU 加速支持（Jetson Orin）
- ✅ 完整的模型部署工具链

**可立即用于**：
- 深度学习模型推理
- 模型优化与转换
- 生产环境部署
- 交互式开发调试

---

**文档版本**: 1.0
**最后更新**: 2025-01-11
**适用设备**: NVIDIA Jetson Orin (JetPack 6.5)
**维护状态**: 活跃维护中

---

## 📞 技术支持

如有问题，请参考：
1. 本文档的故障排除章节
2. jetson-containers GitHub Issues
3. NVIDIA Developer Forums

**祝您使用愉快！** 🚀