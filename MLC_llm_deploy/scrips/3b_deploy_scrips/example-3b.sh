#!/usr/bin/env bash
# ============================================================================
# example-3b.sh — 示例：部署 Qwen2.5-3B 模型
#
# 这是使用参数化脚本部署 3B 模型的完整示例
# ============================================================================

# 设置环境变量切换到 3B 模型
export MODEL_SIZE=3B
export MAX_SEQ_LEN=2048  # 3B 模型需要压低上下文长度以适应 8GB 内存

# 使用镜像站下载 3B 权重（约 1.9GB，62 个 shard）
./jetson-run.sh download

# 启动 3B 服务（后台）
./jetson-run.sh serve

# 等待 JIT 编译完成（可能需要几分钟）
echo "等待服务启动..."
sleep 30

# 查看日志确认服务已启动
sudo docker logs -f mlc-serve

# 当看到日志显示监听 8000 端口后，按 Ctrl+C 退出日志查看
# 然后运行性能测试：
# MODEL_SIZE=3B ./benchmark.sh
