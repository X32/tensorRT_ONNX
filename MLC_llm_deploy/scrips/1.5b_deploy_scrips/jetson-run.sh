#!/usr/bin/env bash
# ============================================================================
# jetson-run.sh — Jetson Orin Nano Super 8GB 上快速验证 MLC-LLM Qwen2.5-1.5B
#
# 用法:
#   ./jetson-run.sh check     # 环境检查（JP 版本/磁盘/Docker/nvpmodel）
#   ./jetson-run.sh shell     # 进入 dustynv/mlc 容器（挂载 ~/work/mlc）
#   ./jetson-run.sh chat      # 容器内交互对话验证
#   ./jetson-run.sh serve     # 容器内启动 OpenAI 兼容服务（后台）
#   ./jetson-run.sh stop      # 停止 serve 容器
#   ./jetson-run.sh swap-on   # 挂 8G swap（防首次 JIT OOM）
#
# 版本锚点: MLC-LLM v0.20.0 / dustynv/mlc:0.20.0-r36.4.0 / JP6.2 (R36.4) / sm_87
# ============================================================================
set -euo pipefail

IMAGE="dustynv/mlc:0.20.0-r36.4.0"
MODEL="HF://mlc-ai/Qwen2.5-1.5B-Instruct-q4f16_1-MLC"
WORKDIR="${HOME}/work/mlc"
PORT=8000
SERVE_NAME="mlc-serve"

cmd="${1:-help}"

do_check() {
    echo "== Jetson 环境检查 =="
    echo "-- L4T 版本（期望 R36.4 / JP6.2）:"
    cat /etc/nv_tegra_release 2>/dev/null || echo "   [警告] 未找到 nv_tegra_release"
    echo "-- 磁盘余量（容器 6.9GB + 模型 ~2GB，建议 ≥ 15GB）:"
    df -h / | tail -1
    echo "-- Docker:"
    docker --version 2>/dev/null || { echo "   [错误] Docker 未安装"; exit 1; }
    docker info 2>/dev/null | grep -i "nvidia" >/dev/null \
        && echo "   nvidia runtime: OK" \
        || echo "   [警告] 未检测到 nvidia runtime（JP6.2 默认 nvidia-container-toolkit）"
    echo "-- 功耗模式（基准测试建议 MAXN 25W: sudo nvpmodel -m 0）:"
    nvpmodel -q 2>/dev/null || echo "   [提示] nvpmodel 不可用"
    echo "-- GPU 架构（期望 sm_87）:"
    ls /proc/device-tree/gpu*/compatible 2>/dev/null || true
    echo "== 检查完成 =="
}

do_shell() {
    mkdir -p "${WORKDIR}"
    sudo docker run --runtime nvidia -it --rm \
        --network=host \
        -v "${WORKDIR}":/workspace \
        -w /workspace \
        "${IMAGE}" bash
}

do_chat() {
    mkdir -p "${WORKDIR}"
    sudo docker run --runtime nvidia -it --rm \
        --network=host \
        -v "${WORKDIR}":/workspace \
        -w /workspace \
        "${IMAGE}" \
        mlc_llm chat "${MODEL}" --device cuda:0
}

do_serve() {
    mkdir -p "${WORKDIR}"
    # 已在运行则先停
    sudo docker rm -f "${SERVE_NAME}" 2>/dev/null || true
    sudo docker run -d --runtime nvidia \
        --name "${SERVE_NAME}" \
        --network=host \
        -v "${WORKDIR}":/workspace \
        -w /workspace \
        --restart unless-stopped \
        "${IMAGE}" \
        mlc_llm serve "${MODEL}" \
            --device cuda:0 \
            --host 0.0.0.0 --port "${PORT}" \
            --max-batch-size 1 \
            --max-total-sequence-length 4096
    echo "serve 已后台启动（容器名 ${SERVE_NAME}，端口 ${PORT}）"
    echo "验证: curl http://localhost:${PORT}/v1/chat/completions \\"
    echo "        -H 'Content-Type: application/json' \\"
    echo "        -d '{\"model\":\"Qwen2.5-1.5B-Instruct-q4f16_1-MLC\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}],\"stream\":true,\"max_tokens\":64}'"
    echo "日志: sudo docker logs -f ${SERVE_NAME}"
}

do_stop() {
    sudo docker rm -f "${SERVE_NAME}" 2>/dev/null && echo "已停止 ${SERVE_NAME}" || echo "无运行中的 serve"
}

do_swap() {
    if swapon --show | grep -q /swapfile; then
        echo "swap 已存在:"; swapon --show
    else
        sudo fallocate -l 8G /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        echo "8G swap 已挂载:"; swapon --show
    fi
}

case "${cmd}" in
    check)   do_check ;;
    shell)   do_shell ;;
    chat)    do_chat ;;
    serve)   do_serve ;;
    stop)    do_stop ;;
    swap-on) do_swap ;;
    *)       grep '^#   ' "$0" | sed 's/^#   //'; exit 1 ;;
esac
