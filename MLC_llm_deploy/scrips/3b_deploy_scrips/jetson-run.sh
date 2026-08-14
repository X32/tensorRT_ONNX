#!/usr/bin/env bash
# ============================================================================
# jetson-run.sh — Jetson Orin Nano Super 8GB 上快速验证 MLC-LLM Qwen2.5-1.5B/3B
#
# 用法:
#   ./jetson-run.sh check     # 环境检查（JP 版本/磁盘/Docker/nvpmodel）
#   ./jetson-run.sh shell     # 进入 dustynv/mlc 容器（挂载 ~/work/mlc）
#   ./jetson-run.sh chat      # 容器内交互对话验证
#   ./jetson-run.sh download  # 宿主机经镜像站下载 MLC 权重（HF 直连不通时用）
#   ./jetson-run.sh serve     # 容器内启动 OpenAI 兼容服务（后台，本地权重优先）
#   ./jetson-run.sh stop      # 停止 serve 容器
#   ./jetson-run.sh swap-on   # 挂 8G swap（防首次 JIT OOM）
#
# 环境变量:
#   MODEL_SIZE=3B             # 切换到 3B 模型（默认 1.5B）
#   MAX_SEQ_LEN=2048          # 调整最大序列长度（默认 4096，3B 建议压到 2048）
#
# 版本锚点: MLC-LLM v0.20.0 / dustynv/mlc:0.20.0-r36.4.0 / JP6.2 (R36.4) / sm_87
# ============================================================================
set -euo pipefail

IMAGE="dustynv/mlc:0.20.0-r36.4.0"
MODEL_SIZE="${MODEL_SIZE:-1.5B}"
MODEL_REPO="mlc-ai/Qwen2.5-${MODEL_SIZE}-Instruct-q4f16_1-MLC"
MODEL="HF://${MODEL_REPO}"
WORKDIR="${HOME}/work/mlc"
PORT=8000
SERVE_NAME="mlc-serve"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"

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
    # 本地已有权重目录则用本地路径（容器内网络无法直连 HF 时的兜底，
    # 用 download 子命令预先下载到 ${WORKDIR}/models/ 下）
    local MODEL_NAME="Qwen2.5-${MODEL_SIZE}-Instruct-q4f16_1-MLC"
    LOCAL_MODEL="${WORKDIR}/models/${MODEL_NAME}"
    if [ -f "${LOCAL_MODEL}/mlc-chat-config.json" ]; then
        MODEL_ARG="/workspace/models/${MODEL_NAME}"
        echo "使用本地权重: ${LOCAL_MODEL}"
    else
        MODEL_ARG="${MODEL}"
        echo "本地权重不存在，将从 HF 下载（国内网络建议先运行: MODEL_SIZE=${MODEL_SIZE} ./jetson-run.sh download）"
    fi
    # 已在运行则先停
    sudo docker rm -f "${SERVE_NAME}" 2>/dev/null || true
    sudo docker run -d --runtime nvidia \
        --name "${SERVE_NAME}" \
        --network=host \
        -v "${WORKDIR}":/workspace \
        -w /workspace \
        --restart unless-stopped \
        "${IMAGE}" \
        mlc_llm serve "${MODEL_ARG}" \
            --device cuda:0 \
            --host 0.0.0.0 --port "${PORT}" \
            --mode local \
            --overrides "max_num_sequence=1;max_total_seq_length=${MAX_SEQ_LEN}"
    echo "serve 已后台启动（容器名 ${SERVE_NAME}，端口 ${PORT}，模型 ${MODEL_SIZE}）"
    echo "验证: curl http://localhost:${PORT}/v1/chat/completions \\"
    echo "        -H 'Content-Type: application/json' \\"
    echo "        -d '{\"model\":\"${MODEL_REPO}\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}],\"stream\":true,\"max_tokens\":64}'"
    echo "日志: sudo docker logs -f ${SERVE_NAME}"
}

do_download() {
    # 宿主机上通过镜像站下载 MLC 权重（绕开容器内 git clone HF 失败问题）
    mkdir -p "${WORKDIR}/models"
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    echo "使用 HF_ENDPOINT=${HF_ENDPOINT}"
    if ! command -v hf >/dev/null 2>&1; then
        pip install -U "huggingface_hub[cli]" -i https://pypi.tuna.tsinghua.edu.cn/simple
    fi
    local MODEL_NAME="Qwen2.5-${MODEL_SIZE}-Instruct-q4f16_1-MLC"
    hf download "${MODEL_REPO}" \
        --local-dir "${WORKDIR}/models/${MODEL_NAME}"
    echo "下载完成:"
    ls -lh "${WORKDIR}/models/${MODEL_NAME}"
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
    download) do_download ;;
    serve)   do_serve ;;
    stop)    do_stop ;;
    swap-on) do_swap ;;
    *)       grep '^#   ' "$0" | sed 's/^#   //'; exit 1 ;;
esac
