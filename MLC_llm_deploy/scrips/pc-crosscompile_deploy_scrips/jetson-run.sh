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
#   SPEC_MODE=small_draft     # 投机解码（默认 disable）。small_draft 用 Qwen2.5-0.5B
#                             # 当 draft（需先按 EAGLE/crosscompile-draft.sh 编译并 scp），
#                             # 8GB 建议 MAX_SEQ_LEN=2048
#                             # 用法: SPEC_MODE=small_draft MODEL_SIZE=1.5B MAX_SEQ_LEN=2048 ./jetson-run.sh serve
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
# 投机解码: disable | small_draft（EAGLE head 路线已核验终止，见 EAGLE/EAGLE-head核验结论-*.md）
SPEC_MODE="${SPEC_MODE:-disable}"
SPEC_DRAFT_SIZE="${SPEC_DRAFT_SIZE:-0.5B}"

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

# 交叉编译产物路径（PC 上 crosscompile.sh 生成并 scp 过来的）：
#   ${WORKDIR}/dist/Qwen2.5-<size>-Instruct-q4f16_1-MLC/            权重目录
#   ${WORKDIR}/dist/libs/...-cuda-jetson_orin_sm87.so               引擎库
# 存在则 serve/chat 追加 --model-lib，跳过设备端 JIT（8GB 内存防 OOM 的关键）
dist_lib_path() {
    local lib
    lib="$(find "${WORKDIR}/dist" -name "*${MODEL_SIZE}*sm87.so" 2>/dev/null | head -1)"
    [ -n "${lib}" ] && echo "${lib#${WORKDIR}/}"
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
    local DIST_MODEL="dist/Qwen2.5-${MODEL_SIZE}-Instruct-q4f16_1-MLC"
    local LIB_ARG=()
    if [ -f "${WORKDIR}/${DIST_MODEL}/mlc-chat-config.json" ] && [ -n "$(dist_lib_path)" ]; then
        echo "使用交叉编译产物: ${DIST_MODEL} + $(dist_lib_path)"
        LIB_ARG=(--model-lib "$(dist_lib_path)")
        # 交叉编译 .so 烘焙的 ModelMetadata 是 32K 窗口/prefill 8192（编译期未压），
        # 8GB Orin 上临时缓冲要 ~10GB —— 必须运行时 overrides 压下去
        sudo docker run --runtime nvidia -it --rm \
            --network=host \
            -v "${WORKDIR}":/workspace \
            -w /workspace \
            "${IMAGE}" \
            mlc_llm chat "/workspace/${DIST_MODEL}" --device cuda:0 "${LIB_ARG[@]}" \
                --overrides "context_window_size=${MAX_SEQ_LEN};prefill_chunk_size=2048"
    else
        sudo docker run --runtime nvidia -it --rm \
            --network=host \
            -v "${WORKDIR}":/workspace \
            -w /workspace \
            "${IMAGE}" \
            mlc_llm chat "${MODEL}" --device cuda:0
    fi
}

do_serve() {
    mkdir -p "${WORKDIR}"
    local MODEL_NAME="Qwen2.5-${MODEL_SIZE}-Instruct-q4f16_1-MLC"
    local LIB_ARG=()
    # 优先级: PC 交叉编译产物（权重+引擎库，免 JIT）> 本地权重（JIT）> HF 下载
    if [ -f "${WORKDIR}/dist/${MODEL_NAME}/mlc-chat-config.json" ] && [ -n "$(dist_lib_path)" ]; then
        MODEL_ARG="/workspace/dist/${MODEL_NAME}"
        LIB_ARG=(--model-lib "$(dist_lib_path)")
        echo "使用交叉编译产物（免 JIT）: ${WORKDIR}/dist/${MODEL_NAME} + $(dist_lib_path)"
    # 本地已有权重目录则用本地路径（容器内网络无法直连 HF 时的兜底，
    # 用 download 子命令预先下载到 ${WORKDIR}/models/ 下）
    elif [ -f "${WORKDIR}/models/${MODEL_NAME}/mlc-chat-config.json" ]; then
        MODEL_ARG="/workspace/models/${MODEL_NAME}"
        echo "使用本地权重: ${WORKDIR}/models/${MODEL_NAME}"
    else
        MODEL_ARG="${MODEL}"
        echo "本地权重不存在，将从 HF 下载（国内网络建议先运行: MODEL_SIZE=${MODEL_SIZE} ./jetson-run.sh download）"
    fi
    # 投机解码参数（small_draft: 0.5B draft 与 1.5B target 配对，需产物齐全才启用）
    local SPEC_ARG=()
    if [ "${SPEC_MODE}" = "small_draft" ]; then
        local DRAFT_NAME="Qwen2.5-${SPEC_DRAFT_SIZE}-Instruct-q4f16_1-MLC"
        local DRAFT_LIB
        DRAFT_LIB="$(find "${WORKDIR}/dist/libs" -name "*${SPEC_DRAFT_SIZE}*sm87.so" 2>/dev/null | head -1 || true)"
        if [ ! -f "${WORKDIR}/dist/${DRAFT_NAME}/mlc-chat-config.json" ] || [ -z "${DRAFT_LIB}" ]; then
            echo "[错误] SPEC_MODE=small_draft 但 draft 产物不齐全:"
            echo "   缺少 ${WORKDIR}/dist/${DRAFT_NAME}/mlc-chat-config.json 或 dist/libs/*${SPEC_DRAFT_SIZE}*sm87.so"
            echo "   先在 PC 上执行 EAGLE/crosscompile-draft.sh 并 scp 到 ~/work/mlc/dist/"
            exit 1
        fi
        [ ${#LIB_ARG[@]} -eq 0 ] && echo "[警告] target 未使用交叉编译产物（将设备端 JIT），叠加 draft 引擎 8GB 上大概率 OOM"
        SPEC_ARG=(--speculative-mode small_draft
                  --additional-models "/workspace/dist/${DRAFT_NAME},/workspace/${DRAFT_LIB#${WORKDIR}/}")
        echo "启用投机解码: small_draft（draft=${DRAFT_NAME}）"
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
            "${LIB_ARG[@]}" \
            "${SPEC_ARG[@]}" \
            --host 0.0.0.0 --port "${PORT}" \
            --mode local \
            --overrides "max_num_sequence=1;max_total_seq_length=${MAX_SEQ_LEN};prefill_chunk_size=2048"
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
