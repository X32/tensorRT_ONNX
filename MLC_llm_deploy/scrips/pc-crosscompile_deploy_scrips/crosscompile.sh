#!/usr/bin/env bash
# ============================================================================
# crosscompile.sh — PC 上执行：权重转换 + 交叉编译 Jetson sm_87 引擎库
#
# 前置: 已构建镜像 mlc-llm-jetson:cu126（见同目录 Dockerfile）
# 用法: ./crosscompile.sh            # 默认 Qwen2.5-1.5B-Instruct
#       ./crosscompile.sh Qwen2.5-3B-Instruct
# 产物: ./dist/<model>-q4f16_1-MLC/           （MLC 权重目录）
#       ./dist/libs/<model>-q4f16_1-cuda-jetson_orin_sm87.so（引擎库）
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_NAME="${1:-Qwen2.5-1.5B-Instruct}"
QUANT="q4f16_1"
CONV_TEMPLATE="qwen2"
IMAGE="mlc-llm-jetson:cu126"

mkdir -p dist/models dist/libs

# ---- ① 下载原始模型（宿主机，目标无关）----
# 完整性判据: 仅 config.json 存在不够(上次下载中断也会留下它),必须至少
# 有一个 >1GB 的 safetensors 分片,否则残缺目录会被跳过下载直接进转换
if [ ! -f "dist/models/${MODEL_NAME}/config.json" ] \
   || [ -z "$(find "dist/models/${MODEL_NAME}" -name '*.safetensors' -size +1G 2>/dev/null | head -1)" ]; then
    echo "== 下载 ${MODEL_NAME} =="
    uvx hf download "Qwen/${MODEL_NAME}" --local-dir "dist/models/${MODEL_NAME}"
fi

# ---- ② 转换权重 + 生成配置（容器内执行，与第③步编译及 Jetson 运行时同源）----
# 注: MLC-LLM 不在 PyPI 发布（官方 wheel 索引 mlc.ai/wheels 无 aarch64 包），
#     宿主机不装 mlc_llm，转换直接在交叉编译镜像内跑
if [ ! -f "dist/${MODEL_NAME}-${QUANT}-MLC/mlc-chat-config.json" ]; then
    echo "== 转换权重 (${QUANT})，容器内执行 =="
    docker run --rm \
      --gpus all --runtime=nvidia \
      -e LOCAL_UID="$(id -u)" -e LOCAL_GID="$(id -g)" \
      --mount type=bind,src="$PWD",dst=/workspace \
      -w /workspace \
      "${IMAGE}" bash -c "
        mlc_llm convert_weight ./dist/models/${MODEL_NAME}/ \
          --quantization ${QUANT} \
          -o dist/${MODEL_NAME}-${QUANT}-MLC
        mlc_llm gen_config ./dist/models/${MODEL_NAME}/ \
          --quantization ${QUANT} \
          --conv-template ${CONV_TEMPLATE} \
          --context-window-size 4096 \
          --prefill-chunk-size 2048 \
          --max-batch-size 1 \
          -o dist/${MODEL_NAME}-${QUANT}-MLC/
      "
fi

# ---- ③ 容器内交叉编译（核心步骤）----
echo "== 交叉编译 sm_87 引擎库 =="
docker run --rm \
  --gpus all --runtime=nvidia \
  -e LOCAL_UID="$(id -u)" -e LOCAL_GID="$(id -g)" \
  --mount type=bind,src="$PWD",dst=/workspace \
  -w /workspace \
  "${IMAGE}" bash -c "
    source /usr/local/bin/jetson-aarch64-env.sh
    mlc_llm compile ./dist/${MODEL_NAME}-${QUANT}-MLC/mlc-chat-config.json \
      --device '{\"kind\":\"cuda\",\"tag\":\"\",\"keys\":[\"cuda\",\"gpu\"],\"max_num_threads\":1024,\"thread_warp_size\":32,\"arch\":\"sm_87\",\"max_threads_per_block\":1024,\"max_shared_memory_per_block\":49152}' \
      --host aarch64-unknown-linux-gnu \
      --opt 'flashinfer=0;cublas_gemm=0;faster_transformer=0;cudagraph=1;cutlass=1;ipc_allreduce_strategy=NONE' \
      -o ./dist/libs/${MODEL_NAME}-${QUANT}-cuda-jetson_orin_sm87.so
  "

# ---- ④ 产物符号版本检查（Jetson 宿主机/容器为 glibc 2.35 / GLIBCXX 3.4.30）----
LIB_OUT="./dist/libs/${MODEL_NAME}-${QUANT}-cuda-jetson_orin_sm87.so"
SYMS="$(objdump -T "${LIB_OUT}" | grep -oE 'GLIBCXX_[0-9.]+|GLIBC_[0-9.]+' | sort -uV)"
echo "== 产物 ELF 架构与符号版本检查 =="
file -b "${LIB_OUT}" | grep -q aarch64 || { echo "❌ 不是 aarch64 产物"; exit 1; }
BAD="$(echo "${SYMS}" | awk -F. '/GLIBC_/  && $NF>35 {print} /GLIBCXX_/ {split($0,a,"."); if (a[3]>30 || (a[3]==30 && a[4]>0)) print}')"
echo "${SYMS}"
if [ -n "${BAD}" ]; then
    echo "❌ 超出 Jetson glibc 2.35 / GLIBCXX 3.4.30 上限: ${BAD}"
    exit 1
fi
echo "✅ 符号版本均 ≤ Jetson 上限（glibc 2.35 / GLIBCXX 3.4.30）"

echo ""
echo "== 完成。拷贝到 Jetson（两份都要）: =="
echo "scp -r ./dist/${MODEL_NAME}-${QUANT}-MLC jetson:~/work/mlc/dist/"
echo "scp ./dist/libs/${MODEL_NAME}-${QUANT}-cuda-jetson_orin_sm87.so jetson:~/work/mlc/dist/libs/"
