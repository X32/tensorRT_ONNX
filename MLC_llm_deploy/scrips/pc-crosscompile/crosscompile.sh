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

MODEL_NAME="${1:-Qwen2.5-1.5B-Instruct}"
QUANT="q4f16_1"
CONV_TEMPLATE="qwen2"
IMAGE="mlc-llm-jetson:cu126"

mkdir -p dist/models dist/libs

# ---- ① 下载原始模型（宿主机，目标无关）----
if [ ! -f "dist/models/${MODEL_NAME}/config.json" ]; then
    echo "== 下载 ${MODEL_NAME} =="
    uvx hf download "Qwen/${MODEL_NAME}" --local-dir "dist/models/${MODEL_NAME}"
fi

# ---- ② 转换权重 + 生成配置（目标无关，宿主机 uvx 环境即可）----
if [ ! -f "dist/${MODEL_NAME}-${QUANT}-MLC/mlc-chat-config.json" ]; then
    echo "== 转换权重 (${QUANT}) =="
    mlc_llm convert_weight "./dist/models/${MODEL_NAME}/" \
      --quantization "${QUANT}" \
      -o "dist/${MODEL_NAME}-${QUANT}-MLC"
    mlc_llm gen_config "./dist/models/${MODEL_NAME}/" \
      --quantization "${QUANT}" \
      --conv-template "${CONV_TEMPLATE}" \
      -o "dist/${MODEL_NAME}-${QUANT}-MLC/"
fi

# ---- ③ 容器内交叉编译（核心步骤）----
echo "== 交叉编译 sm_87 引擎库 =="
docker run --rm -it \
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

echo ""
echo "== 完成。拷贝到 Jetson（两份都要）: =="
echo "scp -r ./dist/${MODEL_NAME}-${QUANT}-MLC jetson:~/work/mlc/dist/"
echo "scp ./dist/libs/${MODEL_NAME}-${QUANT}-cuda-jetson_orin_sm87.so jetson:~/work/mlc/dist/libs/"
