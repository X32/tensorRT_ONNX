#!/usr/bin/env bash
# ============================================================================
# jetson-aarch64-env.sh — 交叉编译容器内环境（进入容器后 source 此脚本）
# 作用: 让 nvcc + 交叉 GCC 产出 aarch64 / sm_87 的 Jetson 引擎库
# ============================================================================
export PATH="/opt/toolchains/aarch64--glibc--stable-2022.08-1/bin:$PATH"
export CC=aarch64-linux-gcc
export CXX=aarch64-linux-g++
export HOST_CUDA_ARCH=61          # PC 显卡架构（GTX 1080 Ti = Pascal sm_61）
export TVM_CUDA_ARCHS="61;87"     # 编译 TVM 时同时启用两档
echo "[env] aarch64 交叉编译环境已加载 (target: Jetson Orin sm_87)"
