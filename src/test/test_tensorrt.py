#!/usr/bin/env python3
"""
TensorRT 环境测试脚本
用于验证 TensorRT 安装和基本功能
"""

import tensorrt as trt
import sys

def test_tensorrt_basic():
    """测试TensorRT基本功能"""
    print("=" * 60)
    print("TensorRT 环境测试")
    print("=" * 60)

    # 检查版本
    print(f"\n✓ TensorRT 版本: {trt.__version__}")
    print(f"✓ Python 版本: {sys.version}")

    # 检查可用的构建器
    print(f"\n✓ 可用的构建器:")
    print(f"  - CUDA: {hasattr(trt, 'Builder') and trt.Builder is not None}")
    print(f"  - ONNX Parser: {hasattr(trt, 'OnnxParser') and trt.OnnxParser is not None}")

    # 测试基本功能
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        print(f"✓ Logger 创建成功")

        builder = trt.Builder(logger)
        print(f"✓ Builder 创建成功")

        network = builder.create_network()
        print(f"✓ Network 创建成功")

        print("\n" + "=" * 60)
        print("🎉 所有基本测试通过！TensorRT 环境正常工作")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_tensorrt_basic()
    sys.exit(0 if success else 1)
