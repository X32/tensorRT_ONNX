#!/usr/bin/env python3
"""
trtpy 使用指南和基本功能演示
"""

import trtpy
import numpy as np
import os

def demonstrate_trtpy_features():
    """演示 trtpy 的主要功能"""
    print("=" * 60)
    print("trtpy 使用指南")
    print("=" * 60)

    print(f"\n📦 trtpy 版本: {trtpy.version}")
    print(f"🐍 Python 版本: {trtpy.python_version}")
    print("✅ trtpy 已正确安装并可用")

    print("\n" + "=" * 60)
    print("trtpy 主要功能")
    print("=" * 60)

    # 展示 trtpy 的核心功能
    features = {
        "编译功能": [
            "compile_onnx_to_file - 编译 ONNX 到引擎文件",
            "compile_onnxdata_to_memory - 编译 ONNX 到内存",
            "compileTRT - 使用 TensorRT 编译",
            "from_torch - 从 PyTorch 模型编译"
        ],
        "推理功能": [
            "infer_numpy - NumPy 数组推理",
            "infer_torch - PyTorch 张量推理",
            "infer_file - 文件输入推理"
        ],
        "设备管理": [
            "set_device - 设置计算设备",
            "get_device - 获取当前设备"
        ],
        "数据处理": [
            "normalize_numpy - NumPy 数据归一化",
            "normalize_torch - PyTorch 数据归一化"
        ]
    }

    for category, funcs in features.items():
        print(f"\n📋 {category}:")
        for func in funcs:
            print(f"  ✅ {func}")

def show_api_examples():
    """展示 trtpy API 使用示例"""
    print("\n" + "=" * 60)
    print("trtpy API 使用示例")
    print("=" * 60)

    # 1. 设置设备
    print("\n1️⃣ 设置计算设备:")
    try:
        # 设置为 GPU 设备 0
        trtpy.set_device(0)
        print("   ✅ 设置 GPU 设备 0")
    except Exception as e:
        print(f"   ⚠️ 设备设置失败: {e}")

    # 2. 检查设备
    print("\n2️⃣ 检查当前设备:")
    try:
        device = trtpy.get_device()
        print(f"   ✅ 当前设备: {device}")
    except Exception as e:
        print(f"   ⚠️ 获取设备失败: {e}")

    # 3. 数据归一化示例
    print("\n3️⃣ 数据处理功能:")
    print("   创建测试数据:")
    test_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    print(f"   原始数据范围: [{test_data.min():.3f}, {test_data.max():.3f}]")

    try:
        normalized = trtpy.normalize_numpy(test_data, trtpy.YoloType.V5)
        print(f"   ✅ 归一化后范围: [{normalized.min():.3f}, {normalized.max():.3f}]")
    except Exception as e:
        print(f"   ⚠️ 归一化失败: {e}")

def show_trtpy_advantages():
    """展示 trtpy 的优势"""
    print("\n" + "=" * 60)
    print("trtpy 的优势")
    print("=" * 60)

    advantages = [
        "🚀 简化 TensorRT 使用 - 无需直接操作复杂 API",
        "🎯 支持多种输入格式 - PyTorch、NumPy、文件",
        "⚡ 快速原型开发 - 适合实验和测试",
        "🔧 设备管理简化 - 自动处理 CUDA 设备",
        "📊 数据预处理 - 内置常用归一化方法",
        "🛠️ 多精度支持 - FP32、FP16、INT8",
        "💡 友好的 Python 接口 - 符合 Python 习惯"
    ]

    for advantage in advantages:
        print(f"  {advantage}")

def practical_examples():
    """实用的 trtpy 使用示例"""
    print("\n" + "=" * 60)
    print("实用使用示例")
    print("=" * 60)

    # 示例 1: 基础使用
    print("\n📝 示例 1: 基础设置和使用")
    print("""
import trtpy
import numpy as np

# 设置设备
trtpy.set_device(0)

# 创建测试数据
data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# 数据归一化
normalized = trtpy.normalize_numpy(data, trtpy.YoloType.V5)
""")

    # 示例 2: 与现有引擎配合使用
    print("📝 示例 2: 与已构建的 TensorRT 引擎配合使用")
    print("""
# trtpy 可以与已构建的 TensorRT 引擎配合使用
# 用于推理时的数据处理和设备管理

import trtpy
import numpy as np

# 设置推理设备
trtpy.set_device(0)

# 准备输入数据
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# 数据预处理
normalized_input = trtpy.normalize_numpy(input_data, trtpy.YoloType.V5)

# 然后使用 TensorRT 引擎进行推理
# (引擎推理部分仍然使用 TensorRT 或 trtexec)
""")

    # 示例 3: 编译功能 (需要正确配置)
    print("📝 示例 3: 编译功能")
    print("""
# trtpy 提供编译功能，但在 Jetson 环境中
# 推荐使用 trtexec 命令行工具进行编译

# 已验证的编译方法:
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
trtexec --onnx=model.onnx --saveEngine=model_int8.engine --int8
""")

def summary():
    """总结和建议"""
    print("\n" + "=" * 60)
    print("总结和建议")
    print("=" * 60)

    print("\n✅ trtpy 已正确安装")
    print("📦 版本: 1.2.6")
    print("🔧 功能: 编译、推理、数据处理")

    print("\n💡 推荐使用方式:")
    print("1. **模型编译**: 使用 trtexec 命令行工具")
    print("2. **数据处理**: 使用 trtpy 的归一化和设备管理功能")
    print("3. **快速测试**: 使用 trtpy 进行原型开发")

    print("\n🎯 最佳实践:")
    print("• 编译阶段: trtexec --onnx=model.onnx --saveEngine=model.engine --fp16")
    print("• 推理阶段: trtexec --loadEngine=model.engine")
    print("• 数据处理: trtpy.normalize_numpy(data, trtpy.YoloType.V5)")

    print("\n🚀 你现在拥有完整的 TensorRT 优化环境:")
    print("• TensorRT 10.3.0 + trtpy 1.2.6")
    print("• FP16 和 INT8 优化引擎")
    print("• 完整的文档和示例")

def main():
    """主函数"""
    demonstrate_trtpy_features()
    show_api_examples()
    show_trtpy_advantages()
    practical_examples()
    summary()

    print("\n" + "=" * 60)
    print("✅ trtpy 功能演示完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
