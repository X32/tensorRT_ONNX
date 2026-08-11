#!/usr/bin/env python3
"""
TensorRT 简单网络示例
创建一个简单的卷积神经网络
"""

import tensorrt as trt
import numpy as np

def create_simple_network():
    """创建一个简单的卷积神经网络"""
    print("创建简单的 TensorRT 网络...")

    # 创建 logger
    logger = trt.Logger(trt.Logger.WARNING)

    # 创建 builder
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # 添加输入层
    input_tensor = network.add_input("input", trt.DataType.FLOAT, (1, 3, 224, 224))
    print(f"✓ 添加输入层: {input_tensor.shape}")

    # 添加卷积层
    convolution = network.add_convolution(
        input=input_tensor,
        num_output_maps=64,
        kernel_shape=(3, 3),
        kernel=np.zeros((64, 3, 3, 3), dtype=np.float32),
        bias=np.zeros(64, dtype=np.float32)
    )
    convolution.stride = (1, 1)
    convolution.padding = (1, 1)
    print(f"✓ 添加卷积层")

    # 添加激活层
    activation = network.add_activation(convolution.get_output(0), trt.ActivationType.RELU)
    print(f"✓ 添加 ReLU 激活层")

    # 添加池化层
    pooling = network.add_pooling(activation.get_output(0), trt.PoolingType.MAX, (2, 2))
    pooling.stride = (2, 2)
    print(f"✓ 添加池化层")

    # 标记输出
    pooling.get_output(0).name = "output"
    network.mark_output(pooling.get_output(0))
    print(f"✓ 标记输出层: {pooling.get_output(0).shape}")

    # 构建引擎
    print("\n构建引擎...")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    engine = builder.build_serialized_network(network, config)
    print(f"✓ 引擎构建成功！")

    return engine

if __name__ == "__main__":
    try:
        engine = create_simple_network()
        print("\n🎉 网络创建成功！")
        print(f"引擎大小: {len(engine)} 字节")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
