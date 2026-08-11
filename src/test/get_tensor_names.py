#!/usr/bin/env python3
"""
获取 TensorRT Engine 的输入输出 tensor 名称
"""

import tensorrt as trt
import numpy as np

def get_tensor_info():
    """获取引擎的 tensor 信息"""
    logger = trt.Logger(trt.Logger.INFO)

    # 加载引擎
    with open("resnet18.engine", 'rb') as f:
        engine_data = f.read()

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)

    print("📊 TensorRT Engine Tensor 信息:")
    print("=" * 60)

    # 获取 IO tensors 数量
    num_io_tensors = engine.num_io_tensors
    print(f"总 IO tensors 数量: {num_io_tensors}")

    # 遍历所有 tensors
    for i in range(num_io_tensors):
        name = engine.get_tensor_name(i)
        dtype = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name)
        mode = engine.get_tensor_mode(name)
        location = engine.get_tensor_location(name)

        print(f"\nTensor {i}:")
        print(f"  名称: {name}")
        print(f"  数据类型: {dtype}")
        print(f"  形状: {shape}")
        print(f"  模式: {mode}")
        print(f"  位置: {location}")

        # 判断是输入还是输出
        if mode == trt.TensorIOMode.INPUT:
            print(f"  类型: ✅ 输入")
        elif mode == trt.TensorIOMode.OUTPUT:
            print(f"  类型: ✅ 输出")
        else:
            print(f"  类型: ❓ 未知")

if __name__ == "__main__":
    get_tensor_info()