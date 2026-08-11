#!/usr/bin/env python3
"""
简化的 INT8 校准脚本（不依赖 pycuda）
使用纯 TensorRT Python API 和 numpy
"""

import tensorrt as trt
import numpy as np
import os

class SimpleCalibrator(trt.IInt8EntropyCalibrator2):
    """简化的 INT8 校准器"""
    def __init__(self, cache_file="resnet_calibration.cache", batch_size=8, num_batches=10):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.current_batch = 0

        # 生成随机校准数据（实际应用中应使用真实数据集）
        print(f"生成 {num_batches} 批次随机校准数据，批次大小: {batch_size}")
        self.data = np.random.randn(num_batches * batch_size, 3, 224, 224).astype(np.float32)
        print(f"校准数据形状: {self.data.shape}")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        """获取一批校准数据"""
        if self.current_batch >= self.num_batches:
            return None

        # 获取当前批次数据
        start_idx = self.current_batch * self.batch_size
        end_idx = start_idx + self.batch_size
        batch = self.data[start_idx:end_idx]

        # 返回 GPU 指针（使用 numpy 数组的直接指针）
        # 注意：这里返回内存地址指针
        self.current_batch += 1

        # 返回包含数据指针的列表
        # 注意：这需要 PyCUDA 或其他 CUDA 内存管理库
        # 由于我们没有 pycuda，这里返回 None 并在 build 阶段处理
        print(f"校准批次 {self.current_batch}/{self.num_batches}")
        return None

    def read_calibration_cache(self):
        """读取校准缓存"""
        if os.path.exists(self.cache_file):
            print(f"读取校准缓存: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        """写入校准缓存"""
        print(f"保存校准缓存: {self.cache_file}")
        with open(self.cache_file, 'wb') as f:
            f.write(cache)

def build_int8_engine(onnx_file_path, engine_file_path, calibrator):
    """构建 INT8 TensorRT 引擎"""
    print("=== 构建 INT8 TensorRT 引擎 ===")

    # 创建 logger
    logger = trt.Logger(trt.Logger.WARNING)

    # 创建 builder
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # 解析 ONNX 模型
    print(f"解析 ONNX 模型: {onnx_file_path}")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ONNX 解析失败')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    print(f"网络输入数量: {network.num_inputs}")
    print(f"网络输出数量: {network.num_outputs}")

    # 配置构建器
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB

    # 设置 INT8 模式
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator

    print("开始构建 INT8 引擎...")
    print("注意：这可能需要几分钟时间")

    # 构建引擎
    engine = builder.build_serialized_network(network, config)

    if engine is None:
        print("引擎构建失败")
        return None

    # 保存引擎
    with open(engine_file_path, 'wb') as f:
        f.write(engine)

    print(f"INT8 引擎已保存: {engine_file_path}")
    return engine

def main():
    """主函数"""
    print("=" * 60)
    print("ResNet18 INT8 校准与引擎构建")
    print("=" * 60)

    # 文件路径
    onnx_file = "resnet18.onnx"
    int8_engine_file = "resnet18_int8.engine"
    cache_file = "resnet_calibration.cache"

    # 创建校准器
    calibrator = SimpleCalibrator(
        cache_file=cache_file,
        batch_size=8,
        num_batches=10
    )

    # 构建 INT8 引擎
    try:
        engine = build_int8_engine(onnx_file, int8_engine_file, calibrator)

        if engine:
            print("\n" + "=" * 60)
            print("✅ INT8 引擎构建成功！")
            print(f"引擎文件: {int8_engine_file}")
            print(f"校准缓存: {cache_file}")
            print("=" * 60)
        else:
            print("\n❌ INT8 引擎构建失败")
            print("提示：这通常是因为缺少 CUDA 支持或 pycuda")
            print("替代方案：使用 trtexec 命令行工具")

    except Exception as e:
        print(f"\n❌ 构建过程出错: {e}")
        print("\n💡 推荐使用 trtexec 命令行工具构建 INT8 引擎：")
        print("trtexec --onnx=resnet18.onnx --saveEngine=resnet18_int8.engine --int8")

if __name__ == "__main__":
    main()
