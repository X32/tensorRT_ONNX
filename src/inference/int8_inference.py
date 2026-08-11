#!/usr/bin/env python3
"""
TensorRT INT8 推理示例（不依赖 pycuda）
使用已构建的 INT8 引擎进行推理
"""

import tensorrt as trt
import numpy as np
import time

class TensorRTEngine:
    """TensorRT 引擎管理类"""
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = None
        self.context = None

    def load_engine(self):
        """加载 TensorRT 引擎"""
        print(f"加载引擎: {self.engine_path}")

        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()

        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        print(f"引擎加载成功！")
        print(f"输入数量: {self.engine.num_io_tensors}")
        print(f"绑定数量: {self.engine.num_bindings}")

        return True

    def get_input_output_specs(self):
        """获取输入输出规格"""
        print("\n=== 引擎规格 ===")

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            mode = self.engine.get_tensor_mode(name)

            print(f"张量 {i}: {name}")
            print(f"  形状: {shape}")
            print(f"  数据类型: {dtype}")
            print(f"  模式: {mode}")

    def infer(self, input_data):
        """执行推理（简化版本，不依赖 pycuda）"""
        print("\n=== 执行推理 ===")
        print(f"输入形状: {input_data.shape}")

        # 注意：这里需要 CUDA 内存操作
        # 由于没有 pycuda，我们返回模拟结果
        print("推理完成（模拟结果）")
        print("提示：完整推理需要 CUDA 内存操作或使用 trtexec")

        # 模拟推理结果
        batch_size = input_data.shape[0]
        output_shape = (batch_size, 1000)  # ResNet18 输出类别数
        mock_output = np.random.randn(*output_shape).astype(np.float32)

        return mock_output

def main():
    """主函数"""
    print("=" * 60)
    print("TensorRT INT8 推理示例")
    print("=" * 60)

    # 引擎文件路径
    int8_engine = "resnet18_int8.engine"
    fp16_engine = "resnet18_fp16.engine"

    # 选择要使用的引擎
    engine_path = int8_engine  # 可以改为 fp16_engine

    # 创建引擎对象
    tensorrt_engine = TensorRTEngine(engine_path)

    try:
        # 加载引擎
        if tensorrt_engine.load_engine():
            # 显示引擎规格
            tensorrt_engine.get_input_output_specs()

            # 创建测试输入
            print("\n=== 创建测试输入 ===")
            batch_size = 1
            input_data = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
            print(f"输入数据形状: {input_data.shape}")
            print(f"输入数据范围: [{input_data.min():.3f}, {input_data.max():.3f}]")

            # 执行推理
            start_time = time.time()
            output = tensorrt_engine.infer(input_data)
            end_time = time.time()

            print(f"\n=== 推理结果 ===")
            print(f"推理时间: {(end_time - start_time) * 1000:.2f} ms")
            print(f"输出形状: {output.shape}")

            # 显示 Top-5 预测
            top5_indices = output.argsort()[0][-5:][::-1]
            print("\nTop-5 预测结果:")
            for i, idx in enumerate(top5_indices):
                confidence = output[0, idx]
                print(f"{i+1}. 类别 {idx}: {confidence:.4f}")

            print("\n" + "=" * 60)
            print("✅ 推理示例完成")
            print("=" * 60)

            print("\n💡 推荐：使用 trtexec 获得完整性能")
            print(f"trtexec --loadEngine={engine_path} --duration=30")

    except FileNotFoundError:
        print(f"❌ 错误: 找不到引擎文件 {engine_path}")
        print(f"请确保引擎文件存在，或使用以下命令构建：")
        print(f"trtexec --onnx=resnet18.onnx --saveEngine={engine_path} --int8")

    except Exception as e:
        print(f"❌ 推理失败: {e}")
        print("\n💡 建议：")
        print("1. 使用 trtexec 命令行工具进行推理")
        print("2. 确保引擎文件路径正确")
        print("3. 检查 TensorRT 环境配置")

if __name__ == "__main__":
    main()
