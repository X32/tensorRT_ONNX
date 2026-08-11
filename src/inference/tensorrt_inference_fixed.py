#!/usr/bin/env python3
"""
TensorRT 推理类 - TensorRT 10.3.0 兼容版本
"""

import tensorrt as trt
import pycuda.driver
import pycuda.autoinit
import numpy as np

class TensorRTInference:
    def __init__(self, engine_path):
        """
        初始化 TensorRT 推理

        Args:
            engine_path: TensorRT Engine 文件路径
        """
        self.logger = trt.Logger(trt.Logger.INFO)

        # 加载引擎
        with open(engine_path, 'rb') as f:
            engine_data = f.read()

        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        # 获取输入输出信息
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None

        self._setup_io()

    def _setup_io(self):
        """设置输入输出信息"""
        num_io_tensors = self.engine.num_io_tensors

        print(f"📊 Engine 包含 {num_io_tensors} 个 IO tensors:")

        for i in range(num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)

            print(f"  Tensor {i}: {name} | 形状: {shape} | 类型: {dtype} | 模式: {mode}")

            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.input_shape = shape
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_name = name
                self.output_shape = shape

        if self.input_name is None or self.output_name is None:
            raise ValueError("无法确定输入输出 tensor 名称")

        print(f"✅ 输入: {self.input_name} {self.input_shape}")
        print(f"✅ 输出: {self.output_name} {self.output_shape}")

    def infer(self, input_data):
        """
        执行推理

        Args:
            input_data: 输入数据 (numpy array, shape: [1, 3, 224, 224])

        Returns:
            output: 输出数据 (numpy array)
        """
        # 验证输入形状
        if input_data.shape != tuple(self.input_shape):
            raise ValueError(f"输入形状错误! 期望: {self.input_shape}, 实际: {input_data.shape}")

        # 分配 GPU 内存
        input_size = int(input_data.nbytes)
        output_size = int(np.prod(self.output_shape) * 4)  # float32 = 4 bytes

        d_input = pycuda.driver.mem_alloc(input_size)
        d_output = pycuda.driver.mem_alloc(output_size)

        # 拷贝输入数据到 GPU
        pycuda.driver.memcpy_htod(d_input, input_data.astype(np.float32))

        # 创建绑定列表
        bindings = [int(d_input), int(d_output)]

        # 执行推理
        self.context.execute_v2(bindings)

        # 拷贝输出数据从 GPU
        output_data = np.empty(self.output_shape, dtype=np.float32)
        pycuda.driver.memcpy_dtoh(output_data, d_output)

        return output_data

def test_inference():
    """测试推理功能"""
    print("🚀 测试 TensorRT 推理功能")
    print("=" * 60)

    # 初始化推理引擎
    inferencer = TensorRTInference("resnet18.engine")

    # 创建测试输入
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    print(f"\n📝 创建测试输入: {input_data.shape}")

    # 执行推理
    print("⏳ 执行推理...")
    output = inferencer.infer(input_data)

    print(f"✅ 推理成功! 输出形状: {output.shape}")
    print(f"📊 输出示例 (前5个值): {output[0][:5]}")

    # 获取预测结果
    predicted_class = np.argmax(output[0])
    confidence = output[0][predicted_class]

    print(f"🎯 预测类别: {predicted_class}, 置信度: {confidence:.4f}")

    return output

if __name__ == "__main__":
    test_inference()