import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os

class ResNetCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file, batch_size=8, num_batches=10):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.current_batch = 0

        # 分配 device 内存
        input_size = batch_size * 3 * 224 * 224 * 4  # float32
        self.device_input = cuda.mem_alloc(input_size)

        # 生成随机校准数据（实际项目用真实数据集）
        self.data = np.random.randn(num_batches * batch_size, 3, 224, 224).astype(np.float32)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_batch >= self.num_batches:
            return None
        batch = self.data[self.current_batch * self.batch_size:
                          (self.current_batch + 1) * self.batch_size]
        cuda.memcpy_htod(self.device_input, batch)
        self.current_batch += 1
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)

# 这段在 trtexec 命令行模式下无法直接用
# 需要用 Python API 编译（见 A2.4）
