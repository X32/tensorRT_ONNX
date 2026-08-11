# TensorRT Engine 文件生成方法详解

**文档创建时间**: 2026-08-11
**适用版本**: TensorRT 10.3.0
**设备平台**: Jetson Orin + JetPack 6.x

---

## 📋 生成方法概览

TensorRT Engine 文件主要有 **3种生成方法**：

```bash
方法1: trtexec 命令行工具 ⭐ 最常用
方法2: Python API 编程生成  
方法3: C++ API 编程生成
```

---

## 🔧 方法1: trtexec 命令行工具 (推荐)

### 基础用法

```bash
# 从 ONNX 生成 FP32 Engine
/usr/src/tensorrt/bin/trtexec \
  --onnx=resnet18.onnx \              # 输入 ONNX 模型
  --saveEngine=resnet18.engine \     # 输出 Engine 文件
  --fp16                               # FP16 精度
```

### 项目中的实际使用方法

#### 1. FP32 Engine (基准精度)

```bash
trtexec \
  --onnx=resnet18.onnx \
  --saveEngine=resnet18.engine \
  
```

**特点**:

- 最高精度，与原始模型一致
- 推理速度较慢
- 显存占用较高
- 适合精度要求极高的场景

#### 2. FP16 Engine (半精度 - 推荐⭐)

```bash
trtexec \
  --onnx=resnet18.onnx \
  --saveEngine=resnet18_fp16.engine \
  --fp16 \
  --memPoolSize=workspace:2048MiB \
  --useSpinWait
```

**特点**:

- 精度损失极小（<1%）
- 速度提升2倍
- 显存占用不变
- **生产环境首选**

#### 3. INT8 Engine (8位量化 - 极限性能)

```bash
trtexec \
  --onnx=resnet18.onnx \
  --saveEngine=resnet18_int8.engine \
  --int8 \
  --memPoolSize=workspace:1024MiB
```

**特点**:

- 精度略有损失（1-2%）
- 速度提升4倍
- 显存占用减少50%
- 适合实时性要求极高的场景

### trtexec 常用参数

#### 基础参数


| 参数                  | 说明                 | 示例                        |
| --------------------- | -------------------- | --------------------------- |
| `--onnx=<file>`       | 输入 ONNX 模型文件   | `--onnx=resnet18.onnx`      |
| `--saveEngine=<file>` | 保存 Engine 文件     | `--saveEngine=model.engine` |
| `--loadEngine=<file>` | 加载已有 Engine 文件 | `--loadEngine=model.engine` |
| `--fp32`              | 使用 FP32 精度       | `--fp32`                    |
| `--fp16`              | 使用 FP16 半精度     | `--fp16`                    |
| `--int8`              | 使用 INT8 量化       | `--int8`                    |

#### 性能测试参数


| 参数                   | 说明                      | 示例                      |
| ---------------------- | ------------------------- | ------------------------- |
| `--duration=<seconds>` | 运行时间（秒）            | `--duration=60`           |
| `--iterations=<num>`   | 迭代次数                  | `--iterations=100`        |
| `--batch=<size>`       | 批处理大小                | `--batch=4`               |
| `--useSpinWait`        | 使用 Spin Wait 提高稳定性 | `--useSpinWait`           |
| `--verbose`            | 详细输出                  | `--verbose`               |
| `--exportTimes=<file>` | 导出性能数据              | `--exportTimes=perf.json` |

#### 内存参数


| 参数                          | 说明           | 示例                              |
| ----------------------------- | -------------- | --------------------------------- |
| `--memPoolSize=<pool>:<size>` | 设置内存池大小 | `--memPoolSize=workspace:2048MiB` |
| `--workspace=<size>`          | 工作空间大小   | `--workspace=1024MiB`             |

### 性能测试命令

#### 基准性能测试

```bash
# 测试 FP32 引擎性能
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=resnet18.engine \
  --duration=60

# 测试 FP16 引擎性能  
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=resnet18_fp16.engine \
  --duration=60

# 测试 INT8 引擎性能
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=resnet18_int8.engine \
  --duration=60
```

#### 批处理性能测试

```bash
# 测试批处理性能
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=resnet18_fp16.engine \
  --batch=4 \
  --duration=60
```

#### 详细性能分析

```bash
# 详细输出 + 导出性能数据
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=resnet18_fp16.engine \
  --verbose \
  --duration=30 \
  --exportTimes=performance.json
```

---

## 🔧 方法2: Python API 生成

### 原生 TensorRT Python API

```python
import tensorrt as trt
import os

# 创建 TensorRT 日志
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def create_fp16_engine(onnx_path, engine_path):
    """使用 Python API 创建 FP16 Engine"""
  
    # 1. 创建 builder 和 network
    builder = trt.Builder(TRT_LOGGER)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
  
    # 2. 解析 ONNX 模型
    parser = trt.ONNXParser(network, TRT_LOGGER)
  
    if not os.path.exists(onnx_path):
        print(f"错误: ONNX 文件不存在: {onnx_path}")
        return False
      
    if not parser.parse_from_file(onnx_path):
        print("错误: ONNX 解析失败")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        return False
  
    print(f"✅ 成功解析 ONNX 模型: {onnx_path}")
  
    # 3. 构建配置 - FP16
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
  
    # 设置工作空间大小
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB
  
    # 4. 构建序列化引擎
    print("🔧 正在构建 TensorRT Engine...")
    engine = builder.build_serialized_network(network, config)
  
    if engine is None:
        print("❌ Engine 构建失败")
        return False
  
    # 5. 保存到文件
    with open(engine_path, "wb") as f:
        f.write(engine)
  
    print(f"✅ 成功保存 Engine 文件: {engine_path}")
    return True

# 使用示例
if __name__ == "__main__":
    onnx_file = "resnet18.onnx"
    engine_file = "resnet18_fp16_py.engine"
  
    create_fp16_engine(onnx_file, engine_file)
```

### INT8 量化示例

```python
def create_int8_engine(onnx_path, engine_path, calibrator=None):
    """创建 INT8 量化 Engine"""
  
    builder = trt.Builder(TRT_LOGGER)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
  
    parser = trt.ONNXParser(network, TRT_LOGGER)
    parser.parse_from_file(onnx_path)
  
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
  
    # 设置 INT8 校准器
    if calibrator:
        config.int8_calibrator = calibrator
  
    # 构建引擎
    engine = builder.build_serialized_network(network, config)
  
    with open(engine_path, "wb") as f:
        f.write(engine)
  
    print(f"✅ 成功创建 INT8 Engine: {engine_path}")
```

### 使用 trtpy (简化版 Python API)

```python
import trtpy

# 从 ONNX 编译到 Engine 文件
trtpy.compile_onnx_to_file(
    "resnet18.onnx",
    "resnet18_fp16_trtpy.engine",
    precision="fp16"  # 可选: "fp32", "fp16", "int8"
)

# 批量编译多个模型
models = [
    ("resnet18.onnx", "resnet18_fp16.engine", "fp16"),
    ("resnet18.onnx", "resnet18_int8.engine", "int8"),
    ("mobilenet.onnx", "mobilenet_fp16.engine", "fp16"),
]

for onnx, engine, precision in models:
    print(f"编译: {onnx} -> {engine} ({precision})")
    trtpy.compile_onnx_to_file(onnx, engine, precision=precision)
    print(f"✅ 完成: {engine}")
```

### Python API 推理示例

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
      
        # 加载引擎
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
      
        self.context = self.engine.create_execution_context()
      
        # 分配内存
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
  
    def allocate_buffers(self):
        """分配 GPU 内存"""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
      
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
          
            # 分配 CPU 和 GPU 内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
          
            bindings.append(int(device_mem))
          
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
      
        return inputs, outputs, bindings, stream
  
    def infer(self, input_data):
        """执行推理"""
        # 复制输入数据到 GPU
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
      
        # 执行推理
        self.context.execute_v2(bindings=self.bindings)
      
        # 复制输出数据到 CPU
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
      
        return self.outputs[0]['host']

# 使用示例
inferencer = TensorRTInference("resnet18_fp16.engine")
result = inferencer.infer(np.random.randn(1, 3, 224, 224).astype(np.float32))
print(f"推理结果形状: {result.shape}")
```

---

## 🔧 方法3: C++ API 生成

### 基础 C++ 示例

```cpp
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <iostream>
#include <fstream>
#include <vector>

using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};

class TensorRTEngineBuilder {
public:
    bool buildFromONNX(const std::string& onnxPath, const std::string& enginePath) {
        Logger logger;
      
        // 1. 创建 builder
        IBuilder* builder = createInferBuilder(logger);
        if (!builder) {
            std::cerr << "Failed to create builder" << std::endl;
            return false;
        }
      
        // 2. 创建 network
        INetworkDefinition* network = builder->createNetworkV2(0U);
        if (!network) {
            std::cerr << "Failed to create network" << std::endl;
            return false;
        }
      
        // 3. 创建 ONNX parser
        IParser* parser = createONNXParser(*network, logger);
        if (!parser) {
            std::cerr << "Failed to create parser" << std::endl;
            return false;
        }
      
        // 4. 解析 ONNX 模型
        if (!parser->parseFromFile(onnxPath.c_str())) {
            std::cerr << "Failed to parse ONNX file" << std::endl;
            for (int i = 0; i < parser->getNbErrors(); ++i) {
                std::cerr << "Error " << i << ": " << parser->getError(i)->desc() << std::endl;
            }
            return false;
        }
      
        std::cout << "Successfully parsed ONNX model: " << onnxPath << std::endl;
      
        // 5. 创建 builder config
        IBuilderConfig* config = builder->createBuilderConfig();
      
        // 设置 FP16 精度
        config->setFlag(BuilderFlag::kFP16);
      
        // 设置工作空间大小 (1GB)
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 30);
      
        // 6. 构建序列化引擎
        IHostMemory* engine = builder->buildSerializedNetwork(*network, *config);
        if (!engine) {
            std::cerr << "Failed to build engine" << std::endl;
            return false;
        }
      
        // 7. 保存引擎到文件
        std::ofstream file(enginePath, std::ios::binary);
        file.write(reinterpret_cast<const char*>(engine->data()), engine->size());
      
        std::cout << "Successfully saved engine: " << enginePath << std::endl;
      
        // 清理资源
        delete parser;
        delete network;
        delete config;
        delete builder;
      
        return true;
    }
};

int main() {
    TensorRTEngineBuilder builder;
  
    std::string onnxPath = "resnet18.onnx";
    std::string enginePath = "resnet18_fp16_cpp.engine";
  
    if (builder.buildFromONNX(onnxPath, enginePath)) {
        std::cout << "Engine built successfully!" << std::endl;
        return 0;
    } else {
        std::cerr << "Failed to build engine" << std::endl;
        return 1;
    }
}
```

### C++ 推理示例

```cpp
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>

class TensorRTInferencer {
private:
    ICudaEngine* engine_;
    IExecutionContext* context_;
    void* buffers_[2];
    cudaStream_t stream_;
  
public:
    TensorRTInferencer(const std::string& enginePath) {
        // 加载引擎
        std::ifstream file(enginePath, std::ios::binary);
        std::vector<char> engineData((std::istreambuf_iterator<char>(file)), 
                                     std::istreambuf_iterator<char>());
      
        // 运行时和引擎
        IRuntime* runtime = createInferRuntime(gLogger);
        engine_ = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
        context_ = engine_->createExecutionContext();
      
        // 分配 GPU 内存
        cudaMalloc(&buffers_[0], 3 * 224 * 224 * sizeof(float));  // 输入
        cudaMalloc(&buffers_[1], 1000 * sizeof(float));             // 输出
      
        cudaStreamCreate(&stream_);
    }
  
    ~TensorRTInferencer() {
        cudaFree(buffers_[0]);
        cudaFree(buffers_[1]);
        cudaStreamDestroy(stream_);
    }
  
    std::vector<float> infer(const std::vector<float>& input) {
        // 复制输入到 GPU
        cudaMemcpyAsync(buffers_[0], input.data(), input.size() * sizeof(float),
                       cudaMemcpyHostToDevice, stream_);
      
        // 执行推理
        context_->enqueueV2(buffers_, stream_, nullptr);
      
        // 获取输出
        std::vector<float> output(1000);
        cudaMemcpyAsync(output.data(), buffers_[1], output.size() * sizeof(float),
                       cudaMemcpyDeviceToHost, stream_);
      
        cudaStreamSynchronize(stream_);
      
        return output;
    }
};

int main() {
    TensorRTInferencer inferencer("resnet18_fp16_cpp.engine");
  
    // 测试推理
    std::vector<float> input(3 * 224 * 224, 1.0f);
    auto output = inferencer.infer(input);
  
    std::cout << "Inference completed! Output size: " << output.size() << std::endl;
  
    return 0;
}
```

---

## 📊 三种方法对比


| 方法           | 难度        | 灵活性 | 性能 | 开发效率 | 适用场景             |
| -------------- | ----------- | ------ | ---- | -------- | -------------------- |
| **trtexec**    | ⭐ 简单     | 中     | 最高 | 最高     | 日常使用、快速测试   |
| **Python API** | ⭐⭐ 中等   | 高     | 高   | 中       | 定制化需求、批量处理 |
| **C++ API**    | ⭐⭐⭐ 复杂 | 最高   | 最高 | 低       | 生产环境、极致性能   |

### 详细对比

#### trtexec 命令行工具

**✅ 优势**:

- 使用简单，一行命令搞定
- 官方优化，性能最佳
- 支持所有 TensorRT 特性
- 内置性能分析工具

**❌ 劣势**:

- 灵活性相对较低
- 不适合复杂的预处理逻辑

#### Python API

**✅ 优势**:

- 灵活性高，可定制化
- 与 PyTorch/TensorFlow 集成方便
- 适合原型开发和实验
- 易于调试和测试

**❌ 劣势**:

- 性能略低于 C++
- 需要处理 Python 环境依赖

#### C++ API

**✅ 优势**:

- 性能最优
- 完全控制内存和执行流程
- 适合生产环境部署
- 无 Python 解释器开销

**❌ 劣势**:

- 开发复杂度高
- 编译和调试困难
- 开发周期长

---

## 🎯 推荐工作流程

### 开发阶段

```bash
# 1. 使用 trtexec 快速生成和测试
trtexec --onnx=resnet18.onnx --saveEngine=test.engine --fp16
trtexec --loadEngine=test.engine --duration=30

# 2. 测试不同精度
trtexec --onnx=model.onnx --saveEngine=fp32.engine --fp32
trtexec --onnx=model.onnx --saveEngine=fp16.engine --fp16  
trtexec --onnx=model.onnx --saveEngine=int8.engine --int8
```

### 批量处理

```python
# 使用 Python API 批量生成
models = ["resnet18.onnx", "mobilenet.onnx", "efficientnet.onnx"]

for model in models:
    engine_name = model.replace(".onnx", "_fp16.engine")
    trtpy.compile_onnx_to_file(model, engine_name, precision="fp16")
    print(f"✅ 生成完成: {engine_name}")
```

### 生产部署

```cpp
// 使用 C++ API 进行生产部署
TensorRTEngineBuilder builder;
builder.buildFromONNX("model.onnx", "production.engine");
```

### 性能调优

```bash
# 对比不同精度的性能
trtexec --loadEngine=fp32.engine --duration=60 --exportTimes=fp32_perf.json
trtexec --loadEngine=fp16.engine --duration=60 --exportTimes=fp16_perf.json  
trtexec --loadEngine=int8.engine --duration=60 --exportTimes=int8_perf.json

# 测试批处理性能
trtexec --loadEngine=fp16.engine --batch=4 --duration=60
```

---

## 🎯 项目推荐方案

根据您的项目特点（Jetson Orin + TensorRT 10.3.0），推荐以下方案：

### 日常开发使用 trtexec

```bash
# 快速生成和测试
/usr/src/tensorrt/bin/trtexec \
  --onnx=resnet18.onnx \
  --saveEngine=resnet18_fp16.engine \
  --fp16 \
  --useSpinWait

# 性能测试
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=resnet18_fp16.engine \
  --duration=60 \
  --exportTimes=performance.json
```

### 特殊需求使用 Python API

```python
# 需要定制化时使用 Python API
import trtpy

trtpy.compile_onnx_to_file(
    "resnet18.onnx", 
    "resnet18_fp16_trtpy.engine",
    precision="fp16"
)
```

---

## 💡 总结

**您项目中最常用的就是 trtexec 方法**，它：

- ✅ **简单易用**: 一行命令搞定所有操作
- ✅ **性能最优**: 官方优化，效果最佳
- ✅ **功能完整**: 支持所有精度和特性
- ✅ **内置工具**: benchmark 和性能分析
- ✅ **文档完善**: 官方支持丰富

**建议**:

- **日常使用**: trtexec 命令行工具
- **批量处理**: Python API (trtpy)
- **生产部署**: C++ API

---

## 📚 参考资料

- **TensorRT 官方文档**: https://docs.nvidia.com/deeplearning/tensorrt/
- **trtexec 使用指南**: `/usr/src/tensorrt/bin/trtexec --help`
- **项目性能报告**: `PERFORMANCE_REPORT.md`
- **项目快速指南**: `QUICK_START_GUIDE.md`

---

**文档维护**: 请根据实际使用情况和 TensorRT 版本更新及时更新本文档。
**最后更新**: 2026-08-11
**适用版本**: TensorRT 10.3.0+
