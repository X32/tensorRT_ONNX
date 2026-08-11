#!/usr/bin/env python3
"""
TensorRT Engine 加载和推理测试程序
用于诊断 TensorRT API 兼容性问题
"""

import tensorrt as trt
import pycuda.driver
import pycuda.autoinit
import numpy as np
import sys

def check_engine_methods(engine):
    """检查引擎对象可用的所有方法"""
    print("\n🔍 引擎对象可用的方法:")
    methods = [m for m in dir(engine) if not m.startswith('_')]
    for method in methods:
        print(f"  - {method}")
    return methods

def test_tensorrt_api():
    """测试 TensorRT API 兼容性"""
    print("=" * 60)
    print("TensorRT Engine 加载和推理测试")
    print("=" * 60)

    # 检查 TensorRT 版本
    print(f"\n📌 TensorRT 版本: {trt.__version__}")

    # 创建 Logger
    logger = trt.Logger(trt.Logger.INFO)
    print("✅ Logger 创建成功")

    # 加载 Engine
    engine_path = "resnet18.engine"
    print(f"\n📂 加载引擎文件: {engine_path}")

    try:
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        print(f"✅ 引擎文件读取成功 ({len(engine_data) / (1024*1024):.1f} MB)")
    except Exception as e:
        print(f"❌ 读取引擎文件失败: {e}")
        return False

    # 反序列化引擎
    try:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_data)
        print("✅ 引擎反序列化成功")
    except Exception as e:
        print(f"❌ 引擎反序列化失败: {e}")
        return False

    # 检查引擎对象的方法
    methods = check_engine_methods(engine)

    # 检查各种 API 方法的可用性
    print("\n🔍 检查 API 方法可用性:")

    # 新 API (TensorRT 8.5+)
    has_new_api = all(hasattr(engine, method) for method in [
        'num_io_tensors', 'get_tensor_name', 'get_tensor_dtype'
    ])
    print(f"  新 API (num_io_tensors, get_tensor_name): {'✅ 可用' if has_new_api else '❌ 不可用'}")

    # 旧 API (TensorRT 8.x)
    has_old_api = all(hasattr(engine, method) for method in [
        'num_bindings', 'get_binding_name', 'get_binding_dtype'
    ])
    print(f"  旧 API (num_bindings, get_binding_name): {'✅ 可用' if has_old_api else '❌ 不可用'}")

    # 其他可能的 API
    print(f"  has 'name': {hasattr(engine, 'name')}")
    print(f"  has 'max_batch_size': {hasattr(engine, 'max_batch_size')}")

    # 尝试获取输入输出信息
    print("\n📊 尝试获取引擎信息:")

    try:
        # 尝试不同的方法来获取 IO 数量
        if hasattr(engine, 'num_io_tensors'):
            num_io = engine.num_io_tensors
            print(f"  使用 num_io_tensors: {num_io} 个 IO")
        elif hasattr(engine, 'num_bindings'):
            num_io = engine.num_bindings
            print(f"  使用 num_bindings: {num_io} 个 bindings")
        elif hasattr(engine, 'get_num_bindings'):
            num_io = engine.get_num_bindings()
            print(f"  使用 get_num_bindings(): {num_io} 个 bindings")
        else:
            print("  ❌ 无法获取 IO 数量")
            # 尝试直接遍历
            print("  尝试直接遍历引擎对象...")
            for i in range(10):  # 尝试最多 10 个
                try:
                    if hasattr(engine, 'get_binding_name'):
                        name = engine.get_binding_name(i)
                        print(f"    Binding {i}: {name}")
                    elif hasattr(engine, 'get_io_tensor_name'):
                        name = engine.get_io_tensor_name(i)
                        print(f"    IO Tensor {i}: {name}")
                    else:
                        break
                except:
                    break
    except Exception as e:
        print(f"  ❌ 获取引擎信息失败: {e}")

    # 创建执行上下文
    try:
        context = engine.create_execution_context()
        print("✅ 执行上下文创建成功")
    except Exception as e:
        print(f"❌ 执行上下文创建失败: {e}")
        return False

    # 检查执行上下文的方法
    print("\n🔍 执行上下文可用的方法:")
    try:
        context_methods = [m for m in dir(context) if not m.startswith('_')]
        for method in context_methods[:10]:  # 只显示前10个
            print(f"  - {method}")
        if len(context_methods) > 10:
            print(f"  ... 还有 {len(context_methods) - 10} 个方法")
    except Exception as e:
        print(f"  ⚠️  无法获取方法列表: {e}")

    # 尝试推理
    print("\n🚀 尝试推理:")
    try:
        # 创建测试输入 (ResNet18: 1x3x224x224)
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        print(f"  创建输入数据: {input_data.shape}")

        # 分配 GPU 内存
        d_input = pycuda.driver.mem_alloc(input_data.nbytes)
        print(f"  分配 GPU 内存: {input_data.nbytes} bytes")

        # 拷贝数据到 GPU
        pycuda.driver.memcpy_htod(d_input, input_data)
        print("  拷贝数据到 GPU")

        # 这里需要知道实际的输入输出名称才能继续
        print("  ⚠️  需要知道输入输出的 tensor 名称才能执行推理")
        print("  💡 尝试使用 trtexec 查看引擎信息:")
        print(f"     trtexec --loadEngine={engine_path}")

    except Exception as e:
        print(f"  ❌ 推理准备失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)

    return True

def use_trtexec_to_inspect():
    """使用 trtexec 来查看引擎信息"""
    print("\n💡 使用 trtexec 查看引擎详细信息:")
    print("运行以下命令:")
    print(f"  trtexec --loadEngine=resnet18.engine --dumpOutput --profilingEnabled=false")
    print("\n这会显示:")
    print("  - 输入输出的名称和形状")
    print("  - 数据类型")
    print("  - 内存使用情况")

if __name__ == "__main__":
    try:
        test_tensorrt_api()
        use_trtexec_to_inspect()
    except Exception as e:
        print(f"\n❌ 测试程序异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
