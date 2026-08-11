import torch
import torchvision.models as models

# 加载预训练 ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()  # 必须！否则 BatchNorm/Dropout 行为不对

# 创建 dummy input（ResNet 输入：1x3x224x224）
dummy_input = torch.randn(1, 3, 224, 224)

# 导出 ONNX（包含所有权重在单个文件中）
torch.onnx.export(
    model,
    dummy_input,
    "resnet18.onnx",
    export_params=True,           # 导出权重
    opset_version=17,             # 用 17+，支持现代算子
    do_constant_folding=True,     # 常量折叠优化
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={                # 支持动态 batch
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    # 🔧 关键修复：确保所有权重保存在 ONNX 文件内部
    keep_initializers_as_inputs=False,  # 不要将初始化器作为输入
    # 新版本 PyTorch 中，export_params=True 已经会包含权重
    # 但我们可以显式确保这一点
)
print("ONNX 导出成功: resnet18.onnx (包含所有权重)")

# 验证生成的 ONNX 文件
import onnx
model_onnx = onnx.load("resnet18.onnx")
print(f"ONNX 模型验证成功")
print(f"模型包含 {len(model_onnx.graph.initializer)} 个初始化器（权重）")
print(f"文件大小: {len(model_onnx.SerializeToString()) / (1024*1024):.1f} MB")
