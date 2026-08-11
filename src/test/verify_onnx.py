import torch
import onnx
import onnxruntime as ort
import numpy as np
import torchvision.models as models

# 1. 检查 ONNX 格式合法性
onnx_model = onnx.load("resnet18.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX 模型格式正确")

# 2. 对比 PyTorch 和 ONNX Runtime 输出
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

dummy = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    torch_out = model(dummy).numpy()

sess = ort.InferenceSession("resnet18.onnx")
onnx_out = sess.run(None, {'input': dummy.numpy()})[0]

diff = np.abs(torch_out - onnx_out).max()
print(f"PyTorch vs ONNX 最大误差: {diff:.2e}")
# 应 < 1e-5，说明导出无损
