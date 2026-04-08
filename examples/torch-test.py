import torch
import torch_npu

# 测试基本 matmul
a = torch.randn(2, 3, dtype=torch.bfloat16).npu()
b = torch.randn(3, 4, dtype=torch.bfloat16).npu()
c = torch.matmul(a, b)
print("NPU matmul 测试成功:", c.shape)