# Unsloth-PTO NPU 适配总结与 Roadmap

> 本文档总结了 Unsloth-PTO 项目中针对华为 Ascend NPU 的适配工作，基于代码中的 `Unsloth-PTO-XXX` 标记进行整理。

## 目录

- [修改标记统计](#修改标记统计)
- [已完成的 NPU 适配](#已完成的-npu-适配-verify-状态)
- [待修复项](#待修复项-fixme-状态)
- [NPU 生态 Roadmap](#npu-生态-roadmap)
- [关键依赖关系](#关键依赖关系)
- [下一步行动建议](#下一步行动建议)

---

## 修改标记统计

根据代码库搜索，发现以下标记类型：

| 标记类型 | 数量 | 说明 |
|---------|------|------|
| `Unsloth-PTO-VERIFY` | ~15 | 需要验证的 NPU 适配代码 |
| `Unsloth-PTO-FIXME` | ~20 | 需要修复的 NPU 相关问题 |
| `Unsloth-PTO-TODO` | ~3 | 待完成的 NPU 功能 |
| 通用 `TODO` | ~30+ | 原有的待办项 |

---

## 已完成的 NPU 适配 (VERIFY 状态)

### 1. 设备检测与初始化

**文件**: `unsloth/device_type.py`

```python
# Line 44-45
elif hasattr(torch, "npu") and torch.npu.is_available(): # Unsloth-PTO-VERIFY: support torch_npu
    return "npu"

# Line 77-78
elif DEVICE_TYPE == "npu": # Unsloth-PTO-VERIFY: support torch_npu
    return torch.npu.device_count()
```

**状态**: ✅ 已实现，需验证

### 2. Stream 和设备管理

**文件**: `unsloth/kernels/utils.py`

```python
# Line 137-140 - Ascend NPU Specific Logic
elif DEVICE_TYPE == "npu":  # Unsloth-PTO-VERIFY: support torch_npu
    import torch_npu
    _gpu_getCurrentRawStream = torch_npu._C._npu_getCurrentRawStream

# Line 173-187 - NPU_STREAMS 初始化
elif DEVICE_TYPE == "npu":  # Unsloth-PTO-VERIFY: support torch_npu
    import torch_npu
    _NPU_STREAMS = {
        (index := torch.npu.device(i).idx): ctypes.c_void_p(
            torch_npu._C._npu_getCurrentRawStream(index)
        )
        for i in range(DEVICE_COUNT)
    }
    NPU_STREAMS = [None] * (max(_NPU_STREAMS.keys()) + 1)
    # ... 初始化完成
```

**状态**: ✅ 已实现，需验证

### 3. BFloat16 支持

**文件**: `unsloth/__init__.py`

```python
# Line 194-195
elif DEVICE_TYPE == "npu": # Unsloth-PTO-VERIFY: support torch_npu
    SUPPORTS_BFLOAT16 = torch.npu.is_bf16_supported()
```

**状态**: ✅ 已实现，需验证

### 4. AMP (混合精度训练)

**文件**: `unsloth/models/_utils.py`

```python
# Line 612-618
# Unsloth-PTO-VERIFY: check the torch_npu version range for unsloth
elif DEVICE_TYPE == "npu":
    if Version(torch_version) < Version("2.4.0"):
        raise RuntimeError("torch.npu currently only supports torch.version >= 2.4.0")
    else:
        torch_amp_custom_fwd = torch.npu.amp.custom_fwd
        torch_amp_custom_bwd = torch.npu.amp.custom_bwd
```

**状态**: ✅ 已实现，需验证

### 5. empty_cache 和基础 API

**涉及文件**: `llama.py`, `vision.py`

| API | 状态 |
|-----|------|
| `torch.npu.empty_cache()` | ✅ 已添加 |
| `torch.npu.current_device()` | ✅ 已适配 |
| `torch.npu.get_device_properties()` | ✅ 已适配 |

### 6. Device Stream API

**文件**: `unsloth/kernels/utils.py`

```python
# Line 225-231
# Unsloth-PTO-VERIFY: check the device stream API of npu
if DEVICE_TYPE == "npu":
    torch_device_stream = torch.npu.current_stream
elif DEVICE_TYPE == "xpu":
    torch_device_stream = torch.xpu.current_stream
else:
    torch_device_stream = torch.cuda.current_stream
```

**状态**: ✅ 已实现，需验证

---

## 待修复项 (FIXME 状态)

### 1. Triton Kernels 适配 (高优先级)

需要使用 `PyPTO` / `PTO-ISA` 重写以下内核（不使用 triton-npu）：

| 内核文件 | 说明 | 当前状态 |
|---------|------|---------|
| `kernels/rms_layernorm.py` | RMS LayerNorm Triton 内核 | 回退到 PyTorch 原生 |
| `kernels/rope_embedding.py` | RoPE Embedding Triton 内核 | 回退到 PyTorch 原生 |
| `kernels/cross_entropy_loss.py` | 交叉熵损失 Triton 内核 | 回退到 PyTorch 原生 |
| `kernels/swiglu.py` | SwiGLU 激活函数内核 | 回退到 PyTorch 原生 |
| `kernels/geglu.py` | GeGLU 激活函数内核 | 回退到 PyTorch 原生 |
| `kernels/layernorm.py` | 标准 LayerNorm 内核 | 回退到 PyTorch 原生 |

**示例代码** (`kernels/rms_layernorm.py`):

```python
# Line 20 - 代码中的注释（将更新为 PyPTO/PTO-ISA）
# Unsloth-PTO-FIXME: Update the triton kernels with triton-npu/PyPTO/PTO-ISA on Ascend NPU

# Line 243-245 - 当前回退到 PyTorch 原生实现
# Unsloth-PTO-FIXME: check the support of fast_rms_layernorm on npu with triton-npu
# Use PyTorch native implementation on NPU (Triton not supported)
if hasattr(torch, 'npu') and torch.npu.is_available():
    # 使用原生实现（后续将用 PyPTO/PTO-ISA 替换）
```

> 📝 **注意**: 代码中的注释提到 `triton-npu`，但实际技术路线是使用 **PyPTO/PTO-ISA**。后续需要更新这些注释。

### 2. torch.compile 支持 (中优先级)

| 文件 | 问题描述 |
|------|---------|
| `models/llama.py` | `torch.compile` 装饰器已注释 |
| `kernels/flex_attention.py` | 需要 NPU 的 compile 选项 |
| `kernels/fast_lora.py` | LoRA 的 compile 支持 |
| `kernels/cross_entropy_loss.py` | CE loss 的 compile 支持 |
| `models/rl.py` | GRPO torch_compile_options (torchair?) |

**示例代码** (`models/llama.py`):

```python
# Line 580-582
# Unsloth-PTO-FIXME: Fix the torch.compile support for NPU
# Currently, we commmented the torch.compile deracotor for the funning.
# @torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
```

**示例代码** (`models/rl.py`):

```python
# Line 1111
# Unsloth-PTO-FIXME: check the NPU/XPU/HIP config with torch_compile_options (torchair?)
```

### 3. vLLM-Ascend 集成 (中优先级)

**文件**: `unsloth/models/llama.py`

```python
# Line 2184-2186
# Unsloth-PTO-FIXME: Fix the vLLM-Ascend support for NPU: here or is_vLLM_available()
elif DEVICE_TYPE == "npu":
    fast_inference = True

# Line 2233-2234
# Unsloth-PTO-FIXME: Fix the vLLM-Ascend support for NPU: here or is_vLLM_available()
elif DEVICE_TYPE == "npu":
    gpu_stats = torch.npu.get_device_properties(0)

# Line 2242
vllm_version = f" vLLM: {importlib_version('vllm')}." # Unsloth-PTO-FIXME: vLLM or vLLM-Ascend
```

**待修复项**:
- vLLM 版本检测（vLLM vs vLLM-Ascend）
- `is_vLLM_available()` 对 NPU 的支持
- Vision 模型的 vLLM-Ascend 支持 (`vision.py` Line 430)

### 4. bitsandbytes NPU 实现 (中优先级)

**文件**: `unsloth/device_type.py`, `unsloth/__init__.py`

```python
# device_type.py Line 86
# Unsloth-PTO-TODO: update the bitsandbytes implementations of NPU

# __init__.py Line 284
elif DEVICE_TYPE == "npu": # Unsloth-PTO-FIXME: update the bitsandbytes implementations
    import bitsandbytes as bnb
```

### 5. FP8 量化支持

**文件**: `unsloth/kernels/fp8.py`

```python
# Line 26-27
# Unsloth-PTO-FIXME: Disable the FP8 support for 910B/C.
# Update the triton kernels with triton-npu/PyPTO/PTO-ISA on Ascend NPU
```

**备注**: 910B/C 不支持 FP8，需要禁用相关功能

### 6. Fast Inference 实现

**文件**: `unsloth/kernels/utils.py`

```python
# Line 599
# Unsloth-PTO-FIXME: update NPU implmentations of fast inference

# Line 887
# Unsloth-PTO-FIXME: update NPU implmentations of fast inference
```

---

## NPU 生态 Roadmap

### Phase 1: 基础适配 ✅ (当前阶段)

```
torch_npu 集成
├── ✅ 设备检测 (device_type.py)
├── ✅ Stream 管理 (kernels/utils.py)
├── ✅ BFloat16 支持
├── ✅ AMP 混合精度
├── ✅ empty_cache/基础 API
└── ⚠️ 基础训练流程 (需验证)
```

**目标**: 确保 Unsloth 可以在 NPU 上运行基础训练流程

### Phase 2: 性能优化 🔄 (下一阶段)

```
torch_compile 支持 (torchair)
├── 🔧 flex_attention 的 compile 选项
├── 🔧 fast_lora compile 支持
├── 🔧 cross_entropy_loss compile 支持
├── 🔧 GRPO torch_compile_options
└── 🔧 llama layernorm torch.compile 装饰器
```

**目标**: 通过 `torchair` 实现 `torch.compile` 加速

**关键技术**:
- `torchair`: 华为的 torch.compile 后端
- NPU Graph 编译优化

### Phase 3: 高性能算子开发 📋 (中期目标)

使用 **PyPTO / PTO-ISA** 实现关键算子（不依赖 triton-npu）：

```
PyPTO / PTO-ISA 算子实现
│
├── Attention 算子 ⭐ 高优先级
│   ├── 📋 flash_attention - Flash Attention 实现
│   │   ├── 当前状态: NPU SDPA 接口不兼容，使用 eager attention
│   │   ├── 目标: 实现 NPU 原生 Flash Attention
│   │   └── 预期收益: 速度提升 2-3x，显存减少 50%+
│   └── 📋 paged_attention - 分页注意力 (vLLM)
│
├── Unsloth 核心算子
│   ├── 📋 rms_layernorm.py - RMS LayerNorm
│   ├── 📋 rope_embedding.py - RoPE Embedding
│   ├── 📋 cross_entropy_loss.py - 交叉熵损失
│   ├── 📋 swiglu.py - SwiGLU 激活
│   ├── 📋 geglu.py - GeGLU 激活
│   └── 📋 layernorm.py - 标准 LayerNorm
│
├── bitsandbytes 关键算子替换
│   ├── 📋 4-bit dequantize - NF4/FP4 反量化
│   ├── 📋 8-bit matmul - INT8 矩阵乘法
│   └── 📋 blockwise quantize - 分块量化
│
└── vLLM-Ascend 关键算子替换
    ├── 📋 rotary_embedding - 旋转位置编码
    └── 📋 layer_norm - 层归一化
```

**目标**: 使用 PyPTO/PTO-ISA 实现高性能 NPU 算子

**关键技术**:
- `PyPTO`: Python 级别的 PTO 编程接口，提供高层抽象
- `PTO-ISA`: NPU 底层指令集架构，直接操作硬件能力

**技术路线选择**:
> ⚠️ 注意：我们选择 **PyPTO/PTO-ISA** 而非 triton-npu，原因是：
> - PyPTO 提供更直接的 NPU 硬件控制
> - PTO-ISA 可以充分发挥 Ascend NPU 的算力
> - 避免 triton-npu 的兼容性和成熟度问题

### Phase 4: 推理优化 📋 (长期目标)

```
vllm_ascend 集成与算子替换
│
├── 集成工作
│   ├── 📋 vLLM-Ascend 版本检测
│   ├── 📋 is_vLLM_available() NPU 支持
│   ├── 📋 Fast inference 实现
│   ├── 📋 LoRA 推理优化
│   └── 📋 Vision 模型推理支持
│
└── 关键算子替换 (PyPTO/PTO-ISA)
    ├── 📋 paged_attention_v1/v2 - 分页注意力核心
    ├── 📋 rotary_embedding - RoPE 位置编码
    ├── 📋 rms_norm - RMS 归一化
    ├── 📋 silu_and_mul - SiLU 激活 + 门控
    └── 📋 fused_moe - 融合 MoE 算子
```

**目标**: 集成 vLLM-Ascend 并使用 PyPTO/PTO-ISA 替换关键算子

**关键技术**:
- `vllm-ascend`: vLLM 的 NPU 适配版本
- PyPTO/PTO-ISA 实现的高性能算子
- PagedAttention NPU 优化实现
- Continuous Batching

### Phase 5: 量化支持 📋 (长期目标)

```
bitsandbytes 算子替换与量化支持
│
├── INT4 量化 (4-bit)
│   ├── 📋 NF4 (Normal Float 4) - QLoRA 默认格式
│   │   ├── nf4_quantize - NF4 量化
│   │   ├── nf4_dequantize - NF4 反量化
│   │   └── nf4_linear - NF4 线性层
│   ├── 📋 FP4 (Float Point 4) - 浮点4位
│   │   ├── fp4_quantize - FP4 量化
│   │   └── fp4_dequantize - FP4 反量化
│   └── 📋 INT4 通用算子
│       ├── cgemm_4bit_inference - 4-bit 推理矩阵乘
│       ├── cgemv_4bit - 4-bit 向量矩阵乘
│       └── cdequantize_blockwise_4bit - 分块反量化
│
├── INT8 量化 (8-bit)
│   ├── 📋 动态量化
│   │   ├── int8_dynamic_quantize - 动态 INT8 量化
│   │   └── int8_dynamic_linear - 动态量化线性层
│   ├── 📋 静态量化
│   │   ├── int8_static_quantize - 静态 INT8 量化
│   │   └── int8_calibration - 校准数据收集
│   └── 📋 INT8 通用算子
│       ├── int8_matmul - INT8 矩阵乘法
│       ├── int8_gemm - INT8 通用矩阵乘
│       └── cdequantize_blockwise_8bit - 分块反量化
│
├── 混合精度量化
│   ├── 📋 W4A16 - 权重 INT4，激活 FP16
│   ├── 📋 W8A8 - 权重 INT8，激活 INT8
│   └── 📋 W4A8 - 权重 INT4，激活 INT8
│
└── 量化训练功能
    ├── 📋 QLoRA - 4-bit 量化 LoRA 训练
    ├── 📋 GPTQ 模型加载 - 预量化模型支持
    └── 📋 AWQ 模型加载 - 激活感知量化模型

```

**目标**: 使用 PyPTO/PTO-ISA 替换 bitsandbytes 关键算子，实现 NPU 原生量化

**关键技术**:
- PyPTO/PTO-ISA 实现的量化/反量化算子
- NPU 原生 INT8/INT4 计算单元 (Cube Unit)
- 分块量化策略优化 (blocksize=64/128)
- 量化感知训练 (QAT) 支持

**INT4/INT8 硬件支持情况**:

| 芯片型号 | INT4 | INT8 | FP8 | 备注 |
|---------|------|------|-----|------|
| 910B | ✅ | ✅ | ❌ | 主力开发平台 |
| 910C | ✅ | ✅ | ❌ | 推理优化 |
| 910D+ | ✅ | ✅ | 🔄 | FP8 待评估 |

**备注**:
> ⚠️ **FP8 支持说明**: 
> - 910B/C: 硬件不支持 FP8，已禁用

---

## 关键依赖关系

```
Unsloth-PTO NPU 生态依赖图
│
├── torch_npu (基础层)
│   ├── ✅ 已集成基础 API
│   ├── 版本要求: >= 2.4.0
│   └── 提供: 设备管理、张量操作、AMP
│
├── torch_compile / torchair (编译优化层)
│   ├── 🔧 需要验证和适配 compile options
│   ├── 依赖: torch_npu
│   └── 提供: 图编译优化、算子融合
│
├── PyPTO / PTO-ISA (算子加速层) ⭐ 核心技术路线
│   ├── 📋 待开发
│   ├── 依赖: torch_npu, CANN
│   ├── 组件:
│   │   ├── PyPTO - Python 级别 PTO 编程接口
│   │   └── PTO-ISA - NPU 底层指令集架构
│   ├── 提供: 高性能自定义算子，直接控制 NPU 硬件
│   └── 算子替换范围:
│       ├── Unsloth 核心算子 (rms_layernorm, rope, ce_loss, etc.)
│       ├── bitsandbytes 量化算子 (dequantize, gemm_4bit, etc.)
│       └── vLLM-Ascend 推理算子 (paged_attn, fused_moe, etc.)
│
├── vllm_ascend (推理加速层)
│   ├── 📋 需要集成和测试
│   ├── 依赖: torch_npu
│   ├── 关键算子: 使用 PyPTO/PTO-ISA 替换
│   └── 提供: 高效推理、LoRA 服务
│
└── bitsandbytes-npu (量化层)
    ├── 📋 待调研和实现
    ├── 依赖: torch_npu
    ├── 关键算子: 使用 PyPTO/PTO-ISA 替换
    ├── 提供: 4-bit/8-bit 量化
    └── FP8: ❌ 暂不支持
```

---

## 下一步行动建议

### 短期目标 (1-2 周)

1. **验证基础功能**
   - 在 910B/C 上运行完整训练流程
   - 确认所有 `VERIFY` 标记的代码工作正常
   - 记录性能基准数据

2. **修复关键问题**
   - 解决 `import torch_npu` 的条件导入问题
   - 确保 `empty_cache()` 正确释放内存

### 中期目标 (1-2 月)

3. **torch.compile 适配**
   - 研究 `torchair` 对 Unsloth 的支持
   - 测试 `torch.compile` 在 NPU 上的效果
   - 逐步启用 `torch.compile` 装饰器

4. **vLLM-Ascend 集成**
   - 测试 `vllm-ascend` 包的兼容性
   - 更新 `is_vLLM_available()` 检测逻辑
   - 实现 `for_inference()` 的 NPU 版本

### 长期目标 (3-6 月)

5. **PyPTO/PTO-ISA 算子开发**
   - 优先实现 `rms_layernorm` (最常用)
   - 掌握 PyPTO 编程接口和 PTO-ISA 指令集
   - 逐步替换 PyTorch 原生实现
   - 性能对标 CUDA Triton 内核

6. **文档和测试**
   - 添加 NPU 特定的测试用例
   - 更新 README 添加 NPU 支持说明
   - 编写 NPU 使用指南

---

## 技术路线说明: PyPTO/PTO-ISA vs triton-npu

### 为什么选择 PyPTO/PTO-ISA？

| 对比项 | triton-npu | PyPTO/PTO-ISA |
|-------|-----------|---------------|
| 成熟度 | 发展中，API 可能变化 | 华为官方支持，稳定 |
| 硬件控制 | 抽象层较高 | 直接控制 NPU 硬件 |
| 性能上限 | 受 Triton 抽象限制 | 可达硬件理论峰值 |
| 学习曲线 | 熟悉 Triton 可快速上手 | 需要学习 PTO 编程模型 |
| 生态依赖 | 依赖 Triton 社区 | 依赖华为 CANN 生态 |

### PyPTO 简介

PyPTO (Python Parallel Template Operator) 是华为提供的 Python 级别算子开发接口：

```python
# PyPTO 示例 (概念性)
import pypto

@pypto.kernel
def rms_layernorm_kernel(x, weight, eps):
    # 使用 PyPTO 原语实现 RMS LayerNorm
    variance = pypto.reduce_mean(x * x, axis=-1, keepdims=True)
    x_normed = x * pypto.rsqrt(variance + eps)
    return x_normed * weight
```

### PTO-ISA 简介

PTO-ISA (Parallel Template Operator Instruction Set Architecture) 是 Ascend NPU 的底层指令集：

- **向量运算**: 高效的 SIMD 操作
- **矩阵运算**: 利用 Cube 单元进行矩阵乘法
- **内存管理**: 精细控制 HBM/L1/L0 缓存层级
- **流水线**: 多级流水线并行执行

### 开发计划

```
PyPTO/PTO-ISA 算子开发优先级
│
├── P0 (最高优先级) - Attention & 核心算子 ⭐
│   ├── flash_attention - NPU Flash Attention 实现 (当前使用 eager，性能瓶颈)
│   ├── rms_layernorm - 每层都使用，性能敏感
│   ├── rope_embedding - Attention 核心组件
│   └── cross_entropy_loss - 训练必需
│
├── P1 (高优先级) - Unsloth + bitsandbytes 基础
│   ├── swiglu/geglu - MLP 激活函数
│   ├── layernorm - 部分模型使用
│   ├── nf4_dequantize - NF4 反量化 (QLoRA 核心)
│   └── cgemm_4bit_inference - 4-bit 推理矩阵乘
│
├── P2 (高优先级) - INT4/INT8 量化算子
│   ├── int4_quantize/dequantize - INT4 量化/反量化
│   ├── int8_quantize/dequantize - INT8 量化/反量化
│   ├── int8_matmul - INT8 矩阵乘法
│   ├── int4_linear - INT4 量化线性层
│   └── int8_linear - INT8 量化线性层
│
├── P3 (中优先级) - vLLM-Ascend
│   ├── paged_attention - 分页注意力
│   ├── silu_and_mul - SiLU 门控激活
│   └── fused_moe - MoE 融合算子
│
└── P4 (低优先级) - 高级量化功能
    ├── gptq_gemm - GPTQ 模型推理
    ├── awq_gemm - AWQ 模型推理
    └── mixed_precision_matmul - 混合精度矩阵乘
```

**说明**: 
- Flash Attention 是性能关键，当前 eager attention 导致速度慢 2-3x，显存占用高
- INT4/INT8 量化是 QLoRA 和推理优化的基础，优先级较高

---

## 附录: 标记规范

### 标记格式

```
Unsloth-PTO-<TYPE>: <description>
```

### 标记类型

| 类型 | 含义 | 使用场景 |
|------|------|---------|
| `VERIFY` | 需要验证 | 已实现但未在真机测试 |
| `FIXME` | 需要修复 | 已知问题或缺失功能 |
| `TODO` | 待办事项 | 计划中的功能 |

### 示例

```python
# Unsloth-PTO-VERIFY: support torch_npu
# Unsloth-PTO-FIXME: Update the triton kernels with triton-npu
# Unsloth-PTO-TODO: update the bitsandbytes implementations of NPU
```

---

## 更新日志

| 日期 | 版本 | 更新内容 |
|------|------|---------|
| 2026-01-31 | v1.4 | 将 Flash Attention 加入 P0 优先级 TODO，使用 PyPTO/PTO-ISA 实现 NPU 原生 Flash Attention |
| 2026-01-31 | v1.3 | 添加详细的 INT4/INT8 量化算子支持，包括 NF4、动态/静态量化、混合精度等 |
| 2026-01-31 | v1.2 | 扩展 PyPTO/PTO-ISA 算子替换范围至 bitsandbytes 和 vLLM-Ascend；FP8 暂不支持 |
| 2026-01-31 | v1.1 | 明确技术路线：使用 PyPTO/PTO-ISA 而非 triton-npu |
| 2026-01-31 | v1.0 | 初始版本，基于代码库标记整理 |

---

*本文档由 Unsloth-PTO 项目维护，如有问题请提交 Issue。*

