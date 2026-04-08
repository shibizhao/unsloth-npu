# Ascend NPU Milestone / Issue 任务清单

## 文档用途
这份文档是 `ASCEND_NPU_ROADMAP.md` 的执行拆解版，目标是把路线图转换成更容易落到项目管理系统中的任务清单。

默认使用方式：

- 每个一级章节对应一个 milestone
- 每个 issue 条目可直接作为 GitHub issue 标题和内容草案
- 建议按本文顺序推进，而不是机械按源码目录拆任务

说明：

- 这里的 milestone 顺序按“推荐执行优先级”排列
- 因此与 `ASCEND_NPU_ROADMAP.md` 中的阶段编号不完全一一对应

## 建议标签
建议统一使用以下标签组合，便于后续筛选：

- `backend:npu`
- `area:core`
- `area:rl`
- `area:compile`
- `area:quantization`
- `area:kernel`
- `area:attention`
- `area:vision`
- `area:studio`
- `area:test`
- `type:bug`
- `type:docs`
- `type:feature`
- `type:perf`

## Milestone 概览

| Milestone | 优先级 | 目标 | 预期产出 |
| --- | --- | --- | --- |
| M1. 运行稳定化 | P0 | 从“示例可跑”收口到“默认可复现” | 明确使用文档、兼容矩阵、CUDA-only 假设清理 |
| M2. 编译与 RL 稳定化 | P0 | 让 NPU compile 路径可控可复用 | 至少一条 LoRA / GRPO compile 路径稳定 |
| M3. 4bit 与 NPU Kernel | P1 | 从通用回退走向 Ascend 专属快路径 | 4bit 可验证路径、dequant / gemv 优化 |
| M4. Studio 与回归测试 | P1 | 让 Ascend 成为可维护的 Studio 后端 | NPU 设备选择测试、smoke test、回归基线 |
| M5. Attention 与多模态收口 | P1 | 统一 attention 推荐策略并补齐 vision 边角 | 文本 / 视觉路径的推荐配置 |
| M6. FP8 与高级优化 | P2 | 在基线稳定后推动高阶能力 | FP8 评估、fused kernel 与高级性能项 |

## M1. 运行稳定化

### Milestone 目标
把当前依赖 example 规避问题的运行方式，沉淀成明确、可复现、可传播的工程基线。

### Milestone 完成标准

- 不看 examples 也能按文档把 Unsloth 在 Ascend 上跑起来
- 主要训练路径中不再残留明显 CUDA-only 假设
- `torch` / `torch_npu` / CANN / `unsloth_zoo` 的搭配关系被明确记录

### Issue 1
- 标题：`[Ascend][Core] 清理 RL 路径中的 CUDA-only 假设`
- 优先级：P0
- 类型：`type:bug`
- 主要文件：
  - [`unsloth/models/rl_replacements.py`](unsloth/models/rl_replacements.py)
- 建议内容：
  - 清理 NPU 路径中仍然写死 `device_type="cuda"` 的 autocast 或相关分支
  - 确保 RL / GRPO 核心路径在 NPU 下不依赖 CUDA-only 参数
- 完成标准：
  - NPU 训练路径中不再出现明显错误的 CUDA-only 设备分支

### Issue 2
- 标题：`[Ascend][Docs] 补齐 Ascend bring-up 文档与最小运行说明`
- 优先级：P0
- 类型：`type:docs`
- 主要文件：
  - [`ASCEND_NPU_ROADMAP.md`](ASCEND_NPU_ROADMAP.md)
  - [`README.md`](README.md)
  - [`examples/unsloth-grpo.py`](examples/unsloth-grpo.py)
- 建议内容：
  - 整理 Ascend 环境准备、依赖安装、最小 smoke test、最小训练示例
  - 明确当前推荐的 dtype、attention、compile 开关
- 完成标准：
  - 新接手同学可根据文档完成首次 bring-up

### Issue 3
- 标题：`[Ascend][Env] 梳理 torch / torch_npu / CANN / unsloth_zoo 兼容矩阵`
- 优先级：P0
- 类型：`type:docs`
- 主要文件：
  - [`unsloth/__init__.py`](unsloth/__init__.py)
  - [`unsloth/models/llama.py`](unsloth/models/llama.py)
  - [`examples/torch-test.py`](examples/torch-test.py)
- 建议内容：
  - 固化当前建议的版本组合
  - 标记已验证组合与未验证组合
- 完成标准：
  - 团队内部不再通过口口相传维持环境搭配

### Issue 4
- 标题：`[Ascend][Defaults] 收口 compile / attention / dtype 的默认推荐策略`
- 优先级：P0
- 类型：`type:feature`
- 主要文件：
  - [`examples/unsloth-grpo.py`](examples/unsloth-grpo.py)
  - [`examples/unsloth-grpo-8bit.py`](examples/unsloth-grpo-8bit.py)
  - [`unsloth/utils/attention_dispatch.py`](unsloth/utils/attention_dispatch.py)
- 建议内容：
  - 把当前 examples 里分散的推荐做法汇总成明确规则
  - 确定默认推荐是 `eager` 还是 `sdpa`
  - 确定 compile 默认开关策略
- 完成标准：
  - 文档与示例的默认建议一致，不再互相矛盾

## M2. 编译与 RL 稳定化

### Milestone 目标
让 Ascend 上的 compile 路径从“实验能力”走向“可控可复用能力”。

### Milestone 完成标准

- 至少一条 LoRA 或 GRPO 路径可以稳定启用 compile
- 示例级 monkey patch 不是主要接入方式

### Issue 5
- 标题：`[Ascend][Compile] 收口 NPU compile options 与 torchair 默认配置`
- 优先级：P0
- 类型：`type:feature`
- 主要文件：
  - [`unsloth/models/_utils.py`](unsloth/models/_utils.py)
  - [`examples/unsloth-grpo-8bit.py`](examples/unsloth-grpo-8bit.py)
- 建议内容：
  - 评估现有 `torch_compile_options` 在 NPU 上的适用性
  - 收口 torchair backend 所需的缺省参数与禁用项
  - 明确是否需要统一 `fullgraph=False`
- 完成标准：
  - 至少一条 compile 路径不再依赖 example 中的临时 patch

### Issue 6
- 标题：`[Ascend][RL] 收口 RLTrainer 与 GRPO 的 NPU compile 逻辑`
- 优先级：P0
- 类型：`type:feature`
- 主要文件：
  - [`unsloth/models/rl.py`](unsloth/models/rl.py)
  - [`unsloth/models/rl_replacements.py`](unsloth/models/rl_replacements.py)
- 建议内容：
  - 重新设计 NPU 下 RLTrainer compile 选项注入
  - 评估哪些 RL 子路径可以启用 compile，哪些需要保留回退
- 完成标准：
  - RL 主路径在 NPU 下具备稳定的 compile 策略，而不是“一律关掉”

### Issue 7
- 标题：`[Ascend][GRPO] 解除 NPU 下 GRPO 对 compile 的全量跳过`
- 优先级：P1
- 类型：`type:perf`
- 主要文件：
  - 本地配套 `unsloth-zoo/unsloth_zoo/rl_replacements.py`
  - [`examples/unsloth-grpo.py`](examples/unsloth-grpo.py)
  - [`examples/unsloth-grpo-8bit.py`](examples/unsloth-grpo-8bit.py)
- 建议内容：
  - 细化 GRPO 的 NPU compile 兼容范围
  - 拆分可编译和不可编译的子路径
- 完成标准：
  - GRPO 在 NPU 上至少有一条 compile 可用路径

## M3. 4bit 与 NPU Kernel

### Milestone 目标
从通用 fallback 走向 Ascend 专属性能路径。

### Milestone 完成标准

- 至少一条 4bit / QLoRA 路径在 Ascend 上被验证
- 核心量化快路径不再完全依赖通用 fallback

### Issue 8
- 标题：`[Ascend][Kernel] 为 NPU 增加 fast_dequantize 路径`
- 优先级：P1
- 类型：`type:perf`
- 主要文件：
  - [`unsloth/kernels/utils.py`](unsloth/kernels/utils.py)
- 建议内容：
  - 增加 Ascend 专属 dequantize 快路径
  - 验证正确性、数值稳定性与吞吐收益
- 完成标准：
  - NPU 不再只能走通用 dequantize fallback

### Issue 9
- 标题：`[Ascend][Kernel] 为 NPU 增加 fast_gemv 路径`
- 优先级：P1
- 类型：`type:perf`
- 主要文件：
  - [`unsloth/kernels/utils.py`](unsloth/kernels/utils.py)
- 建议内容：
  - 增加 Ascend 专属 gemv 快路径
  - 明确 seq_len == 1 等关键场景的收益
- 完成标准：
  - NPU 在相关场景下具备专属 fast_gemv 实现

### Issue 10
- 标题：`[Ascend][QLoRA] 验证并开放 Ascend 4bit 训练路径`
- 优先级：P1
- 类型：`type:feature`
- 主要文件：
  - [`examples/unsloth-grpo.py`](examples/unsloth-grpo.py)
  - [`examples/unsloth-grpo-8bit.py`](examples/unsloth-grpo-8bit.py)
  - [`tests/qlora/README.md`](tests/qlora/README.md)
- 建议内容：
  - 明确 Ascend 下 4bit 的最小支持范围
  - 用实际验证路径替换 `load_in_4bit=False` 的长期占位
- 完成标准：
  - 至少一条 4bit 或 QLoRA 训练路径可复现

## M4. Studio 与回归测试

### Milestone 目标
让 Ascend 成为可维护的 Studio 后端，而不是只能识别的设备类型。

### Milestone 完成标准

- Studio 在 Ascend 上的行为可预测
- NPU 相关回归能被自动化测试捕获

### Issue 11
- 标题：`[Ascend][Studio] 完善 NPU 设备检测、监控与可见设备逻辑`
- 优先级：P1
- 类型：`type:feature`
- 主要文件：
  - [`studio/backend/utils/hardware/hardware.py`](studio/backend/utils/hardware/hardware.py)
- 建议内容：
  - 强化 Ascend 设备信息采集
  - 明确 visible device、显存统计与设备摘要策略
- 完成标准：
  - Studio 的 Ascend 设备信息足以支持稳定展示和调试

### Issue 12
- 标题：`[Ascend][Studio][Test] 增加 NPU 设备选择与拒绝测试`
- 优先级：P1
- 类型：`type:test`
- 主要文件：
  - [`studio/backend/tests/test_gpu_selection.py`](studio/backend/tests/test_gpu_selection.py)
- 建议内容：
  - 增加 `TestNpuRejection` 或等价覆盖
  - 明确 NPU 下哪些 GPU selection 功能应拒绝，哪些可继续支持
- 完成标准：
  - Studio 设备选择逻辑对 NPU 有清晰测试覆盖

### Issue 13
- 标题：`[Ascend][Test] 建立 NPU smoke test 与训练/导出最小回归集`
- 优先级：P1
- 类型：`type:test`
- 主要文件：
  - [`examples/torch-test.py`](examples/torch-test.py)
  - [`examples/unsloth-grpo.py`](examples/unsloth-grpo.py)
  - [`tests/qlora/README.md`](tests/qlora/README.md)
- 建议内容：
  - 建立最小 smoke test
  - 建立最小训练回归
  - 建立最小 save/export 回归
- 完成标准：
  - 核心 Ascend 回归问题不再只能靠手工运行 examples 发现

## M5. Attention 与多模态收口

### Milestone 目标
让 attention 默认策略与 vision 路径在 Ascend 上可解释、可落地。

### Milestone 完成标准

- 文本模型与 vision 模型具备一致的 Ascend 推荐配置
- 用户不再需要自己在 `eager` 与 `sdpa` 之间试错

### Issue 14
- 标题：`[Ascend][Attention] 统一 eager / sdpa 的推荐策略与默认边界`
- 优先级：P1
- 类型：`type:feature`
- 主要文件：
  - [`unsloth/utils/attention_dispatch.py`](unsloth/utils/attention_dispatch.py)
  - [`examples/unsloth-grpo.py`](examples/unsloth-grpo.py)
  - [`examples/unsloth-grpo-8bit.py`](examples/unsloth-grpo-8bit.py)
- 建议内容：
  - 明确 NPU 下 attention backend 的选择顺序
  - 明确训练、推理、长上下文等场景的推荐差异
- 完成标准：
  - 文档、代码默认行为与示例策略能够对齐

### Issue 15
- 标题：`[Ascend][Vision] 收口 vision 路径中的 NPU TODO 与配置分歧`
- 优先级：P1
- 类型：`type:feature`
- 主要文件：
  - [`unsloth/models/vision.py`](unsloth/models/vision.py)
- 建议内容：
  - 清理 NPU cache、vLLM 假设、quantization 相关 TODO
  - 明确 Ascend 上 vision 模型的推荐配置
- 完成标准：
  - vision 路径对 Ascend 有清晰、稳定的推荐方案

## M6. FP8 与高级性能优化

### Milestone 目标
在基线已经稳定后，推进高阶能力和高阶性能优化。

### Milestone 完成标准

- 对 Ascend 高端型号的 FP8 能力有明确评估结论
- 至少形成一轮高级性能优化方向的技术评审结果

### Issue 16
- 标题：`[Ascend][FP8] 评估 Ascend 950+ 的 FP8 支持范围与依赖条件`
- 优先级：P2
- 类型：`type:feature`
- 主要文件：
  - [`unsloth/models/_utils.py`](unsloth/models/_utils.py)
- 建议内容：
  - 评估 FP8 在 Ascend 上的硬件、软件和依赖限制
  - 明确当前是否只做能力判断，还是进入可用实现阶段
- 完成标准：
  - 对 Ascend FP8 支持是否进入开发有明确结论

### Issue 17
- 标题：`[Ascend][Perf] 评估 fused kernel、长上下文与高级 compile 优化机会`
- 优先级：P2
- 类型：`type:perf`
- 主要文件：
  - [`unsloth/models/_utils.py`](unsloth/models/_utils.py)
  - [`unsloth/kernels/utils.py`](unsloth/kernels/utils.py)
  - [`unsloth/utils/attention_dispatch.py`](unsloth/utils/attention_dispatch.py)
- 建议内容：
  - 评估长上下文、fused kernel、compile 深化优化的收益和前提
  - 为下一轮性能专项提供方向输入
- 完成标准：
  - 输出一轮明确的高级性能优化候选清单

## 建议首批建单顺序
如果现在准备开始真正建 issue，建议第一批先建以下 6 个：

1. `[Ascend][Core] 清理 RL 路径中的 CUDA-only 假设`
2. `[Ascend][Docs] 补齐 Ascend bring-up 文档与最小运行说明`
3. `[Ascend][Env] 梳理 torch / torch_npu / CANN / unsloth_zoo 兼容矩阵`
4. `[Ascend][Defaults] 收口 compile / attention / dtype 的默认推荐策略`
5. `[Ascend][Compile] 收口 NPU compile options 与 torchair 默认配置`
6. `[Ascend][RL] 收口 RLTrainer 与 GRPO 的 NPU compile 逻辑`

原因：

- 这 6 个任务能先把“可复现基线”和“可持续优化入口”立住
- 后面的 4bit、Studio、attention、FP8 任务会更依赖这一层基线

## 建议首个 Milestone
如果只先开一个 milestone，建议先开：

- `M1. 运行稳定化`

建议描述：

- 目标是把当前 Ascend NPU 支持从“依赖 examples 和本地经验可跑”提升到“按文档即可复现，主路径不再残留明显 CUDA-only 假设”。
