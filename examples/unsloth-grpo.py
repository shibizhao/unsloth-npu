# -*- coding: utf-8 -*-
"""
GRPO (Group Relative Policy Optimization) 训练示例
模型: Qwen3-8B
数据集: GSM8K (数学推理任务)
平台: Ascend NPU
"""

import os

# ==================== 环境配置 ====================
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "4"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

from unsloth import FastModel
import torch
import re
import gc
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

# ==================== 配置 ====================
max_seq_length = 4096
max_prompt_length = 512
lora_rank = 64

# ==================== 加载模型 ====================
print("=" * 60)
print("🚀 加载 Qwen3-8B 模型...")
print("=" * 60)

from modelscope import snapshot_download
model_path = snapshot_download("Qwen/Qwen3-8B", cache_dir="./try_models")
print(f"✅ 模型路径: {model_path}")

model, tokenizer = FastModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    load_in_4bit=False,           # TODO: NPU 4-bit 量化支持
    load_in_8bit=False,
    load_in_16bit=True,
    full_finetuning=False,
    fast_inference=False,
    dtype=torch.bfloat16,
    attn_implementation="eager",  # NPU 必须使用 eager attention
)
print("✅ 模型加载完成!")

# ==================== 添加 LoRA ====================
print("\n🔧 添加 LoRA 适配器...")

model = FastModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_rank,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    max_seq_length=max_seq_length,
)
print("✅ LoRA 适配器添加完成!")

# ==================== 数据集准备 ====================
print("\n📊 准备 GSM8K 数据集...")

reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"

def extract_hash_answer(text):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def prepare_gsm8k_dataset(dataset):
    system_prompt = (
        f"你是一个数学问题解决专家。请仔细思考问题并逐步推理。"
        f"将你的思考过程放在 {reasoning_start} 和 {reasoning_end} 之间。"
        f"然后在 {solution_start} 和 {solution_end} 之间给出你的最终数值答案。"
    )
    
    def format_example(example):
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["question"]},
            ],
            "answer": extract_hash_answer(example["answer"]),
        }
    return dataset.map(format_example)

gsm8k_dataset = load_dataset("openai/gsm8k", "main", split="train", cache_dir="./try_datasets")
gsm8k_train = prepare_gsm8k_dataset(gsm8k_dataset)
gsm8k_train = gsm8k_train.filter(lambda x: x["answer"] is not None)
print(f"✅ 数据集准备完成! 共 {len(gsm8k_train)} 个样本")

# ==================== 奖励函数 ====================
print("\n🎯 定义奖励函数...")

def match_format_exactly(completions, **kwargs):
    """精确格式匹配"""
    pattern = rf"^[\s]*{re.escape(reasoning_start)}.+?{re.escape(reasoning_end)}.*?{re.escape(solution_start)}.+?{re.escape(solution_end)}[\s]*$"
    responses = [c[0]["content"] for c in completions]
    return [3.0 if re.match(pattern, r, re.DOTALL) else 0.0 for r in responses]

def match_format_approximately(completions, **kwargs):
    """近似格式匹配"""
    scores = []
    for c in completions:
        r = c[0]["content"]
        score = sum([
            0.5 if r.count(tag) == 1 else -1.0
            for tag in [reasoning_start, reasoning_end, solution_start, solution_end]
        ])
        scores.append(score)
    return scores

def check_answer_correctness(prompts, completions, answer, **kwargs):
    """答案正确性检查"""
    def extract_answer(text):
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        return re.sub(r"[%$,]", "", match.group(1)).strip() if match else ""
    
    responses = [c[0]["content"] for c in completions]
    extracted = [extract_answer(r) for r in responses]
    
    scores = []
    for guess, true_ans in zip(extracted, answer):
        if not guess:
            scores.append(0)
        elif guess == true_ans:
            scores.append(3.0)
        elif guess.strip() == true_ans.strip():
            scores.append(1.5)
        else:
            try:
                ratio = float(guess) / float(true_ans)
                scores.append(1.0 if 0.9 <= ratio <= 1.1 else -1.5)
            except:
                scores.append(-1.5)
    return scores

match_numbers = re.compile(solution_start + r".*?([\d\.\,]{1,})", flags=re.DOTALL)
PRINT_COUNTER = 0

def check_numbers(prompts, completions, answer, **kwargs):
    """数值验证"""
    global PRINT_COUNTER
    question = prompts[0][-1]["content"]
    responses = [c[0]["content"] for c in completions]
    extracted = [m.group(1) if (m := match_numbers.search(r)) else None for r in responses]
    
    if PRINT_COUNTER % 10 == 0:
        print(f"\n{'='*40}\n问题: {question}\n答案: {answer[0]}\n提取: {extracted[0]}")
    PRINT_COUNTER += 1
    
    scores = []
    for guess, true_ans in zip(extracted, answer):
        if guess is None:
            scores.append(0)
        else:
            try:
                scores.append(1.5 if float(guess.replace(",", "")) == float(true_ans) else -0.5)
            except:
                scores.append(0)
    return scores

print("✅ 奖励函数定义完成!")

# ==================== GRPO 训练 ====================
print("\n🎯 开始 GRPO 训练...")

training_args = GRPOConfig(
    output_dir="try_outputs/grpo-qwen3-8b",
    learning_rate=5e-6,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_generations=4,
    max_prompt_length=max_prompt_length,
    max_completion_length=128,
    max_steps=500,
    logging_steps=1,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="none",
    bf16=True,
    gradient_checkpointing=False,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[match_format_exactly, match_format_approximately, check_answer_correctness, check_numbers],
    args=training_args,
    train_dataset=gsm8k_train,
)

print(f"🚂 开始训练，共 {len(gsm8k_train)} 个样本...")
trainer.train()
print("✅ GRPO 训练完成!")

# ==================== 保存模型 ====================
print("\n💾 保存模型...")
model.save_pretrained("try_outputs/grpo-qwen3-8b/lora_adapter")
tokenizer.save_pretrained("try_outputs/grpo-qwen3-8b/lora_adapter")
print("✅ LoRA 适配器已保存到 try_outputs/grpo-qwen3-8b/lora_adapter")

# ==================== 清理显存 ====================
def cleanup_memory():
    gc.collect()
    if hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.empty_cache()
        print(f"NPU 显存 - 已分配: {torch.npu.memory_allocated()/1024**3:.2f} GB")
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU 显存 - 已分配: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

cleanup_memory()
print("\n🎉 GRPO 训练全部完成!")
