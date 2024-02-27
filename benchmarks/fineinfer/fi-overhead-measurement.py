import argparse
import gc
import os
import time

import torch
from transformers import (
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="meta-llama/Llama-2-7b-hf", help="model name or path")
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")), help="local rank for distributed inference")
    parser.add_argument("--cache_dir", type=str, default="/scratch/yonghe", help="cache dir for model name")
    args = parser.parse_args()

    model_name = args.model_name
    cache_dir = args.cache_dir
    dtype = torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=dtype)
    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.01,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model = model.eval()


    # Warmup
    from peft.tuners.tuners_utils import BaseTunerLayer
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            module.lora_A.to(torch.cuda.current_device())
            module.lora_B.to(torch.cuda.current_device())
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            module.lora_A.to("cpu")
            module.lora_B.to("cpu")

    start = time.time()
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            module.lora_A.to(torch.cuda.current_device())
            module.lora_B.to(torch.cuda.current_device())
    end = time.time()
    print("CPU-to-GPU data movement (seconds):", end - start)

    start = time.time()
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            module.lora_A.to("cpu")
            module.lora_B.to("cpu")
    end = time.time()
    print("GPU-to-CPU data movement (seconds):", end - start)
