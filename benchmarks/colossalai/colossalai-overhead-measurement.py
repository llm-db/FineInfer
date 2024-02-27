import argparse
import datetime
import gc
import os
import time

import colossalai
from colossalai.inference import InferenceEngine
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
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

    colossalai.launch(config={}, rank=0, world_size=1, host="localhost", port=29502, backend="nccl")
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
    optimizer = torch.optim.AdamW(model.parameters(), 3e-4)


    start = time.time()
    model.to(torch.cuda.current_device())
    end = time.time()
    print("CPU-to-GPU data movement (seconds):", end - start)

    start = time.time()
    engine = InferenceEngine(model=model)
    # plugin = LowLevelZeroPlugin(stage=2)
    # booster = Booster(plugin=plugin)
    # model, optimizer, _, _, _ = booster.boost(model=model, optimizer=optimizer)
    end = time.time()
    print("Task initialization (seconds):", end - start)

    start = time.time()
    model.to("cpu")
    end = time.time()
    print("GPU-to-CPU data movement (seconds):", end - start)

    print("Task cleanup begins at", datetime.datetime.now())
