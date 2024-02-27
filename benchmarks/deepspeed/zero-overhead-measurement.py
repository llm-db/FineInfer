import argparse
import datetime
import gc
import os
import time

import deepspeed
import torch
from transformers import (
    AutoModelForCausalLM,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="meta-llama/Llama-2-7b-hf", help="model name or path")
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")), help="local rank for distributed inference")
    parser.add_argument("--cache_dir", type=str, default="/scratch/yonghe", help="cache dir for model name")
    args = parser.parse_args()

    model_name = args.model_name
    cache_dir = args.cache_dir
    dtype = torch.float16
    ds_config = {
        "fp16": {
            "enabled": dtype == torch.float16,
        },
        "bf16": {
            "enabled": dtype == torch.bfloat16,
        },
        "zero_optimization": {
            "stage": 0,
        },
        "steps_per_print": 2000,
        "train_batch_size": 1,
        "hybrid_engine": {"enabled": True},
    }

    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=dtype)
    model = model.eval()


    start = time.time()
    model.to(torch.cuda.current_device())
    end = time.time()
    print("CPU-to-GPU data movement (seconds):", end - start)

    start = time.time()
    ds_engine = deepspeed.init_inference(model=model)
    # ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    end = time.time()
    print("Task initialization (seconds):", end - start)

    start = time.time()
    model.to("cpu")
    end = time.time()
    print("GPU-to-CPU data movement (seconds):", end - start)

    print("Task cleanup begins at", datetime.datetime.now())
