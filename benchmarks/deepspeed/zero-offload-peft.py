"""
Run Llama2 with deepspeed.

Reference:
https://github.com/microsoft/DeepSpeedExamples/blob/master/inference/huggingface/zero_inference/run_model.py
"""

import argparse
import gc
import itertools
import os
import time

import datasets
import deepspeed
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from transformers.deepspeed import HfDeepSpeedConfig
from peft import LoraConfig, get_peft_model

import sys
sys.path.append("..")
import utils


def get_ds_model(
    model_name,
    batch_size,
    pin_memory,
    cpu_offload,
    disk_offload,
    offload_dir,
    quant_bits,
    quant_group_size,
    cache_dir,
):
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    pin_memory = bool(args.pin_memory)
    dtype = torch.float16

    ds_config = {
        "fp16": {
            "enabled": dtype == torch.float16,
        },
        "bf16": {
            "enabled": dtype == torch.bfloat16,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
            "lr": 3e-4,
            }
        },
        "zero_optimization": {
            "stage": 3,
            "stage3_prefetch_bucket_size": pow(config.hidden_size, 2) * 2, # 0,
            "stage3_param_persistence_threshold": config.hidden_size,
            "stage3_max_live_parameters": pow(config.hidden_size, 2) * 2,
        },
        "steps_per_print": 2000,
        "train_batch_size": batch_size,
        "wall_clock_breakdown": False,
    }

    if quant_bits == 4:
        quant_config = utils.get_quant_config(config, quant_bits=quant_bits, quant_group_size=quant_group_size)
        ds_config.update(quant_config)
    if cpu_offload:
        ds_config["zero_optimization"]["offload_param"] = dict(
            device="cpu", pin_memory=pin_memory
        )
        ds_config["zero_optimization"]["offload_optimizer"] = dict(
            device="cpu", pin_memory=pin_memory
        )
    if disk_offload:
        ds_config["zero_optimization"]["offload_param"] = dict(
            device="nvme",
            pin_memory=pin_memory,
            nvme_path=offload_dir,
            buffer_count=5,
            buffer_size=2 * utils.GB,
        )
        ds_config["zero_optimization"]["offload_optimizer"] = dict(
            device="nvme",
            pin_memory=pin_memory,
            nvme_path=offload_dir,
        )
        ds_config["aio"] = {
            "block_size": 1048576,
            "queue_depth": 8,
            "thread_count": 1,
            "single_submit": False,
            "overlap_events": True,
        }

    # this tells from_pretrained to instantiate directly on gpus
    dschf = HfDeepSpeedConfig(ds_config) # keep this object alive

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
    for name, module in model.named_modules():
        module.training = True

    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    return ds_engine


def run_peft(
    model_name,
    dataset_name,
    trials,
    batch_size,
    gradient_accumulation_steps,
    seq_len,
    local_rank,
    pin_memory,
    cpu_offload,
    disk_offload,
    offload_dir,
    kv_offload,
    quant_bits,
    quant_group_size,
    pin_kv_cache,
    async_kv_offload,
    cache_dir
):
    # Load tokenizer
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', cache_dir=cache_dir)
    tokenizer.pad_token = '[PAD]'

    print("load model")
    model = get_ds_model(
        model_name,
        batch_size,
        pin_memory,
        cpu_offload,
        disk_offload,
        offload_dir,
        quant_bits,
        quant_group_size,
        cache_dir,
    )

    if kv_offload:
        model.set_kv_cache_offload(True, seq_len, pin_kv_cache, async_kv_offload)

    def prepare_alpaca(sample_raw):
        template = {
            "description": "A shorter template to experiment with.",
            "prompt_input": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
            "prompt_no_input": "### Instruction:\n{instruction}\n\n### Response:\n",
            "response_split": "### Response:"
        }
        if len(sample_raw["input"]):
            sample_text = template["prompt_input"].format(
                instruction=sample_raw["instruction"], input=sample_raw["input"]
            )
        else:
            sample_text = template["prompt_no_input"].format(
                instruction=sample_raw["instruction"]
            )
        if len(sample_raw["output"]):
            sample_text += sample_raw["output"]
        sample_tokens = tokenizer(sample_text, padding='max_length', truncation=True, max_length=seq_len)
        return sample_tokens

    dataset = datasets.load_dataset(dataset_name, cache_dir=cache_dir)
    dataset = dataset.map(lambda sample_raw: prepare_alpaca(sample_raw), remove_columns=dataset["train"].column_names)
    dataloader = torch.utils.data.DataLoader(
        dataset["train"], shuffle=True, collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        batch_size=batch_size, pin_memory=pin_memory,
    )
    dataloader_iter = itertools.cycle(iter(enumerate(dataloader)))

    # Run peft
    # print(f"benchmark, seq_len = {seq_len}, input_ids.shape = {inputs.input_ids.shape}")

    forward_timings = []
    total_timings = []
    for _ in range(trials):
        start = time.time()

        step, inputs = next(dataloader_iter)
        inputs.to(torch.cuda.current_device())
        outputs = model(**inputs)
        forward_timings.append(time.time() - start)

        loss = outputs.loss
        model.backward(loss)
        if step % gradient_accumulation_steps == 0:
            model.step()

        end = time.time()
        total_timings.append(end - start)

    if local_rank != 0:
        return

    # Check lengths
    input_lens = [len(x) for x in inputs.input_ids]
    output_lens = [len(x) for x in outputs.logits]
    assert all(x == seq_len for x in input_lens)
    assert all(x == seq_len for x in output_lens)

    # Log output
    print(f"Summary:")
    print(f"total_timings = {total_timings}")
    print(f"forward_timings = {forward_timings}")
    total_latency = total_timings[-1]
    forward_latency = forward_timings[-1]
    backward_latency = total_latency - forward_latency

    total_throughput = batch_size / total_latency
    forward_throughput = batch_size / forward_latency
    backward_throughput = batch_size / backward_latency
    gpu_peak_mem = torch.cuda.max_memory_allocated(torch.device("cuda"))

    model_size = utils.model_bytes(config)
    log_str = utils.write_peft_benchmark_log(
        model_size,
        0,
        gpu_peak_mem,
        forward_latency,
        forward_throughput,
        backward_latency,
        backward_throughput,
        total_latency,
        total_throughput,
    )
    print(log_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="meta-llama/Llama-2-7b-hf", help="model name or path")
    parser.add_argument("--dataset_name", type=str, default="yahma/alpaca-cleaned", help="dataset name or path")
    parser.add_argument("--trials", type=int, default=3,  help="Number of token generation iterations")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=256,  help="sequence length")
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")), help="local rank for distributed inference")
    parser.add_argument("--pin_memory", type=int, default=0, help="whether to pinned CPU memory for ZeRO offloading")
    parser.add_argument("--cpu_offload", action="store_true", help="Use cpu offload.")
    parser.add_argument("--disk_offload", action="store_true", help="Use disk offload.")
    parser.add_argument("--offload_dir", type=str, default="/scratch/yonghe/offload_dir", help="Directory to store offloaded cache.")
    parser.add_argument("--kv_offload", action="store_true", help="Use kv cache cpu offloading.")
    parser.add_argument("--quant_bits", type=int, default=16, help="model weight quantization bits; either 4 or 8")
    parser.add_argument("--quant_group_size", type=int, default=64, help="model weight quantization group size")
    parser.add_argument("--pin_kv_cache", action="store_true", help="Allocate kv cache in pinned memory for offloading.")
    parser.add_argument("--async_kv_offload", action="store_true", help="Using non_blocking copy for kv cache offloading.")
    parser.add_argument("--cache_dir", type=str, default="/scratch/yonghe", help="cache dir for model name")
    args = parser.parse_args()

    deepspeed.init_distributed(verbose=False)
    # clear cache / free memory
    deepspeed.accelerator.get_accelerator().empty_cache()
    gc.collect()

    run_peft(
        args.model_name,
        args.dataset_name,
        args.trials,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.seq_len,
        args.local_rank,
        args.pin_memory,
        args.cpu_offload,
        args.disk_offload,
        args.offload_dir,
        args.kv_offload,
        args.quant_bits,
        args.quant_group_size,
        args.pin_kv_cache,
        args.async_kv_offload,
        args.cache_dir,
    )
