"""
Run Llama2 with colossalai

Reference:
https://github.com/microsoft/DeepSpeedExamples/blob/master/inference/huggingface/zero_inference/run_model.py
"""

import argparse
import gc
import itertools
import os
import time

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
import datasets
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

import sys
sys.path.append("..")
import utils


def get_colossalai_model(
    model_name,
    pin_memory,
    quant_bits,
    quant_group_size,
    cache_dir,
):
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    pin_memory = bool(args.pin_memory)
    dtype = torch.float16

    if quant_bits == 4:
        raise NotImplementedError()

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

    plugin = LowLevelZeroPlugin(stage=2)
    booster = Booster(plugin=plugin)
    model, optimizer, _, _, _ = booster.boost(model=model, optimizer=optimizer)

    return model, optimizer


def run_peft(
    model_name,
    dataset_name,
    trials,
    batch_size,
    gradient_accumulation_steps,
    seq_len,
    local_rank,
    pin_memory,
    quant_bits,
    quant_group_size,
    cache_dir
):
    # Load tokenizer
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', cache_dir=cache_dir)
    tokenizer.pad_token = '[PAD]'

    print("load model")
    model, optimizer = get_colossalai_model(
        model_name,
        pin_memory,
        quant_bits,
        quant_group_size,
        cache_dir,
    )

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
        optimizer.backward(loss)
        if step % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

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
    parser.add_argument("--trials", type=int, default=5,  help="Number of token generation iterations")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=256,  help="sequence length")
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")), help="local rank for distributed inference")
    parser.add_argument("--pin_memory", type=int, default=0, help="whether to pinned CPU memory for ZeRO offloading")
    parser.add_argument("--quant_bits", type=int, default=16, help="model weight quantization bits; either 4 or 8")
    parser.add_argument("--quant_group_size", type=int, default=64, help="model weight quantization group size")
    parser.add_argument("--cache_dir", type=str, default="/scratch/yonghe", help="cache dir for model name")
    args = parser.parse_args()

    colossalai.launch(config={}, rank=0, world_size=1, host="localhost", port=29502, backend="nccl")
    # clear cache / free memory
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
        args.quant_bits,
        args.quant_group_size,
        args.cache_dir,
    )

