"""
Run Llama2 with colossalai

Reference:
https://github.com/microsoft/DeepSpeedExamples/blob/master/inference/huggingface/zero_inference/run_model.py
"""

import argparse
import gc
import os
import time

import colossalai
from colossalai.inference import InferenceEngine
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

import sys
sys.path.append("..")
import utils


def get_colossalai_engine(
    model_name,
    prompt_len,
    gen_len,
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

    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=dtype)
    model.to(torch.cuda.current_device())

    engine = InferenceEngine(model=model, max_input_len=prompt_len, max_output_len=gen_len)
    engine.model.eval()

    return engine


def run_generation(
    model_name,
    trials,
    batch_size,
    prompt_len,
    gen_len,
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
    with torch.no_grad():
        engine = get_colossalai_engine(
            model_name,
            prompt_len,
            gen_len,
            pin_memory,
            quant_bits,
            quant_group_size,
            cache_dir,
        )

    utils.add_model_hooks(engine.model)

    prompts = ["Paris is the capital city of"] * batch_size
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt",
        padding="longest", max_length=prompt_len, truncation=True)
    inputs = {k: v.to(torch.cuda.current_device()) for k, v in input_tokens.items()}

    # NOTE(yongjunhe): ColossalAI InferenceEngine does not work with padding="max_length" correctly
    prompt_len = input_tokens.input_ids.shape[1]
    # Run generation
    print(f"benchmark, prompt_len = {prompt_len}, gen_len = {gen_len}, input_ids.shape = {input_tokens.input_ids.shape}")

    prefill_timings = []
    total_timings = []
    for _ in range(trials):
        start = time.time()
        with torch.no_grad():
            engine.model.stage = "prefill"
            output_ids = engine.generate(inputs)
            prefill_timings.append(engine.model.__duration__)
        end = time.time()
        total_timings.append(end - start)

    if local_rank != 0:
        return

    utils.remove_model_hooks(engine.model)
    # Check lengths
    for idx in range(len(output_ids)):
        output_ids[idx] = input_tokens.input_ids[idx].tolist() + output_ids[idx]
    input_lens = [len(x) for x in input_tokens.input_ids]
    output_lens = [len(x) for x in output_ids]
    assert all(x == prompt_len for x in input_lens)
    assert all(x == prompt_len + gen_len for x in output_lens)

    # Log output
    print(f"Summary:")
    print(f"total_timings = {total_timings}")
    print(f"prefill_timings = {prefill_timings}")
    total_latency = total_timings[-1]
    prefill_latency = prefill_timings[-1]

    prefill_throughput = batch_size * prompt_len / prefill_latency
    decode_latency = total_latency - prefill_latency
    decode_throughput = batch_size * (gen_len - 1) / max(decode_latency, 1e-10)
    num_generated_tokens = batch_size * gen_len
    total_throughput = num_generated_tokens / total_latency
    gpu_peak_mem = torch.cuda.max_memory_allocated(torch.device("cuda"))

    model_size = utils.model_bytes(config)
    cache_size = utils.cache_bytes(config, batch_size, prompt_len + gen_len)
    log_str = utils.write_gen_benchmark_log(
        model_size,
        cache_size,
        gpu_peak_mem,
        prefill_latency,
        prefill_throughput,
        decode_latency,
        decode_throughput,
        total_latency,
        total_throughput,
    )
    print(log_str)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    show_str = "Outputs:\n" + 30 * "-" + "\n"
    for i in [0, (len(outputs) - 1) // 2, len(outputs) - 1]:
        show_str += f"{i}: {outputs[i]}\n"
        show_str += 30 * "-" + "\n"
    print(show_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="meta-llama/Llama-2-7b-hf", help="model name or path")
    parser.add_argument("--trials", type=int, default=3,  help="Number of token generation iterations")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prompt_len", type=int, default=512,  help="prompt length")
    parser.add_argument("--gen_len", type=int, default=32,  help="number of tokens to generate")
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")), help="local rank for distributed inference")
    parser.add_argument("--pin_memory", type=int, default=0, help="whether to pinned CPU memory for ZeRO offloading")
    parser.add_argument("--quant_bits", type=int, default=16, help="model weight quantization bits; either 4 or 8")
    parser.add_argument("--quant_group_size", type=int, default=64, help="model weight quantization group size")
    parser.add_argument("--cache_dir", type=str, default="/scratch/yonghe", help="cache dir for model name")
    args = parser.parse_args()

    colossalai.launch(config={}, rank=0, world_size=1, host="localhost", port=29502, backend="nccl")
    # clear cache / free memory
    gc.collect()

    run_generation(
        args.model_name,
        args.trials,
        args.batch_size,
        args.prompt_len,
        args.gen_len,
        args.local_rank,
        args.pin_memory,
        args.quant_bits,
        args.quant_group_size,
        args.cache_dir,
    )
