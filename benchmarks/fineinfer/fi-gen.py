"""
Run Llama2 with FineInfer

Reference:
https://github.com/microsoft/DeepSpeedExamples/blob/master/inference/huggingface/zero_inference/run_model.py
"""

import argparse
import copy
import gc
import inspect
import os
import time

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

import sys
sys.path.append("../..")
from benchmarks import utils
from fineinfer.engine import llm_engine


def get_hf_model(
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

    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=dtype)
    model.to(torch.cuda.current_device())
    model = model.eval()

    return model


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
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    print("load model")
    with torch.no_grad():
        model = get_hf_model(
            model_name,
            pin_memory,
            quant_bits,
            quant_group_size,
            cache_dir,
        )

    utils.add_model_hooks(model)

    prompts = ["Paris is the capital city of"] * batch_size
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt",
        padding="max_length", max_length=prompt_len)
    input_tokens.to(torch.cuda.current_device())

    # Run generation
    print(f"benchmark, prompt_len = {prompt_len}, gen_len = {gen_len}, input_ids.shape = {input_tokens.input_ids.shape}")

    # Prepare
    prepare_output = llm_engine.prepare_inputs_and_config(self=model,
        **input_tokens, max_new_tokens=gen_len, do_sample=False)

    input_ids = copy.deepcopy(prepare_output.input_ids)
    model_kwargs = copy.deepcopy(prepare_output.model_kwargs)
    batch_meta = llm_engine.BatchMeta(
        prompt_lens = torch.full(size=(batch_size,), fill_value=prompt_len,
            dtype=torch.long, device=torch.cuda.current_device()),
        gen_lens = torch.full(size=(batch_size,), fill_value=gen_len,
            dtype=torch.long, device=torch.cuda.current_device()),
        cur_lens = torch.full(size=(batch_size,), fill_value=prompt_len,
            dtype=torch.long, device=torch.cuda.current_device()),
    )
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=torch.cuda.current_device())
    this_peer_finished = False

    output_ids = []
    prefill_timings = []
    total_timings = []

    start = time.time()
    with torch.no_grad():
        model.stage = "prefill"
        model_kwargs = model._get_initial_cache_position(input_ids, model_kwargs)

        while True:
            if input_ids.shape[0] + len(output_ids) < trials * batch_size \
                and torch.min(batch_meta.cur_lens) == prompt_len + 5:
                new_unfinished_sequences = torch.ones(prepare_output.input_ids.shape[0],
                    dtype=torch.long, device=input_ids.device)
                new_input_ids = copy.deepcopy(prepare_output.input_ids)
                new_model_kwargs = copy.deepcopy(prepare_output.model_kwargs)

                prefill_timings.append(model.__duration__)
                model.stage = "prefill"
                new_model_kwargs = model._get_initial_cache_position(new_input_ids, new_model_kwargs)

                new_unfinished_sequences, new_input_ids, new_model_kwargs = llm_engine.generate_step(
                    self=model,
                    unfinished_sequences=new_unfinished_sequences,
                    input_ids=new_input_ids,
                    logits_processor=prepare_output.logits_processor,
                    stopping_criteria=prepare_output.stopping_criteria,
                    generation_config=prepare_output.generation_config,
                    **new_model_kwargs
                )

                unfinished_sequences, input_ids, model_kwargs = llm_engine.add_new_request(
                    unfinished_sequences = unfinished_sequences,
                    input_ids = input_ids,
                    model_kwargs = model_kwargs,
                    new_unfinished_sequences = new_unfinished_sequences,
                    new_input_ids = new_input_ids,
                    new_model_kwargs = new_model_kwargs,
                )

                batch_meta.prompt_lens = torch.cat(tensors=(batch_meta.prompt_lens, \
                    torch.full(size=(batch_size,), fill_value=prompt_len, \
                    dtype=torch.long, device=torch.cuda.current_device())), dim=0)
                batch_meta.gen_lens = torch.cat(tensors=(batch_meta.gen_lens, \
                    torch.full(size=(batch_size,), fill_value=gen_len, \
                    device=torch.cuda.current_device(), dtype=torch.long)), dim=0)
                batch_meta.cur_lens = torch.cat(tensors=(batch_meta.cur_lens, \
                    torch.full(size=(batch_size,), fill_value=prompt_len + 1, \
                    device=torch.cuda.current_device(), dtype=torch.long)), dim=0)

            unfinished_sequences, input_ids, model_kwargs = llm_engine.generate_step(
                self=model,
                unfinished_sequences=unfinished_sequences,
                input_ids=input_ids,
                logits_processor=prepare_output.logits_processor,
                stopping_criteria=prepare_output.stopping_criteria,
                generation_config=prepare_output.generation_config,
                **model_kwargs
            )

            batch_meta.cur_lens += 1

            this_peer_finished = unfinished_sequences.max() == 0

            if not model._has_unfinished_sequences(this_peer_finished, prepare_output.synced_gpus, input_ids.device):
                unfinished_sequences, input_ids, model_kwargs, batch_meta, output_ids = llm_engine.remove_old_request(
                    unfinished_sequences=unfinished_sequences,
                    input_ids=input_ids,
                    model_kwargs=model_kwargs,
                    batch_meta=batch_meta,
                    output_ids=output_ids,
                )

                if len(output_ids) >= trials * batch_size:
                    break

    end = time.time()
    total_timings.append(end - start)
    prefill_timings.append(model.__duration__)

    if local_rank != 0:
        return

    utils.remove_model_hooks(model)
    # Check lengths
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
    decode_latency = total_latency - sum(prefill_timings)
    decode_throughput = trials * batch_size * (gen_len - 1) / max(decode_latency, 1e-10)
    num_generated_tokens = trials * batch_size * gen_len
    total_throughput = num_generated_tokens / total_latency
    gpu_peak_mem = torch.cuda.max_memory_allocated(torch.cuda.current_device())

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

    outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in output_ids]
    show_str = "Outputs:\n" + 30 * "-" + "\n"
    for i in [0, (len(outputs) - 1) // 2, len(outputs) - 1]:
        show_str += f"{i}: {outputs[i]}\n"
        show_str += 30 * "-" + "\n"
    print(show_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="meta-llama/Meta-Llama-3-8B", help="model name or path")
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
