"""
Run Llama2 with huggingface
Reference:
https://github.com/microsoft/DeepSpeedExamples/blob/master/inference/huggingface/zero_inference/run_model.py
"""

import argparse
import copy
import gc
import itertools
import os
import time

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
sys.path.append("../..")
from benchmarks import utils
from fineinfer.engine import llm_engine


def get_hf_model(
    model_name,
    adapter_names,
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
    model = get_peft_model(base_model, lora_config, adapter_name=adapter_names[0])
    for idx, name in enumerate(adapter_names):
        if idx:
            model.add_adapter(name, lora_config)
    model.to(torch.cuda.current_device())

    return model


def run_ht(
    model_name,
    adapter_size,
    dataset_name,
    batch_size,
    prompt_len,
    gen_len,
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
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    print("load model")
    adapter_names = ["adapter_" + str(i) for i in range(adapter_size)]
    model = get_hf_model(
        model_name,
        adapter_names,
        pin_memory,
        quant_bits,
        quant_group_size,
        cache_dir,
    )

    prompts = ["Paris is the capital city of"] * batch_size
    gen_inputs = tokenizer.batch_encode_plus(prompts, return_tensors="pt",
        padding="max_length", max_length=prompt_len)
    gen_inputs.to(torch.cuda.current_device())
    prepare_output = llm_engine.prepare_inputs_and_config(self=model,
        **gen_inputs, max_new_tokens=gen_len, do_sample=False)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Run heterogeneous workload
    print(f"benchmark, seq_len = {seq_len}, prompt_len = {prompt_len}, gen_len = {gen_len}")

    total_latency = 60.0
    req_gap_bound = 2.0
    ht_workloads = utils.get_ht_workloads(total_latency, req_gap_bound)
    gen_trials = len(ht_workloads)
    cursor = 0

    gen_outputs = []
    gen_timings = []
    peft_timings = []
    start = time.time()
    while time.time() - start < total_latency:
        if cursor < gen_trials and ht_workloads[cursor] <= time.time() - start:
            cursor += 1
            with torch.no_grad():
                input_ids = copy.deepcopy(prepare_output.input_ids)
                model_kwargs = copy.deepcopy(prepare_output.model_kwargs)
                model_kwargs = model._get_initial_cache_position(input_ids, model_kwargs)
                batch_meta = llm_engine.BatchMeta(
                    prompt_lens = torch.full(size=(batch_size,), fill_value=prompt_len,
                        dtype=torch.long, device=torch.cuda.current_device()),
                    gen_lens = torch.full(size=(batch_size,), fill_value=gen_len,
                        dtype=torch.long, device=torch.cuda.current_device()),
                    cur_lens = torch.full(size=(batch_size,), fill_value=prompt_len,
                        dtype=torch.long, device=torch.cuda.current_device()),
                )
                unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=torch.cuda.current_device())

                while True:
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
                        gen_timings.append(time.time() - start - ht_workloads[int(len(gen_outputs) / batch_size)])
                        unfinished_sequences, input_ids, model_kwargs, batch_meta, output_ids = llm_engine.remove_old_request(
                            unfinished_sequences=unfinished_sequences,
                            input_ids=input_ids,
                            model_kwargs=model_kwargs,
                            batch_meta=batch_meta,
                            output_ids=gen_outputs,
                        )

                    if batch_meta.cur_lens.shape[0] == 0:
                        break

                    if cursor < gen_trials and ht_workloads[cursor] <= time.time() - start:
                        cursor += 1
                        new_unfinished_sequences = torch.ones(prepare_output.input_ids.shape[0],
                            dtype=torch.long, device=input_ids.device)
                        new_input_ids = copy.deepcopy(prepare_output.input_ids)
                        new_model_kwargs = copy.deepcopy(prepare_output.model_kwargs)
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

        peft_start_time = time.time()
        step, peft_inputs = next(dataloader_iter)
        peft_inputs.to(torch.cuda.current_device())
        model.set_adapter(adapter_names[0])
        peft_outputs = model(**peft_inputs)

        loss = peft_outputs.loss
        loss.backward()
        if step % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        peft_timings.append(time.time() - peft_start_time)

    total_latency = time.time() - start

    if local_rank != 0:
        return

    # Check lengths
    gen_input_lens = [len(x) for x in gen_inputs.input_ids]
    gen_output_lens = [len(x) for x in gen_outputs]
    assert all(x == prompt_len for x in gen_input_lens)
    assert all(x == prompt_len + gen_len for x in gen_output_lens)
    peft_input_lens = [len(x) for x in peft_inputs.input_ids]
    peft_output_lens = [len(x) for x in peft_outputs.logits]
    assert all(x == seq_len for x in peft_input_lens)
    assert all(x == seq_len for x in peft_output_lens)

    # Log output
    print(f"Summary:")
    print(f"gen_timings = {gen_timings[-3:]}")
    print(f"peft_timings = {peft_timings[-3:]}")

    gen_exec_total_latency = total_latency - sum(peft_timings)
    gen_exec_throughput = gen_trials * batch_size * gen_len / gen_exec_total_latency

    peft_trials = len(peft_timings)
    peft_total_latency = sum(peft_timings)
    peft_throughput = peft_trials * batch_size / peft_total_latency

    gpu_peak_mem = torch.cuda.max_memory_allocated(torch.device("cuda"))
    model_size = utils.model_bytes(config)

    log_str = utils.write_ht_benchmark_log(
        model_size,
        0,
        gpu_peak_mem,
        gen_trials,
        gen_exec_total_latency,
        gen_exec_throughput,
        peft_trials,
        peft_total_latency,
        peft_throughput,
        total_latency,
    )
    print(log_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="meta-llama/Meta-Llama-3-8B", help="model name or path")
    parser.add_argument("--adapter_size", type=int, default=2, help="lora adapters swapping")
    parser.add_argument("--dataset_name", type=str, default="yahma/alpaca-cleaned", help="dataset name or path")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prompt_len", type=int, default=512,  help="prompt length")
    parser.add_argument("--gen_len", type=int, default=32,  help="number of tokens to generate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=256,  help="sequence length")
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")), help="local rank for distributed inference")
    parser.add_argument("--pin_memory", type=int, default=0, help="whether to pinned CPU memory for ZeRO offloading")
    parser.add_argument("--quant_bits", type=int, default=16, help="model weight quantization bits; either 4 or 8")
    parser.add_argument("--quant_group_size", type=int, default=64, help="model weight quantization group size")
    parser.add_argument("--cache_dir", type=str, default="/scratch/yonghe", help="cache dir for model name")
    args = parser.parse_args()

    gc.collect()

    run_ht(
        args.model_name,
        args.adapter_size,
        args.dataset_name,
        args.batch_size,
        args.prompt_len,
        args.gen_len,
        args.gradient_accumulation_steps,
        args.seq_len,
        args.local_rank,
        args.pin_memory,
        args.quant_bits,
        args.quant_group_size,
        args.cache_dir,
    )
