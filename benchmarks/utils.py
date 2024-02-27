import random
import torch
import time

KB = 1 << 10
MB = 1 << 20
GB = 1 << 30
TB = 1e12


def get_ht_workloads(
    total_latency: float,
    req_gap_bound: float,
):
    random.seed(2024)

    workloads = [0]
    while True:
        req_gap = random.uniform(0, req_gap_bound)
        if len(workloads):
            # avoid the total latency is larger than the setup
            if workloads[-1] + req_gap  > total_latency - req_gap_bound:
                break
            workloads.append(workloads[-1] + req_gap)

    return workloads

def get_quant_config(model_config, quant_bits: int, quant_group_size: int):
    qaunt_config = {
        'weight_quantization': {
            'quantized_initialization' : {
                'num_bits': quant_bits,
                'group_size': quant_group_size,
                "group_dim": 1,
                "symmetric": False
            }
        }
    }

    return qaunt_config

def model_bytes(config):
    h = config.hidden_size
    return 	2 * (config.num_hidden_layers * (
    # config-attention
    h * (3 * h + 1) + h * (h + 1) +
    # mlp
    h * (4 * h + 1) + h * 4 * (h + 1) +
    # layer norm
    h * 4) +
    # embedding
    config.vocab_size * (h + 1))

def cache_bytes(config, batch_size, seq_len):
    return 2 * batch_size * seq_len * config.num_hidden_layers * config.hidden_size * 2

def write_gen_benchmark_log(model_size, cache_size, gpu_peak_mem,
        prefill_latency, prefill_throughput,
        decode_latency, decode_throughput,
        total_latency, total_throughput):

    log_str = (f"model size: {model_size/GB:.3f} GB\t"
               f"cache size: {cache_size/GB:.3f} GB\t"
               f"peak gpu mem: {gpu_peak_mem / GB:.3f} GB\n"
               f"prefill latency: {prefill_latency:.3f} s\t"
               f"prefill throughput: {prefill_throughput:.3f} token/s\n"
               f"decode latency: {decode_latency:.3f} s\t"
               f"decode throughput: {decode_throughput:.3f} token/s\n"
               f"total latency: {total_latency:.3f} s\t"
               f"total throughput: {total_throughput:.3f} token/s")

    return log_str

def write_peft_benchmark_log(model_size, activation_size, gpu_peak_mem,
        forward_latency, forward_throughput,
        backward_latency, backward_throughput,
        total_latency, total_throughput):

    log_str = (f"model size: {model_size/GB:.3f} GB\t"
               f"activation size: {activation_size/GB:.3f} GB\t"
               f"peak gpu mem: {gpu_peak_mem / GB:.3f} GB\n"
               f"forward latency: {forward_latency:.3f} s\t"
               f"forward throughput: {forward_throughput:.3f} sample/s\n"
               f"backward latency: {backward_latency:.3f} s\t"
               f"backward throughput: {backward_throughput:.3f} sample/s\n"
               f"total latency: {total_latency:.3f} s\t"
               f"total throughput: {total_throughput:.3f} sample/s")

    return log_str


def write_ht_benchmark_log(model_size, activation_size, gpu_peak_mem,
        gen_trials, gen_exec_total_latency, gen_exec_throughput,
        peft_trials, peft_total_latency, peft_throughput, total_latency):

    log_str = (f"model size: {model_size/GB:.3f} GB\t"
               f"activation size: {activation_size/GB:.3f} GB\t"
               f"peak gpu mem: {gpu_peak_mem / GB:.3f} GB\n"
               f"gen trials: {gen_trials}\t"
               f"gen exec total latency: {gen_exec_total_latency:.3f} s\t"
               f"gen exec throughput: {gen_exec_throughput:.3f} token/s\n"
               f"peft trials: {peft_trials}\t"
               f"peft total latency: {peft_total_latency:.3f} s\t"
               f"peft throughput: {peft_throughput:.3f} sample/s\n"
               f"total latency: {total_latency:.3f} s")

    return log_str

# add timing hooks
def add_model_hooks(model: torch.nn.Module):

    def start_time_hook(module, input):
        if hasattr(module, 'stage') and module.stage == "decode":
            return
        elif hasattr(module, 'stage') and module.stage == 'prefill':
            torch.cuda.synchronize()
            module.__start_time__ = time.time()

    def end_time_hook(module, input, output):
        if hasattr(module, 'stage') and module.stage == "decode":
            return
        elif hasattr(module, 'stage') and module.stage == 'prefill':
            torch.cuda.synchronize()
            module.__duration__ = time.time() - module.__start_time__
            module.stage = "decode"

    if not hasattr(model, '__start_time_hook_handle'):
        model.__start_time_hook_handle__ = model.register_forward_pre_hook(
            start_time_hook, )

    if not hasattr(model, '__end_time_hook_handle__'):
        model.__end_time_hook_handle__ = model.register_forward_hook(
            end_time_hook, )

# remove timing hooks
def remove_model_hooks(module):
    if hasattr(module, "__start_time_hook_handle__"):
        module.__start_time_hook_handle__.remove()
        del module.__start_time_hook_handle__
    if hasattr(module, "__end_time_hook_handle__"):
        module.__end_time_hook_handle__.remove()
        del module.__end_time_hook_handle__
    if hasattr(module, "stage"):
        del module.stage
    if hasattr(module, "__duration__"):
        del module.__duration__

