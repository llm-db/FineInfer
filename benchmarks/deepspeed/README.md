```
conda create -n FineInfer python=3.11
conda activate FineInfer
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
conda install -c huggingface transformers
conda install scipy
pip install bitsandbytes peft
pip install deepspeed
conda deactivate
```

ZeRO
```
deepspeed --num_gpus 1 zero-gen.py -m meta-llama/Llama-2-7b-hf --batch_size 1
deepspeed --num_gpus 1 zero-peft-gen.py -m meta-llama/Llama-2-7b-hf --batch_size 1
deepspeed --num_gpus 1 zero-peft.py -m meta-llama/Llama-2-7b-hf --batch_size 1
```

ZeRO-Offload
```
deepspeed --num_gpus 1 zero-offload-gen.py -m meta-llama/Llama-2-7b-hf --batch_size 1 --cpu_offload
deepspeed --num_gpus 1 zero-offload-peft-gen.py -m meta-llama/Llama-2-7b-hf --batch_size 1 --cpu_offload
deepspeed --num_gpus 1 zero-offload-peft.py -m meta-llama/Llama-2-7b-hf --batch_size 1 --cpu_offload
```

ZeRO-Heterogeneous
```
deepspeed --num_gpus 1 zero-ht.py -m meta-llama/Llama-2-7b-hf --batch_size 1
deepspeed --num_gpus 1 zero-offload-ht.py -m meta-llama/Llama-2-13b-hf --batch_size 1 --cpu_offload
```

ZeRO-overhead
```
deepspeed --num_gpus 1 zero-overhead-measurement.py -m meta-llama/Llama-2-7b-hf
```
