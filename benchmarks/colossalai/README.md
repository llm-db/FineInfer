```
conda create -n FineInfer python=3.11
conda activate FineInfer
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
conda install -c huggingface transformers
conda install scipy
pip install bitsandbytes peft
pip install colossalai
conda deactivate
```

ColossalAI
```
CUDA_VISIBLE_DEVICES=0 python colossalai-gen.py -m meta-llama/Llama-2-7b-hf --batch_size 1
CUDA_VISIBLE_DEVICES=0 python colossalai-peft-gen.py -m meta-llama/Llama-2-7b-hf --batch_size 1
CUDA_VISIBLE_DEVICES=0 python colossalai-peft.py -m meta-llama/Llama-2-7b-hf --batch_size 1
```

ColossalAI-Gemini
```
CUDA_VISIBLE_DEVICES=0 python colossalai-offload-gen.py -m meta-llama/Llama-2-7b-hf --batch_size 1 --cpu_offload
CUDA_VISIBLE_DEVICES=0 python colossalai-offload-peft-gen.py -m meta-llama/Llama-2-7b-hf --batch_size 1 --cpu_offload
CUDA_VISIBLE_DEVICES=0 python colossalai-offload-peft.py -m meta-llama/Llama-2-7b-hf --batch_size 1 --cpu_offload
```

ColossalAI-Heterogeneous
```
CUDA_VISIBLE_DEVICES=0 python colossalai-ht.py -m meta-llama/Llama-2-7b-hf --batch_size 1
CUDA_VISIBLE_DEVICES=0 python colossalai-offload-ht.py -m meta-llama/Llama-2-13b-hf --batch_size 1 --cpu_offload
```

ColossalAI-overhead
```
colossalai run --nproc_per_node 1 --master_port 29502 colossalai-overhead-measurement.py 
```
