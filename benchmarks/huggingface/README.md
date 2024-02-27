```
conda create -n FineInfer python=3.11
conda activate FineInfer
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
conda install -c huggingface transformers
conda install -c conda-forge accelerate
conda install scipy
pip install bitsandbytes peft
conda deactivate
```

HuggingFace
```
CUDA_VISIBLE_DEVICES=0 python hf-gen.py -m meta-llama/Llama-2-7b-hf --batch_size 1
CUDA_VISIBLE_DEVICES=0 python hf-peft-gen.py -m meta-llama/Llama-2-7b-hf --batch_size 1
CUDA_VISIBLE_DEVICES=0 python hf-peft.py -m meta-llama/Llama-2-7b-hf --batch_size 1
```

HuggingFace-Offload
```
CUDA_VISIBLE_DEVICES=0 python hf-offload-gen.py -m meta-llama/Llama-2-7b-hf --batch_size 1 --cpu_offload
CUDA_VISIBLE_DEVICES=0 python hf-offload-gen.py -m meta-llama/Llama-2-7b-hf --batch_size 1 --disk_offload
CUDA_VISIBLE_DEVICES=0 python hf-offload-peft-gen.py -m meta-llama/Llama-2-7b-hf --batch_size 1 --cpu_offload
CUDA_VISIBLE_DEVICES=0 python hf-offload-peft-gen.py -m meta-llama/Llama-2-7b-hf --batch_size 1 --disk_offload
```
