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

FineInfer
```
CUDA_VISIBLE_DEVICES=0 python fi-gen.py -m meta-llama/Llama-2-7b-hf --batch_size 1
```

FineInfer-Offload
```
```

FineInfer-Heterogeneous
```
CUDA_VISIBLE_DEVICES=0 python baseline-ht.py -m meta-llama/Llama-2-7b-hf --batch_size 1
CUDA_VISIBLE_DEVICES=0 python fi-ht.py -m meta-llama/Llama-2-7b-hf --batch_size 1
deepspeed --num_gpus 1 baseline-offload-ht.py -m meta-llama/Llama-2-13b-hf --batch_size 1 --cpu_offload
deepspeed --num_gpus 1 fi-offload-ht.py -m meta-llama/Llama-2-13b-hf --batch_size 1 --cpu_offload
```

FineInfer-overhead
```
CUDA_VISIBLE_DEVICES=0 python fg-overhead-measurement.py -m meta-llama/Llama-2-7b-hf
```
