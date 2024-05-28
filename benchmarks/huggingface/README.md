```
conda create -n FineInfer python=3.12
conda activate FineInfer
pip install -r requirements.txt
```

HuggingFace
```
CUDA_VISIBLE_DEVICES=0 python hf-gen.py -m meta-llama/Meta-Llama-3-8B --batch_size 1
CUDA_VISIBLE_DEVICES=0 python hf-peft-gen.py -m meta-llama/Meta-Llama-3-8B --batch_size 1
CUDA_VISIBLE_DEVICES=0 python hf-peft.py -m meta-llama/Meta-Llama-3-8B --batch_size 1
```
