```
conda create -n FineInfer python=3.12
conda activate FineInfer
pip install -r requirements.txt
```

FineInfer-inference
```
CUDA_VISIBLE_DEVICES=0 python fi-gen.py -m meta-llama/Meta-Llama-3-8B --batch_size 1
```

FineInfer-heterogeneous
```
CUDA_VISIBLE_DEVICES=0 python baseline-ht.py -m meta-llama/Meta-Llama-3-8B --batch_size 1
CUDA_VISIBLE_DEVICES=0 python fi-ht.py -m meta-llama/Meta-Llama-3-8B --batch_size 1
```
