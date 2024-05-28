<h1 align="center">
FineInfer
</h1>

<p align="center">
| <a href="https://dl.acm.org/doi/10.1145/3642970.3655835"><b>Paper</b></a> |
</p>

FineInfer is a research prototype for fine-tuning and serving large language models.

FineInfer supports concurrent parameter-efficient fine-tuning and inference through the following features:
* Deferred continuous batching
* Hybrid system architecture
* Heterogeneous batching

## Get Started
[Installation and examples](https://github.com/llm-db/FineInfer/tree/main/benchmarks/fineinfer)

The current version removes some previous features and functionalities. If you need them, please download [previous versions](https://github.com/llm-db/FineInfer/releases).

## Citation
```
@inproceedings{FineInfer,
  author = {He, Yongjun and Lu, Yao and Alonso, Gustavo},
  title = {Deferred Continuous Batching in Resource-Efficient Large Language Model Serving},
  year = {2024},
  booktitle = {Proceedings of the 4th Workshop on Machine Learning and Systems},
  pages = {98–106},
  series = {EuroMLSys '24}
}
```
