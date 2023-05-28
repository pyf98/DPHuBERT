# DPHuBERT

This repo contains the code and models for our paper: 

Yifan Peng, Yui Sudo, Shakeel Muhammad, and Shinji Watanabe, “DPHuBERT: Joint Distillation and Pruning of Self-Supervised Speech Models,” in Proc. INTERSPEECH, 2023. (to appear)


## Overview

DPHuBERT is a task-agnostic compression method based on joint distillation and structured pruning. DPHuBERT outperforms previous distillation methods in most tasks of SUPERB. Comprehensive analyses are presented to investigate its performance with less training data or at various sparsity ratios. In addition to HuBERT Base, our method can be directly applied to other speech SSL models such as WavLM and HuBERT Large while still being efficient and effective.

## Citation

```
@inproceedings{dphubert,
    title={{DPHuBERT: Joint Distillation and Pruning of Self-Supervised Speech Models}},
    author={Yifan Peng and Yui Sudo and Shakeel Muhammad and Shinji Watanabe},
    booktitle={Proceedings of the 24th Annual Conference of the International Speech Communication Association (INTERSPEECH)},
    year={2023},
}
```

## Acknowledgments

We thank the authors of the following projects for open-sourcing their code:
- [TorchAudio](https://github.com/pytorch/audio): Our speech SSL models and training pipelines are based on TorchAudio's implementation.
- [FLOP](https://github.com/asappresearch/flop): Our implementation of the Hard Concrete Distribution is modified from FLOP.
- [CoFiPruning](https://github.com/princeton-nlp/CoFiPruning): Some of our training hyper-parameters are inspired by CoFiPruning.
