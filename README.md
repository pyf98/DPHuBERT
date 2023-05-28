# DPHuBERT

This repo contains the code and models for our paper: 

Yifan Peng, Yui Sudo, Shakeel Muhammad, and Shinji Watanabe, “DPHuBERT: Joint Distillation and Pruning of Self-Supervised Speech Models,” in Proc. INTERSPEECH, 2023. (to appear)


## Overview

DPHuBERT is a task-agnostic compression method based on joint distillation and structured pruning. DPHuBERT outperforms previous distillation methods in most tasks of SUPERB. Comprehensive analyses are presented to investigate its performance with less training data or at various sparsity ratios. In addition to HuBERT Base, our method can be directly applied to other speech SSL models such as WavLM and HuBERT Large while still being efficient and effective.


## Requirements


## Train DPHuBERT

### 1. Download and prepare audio data

The following script creates file lists for LibriSpeech in tsv format. `LibriSpeech_PATH` is the path to the downloaded raw data.

```bash
python prepare_data.py --data LibriSpeech_PATH --out data/librispeech
```

The output directory has this structure:

```
data
└── librispeech
    ├── train100.tsv
    ├── train960.tsv
    └── valid.tsv
```

### 2. Download pre-trained SSL (e.g., HuBERT Base) and convert it to our format

We need to download pre-trained SSL checkpoints from fairseq or Hugging Face and then convert them to our own format. These models will be used as the teacher for compression. For example, we can obtain HuBERT Base by executing:

```bash
mkdir -p pretrained
python convert_hubert_from_hf.py
```

The converted checkpoint will be saved as `pretrained/hubert-base-ls960.hf.pth`. The output path can be changed in the python script.

### 3. 

## Citation

Please cite our paper if you use DPHuBERT.

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
- [TorchAudio](https://github.com/pytorch/audio): Our speech SSL models and training pipelines are based on TorchAudio.
- [FLOP](https://github.com/asappresearch/flop): Our implementation of the Hard Concrete Distribution is modified from FLOP.
- [CoFiPruning](https://github.com/princeton-nlp/CoFiPruning): Some of our training hyper-parameters follow CoFiPruning.
