# UniTime: Universal Video Temporal Grounding with Generative Multi-modal Large Language Models
This repository contains the official PyTorch implementation of UniTime: https://arxiv.org/abs/2506.18883.

We have open-sourced our inference code, and expect to gradually open-source the training code!
Please stay tuned! Feel free to reach out for discussions!

<div align="center">
   <img src="./assets/teaser.png">
</div>

## Some Information
[Project Page](https://lzq5.github.io/UniTime/) $\cdot$ [Paper](https://arxiv.org/abs/2506.18883/) $\cdot$ [Model](https://huggingface.co/zeqianli/UniTime)

## News
- [2025.6] We have released Inference code.
- [2025.6] Our pre-print paper is released on arXiv.

## TODO
- [x] Release Paper
- [x] Release Inference Code.
- [ ] Release Code of Data Construction.
- [ ] Release Training and Evaluation Code.

## Requirements
- Python >= 3.10 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch == 2.1.2]
- accelerate == 1.0.1
- transformers == 4.49.0

A suitable [conda](https://conda.io/) environment named `SpatialScore` can be created and activated with:

```
conda env create -f environment.yaml
conda activate SpatialScore
```

## Inference



## Citation
If you use this code and data for your research or project, please cite:

	@article{li2025unitime,
        author    = {Li, Zeqian and Di, Shangzhe and Zhai, Zhonghua and Huang, Weilin and Wang, Yanfeng and Xie, Weidi},
        title     = {Universal Video Temporal Grounding with Generative Multi-modal Large Language Models},
        journal   = {arXiv preprint arXiv:2506.18883},
        year      = {2025},
    }

## Acknowledgements
Many thanks to the code bases from [lmms-finetune](https://github.com/zjysteven/lmms-finetune) and [LamRA](https://github.com/Code-kunkun/LamRA).


## Contact
If you have any questions, please feel free to contact lzq0103@sjtu.edu.cn.