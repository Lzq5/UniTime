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
- PyTorch == 2.1.2
- accelerate == 1.0.1
- transformers == 4.49.0
- peft == 0.14.0

## Inference

1. **Download Model Checkpoints**  
   - Obtain the pretrained checkpoints from [Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) and [UniTime](https://huggingface.co/zeqianli/UniTime).  
   - Set the `model_local_path` to your local path for Qwen2-VL-7B, and `model_finetune_path` to your UniTime checkpoint.

2. **Prepare Input Data**  
   - Create a JSON file for inference in the following format, and specify its path via the `data_path` argument:
   ```json
   [
       {
           "qid": 0, 
           "id": "3MSZA", 
           "annos": [
               {
                   "query": "person turn a light on.",
                   "window": [[24.3, 30.4]]
               }
           ],
           "duration": 30.96,
           "video_path": "./videos/3MSZA.mp4"
       }
   ]
   ```

3. **Run Inference**  
   - Execute the following command to perform inference. The output results will be saved in the `results/` directory.
   ```bash
   python inference.py --model_local_path path_to_qwen2vl7B \
       --model_finetune_path ckpt/unitime \
       --data_path data/test.json
   ```



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