

# ProContEXT: Exploring Progressive Context Transformer for Tracking



### Introduction

In VOT, the ability to predict motion trajectories hinges heavily on context, particularly in scenarios teeming with similar object instances. For instance, consider a situation where objects change rapidly. The traditional VOT methods would falter due to their sole reliance on the target area in the initial frame. ProContEXT, in stark contrast, leverages both temporal (changes over time) and spatial (relative positions) contexts, significantly enhancing tracking precision.

**Fig. 1:** The fast-changing and crowded scenes are omnipresent in visual object tracking. It's evident that harnessing the temporal and spatial context in video sequences is fundamental to precise tracking.


<p align='center'>
  <img src='https://github.com/zhiqic/ProContEXT/assets/65300431/187090d1-cacb-44ec-8a14-1375cbc4fd38' width='678'/>
</p>

**Fig. 2:** A comprehensive visualization of the Progressive Context Encoding Transformer Tracker (ProContEXT) framework.

<p align='center'>
  <img src='https://github.com/zhiqic/ProContEXT/assets/65300431/fe48205c-c811-4671-ae77-cd4f95858182' width='695'/>
</p>

[ProContEXT](https://arxiv.org/abs/2210.15511) showcases state-of-the-art (SOTA) performance across a range of benchmarks.

| Tracker     | GOT-10K (AO) | TrackingNet (AUC) |
|:-----------:|:------------:|:-----------:|
| ProContEXT | 74.6         | 84.6        |



## Getting Started

### Installation
For setting up the environment and data preparation, you can follow instructions from [OSTrack](https://github.com/botaoye/OSTrack).

### Project Configuration
Execute the command below to designate paths for this project:
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

### Model Training
Use models from [MAE](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) as your pre-trained models. Arrange them as:
```
${PROJECT_ROOT}
    -- pretrained_models
      | -- mae_pretrain_vit_base.pth
```
For model training, run:
```shell
python tracking/train.py --script procontext --config procontext_got10k --save_dir ./output --mode multiple --nproc_per_node 4 # GOT-10k model
python tracking/train.py --script procontext --config procontext --save_dir ./output --mode multiple --nproc_per_node 4 # TrackingNet model
```

### Model Testing
Download the trained [model](https://drive.google.com/drive/folders/1d1kKFDoacS67_6mSsWOf3NLEY-Un1gtz?usp=share_link) and organize them as:
```shell
${PROJECT_ROOT}/output/checkpoints/train/procontext/procontext_got10k/ProContEXT_ep0100.pth.tar # GOT-10k model
${PROJECT_ROOT}/output/checkpoints/train/procontext/procontext/ProContEXT_ep0300.pth.tar # TrackingNet model
```
For testing, execute:
```shell
python tracking/test.py procontext procontext_got10k --dataset got10k_test --threads 16 --num_gpus 4
python tracking/test.py procontext procontext --dataset trackingnet --threads 16 --num_gpus 4
```

## Acknowledgment

We owe a significant portion of our implementation to foundational works like [OSTrack](https://github.com/botaoye/OSTrack), [Stark](https://github.com/researchmm/Stark), [pytracking](https://github.com/visionml/pytracking), and [Timm](https://github.com/rwightman/pytorch-image-models). A sincere note of gratitude to their authors for their invaluable contributions. We are also grateful to Alibaba Group's DAMO Academy for their invaluable support.


## Cite Our Work

If our repository aids your research, kindly reference our work:
```bibtex
@inproceedings{lan2023procontext,
  title={Procontext: Exploring progressive context transformer for tracking},
  author={Lan, Jin-Peng and Cheng, Zhi-Qi and He, Jun-Yan and Li, Chenyang and Luo, Bin and Bao, Xu and Xiang, Wangmeng and Geng, Yifeng and Xie, Xuansong},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

## Licensing
The codebase is available under the MIT license. More details can be found in the LICENSE file.
