# Cross-Modality Time-Variant Relation Learning for Generating Dynamic Scene Graphs

[paper](https://arxiv.org/abs/2305.08522) | [video](https://youtu.be/RrL-AwcOBLw) | [slides](https://docs.google.com/presentation/d/1qM7DFlgufzBTr3B76X1EQZs6A8oYGrKJ/edit?usp=sharing&ouid=117596268568819876341&rtpof=true&sd=true)

This repository contains the code implementation for the paper "Cross-Modality Time-Variant Relation Learning for Generating Dynamic Scene Graphs" accepted by ICRA 2023.

## Introduction

Dynamic scene graphs generated from video clips could help enhance the semantic visual understanding in a wide range of challenging tasks. In the process of temporal and spatial modeling during dynamic scene graph generation, it is particularly intractable to learn time-variant relations in dynamic scene graphs among frames. In this paper, we propose a Time-variant Relation-aware TRansformer (TR<sup>2</sup>), which aims to model the temporal change of relations in dynamic scene graphs. Extensive experiments on the Action Genome dataset prove that our TR<sup>2</sup>can effectively model the time-variant relations. TR<sup>2</sup> significantly outperforms previous state-of-the-art methods under two different settings by 2.1% and 2.6% respectively.

![](overall_v3.png)

## Getting Started

To get started with the code and reproduce the results presented in the paper, follow the steps below:

1. Clone this repository:
```
git clone https://github.com/qncsn2016/TR2.git
```

2. Environment:

We use Python 3.7, PyTorch 1.10, and torchvision 0.11. Install the required dependencies:

```
pip install -r requirements.txt
```

3. Dataset:

Our experiments are conducted on the [Action Genome (AG)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ji_Action_Genome_Actions_As_Compositions_of_Spatio-Temporal_Scene_Graphs_CVPR_2020_paper.pdf) dataset, which is the benchmark dataset of dynamic scene graph generation. Download and process the dataset according to the official repository of [AG](https://github.com/JingweiJ/ActionGenome). Please modify the `data_path` in the config.

Following [STTran](https://github.com/yrcong/STTran), we keep bounding boxes with short edges larger than 16 pixels for SGCls and SGDet tasks. Please download the file [object_bbox_and_relationship_filtersmall.pkl](https://drive.google.com/file/d/19BkAwjCw5ByyGyZjFo174Oc3Ud56fkaT/view?usp=sharing) and put it in the ```dataloader```.

4. We borrow some compiled code for bbox operations:
```
cd lib/draw_rectangles
python setup.py build_ext --inplace
cd ..
cd fpn/box_intersections_cpu
python setup.py build_ext --inplace
```

For the object detector part, please follow the compilation from https://github.com/jwyang/faster-rcnn.pytorch. Following [STTran](https://github.com/yrcong/STTran), we use the pretrained FasterRCNN model for Action Genome. Please download [here](https://drive.google.com/file/d/1-u930Pk0JYz3ivS6V_HNTM1D5AxmN5Bs/view?usp=sharing) and put it in 
```
fasterRCNN/models/faster_rcnn_ag.pth
```

For the cross-modality guidance module, please download the [ViT-B/32](https://github.com/openai/CLIP) model and put it in
```
lib/models/clip/ViT-B-32.pt
```
If you want to accelerate the training speed, we recommend downloading [the precomputed features](https://drive.google.com/file/d/12UB12Btac0WMV9sZ9o6vxn3hC_QBXpS6/view?usp=sharing) and modifying the `pre_path` in the config to the corresponding download path.

We borrowed some code from [STTran](https://github.com/yrcong/STTran).

## Train
```
# PredCls
python train.py --mode predcls
# SgCls
python train.py --mode sgcls
# SgDet
python train_amp.py
```

## Evaluation
```
python test.py --mode mode --model_path path_to_ckpt
```

## Results
|  setting  | ckpt | With <br> R@20 | No <br> R@20 | Top 6 <br> R@20 |
|:-------:|:----:|:--------:|:-----:|:----:|
| PredCls | [link](https://drive.google.com/file/d/13InLQEeT_nXy5zFEq1QMI06KOErPvV3K/view?usp=drive_link) | 73.8  | 96.6  | 93.5  |
| SgCls   | [link](https://drive.google.com/file/d/1m7yGWaJRnk91A1R0gmE0_RRdY03qbbjH/view?usp=sharing) | 48.7  | 64.4  | 62.4  |
| SgDet   | [link](https://drive.google.com/file/d/1ihnFp8GPhciAIX0sQ4j9LbUD6pBrTrlw/view?usp=sharing) | 35.5  | 39.2  | 39.1  |



## Citation

If you find this work useful in your research, please consider citing our paper:

```
@inproceedings{tr2,
    title = {Cross-Modality Time-Variant Relation Learning for Generating Dynamic Scene Graphs},
    author = {Jingyi Wang, Jinfa Huang, Can Zhang, and Zhidong Deng},
    booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
    year = {2023}
}
```