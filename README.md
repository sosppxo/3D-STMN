# 3D-STMN
NEWS:🔥3D-STMN is accepted at AAAI 2024!🔥

🔥This branch is for end-to-end training (about 31G of GPU RAM is needed). To save the GPU RAM by preprocessing features before training, please switch to the [feat branch](https://github.com/sosppxo/3D-STMN/tree/feat) (only 7G of GPU RAM is needed for training).🔥

[3D-STMN: Dependency-Driven Superpoint-Text Matching Network for End-to-End 3D Referring Expression Segmentation](https://arxiv.org/abs/2308.16632)

Changli Wu, Yiwei Ma, Qi Chen, Haowei Wang, Gen Luo, Jiayi Ji*, Xiaoshuai Sun

<img src="docs\3D-STMN.png"/>

## Introduction

​In 3D Referring Expression Segmentation (3D-RES), the earlier approach adopts a two-stage paradigm, extracting segmentation proposals and then matching them with referring expressions. However, this conventional paradigm encounters significant challenges, most notably in terms of the generation of lackluster initial proposals and a pronounced deceleration in inference speed.
Recognizing these limitations, we introduce an innovative end-to-end Superpoint-Text Matching Network (3D-STMN) that is enriched by dependency-driven insights. One of the keystones of our model is the Superpoint-Text Matching (STM) mechanism. Unlike traditional methods that navigate through instance proposals, STM directly correlates linguistic indications with their respective superpoints, clusters of semantically related points. This architectural decision empowers our model to efficiently harness cross-modal semantic relationships, primarily leveraging densely annotated superpoint-text pairs, as opposed to the more sparse instance-text pairs.
In pursuit of enhancing the role of text in guiding the segmentation process, we further incorporate the Dependency-Driven Interaction (DDI) module to deepen the network's semantic comprehension of referring expressions. Using the dependency trees as a beacon, this module discerns the intricate relationships between primary terms and their associated descriptors in expressions, thereby elevating both the localization and segmentation capacities of our model.
Comprehensive experiments on the ScanRefer benchmark reveal that our model not only set new performance standards, registering an mIoU gain of 11.7 points but also achieve a staggering enhancement in inference speed, surpassing traditional methods by 95.7 times.

## Installation

Requirements

- Python 3.7 or higher
- Pytorch 1.12
- CUDA 11.3 or higher

The following installation suppose `python=3.8` `pytorch=1.12.1` and `cuda=11.3`.
- Create a conda virtual environment

  ```
  conda create -n 3d-stmn python=3.8
  conda activate 3d-stmn
  ```

- Clone the repository

  ```
  git clone https://github.com/sosppxo/3D-STMN.git
  ```

- Install the dependencies

  Install [Pytorch 1.12.1](https://pytorch.org/)

  ```
  pip install spconv-cu113
  conda install pytorch-scatter -c pyg
  pip install -r requirements.txt
  ```

  Install segmentator from this [repo](https://github.com/Karbo123/segmentator) (We wrap the segmentator in ScanNet).
  
  Install Stanford CoreNLP toolkit from the [official website](https://stanfordnlp.github.io/CoreNLP/download.html).

- Setup, Install stmn and pointgroup_ops.

  ```
  sudo apt-get install libsparsehash-dev
  python setup.py develop
  cd stmn/lib/
  python setup.py develop
  ```

## Data Preparation

### ScanNet v2 dataset

Download the [ScanNet](http://www.scan-net.org/) v2 dataset.

Put the downloaded `scans` folder as follows.

```
3D-STMN
├── data
│   ├── scannetv2
│   │   ├── scans
```

Split and preprocess point cloud data

```
cd data/scannetv2
bash prepare_data.sh
```

The script data into train/val folder and preprocess the data. After running the script the scannet dataset structure should look like below.

```
3D-STMN
├── data
│   ├── scannetv2
│   │   ├── scans
│   │   ├── train
│   │   ├── val
```

### ScanRefer dataset
Download [ScanRefer](https://github.com/daveredrum/ScanRefer) annotations following the instructions.

Put the downloaded `ScanRefer` folder as follows.
```
3D-STMN
├── data
│   ├── ScanRefer
│   │   ├── ScanRefer_filtered_train.json
│   │   ├── ScanRefer_filtered_val.json
```
Preprocess textual data
```
python data/features/save_graph.py --split train --data_root data/ --max_len 78
python data/features/save_graph.py --split val --data_root data/ --max_len 78
```

## Pretrained Backbone

Download [SPFormer](https://stuxmueducn-my.sharepoint.com/:f:/g/personal/22920182204313_stu_xmu_edu_cn/Em7yJHaCHAxFpM15uVwk9cgByDp-67lWQg59vkU-zokHYA?e=IuZV0D) pretrained model (We only use the Sparse 3D U-Net backbone for training).

Move the pretrained model to backbones.
```
mkdir backbones
mv ${Download_PATH}/sp_unet_backbone.pth backbones/
```

## Training
For single GPU (32G):

```
bash scripts/train.sh
```
For multi-GPU (11G * 4 or 24G * 2):
```
bash scripts/train_multi_gpu.sh
```

## Inference

Download [3D-STMN](https://stuxmueducn-my.sharepoint.com/:f:/g/personal/22920182204313_stu_xmu_edu_cn/Em7yJHaCHAxFpM15uVwk9cgByDp-67lWQg59vkU-zokHYA?e=IuZV0D) pretrain model and move it to checkpoints.

```
bash scripts/test.sh
```

## Citation

If you find this work useful in your research, please cite:

```
@misc{2308.16632,
Author = {Changli Wu and Yiwei Ma and Qi Chen and Haowei Wang and Gen Luo and Jiayi Ji and Xiaoshuai Sun},
Title = {3D-STMN: Dependency-Driven Superpoint-Text Matching Network for End-to-End 3D Referring Expression Segmentation},
Year = {2023},
Eprint = {arXiv:2308.16632},
}
```

## Ancknowledgement

Sincerely thanks for [SoftGroup](https://github.com/thangvubk/SoftGroup) [SSTNet](https://github.com/Gorilla-Lab-SCUT/SSTNet) and [SPFormer](https://github.com/sunjiahao1999/SPFormer) repos. This repo is build upon them.

