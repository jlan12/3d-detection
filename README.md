
3D object detection 
===
## Description

---
## Table of Content
- [3D object detection](#3d-object-detection)
  - [Description](#description)
  - [Table of Content](#table-of-content)
  - [Dataset](#dataset)
  - [Prerequisites](#prerequisites)
  - [### System Requirements](#-system-requirements)
  - [### Necessary Packages and Libraries](#-necessary-packages-and-libraries)
  - [Expected Directory Structure](#expected-directory-structure)
  - [Training](#training)
  - [Testing](#testing)
  - [Result](#result)

---

## Dataset
KITII dataset: http://www.cvlibs.net/datasets/kitti/eval_3dobject.php

---

## Prerequisites

### System Requirements
---
* Linux
* Python 3.5.3
* CUDA 10.0.130
* Tensorflow 2.2.4 (?)
* Pytorch 1.4.0


### Necessary Packages and Libraries
---
1. Install [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
2. Install [TensorFlow](https://www.tensorflow.org/install/)
3. Install **Keras**
    ```bash
    sudo pip install keras
    ```
    or
    ```bash
    git clone https://github.com/keras-team/keras.git
    cd keras
    sudo python setup.py install
    ```
4. Install **mmcv** and **pytorch** 

    ```bash
    pip install mmcv(*** Richard HELP!!!!!! ***)
    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
    ```

    Run the following commands in mmdetection to finish installation
    ```bash
    ./compile.sh
    python setup.py develop
    # 'pip install -e' works too apparently
    ```
5. Install Image Processing Libraries
    ```bash
    pip install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely
    pip install imgaug
    ```

---
## Expected Directory Structure
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── kitti
│   │   ├── annotations
│   │   ├── $path to images$

---
## Training 

The config file is located at configs/cascade_rcnn_x101_64x4d_fpn_1x_kitti.py

The config file was modified to provide the following information:
* bbox_head/num_classes=10, # kitti has 9 classes, so 9 + 1
* img_norm_cfg=... # provide the mean and std of each channel in BGR format from the kitti data for normalization (use the script at ??? to get the mean and std)
* imgs_per_gpu=4     # batch size for training 
* workers_per_gpu=4  # match batch size
* dataset_type = 'KittiDataset' # this refers to datasets/kitti.py
* data_root = 'data/kitti/' # data directory, everything is relative to the mmdetection
* ann_file=data_root + 'annotations/instances_runtrain.json', # path to the train labels in coco format
* img_prefix= data_root + 'training/image_2', # path to the image directory

optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0001) #manually adjust to lower learning rate as it begins training begins to stagnate 

Command line to start training:
``` bash
./tools/dist_train.sh configs/cascade_rcnn_x101_64x4d_fpn_1x_kitti.py ${GPU_NUM} --validate 
```
where ${GPU_NUM} is the number of gpus to train on.

---
## Testing 

---
## Result