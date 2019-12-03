
3D object detection 
===
## Description (TODO)

---
## Table of Content
- [3D object detection](#3d-object-detection)
  - [Description (TODO)](#description-todo)
  - [Table of Content](#table-of-content)
  - [Dataset](#dataset)
  - [Prerequisites](#prerequisites)
      - [System Requirements](#-system-requirements)
      - [Necessary Packages and Libraries](#-necessary-packages-and-libraries)
  - [Directory Structure (TODO)](#expected-directory-structure-todo)
  - [Training (TODO)](#training-todo)
  - [Testing (TODO)](#testing-todo)
  - [Result (TODO)](#result-todo)

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
    tensorflow-gpu is preferred
3. Install [Pytorch](https://pytorch.org)
4. Install the mmdetection requirements:[mmcv](https://pypi.org/project/mmcv/)
5. Install Image Processing Libraries
    ```bash
    pip install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely
    pip install imgaug
    ```
6. Install [pycoco tools](https://pypi.org/project/pycocotools/)

---
## Directory Structure (TODO)
|
|
|
|

---
## Training
### For the alpha prediction model:
```bash
cd full_image_alpha_model
python3 driver.py
```
### For the dimension prediction model:
```bash
cd dim_model
python3 driver_dims.py
```

---
## Inferencing 
### For the alpha prediction model:
```bash
python3 driver.py detect <checkpoint> <output directory>
```
### For the dimension prediction model:
```bash
#TODO
```
---
## Result (TODO)