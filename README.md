
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
  - [### System Requirements](#-system-requirements)
  - [### Necessary Packages and Libraries](#-necessary-packages-and-libraries)
  - [Expected Directory Structure (TODO)](#expected-directory-structure-todo)
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
## Expected Directory Structure (TODO)
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── kitti
│   │   ├── annotations
│   │   ├── $path to images$

---
## Training (TODO)

---
## Testing (TODO)

---
## Result (TODO)