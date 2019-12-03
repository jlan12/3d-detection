
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
<<<<<<< HEAD
      - [System Requirements](#-system-requirements)
      - [Necessary Packages and Libraries](#-necessary-packages-and-libraries)
  - [Directory Structure (TODO)](#expected-directory-structure-todo)
=======
  - [### System Requirements](#-system-requirements)
  - [### Necessary Packages and Libraries](#-necessary-packages-and-libraries)
  - [Expected Directory Structure (TODO)](#expected-directory-structure-todo)
>>>>>>> 23f6fe88cb9296b57933261354fc809ceef8ff37
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
<<<<<<< HEAD
    tensorflow-gpu is preferred
3. Install [Pytorch](https://pytorch.org)
4. Install the mmdetection requirements:[mmcv](https://pypi.org/project/mmcv/)
=======
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
>>>>>>> 23f6fe88cb9296b57933261354fc809ceef8ff37
5. Install Image Processing Libraries
    ```bash
    pip install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely
    pip install imgaug
    ```
<<<<<<< HEAD
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
=======

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
>>>>>>> 23f6fe88cb9296b57933261354fc809ceef8ff37
## Result (TODO)