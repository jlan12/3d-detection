
3D Object Detection 
===
## Description
3D object detection is our First-year Innocation and Research Experience(FIRE) Capital One Machine Learning (COML) project, under the guidance of our research educator Dr. Raymond Tu. This project was created contibuted to equally by Richard Gao, Jerry Lan, Vladimir Leung, Siyuan Peng.

We attempt to break the task of monocular 3d object detection into three parts: predicting the angle, the dimention and the location. We were succcessfully able to predict angle and had some progress in predicting dimension.

---
## Table of Contents
- [3D object detection](#3d-object-detection)
  - [Description ](#description-todo)
  - [Table of Contents](#table-of-contents)
  - [Dataset](#dataset)
  - [Prerequisites](#prerequisites)
      - [System Requirements](#-system-requirements)
      - [Necessary Packages and Libraries](#-necessary-packages-and-libraries)
  - [Directory Structure ](#expected-directory-structure-todo)
  - [Training ](#training-todo)
  - [Testing ](#testing-todo)
  - [Result ](#result-todo)

---

## Dataset
KITTI dataset: http://www.cvlibs.net/datasets/kitti/eval_3dobject.php

---

## Prerequisites

### System Requirements
---
* Linux
* Python 3.5+
* CUDA 10
* Tensorflow-gpu
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
7. (Optional:) Run dependency_checker.ipynb to check

---
## Directory Structure 
<pre>
├── Mask_RCNN (git submodule)
├── dependency_checker.ipynb
├── dim_model
│   ├── ckpts/ : For the model weights
│   ├── driver_dims.py : Driver to run the dimension model
│   ├── inputs/ -> (can be symlinked to full_img_alpha_model/inputs)
│   ├── prep_dim.py: Preprocessor for dimension model
│   ├── util_dim.py: Miscellaneous Utilities for dimension model 
│   └── xmodel_dim.py : Modifiled XceptionNet
│
├── full_img_alpha_model
│   ├── Augmentation.ipynb
│   ├── box_3d_iou.py: Utility class for visualization
│   ├── ckpts/: For the model weights 
│   ├── driver.py: Driver used to train and inference model
│   ├── inputs/ 
│   │   ├── instance_segs/: Directory containing segmentation for instances
│   │   ├── train_data.json: Annotations for training
│   │   └── train_segs.json: Full segmentations by image
│   │   
│   ├── output_visualzer.ipynb
│   ├── preprocessor.py: Preprocessor for angle prediction
│   ├── Seg_Associate.ipynb: Identify segmentation for given instance
│   ├── utilities.py: Utilities used for pre/postprocessing 
│   ├── val_output/: Results from inferencing on validation set
│   ├── visualization.ipynb
│   ├── visualization.py: Visualization utility functions
│   └── xmodel.py: Modified XceptionNet model architecture
│
├── Mask_RCNN (git module)
├── mmdetection (git module)
├── rotcatgen.ipynb (used to modify json to include rotation categories)
├── struct2depth (git module)
└── README.md
</pre>

---
## Training
### For the angle prediction model:
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
### For the angle prediction model:
#### Outputs as JSON file
```bash
python3 driver.py detect <checkpoint> <output file path>
```
### For the dimension prediction model:
```bash
#TODO
```
---
## Training results
<details>
<summary>For the alpha model</summary>

|||
--|--
![](https://i.imgur.com/k7dfn4k.png)|![](https://i.imgur.com/IxPuhZd.png)
</details>
<details>
<summary>For the dimension model</summary>

|||
--|--
![](https://i.imgur.com/ewvd5nZ.png)|![](https://i.imgur.com/naKTAtq.png)
</details>
## Visualization results

### The alpha prediction
|||
--|--
![](https://i.imgur.com/AYvdRx7.png)|![](https://i.imgur.com/BW02PQi.png)

---
## Conclusion 

From our alpha prediction results, it appears that the model produces sufficient results for a mostly accurate prediction of the alpha. On the validation set, it had over a .9 accuracy, which is very statistically significant as we predicted on 32 categories. 
On the dimension model, we had more varying results as the model had significant issues with overfitting.

---
## Reference
- ### [struct2depth](https://github.com/tensorflow/models/tree/master/research/struct2depth)
- ### [Mask_RCNN](https://github.com/matterport/Mask_RCNN)
- ### [mmdetection](https://github.com/open-mmlab/mmdetection)