
3D Object Detection 
===
## Description 
3D object detection is our First-year Innocation and Research Experience (FIRE) Capital One Machine Learning (COML) project, under the guidance of our research educator Dr. Raymond Tu. This project was created contibuted to equally by Richard Gao, Jerry Lan, Vladimir Leung, Siyuan Peng.

We attempt to seperate the task of monocular 3d object detection into three parts: predicting the angle, the dimention and the location. We were succcessfully able to predict angle and had some progress in predicting dimension.

For the rotation prediction model, we were inspired by GS3d to formulate the question as a categorical problem instead of a regression problem. We created our own loss function to handle close angles.

Our primary contribution is in angle detection, for which we present a novel input scheme, classification formulation, and quality aware loss function.

---
## Table of Contents
- [3D object detection](#3d-object-detection)
  - [Description ](#description-todo)
  - [Table of Contents](#table-of-contents)
  - [Dataset](#Dataset)
  - [Prerequisites](#Prerequisites)
      - [System Requirements](#System-Requirements)
      - [Necessary Packages and Libraries](#Necessary-Packages-and-Libraries)
  - [Main Structure](#Main-Structure)
  - [Directory Structure ](#Directory-Structure)
  - [Training ](#Training)
  - [Inferencing](#Inferencing)
  - [Results](#Training-Results)
  - [References](#References)

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
## Main Structure
Data is stored in the following way. Refer to Directory Structure for how code is organized.

<pre>
├── 3d_object_detection
│   │
│   ├── project-3d-detection/ 
│   │   └── [SEE DIRECTORY STRUCTURE]
│   │   
│   ├── home/ 
│   │   ├── ubuntu/
│   │   │   ├── kitti-3d-detection-unzipped/
│   │   │   │   ├── testing/
│   │   │   │   │   ├── image_2/: Folder of images mainly used for testing
│   │   │   │   │   ├── image_3/
│   │   │   │   │   ├── prev_2/
│   │   │   │   │   └── prev_3/
│   │   │   │   │   
│   │   │   │   ├── training/
│   │   │   │   │   ├── label_2/: Folder of labels for image_2
│   │   │   │   │   ├── image_2/: Folder of images mainly used for training
│   │   │   │   │   ├── image_3/
│   │   │   │   │   ├── prev_2/: Folder of previous frames from image_2 images
│   │   │   │   │   └── prev_3/

</pre>

---
## Directory Structure 
This is the structure as it appears on GitHub. This structure should be a inside of: 

    3d_object_detection/project-3d-detection/ 

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
│   ├── box_3d_iou.py: Utility class for visualization
│   ├── ckpts/: For the model weights 
│   ├── driver.py: Driver used to train and inference model
│   ├── inputs/ 
│   │   ├── instance_segs/: Directory containing segmentation for instances
│   │   ├── train_data.json: Annotations for training
│   │   └── train_segs.json: Full segmentations by image
│   ├── Demo_12_4.ipynb : used to visualize the results
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
python3 driver.py train <img_dir> <annotation_json_fp> <segmentation_dir> <num_sectors> <vehicle_only>
# <img_dir> is the directory with the images from training/image_2/: Folder of images mainly used for training
# <annotation_json_fp> is the annotions converted from training/label_2/ into json format, and rotation y category id as shown in full_img_alpha_model/inputs/train_data.json.
# <segmentation_dir> contains the RLE masks files of all instances in image_2 generated by seg_associate.py
# <num_sectors> is the number of rotation y categories
# <vehicle_only> y/n select whether to use only vehicle classes
```

## Training from Model Checkpoint
### For the angle prediction model:
```bash
cd full_image_alpha_model
python3 driver.py train_from_ckpt <img_dir> <annotation_json_fp> <segmentation_dir> <num_sectors> <vehicle_only> <ckpt_fp>
# <img_dir> is the directory with the images from training/image_2/: Folder of images mainly used for training
# <annotation_json_fp> is the annotions converted from training/label_2/ into json format, and rotation y category id as shown in full_img_alpha_model/inputs/train_data.json.
# <segmentation_dir> contains the RLE masks files of all instances in image_2 generated by seg_associate.py
# <num_sectors> is the number of rotation y categories
# <vehicle_only> y/n select whether to use only vehicle classes
# <ckpt_fp> is the checkpoint .hdf5 file
```


### For the dimension prediction model:
```bash
# not done, work in progress
cd dim_model
python3 driver_dims.py
```

---
## Inferencing 
### For the angle prediction model:
#### Outputs as JSON file
```bash
python3 driver.py detect <img_dir> <annotation_json_fp> <segmentation_dir> <num_sectors> <vehicle_only> <ckpt_fp> <output_json_path>
# <img_dir> is the directory with the images from training/image_2/: Folder of images mainly used for training
# <annotation_json_fp> is the annotions converted from training/label_2/ into json format, and rotation y category id as shown in full_img_alpha_model/inputs/train_data.json.
# <segmentation_dir> contains the RLE masks files of all instances in image_2 generated by seg_associate.py
# <num_sectors> is the number of rotation y categories
# <vehicle_only> y/n select whether to use only vehicle classes
# <ckpt_fp> is the checkpoint .hdf5 file
# <output_json_path> is the file to output the rotation y categories, 0 sector starts at straight west
```
---
## Training Results
<details>
<summary>For the rotation model</summary>

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

## Visualization Results

### The rotation prediction
|||
--|--
![](https://i.imgur.com/AYvdRx7.png)|![](https://i.imgur.com/BW02PQi.png)


---
## Reference
- ### [struct2depth](https://github.com/tensorflow/models/tree/master/research/struct2depth)
- ### [Mask_RCNN](https://github.com/matterport/Mask_RCNN)
- ### [mmdetection](https://github.com/open-mmlab/mmdetection)
- ### [GS3d](https://arxiv.org/abs/1903.10955)
