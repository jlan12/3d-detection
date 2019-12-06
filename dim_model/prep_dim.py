import json
import numpy as np
import cv2
import os
import skimage
import random
from pycocotools import mask
from util_dim import resize_fit_rect, apply_augmentation

NUMBER_OF_ROT_Y_SECTORS = 32
IMAGE_SIZE = (376,1242)
VIEW_ANGLE_TOTAL_X = 1.4835298642
VIEW_ANGLE_TOTAL_Y = 0.55850536064


def getLoader(mode='training',
              split = 0.1 ,
              ry_cats = NUMBER_OF_ROT_Y_SECTORS, 
              image_dir='../../../home/ubuntu/kitti-3d-detection-unzipped/training/image_2',
              batchsize=4,
              jsonfp = "../inputs/train_data.json", 
              segs_dir = "../inputs/instance_segs/",
              shuffle = True,
             discard_cls=[]):
    
    annos = json.load(open(jsonfp, mode='r'))
    # filter and keep the category_ids in lamda/
    annos = annos if not discard_cls else list(filter(lambda anno:anno['category_id'] not in discard_cls, annos))
    '''
    Recommendations : [0,1,2,4,7] for all vehicles
    [0,1,2,3,4,5,7,8] for only cars
    [1,3,5,6,7,8] for peoples
    '''
    # split annotation for validation set (else case is for validation)
    annos = annos[:round(len(annos)*(1-split))] if (mode=='training') else annos[round(len(annos)*(1-split)):]
    
    while(True):
        if (shuffle):
            random.shuffle(annos)
        batch_stacked_inputs = [] #src image stacked with segmentation for specific instance
        batch_view_angles = []
        batch_dim = [] # rot_y values
        for anno in annos:
            #Full image
            image_name = str(anno['image_id']).zfill(6) # lengthen image_id to 6 digits by adding preceding zeros
            image = cv2.imread(os.path.join(image_dir,str(image_name)+'.png'))
            
            #Segmentation
            segmentation = {}
            with open (os.path.join(segs_dir, str(anno['id']) + '.txt'), "r") as seg_file:
                seg_info = seg_file.read()
                seg_info = seg_info.split("\n")
                segmentation["counts"] = seg_info[0]
                segmentation["size"] = [int(dim) for dim in seg_info[1].strip('][').split(",")] #Turn string representation of int list into actual list of integers
                
            segmentation = mask.decode(segmentation) # get seg in numpy format
            
            #Augmentation 
            #--------------
            aug_im = apply_augmentation([image])
            
            image = resize_fit_rect(aug_im[0], IMAGE_SIZE)
            segmentation = resize_fit_rect(segmentation, IMAGE_SIZE)
            segmentation = np.resize(segmentation, (IMAGE_SIZE[0], IMAGE_SIZE[1], 1)) #Add channel to enable concatenation
            
            stacked = np.concatenate([image,segmentation],axis=-1)
            #--------------
            
            batch_stacked_inputs.append(stacked)
            
            categorical_ry = np.zeros(ry_cats) # initialize a vector of size ry_cats
            distr_cats = np.zeros(ry_cats)
            
            batch_dim.append(anno['dimensions'])
            
            bbox = anno['bbox'] # bbox is a list in [x,y,w,h] format
            #Calculate center of object by taking center of bounding box
            center = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
            width = image.shape[1]
            #Calculate angle of view
            view_angle = center[0] / width * VIEW_ANGLE_TOTAL_X - (VIEW_ANGLE_TOTAL_X / 2)
            batch_view_angles.append(view_angle)
            
            if (len(batch_stacked_inputs) == batchsize):
                batch_stacked_inputs = np.asarray(batch_stacked_inputs)
                batch_dim = np.asarray(batch_dim)
                batch_view_angles = np.asarray(batch_view_angles)
                yield ([batch_stacked_inputs, batch_view_angles], batch_dim)
                batch_stacked_inputs = []
                batch_dim = []
                batch_view_angles = []
                
            if (len(batch_stacked_inputs)>batchsize):
                #Should never happen
                raise Exception("Suspected Concurrency Error")
            
            
#Loader for inferencing
def getDetectionLoader(ry_cats = NUMBER_OF_ROT_Y_SECTORS, 
              split = 0.1,
              image_dir='../../home/ubuntu/kitti-3d-detection-unzipped/training/image_2',
              batchsize=4,
              jsonfp = "inputs/train_data.json", 
              segs_dir = "inputs/instance_segs/",
             discard_cls=[]):
    
    annos = json.load(open(jsonfp, mode='r'))
    # filter and keep the category_ids in lamda/
    #annos = annos if not discard_cls else list(filter(lambda anno:anno['category_id'] not in discard_cls, annos))
    #Only predict on validation
    annos = annos[round(len(annos)*(1-split)):]
    '''
    Recommendations : [0,1,2,4,7] for all vehicles
    [0,1,2,3,4,5,7,8] for only cars
    [1,3,5,6,7,8] for peoples
    '''
    while(True):
        batch_stacked_inputs = [] #src image stacked with segmentation for specific instance
        batch_view_angles = []
        instance_ids = []
        for anno in annos:
            #Full image
            image_name = str(anno['image_id']).zfill(6) # lengthen image_id to 6 digits by adding preceding zeros
            image = cv2.imread(os.path.join(image_dir,str(image_name)+'.png'))
            
            #Segmentation
            segmentation = {}
            with open (os.path.join(segs_dir, str(anno['id']) + '.txt'), "r") as seg_file:
                seg_info = seg_file.read()
                seg_info = seg_info.split("\n")
                segmentation["counts"] = seg_info[0]
                segmentation["size"] = [int(dim) for dim in seg_info[1].strip('][').split(",")] #Turn string representation of int list into actual list of integers
                
            segmentation = mask.decode(segmentation) # get seg in numpy format
            
            segmentation = resize_fit_rect(segmentation, IMAGE_SIZE)
            image = resize_fit_rect(image, IMAGE_SIZE)
            segmentation = np.resize(segmentation, (IMAGE_SIZE[0], IMAGE_SIZE[1], 1)) #Add channel to enable concatenation
            stacked = np.concatenate([image,segmentation],axis=-1)
            batch_stacked_inputs.append(stacked)
            
            bbox = anno['bbox'] # bbox is a list in [x,y,w,h] format
            #Calculate center of object by taking center of bounding box
            center = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
            width = image.shape[1]
            #Calculate angle of view
            view_angle = center[0] / width * VIEW_ANGLE_TOTAL_X - (VIEW_ANGLE_TOTAL_X / 2)
            batch_view_angles.append(view_angle)
            
            instance_ids.append(anno['id'])
            
            if (len(batch_stacked_inputs) == batchsize):
                batch_stacked_inputs = np.asarray(batch_stacked_inputs)
                batch_view_angles = np.asarray(batch_view_angles)
                yield ([batch_stacked_inputs, batch_view_angles], instance_ids)
                batch_stacked_inputs = []
                batch_view_angles = []
                instance_ids = []
                
            if (len(batch_stacked_inputs)>batchsize):
                #Should never happen
                raise Exception("Suspected Concurrency Error")
            