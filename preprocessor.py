import numpy as np
import os
import glob
import math
import json
import random
import skimage.transform
import matplotlib.pyplot as plt
from pycocotools import mask
import cv2
import pickle as pkl

NORMALIZED_INPUT_DIR = '/home/ubuntu/normalized_inputs/'
NORMALIZED_INPUT_TEST = '/home/ubuntu/normalized_inputs_test'
CASCADE_OUTPUT = '/home/ubuntu/cascade-cls-bbox-conv/kitti-training/bboxdump.json'
CASCADE_OUTPUT_TEST = '/home/ubuntu/cascade-cls-bbox-conv/kitti-testing/bboxdump.json'

def generate_crops_train(batch_size, val_split, val_flag, input_json = CASCADE_OUTPUT, norm_dir = NORMALIZED_INPUT_DIR):
    
    #val_split is the number to mod the name by to produce validation data
    with open(input_json) as json_file:
        all_data = json.load(json_file)
    
    inst_tracker = 0
    
    while True:
        final_inputs = []
        stacked_imgs = []
        stacked_arrays = []
        stacked_convs = []
        dim_targets = []
        location_targets = []
        roy_targets = []
            
        while len(stacked_imgs) < batch_size:
            inst = all_data[inst_tracker]
            inst_tracker += 1
            if inst_tracker >= len(all_data):
                inst_tracker = 0
                random.shuffle(all_data)
                
            object_class = inst['pred_category']
            class_array = [int(i == object_class) for i in range(8)]
            name = str(inst['image_id']).zfill(6)
            instance_id = inst['instance_id']
            
            file_path = os.path.join(norm_dir, instance_id + '.pkl')
            
            if os.path.exists(file_path):
                with open(file_path, 'rb') as file:
                    relevant_info = pkl.load(file)

                # either validation or not validation
                if (val_flag and (int(name) % val_split) == 0) or ((not val_flag) and (int(name) % val_split) != 0):
                    dim_targets.append(relevant_info[3][0:3])
                    location_targets.append(relevant_info[3][3:6])
                    roy_targets.append(relevant_info[3][6])
                    stacked_imgs.append(relevant_info[0])
                    stacked_arrays.append(np.concatenate((relevant_info[2], class_array), axis=-1).astype(np.float16))
                    stacked_convs.append(relevant_info[1])

        # right after while len(stacked_imgs) < batch_size:
        stacked_imgs = np.array(stacked_imgs).astype(np.float32)
        stacked_arrays = np.array(stacked_arrays).astype(np.float32)
        stacked_convs = np.array(stacked_convs).astype(np.float32)
        dim_targets = np.array(dim_targets).astype(np.float32)
        location_targets = np.array(location_targets).astype(np.float32)
        roy_targets = np.array(roy_targets).astype(np.float32)
        final_inputs = [stacked_imgs, stacked_arrays, stacked_convs]
        final_output = (final_inputs, [dim_targets, location_targets, roy_targets])
        yield final_output
        
def generate_crops_test(image_range, input_json = CASCADE_OUTPUT_TEST, norm_dir = NORMALIZED_INPUT_TEST):
    #image_range is an iterable containing the numbers of the images
    # this generator assumes the instances in input_json are in ascending order by their image_id
    image_range = sorted(image_range)
    
    with open(input_json) as json_file:
        all_data = json.load(json_file) # json is a list
    
    inst_tracker = 0 # counting the index of the instance
    inst = all_data[inst_tracker] # get the instance
    end_flag = False
    
    for i in image_range:
        final_inputs = []
        stacked_imgs = []
        stacked_arrays = []
        stacked_convs = []
        stacked_classes = []
        stacked_confidences = []
        none_flag = False
            
        while inst['image_id'] != i and inst_tracker < len(all_data) - 1:
            inst_tracker += 1
            inst = all_data[inst_tracker] # get the next until instance image_id match current image_id

        if (inst['image_id'] != i and inst_tracker == len(all_data) - 1):
            inst_tracker = 0
            inst = all_data[inst_tracker]
            yield None
            continue
            
        while inst['image_id'] == i:    # get next matching inst
            object_class = inst['pred_category']
            class_array = [int(i == object_class) for i in range(8)]
            name = str(inst['image_id']).zfill(6)
            instance_id = inst['instance_id']
            confidence = inst['pred_score']
            

            # get the feature maps
            file_path = os.path.join(norm_dir, instance_id + '.pkl')
            if os.path.exists(file_path):
                with open(file_path, 'rb') as file:
                    feature_maps = pkl.load(file)
                # all info is ready to append to output list
                stacked_imgs.append(feature_maps[0])
                stacked_convs.append(feature_maps[1])
                bbox_array = feature_maps[2]
                stacked_arrays.append(np.concatenate((bbox_array, class_array)).astype(np.float16))
                stacked_classes.append(object_class)
                stacked_confidences.append(confidence)

            else:
                inst_tracker += 1
                inst = all_data[inst_tracker]
                yield None
                none_flag = True
                break
                
            if inst_tracker < len(all_data) - 1:
                inst_tracker += 1
                inst = all_data[inst_tracker]
            elif inst_tracker == len(all_data) - 1:
                end_flag = True
                break
        

        if none_flag:
            continue
            
        # convert to float32
        stacked_imgs = np.array(stacked_imgs).astype(np.float32)
        stacked_arrays = np.array(stacked_arrays).astype(np.float32)
        stacked_convs = np.array(stacked_convs).astype(np.float32)
        stacked_classes = np.array(stacked_classes)
        stacked_confidences = np.array(stacked_confidences)
        final_inputs = [stacked_imgs, stacked_arrays, stacked_convs]
        yield (final_inputs, stacked_classes, stacked_confidences)
        
        if end_flag:
            return