import numpy as np
import os
import cv2
import glob
import json
import pickle as pkl
import sys
import math
import skimage
import matplotlib.pyplot as plt
from pycocotools import mask

RESIZE_SHAPE = (224, 224)


def normalize_array(array, means, stds):
    array = np.subtract(array, means)
    array = np.divide(array, stds)
    return array

def IOU(box1, box2):
    y_overlap = max(0, min(box1[1]+box1[3], box2[1]+box2[3]) - max(box1[1], box2[1]));
    x_overlap = max(0, min(box1[0]+box1[2], box2[0]+box2[2]) - max(box1[0], box2[0]));
    interArea = x_overlap * y_overlap;
    unionArea = (box1[3]) * (box1[2]) + (box2[3]) * (box2[2]) - interArea
    iou = interArea / unionArea
    return iou

def resize_fit_square(img, resize_shape):
    #img is numpy array
    #reize_shape is desired size
    #Returns resized image
    if (len(img.shape) == 3):
        new = skimage.transform.rescale(img, resize_shape/max(len(img), len(img[0])), mode = 'constant', multichannel=True, preserve_range=True, anti_aliasing=False)
        new = np.pad(new, ((0, resize_shape-len(new)), (0, resize_shape - len(new[0])), (0,0)), mode = 'constant')
    else:
        new = skimage.transform.rescale(img, resize_shape/max(len(img), len(img[0])), mode = 'constant', multichannel=False, preserve_range=True, anti_aliasing=False)
        new = np.pad(new, ((0, resize_shape-len(new)), (0, resize_shape - len(new[0]))), mode = 'constant')
    return new

def process(inst, depth_path, op_path, image_path, pickles_path, use_predicted_bbox=True):
    '''Normalize and combine the tensors'''
    with open('mean_std_final.json') as file:
        stats = json.load(file)

    #PIPELINE: Read, Crop, Resize, Normalize, Concatenate, Save
    object_class = inst['pred_category']
    if mode == 0:
        rot_y = inst['gt_ry']
        dimensions = inst['gt_dimensions']
        location = inst['gt_location']
    bbox = inst['pred_bbox']
    name = str(inst['image_id']).zfill(6)
    instance_id = inst['instance_id']
    confidence = inst['pred_score']
    
    #Read original
    bbox = [math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2]), math.ceil(bbox[3])]
    original = cv2.imread(os.path.join(image_path, name + '.png'))
    # crop by height, width, channels
    original_crop = original[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]
    # resize by fit instead of stretch
    re_ori_crop = resize_fit_square(original_crop, RESIZE_SHAPE[0]) 
    norm_ori_crop = normalize_array(re_ori_crop, stats['bgr'][0], stats['bgr'][1])
    
    # optical flow is same size as image
    op_fn = os.path.join(op_path, name + "_pwc_fusion.npy")
    if os.path.exists(op_fn):
        op_flow = np.load(op_fn)
        op_crop = op_flow[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]
        # add filler channels for resize
        op_temp = np.concatenate((op_crop, np.zeros((op_crop.shape[0], op_crop.shape[1], 1))), axis=-1)
        re_op_crop = resize_fit_square(op_temp, RESIZE_SHAPE[0])
        re_op_crop = re_op_crop[:,:,0:2]
        norm_op_crop = normalize_array(re_op_crop, stats['op'][0], stats['op'][1])
    else:
        return None
    
    # Depth
    depth_fn = os.path.join(depth_path, name + ".npy")
    if os.path.exists(depth_fn):
        depth = np.load(depth_fn)
        height_ratio = len(depth) / len(original)
        width_ratio = len(depth[0]) / len(original[0])

        # depth map is smaller size than image
        adj_bbox = [bbox[0] * width_ratio, bbox[1] * height_ratio, bbox[2] * width_ratio, bbox[3] * height_ratio]
        adj_bbox = [math.floor(adj_bbox[0]), math.floor(adj_bbox[1]), math.ceil(adj_bbox[2]), math.ceil(adj_bbox[3])]
        
        depth_crop = depth[adj_bbox[1]:adj_bbox[1] + adj_bbox[3], adj_bbox[0]:adj_bbox[0] + adj_bbox[2]]
        re_depth_crop = resize_fit_square(depth_crop, RESIZE_SHAPE[0])
        norm_depth_crop = normalize_array(re_depth_crop, stats['depth'][0], stats['depth'][1])
        # add channel dimension for concat later
        norm_depth_crop = np.reshape(norm_depth_crop, (RESIZE_SHAPE[0], RESIZE_SHAPE[1], 1))
    else:
        return None
        
    #Find segmentations for given image
    iou_list = []
    segs_dict = segs_data[inst['image_id']] # all the segmentations of the image
    anno = segs_dict['annotations'] # all the annotations (bbox, class, score) of the image
    for num in range(len(anno)):
        if use_predicted_bbox:
            seg_bbox = anno[num]['bbox']
        else:
            # calculate bbox by the size of the segmentation
            # TODO
            pass
        iou_list.append(IOU(bbox, seg_bbox))
    max_iou = max(iou_list)
    iou_index = iou_list.index(max_iou)
    segmentation = anno[iou_index]['segmentation']
    segmentation = mask.decode(segmentation) # get seg in numpy format
    # bbox comes from the trained cascade model, crop the seg map
    seg_crop = segmentation[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
    re_seg_crop = resize_fit_square(seg_crop, RESIZE_SHAPE[0])
    re_seg_crop = np.round(re_seg_crop) # round to 0 or 1 after interpolation 
    # add channel dimension for concat later
    re_seg_crop = np.resize(re_seg_crop, (RESIZE_SHAPE[0], RESIZE_SHAPE[1], 1))
    
    final_crops = np.concatenate((norm_ori_crop, norm_op_crop, norm_depth_crop, re_seg_crop), axis = -1)
    
    # Load the convs
    with open(os.path.join(pickles_path,instance_id + '.pkl'), 'rb') as file:
        convs = pkl.load(file)    
    #Swap axes with numpy to make it (7,7,256)
    convs = np.swapaxes(convs, 0, 2)

    
    #Normalizing bbox
    norm_bbox = normalize_array(bbox, stats['bboxes'][0], stats['bboxes'][1])
    
    #Normalizing labels
    if mode == 0:
        norm_label = normalize_array(dimensions + location + [rot_y], stats['labels'][0], stats['labels'][1])
        return [final_crops, convs, norm_bbox, norm_label]
    else:
        return [final_crops, convs, norm_bbox]

if __name__ == '__main__':
    output_dir = '/home/ubuntu/normalized_inputs/'
    output_dir_test = '/home/ubuntu/normalized_inputs_test/'
    cascade_output = '/home/ubuntu/cascade-cls-bbox-conv/kitti-training/bboxdump.json'
    cascade_output_test = '/home/ubuntu/cascade-cls-bbox-conv/kitti-testing/bboxdump.json'
    segs_output = '/home/ubuntu/cascade-cls-bbox-conv/kitti-training/trainbboxsegv2.json'
    segs_output_test = '/home/ubuntu/cascade-cls-bbox-conv/kitti-testing/testbboxsegv2.json'
    
    if len(sys.argv) > 1:
        mode = (int)(sys.argv[1])
    else:
        mode = 0
        
    tracker = 0
    if (mode == 0): #Training
        with open(cascade_output) as json_file:
            all_data = json.load(json_file)
        with open(segs_output) as json_file:
            segs_data = json.load(json_file)
        output_directory = output_dir
        pickles_path = '/home/ubuntu/cascade-cls-bbox-conv/kitti-training/pickles/'
        depth_path = '/home/ubuntu/depth_output'
        op_path = '/home/ubuntu/pwc-net/kitti-optical-flow-training'
        image_path = '/home/ubuntu/kitti-3d-detection-unzipped/training/image_2'
        

    else: #Testing
        with open(cascade_output_test) as json_file:
            all_data = json.load(json_file)
        with open(segs_output_test) as json_file:
            segs_data = json.load(json_file)
        output_directory = output_dir_test
        pickles_path = '/home/ubuntu/cascade-cls-bbox-conv/kitti-testing/pickles/'
        depth_path = '/home/ubuntu/depth_output_testset'
        op_path = '/home/ubuntu/pwc-net/kitti-optical-flow-testing'
        image_path = '/home/ubuntu/kitti-3d-detection-unzipped/testing/image_2'
        
        
            
    for instance in all_data:
        input_data = process(instance, depth_path, op_path, image_path, pickles_path)

        if input_data is not None:
            for i in range(len(input_data)):
                input_data[i] = input_data[i].astype(np.float16)

            pkl.dump(input_data, open(os.path.join(output_directory, instance['instance_id'] + '.pkl'), 'wb'))
            tracker += 1
            if tracker % 1000 == 0:
                print(tracker)
                