# The purpose of this script is to pre-normalize the input and target data, and to group the dataset for efficient I/O access during batch generation

import numpy as np
import os
import cv2
import glob
import json
import pickle as pkl
import math
import skimage
import matplotlib.pyplot as plt
from pycocotools import mask


RESIZE_SHAPE = (224, 224)
output_dir = '/home/ubuntu/normalized_inputs/'
depth_path = '/home/ubuntu/depth_output'
op_path = '/home/ubuntu/pwc-net/kitti-optical-flow-training'
image_path = '/home/ubuntu/kitti-3d-detection-unzipped/training/image_2'
cascade_output = '/home/ubuntu/cascade-cls-bbox-conv/kitti-training/bboxdump.json'
pickles_path = '/home/ubuntu/cascade-cls-bbox-conv/kitti-training/pickles/'
segs_output = '/home/ubuntu/cascade-cls-bbox-conv/kitti-training/trainbboxsegv2.json'
labels_path = '/home/ubuntu/kitti-3d-detection-unzipped/training/label_2/'

with open(cascade_output) as json_file:
    all_data = json.load(json_file)
with open(segs_output) as json_file:
    segs_data = json.load(json_file)

def getImage(image_path):
    image = cv2.imread(image_path)
    return image

def find_BGR_means(dir_path):
    '''Get all images in the dir_path and then output the means and stds for each of the 3 channels, indexed by order of BGR'''
    #Store the sums and stds for each channel
    channel_s=([],[],[])
    channel_d=([],[],[])
    result_mean=[]
    result_std=[]
    image_counter = 0
    total_pixels = 0
    # for each image, get the sum and std for each channel 
    for im_path in glob.glob(dir_path+'/*.png'):
        if image_counter%1000 == 0:
            print(image_counter/1000)
        im = getImage(im_path)
        for channel_num in range(3):
            channel_info=im[:,:,channel_num] 
            channel_s[channel_num].append(channel_info.sum())
            channel_d[channel_num].append(channel_info.std())
            
        total_pixels += channel_info.shape[0]*channel_info.shape[1]
        image_counter += 1
    # for each channel, get the average pixel value and the average std
    for count in range(3):
        result_mean.append(sum(channel_s[count])/total_pixels)
        result_std.append(sum(channel_d[count])/image_counter)
    return [result_mean, result_std]

def find_depth_means(dir_path):
    '''Get all depth images in the dir_path and then output the mean and std'''
    channel_s=[]
    channel_d=[]
    result_mean=[]
    result_std=[]
    image_counter = 0
    total_pixels = 0
    for im_path in glob.glob(dir_path+'/*.npy'):
        if image_counter%1000 == 0:
            print(image_counter/1000)
        im = np.load(im_path) 
        channel_s.append(im.sum())
        channel_d.append(im.std())
            
        total_pixels += im.shape[0]*im.shape[1]
        image_counter += 1
        
    result_mean.append(sum(channel_s)/total_pixels)
    result_std.append(sum(channel_d)/image_counter)
    return [result_mean[0],result_std[0]]

def find_op_means(dir_path):
    '''Get all optical flow images in the dir_path and then output the mean and std for each of the 2 channels, 0 values are filtered'''
    channel_s=([],[])
    channel_d=([],[])
    result_mean=[]
    result_std=[]
    counter = 0
    total_pixels = [0,0]
    for im_path in glob.glob(dir_path+'/*.png'):
        if counter%1000 == 0:
            print(counter/1000)
        im = getImage(im_path)
        
        for channel_num in range(2):
            channel_info=im[:,:,channel_num] 
            channel_s[channel_num].append(channel_info.sum())
            # get the std of the non zero values
            channel_d[channel_num].append(np.extract(channel_info != 0, channel_info).std())
            # count the number of non zero pixels
            total_pixels[channel_num] += np.count_nonzero(channel_info)
        counter += 1
        
    for channel in range(2):
        result_mean.append(sum(channel_s[channel])/total_pixels[channel])
        result_std.append(sum(channel_d[channel])/counter)
    return [result_mean, result_std]

def find_label_means(labels_dir):
    '''Get the means and std of all labels in labels_dir in the order of h, w, l, x, y, z, rot_y'''
    file_list = glob.glob(labels_dir + '*.txt')
    rot_y_list = []
    x_list = []
    y_list = []
    z_list = []
    h_list = []
    w_list = []
    l_list = []
    
    for file in file_list:
        with open(file, 'r') as curr:
            lines = curr.readlines()
            lines = [l.split() for l in lines]
        for line in lines:
            if line[0] != 'DontCare':
                rot_y_list.append(float(line[14]))
                x_list.append(float(line[11]))
                y_list.append(float(line[12]))
                z_list.append(float(line[13]))
                h_list.append(float(line[8]))
                w_list.append(float(line[9]))
                l_list.append(float(line[10]))
            
    label_lists = [h_list, w_list, l_list, x_list, y_list, z_list, rot_y_list]
    label_means = [np.mean(l) for l in label_lists]
    label_std = [np.std(l) for l in label_lists]
    
    return [label_means, label_std]

def find_bbox_means(cascade_bbox_json):
    '''Get the means and std of all bboxes in cascade_bbox_json in the order of [x, y, w, h]'''
    all_bboxes = []
    with open(cascade_bbox_json, 'r') as json_file:
        data = json.load(json_file)
    for item in data:
        all_bboxes.append(item['pred_bbox'])
    mean = np.mean(all_bboxes, axis = 0)
    std = np.std(all_bboxes, axis = 0)
    return [list(mean), list(std)]

def find_conv_means(cascade_bbox_json):
    with open(cascade_bbox_json, 'r') as json_file:
        data = json.load(json_file)
    full_sum = 0
    total = 0
    stds = []
    #Get convolutions
    for item in data:
        instance_id = item['instance_id']
        with open(os.path.join(pickles_path,instance_id + '.pkl'), 'rb') as file:
            convs = pkl.load(file)    
            #Swap axes with numpy to make it (7,7,256)
            convs = np.swapaxes(convs, 0, 2)
            
        full_sum += np.sum(convs)
        total += 7 * 7 * 256
        stds.append(np.std(convs))
    return full_sum / total, np.mean(stds)

bgr_stats = find_BGR_means(image_path)
depth_stats = find_depth_means(depth_path)
label_stats = find_label_means(labels_path)
bbox_stats = find_bbox_means(cascade_output)
op_stats = find_op_means(op_path)

mean_stds = {}
mean_stds['bgr'] = bgr_stats
mean_stds['depth'] = depth_stats
mean_stds['labels'] = label_stats
mean_stds['bboxes'] = bbox_stats
mean_stds['op'] = op_stats
with open('mean_std_final.json', 'w') as fp:
    json.dump(mean_stds, fp)