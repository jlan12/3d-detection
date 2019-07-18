import tensorflow as tf
import preprocessor
import json
import numpy as np
import sys
import os
from typing import List
from io import StringIO

OUTPUT_DIR = "inference_outputs/test/"
CATEGORIES = ['Cyclist','Tram','Person_sitting','Truck','Pedestrian','Van','Car','Misc','DontCare']
STATS_FILE = 'mean_std_final.json'

gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

def detect(model_ckpt, output_dir, images: List[int]):
    images = sorted(images)
    model = tf.keras.models.load_model(model_ckpt)
    test_generator = preprocessor.generate_crops_test(images)
    with open(STATS_FILE) as file:
        stats = json.load(file)
    images_id_counter = 0
    for generator_output in test_generator: # for each image id
        
        if generator_output is None:
            image_id = images[images_id_counter]
            output_fn = os.path.join(OUTPUT_DIR, str(image_id).zfill(6) + ".txt")
            with open(output_fn, "w") as file:
                pass #Make empty file
            images_id_counter += 1
            continue
        
        
        inputs, stacked_classes, confidences = generator_output
        dims, locs, roys = model.predict(inputs) # get the predictions for the current image (list of instances)

        full_labels = np.concatenate((dims, locs, roys), axis=1)
        label_means = stats['labels'][0]
        label_stds = stats['labels'][1]
        full_labels = denormalize(full_labels, label_means, label_stds)

        denormed_roys = full_labels[:, -1]
        xs = full_labels[:, 3]
        zs = full_labels[:, 5]
        alphas =  denormed_roys - np.arctan(xs / zs)
        
        stacked_arrays = inputs[1]    
        bboxes = stacked_arrays[:, 0:4]
        bbox_means = stats['bboxes'][0]
        bbox_stds = stats['bboxes'][1]
        bboxes = denormalize(bboxes, bbox_means, bbox_stds)
        
        with StringIO() as file_text: # the kitti label string for output
            for j in range(len(dims)): # for each instance in the list of outputs
                full_label = full_labels[j]
                full_label = " ".join(list(map(str,full_label))) # add space between each list element
                
                bbox = bboxes[j]
                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                bbox = " ".join(list(map(str, bbox))) # add space between each list element

                confidence = confidences[j]

                ob_class = stacked_classes[j]
                ob_class = CATEGORIES[ob_class]

                alpha = alphas[j]
                
                final_str = ob_class + " " + "0 0 " + str(alpha) + " " + bbox + " " + full_label + " " + str(confidence) + '\n'
                file_text.write(final_str)

            # write our list of outputs to text file
            image_id = images[images_id_counter]
            output_fn = os.path.join(OUTPUT_DIR, str(image_id).zfill(6) + ".txt")
            with open(output_fn, "w") as file:
                file.write(file_text.getvalue())

        images_id_counter += 1
        

def denormalize(array, mean, std):
    denorm = np.multiply(array, std)
    denorm = np.add(array, mean)
    return denorm
    

if __name__ == '__main__':
    model_ckpt = None
    if len(sys.argv) > 1:
        model_ckpt = sys.argv[1]
        if len(sys.argv) > 3:
            #Last index not included!
            start = int(sys.argv[2])
            finish = int(sys.argv[3])
    else:
        print("No model!")
        
    detect(model_ckpt, OUTPUT_DIR, range(start, finish))