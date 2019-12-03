import skimage
import numpy as np
from imgaug import augmenters as iaa
from random import random
from random import seed

#50 / 50 chance to do some opperations
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

#Applied to both image and mask
#pair_aug = iaa.Sequential([
#    sometimes(iaa.Fliplr(0.5)),
#    sometimes(iaa.Flipud(0.1)), 
#    sometimes(iaa.SomeOf((0,2), [iaa.Crop(px = (0,64)), iaa.Pad(px = (0,64))])),
#    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random #strengths)
#    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
#    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
#])

#Applied to just image
input_aug = iaa.Sequential([
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0,3),[
            iaa.OneOf([
               iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0
               iaa.AverageBlur(k=(1, 4)), # blur image using local means with kernel sizes between 2 and 7
               iaa.MedianBlur(k=(1, 5)), # blur image using local medians with kernel sizes between 2 and 7
            ]),
            iaa.Sharpen(alpha=(0, 0.8), lightness=(0.5, 1.0)), # sharpen images
#             search either for all edges or for directed edges,
#             blend the result with the original image using a blobby mask
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03*255), per_channel=0.5), # add gaussian noise to images
            iaa.AddToHueAndSaturation((-5, 5)), # change hue and saturation
            iaa.OneOf([
                iaa.Add((-4, 4), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
    #             either change the brightness of the whole image (sometimes
    #             per channel) or change the brightness of subareas
                iaa.ContrastNormalization((0.65, .9)) # improve or worsen the contrast
                ]),
            ], random_order=True )
    ], random_order=True )

def resize_fit_rect(img, resize_shape):
    #img is numpy array
    #resize_shape is desired size
    #Returns resized image
    x_dif = resize_shape[0] - len(img)
    y_dif = resize_shape[1] - len(img[0])

    resize_ratio = 1 + min(x_dif/resize_shape[0], y_dif/resize_shape[1])
    
    if (len(img.shape) == 3):
        #Stretches image maintaining aspect ratio
        new = skimage.transform.rescale(img, resize_ratio, mode = 'constant', multichannel=True, preserve_range=True, anti_aliasing=False)
        if len(new) != resize_shape[0]: #Pad height
            if (resize_shape[0] - len(new)) % 2 == 0:
                pad1 = pad2 = int((resize_shape[0] - len(new)) / 2)
            else:
                pad1 = int((resize_shape[0] - len(new)) / 2)
                pad2 = pad1 + 1
            new = np.pad(new, [(pad1, pad2), (0,0), (0,0)], mode = 'constant')
        elif len(new[0]) != resize_shape[1]: #Pad width
            if (resize_shape[1] - len(new[0])) % 2 == 0:
                pad1 = pad2 = int((resize_shape[1] - len(new[0])) / 2)
            else:
                pad1 = int((resize_shape[1] - len(new[0])) / 2)
                pad2 = pad1 + 1
            new = np.pad(new, [(0,0), (pad1, pad2), (0,0)], mode = 'constant')
            
    elif (len(img.shape) == 2):
        new = skimage.transform.rescale(img, resize_ratio, mode = 'constant', multichannel=False, preserve_range=True, anti_aliasing=False)
        if len(new) != resize_shape[0]:
            if (resize_shape[0] - len(new)) % 2 == 0:
                pad1 = pad2 = int((resize_shape[0] - len(new)) / 2)
            else:
                pad1 = int((resize_shape[0] - len(new)) / 2)
                pad2 = pad1 + 1
            new = np.pad(new, [(pad1, pad2), (0,0)], mode = 'constant')
        elif len(new[0]) != resize_shape[1]:
            if (resize_shape[1] - len(new[0])) % 2 == 0:
                pad1 = pad2 = int((resize_shape[1] - len(new[0])) / 2)
            else:
                pad1 = int((resize_shape[1] - len(new[0])) / 2)
                pad2 = pad1 + 1
            new = np.pad(new, [(0,0), (pad1, pad2)], mode = 'constant')
            
    return new


def apply_augmentation(images,
                       im_aug = input_aug):

    aug_images = im_aug.augment_images(images)
    
    return np.array(aug_images)