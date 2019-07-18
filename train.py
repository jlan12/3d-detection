"""
Clean and simple Keras implementation of network architectures described in:
    - (ResNet-50) [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf).
    - (ResNeXt-50 32x4d) [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf).
    
Python 3.
"""
import preprocessor
import sys
import numpy as np
import tensorflow as tf
import math
import os



import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import callbacks
from tensorflow.keras.utils import plot_model

dtype='float32'
K.set_floatx(dtype)
# default is 1e-7 which is too small for float16.  Without adjusting the epsilon, we will get NaN predictions because of divide by zero problems
K.set_epsilon(1e-7) 
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

BATCH_SIZE = 32
VAL_DIVISOR = 10

#
# image dimensions
#

img_channels = 3

#
# network params
#

cardinality = 32
gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))


def residual_network(x, x2, x3):
    """
    x is the stacked resized cropped tensor (re_crop_im, re_crop_opflo, re_crop_depth), expected shape is (batch_size,256,256,6)
    x2 is the bbox, class_array, expected shape is (batch_size,5)
    x3 is the final conv output tensor from cascade rnn (batch_size,7,7,256)
    All resized crops should be of shape (32, 32)
    # just an example of input
    stacked_tensor = np.concatenate((im_resized_crops, cropped_resized_depths, cropped_resized_ofs), axis=2)
    
    ResNeXt by default. For ResNet set `cardinality` = 1 above.
    
    """
    def add_common_layers(y):
        y = layers.BatchNormalization(renorm=True)(y)
        y = layers.LeakyReLU()(y)
        return y

    def grouped_convolution(y, nb_channels, _strides):
        # when `cardinality` == 1 this is just a standard convolution
        if cardinality == 1:
            return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        
        assert not nb_channels % cardinality
        _d = nb_channels // cardinality

        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
            groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))
            
        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)

        return y

    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:

        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """
        shortcut = y

        # we modify the residual building block as a bottleneck design to make the network more economical
        y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = add_common_layers(y)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = grouped_convolution(y, nb_channels_in, _strides=_strides)
        y = add_common_layers(y)

        y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = layers.BatchNormalization(renorm=True)(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization(renorm=True)(shortcut)

        y = layers.add([shortcut, y])

        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        y = layers.LeakyReLU()(y)
        return y

    # start building the network, x inputs to resnext-50
    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
    x = add_common_layers(x)
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    # conv2
    for i in range(3):
        project_shortcut = True if i == 0 else False
        x = residual_block(x, 128, 256, _project_shortcut=project_shortcut) # expected output size 
        
    # conv3
    for i in range(4):
        # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 256, 512, _strides=strides) # shrinks the size to
        
     # conv4
    for i in range(6):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 512, 1024, _strides=strides)

    # conv5
    for i in range(3):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 1024, 2048, _strides=strides)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    
    x2 = layers.BatchNormalization(renorm=True)(x2)
    
    x3 = layers.Conv2D(512, (5,5), name='prev_last_step_of_x3')(x3) # expected output size is (batch_size,3,3,512)
    x3 = add_common_layers(x3)
    x3 = layers.Conv2D(512, (3,3), name='last_step_of_x3')(x3) # expected output size is (batch_size,1,1,512)
    x3 = add_common_layers(x3)
    x3 = layers.Flatten()(x3) # expected output size is (batch_size,512)
    
    # merge starts here
    merged = layers.concatenate([x, x2, x3])
    merged = layers.BatchNormalization(renorm=True)(merged)
    merged = layers.Dense(1024, kernel_initializer='he_uniform')(merged)
    merged = layers.Dropout(0.5)(merged)
    
    # output branch for dimensions
    dim_branch = layers.Dense(512, kernel_initializer='he_uniform')(merged)
    dim_branch = layers.Dropout(0.5)(dim_branch)
    dim_output = layers.Dense(3, name='dim_output')(dim_branch)
    
    # output branch for location
    loc_branch = layers.Dense(512, kernel_initializer='he_uniform')(merged)
    loc_branch = layers.Dropout(0.5)(loc_branch)
    loc_output = layers.Dense(3, name='loc_output')(loc_branch)
    
    # output branch for rotation y
    roy_branch = layers.Dense(256, kernel_initializer='he_uniform')(merged)
    roy_branch = layers.Dropout(0.5)(roy_branch)
    roy_output = layers.Dense(1, name='roy_output')(roy_branch)
    
    return dim_output, loc_output, roy_output

def train(model_ckpt):
    if model_ckpt is not None:
        model = models.load_model(model_ckpt)
    else:
        image_inputs = layers.Input(shape=(224, 224, 7), name='image_inputs') # concat(im_crop, of_crop, d_crop, seg_crop)
        array_inputs = layers.Input(shape=(12,), name='array_inputs') # concat(x, y, w, h, ohe_8_classes)
        conv_inputs = layers.Input(shape=(7,7,256), name='conv_inputs') # conv from trained 2d bbox
        dim_output, loc_output, roy_output = residual_network(image_inputs, array_inputs, conv_inputs)

        model = models.Model(inputs=[image_inputs, array_inputs, conv_inputs], outputs=[dim_output, loc_output, roy_output])
        
        print(model.summary())
        
        #confirm dtype is float16
        #print("Model output dtype type is: ", K.dtype(dim_output.kernel))
        
        plot_model(model, to_file='/home/ubuntu/fusion-3d-detection/model.png')
        
        losses = {
            "dim_output": "huber_loss",
            "loc_output": "huber_loss",
            "roy_output": "huber_loss"
        }
        
        lossWeights = {"dim_output": 0.3333, "loc_output": 0.3333, "roy_output": 0.3333}
        opt = optimizers.Adam(clipnorm=1.0)
        
        model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights)

    save_callback = callbacks.ModelCheckpoint('model_ckpts/modelv4_weights.{epoch:d}-{val_loss:.4f}.hdf5')
    terminate_nan_callback = callbacks.TerminateOnNaN()
    
    class AdamLearningRateTrackerCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            optimizer = self.model.optimizer
            lr = optimizer.lr
            decay = optimizer.decay
            iterations = optimizer.iterations
            lr = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
            t = K.cast(iterations, K.floatx()) + 1
            lr_t = lr * (K.sqrt(1. - K.pow(optimizer.beta_2, t)) / (1. - K.pow(optimizer.beta_1, t)))
            print('\nLR: {:.6f}\n'.format(session.run(lr_t)))
            
    history = model.fit_generator(preprocessor.generate_crops_train(BATCH_SIZE, VAL_DIVISOR, False), steps_per_epoch=1200, epochs=100, callbacks=[save_callback, terminate_nan_callback, AdamLearningRateTrackerCallback()], validation_data=preprocessor.generate_crops_train(BATCH_SIZE, VAL_DIVISOR, True), validation_steps=480)
    
    np_loss_history = np.array(history)
    np.savetxt("loss_history.txt", np_loss_history, delimiter=",")

if __name__ == '__main__':
    model_ckpt = None
    if len(sys.argv) > 1:
        model_ckpt = sys.argv[1]
    train(model_ckpt)