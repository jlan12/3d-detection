import xmodel
import sys
import preprocessor
import os
import numpy as np
import json
from tensorflow.keras import callbacks
import tensorflow.keras.backend as K
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D,Dense,Dropout,Flatten, BatchNormalization, Input, GlobalAveragePooling2D, concatenate, MaxPool2D, LeakyReLU, Lambda, add
import tensorflow.keras.layers as layers
import tensorflow as tf

K.set_floatx('float32')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
BATCH_SIZE = 4
PREDICT_BATCH_SIZE = 16
_EPSILON = 1e-7
SECTORS = 32
NUM_INSTANCES = 40570 #Total number of instances, before accounting for val split or discarded classes

#Trains model from scratch, with quality-aware loss
def train():
    model = xmodel.getModel()
    generator = preprocessor.getLoader(batchsize=BATCH_SIZE)
    valgenerator = preprocessor.getLoader(mode='validation',batchsize=BATCH_SIZE)
    #sgd = optimizers.SGD(learning_rate=0.05, momentum=0.01, nesterov=True)
    nadam = optimizers.Nadam()# apparently according to keras, the default is the recommended value
    model.compile(optimizer='nadam', loss=quality_loss)
    save_callback = callbacks.ModelCheckpoint('ckpts/weights.{epoch:d}-{loss:.2f}-{val_loss:.2f}.hdf5')# add validation
    tb = callbacks.TensorBoard(log_dir='./logs')
    model.fit_generator(generator, steps_per_epoch=int(36450/BATCH_SIZE), epochs=100, verbose=1, validation_freq=1, validation_data=valgenerator,validation_steps=int(4050/BATCH_SIZE), callbacks=[save_callback,tb])
    

#Trains model from a checkpoint with output vector of size 8
def train_from_8_ckpt(ckpt, only_vehicles = True):
    backbone = load_model(ckpt)
    backbone.layers.pop()
    predictions = Dense(SECTORS, activation = 'softmax', name="final_layer")(backbone.layers[-1].output)
    model = Model(backbone.input,outputs=predictions)
    if only_vehicles:
        generator = preprocessor.getLoader(batchsize=BATCH_SIZE, discard_cls = [0,1,2,4,7])
        valgenerator = preprocessor.getLoader(mode='validation',batchsize=BATCH_SIZE, discard_cls = [0,1,2,4,7])
    else:
        generator = preprocessor.getLoader(batchsize=BATCH_SIZE)
        valgenerator = preprocessor.getLoader(mode='validation',batchsize=BATCH_SIZE)
    #sgd = optimizers.SGD(learning_rate=0.05, momentum=0.01, nesterov=True)
    nadam = optimizers.Nadam()
    model.compile(optimizer='nadam', loss=base_categorical_crossentropy)
    save_callback = callbacks.ModelCheckpoint('ckpts/weights.{epoch:d}-{loss:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5')# add validation
    tb = callbacks.TensorBoard(log_dir='./logs')
    model.fit_generator(generator, steps_per_epoch=int(36450/BATCH_SIZE), epochs=100, verbose=1, validation_freq=1, validation_data=valgenerator,validation_steps=int(4050/BATCH_SIZE), callbacks=[save_callback,tb])
    
def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.
    # Returns
        A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x
    
#Loss function expects the ground truths to be the concatenation of the One-Hot Encoded vector and the vector with distributed weights
def quality_loss(targets, output):
    #Make output add up to 1
    output /= tf.reduce_sum(output,
                                axis=len(output.get_shape()) - 1,
                                keepdims=True)
    
    #Separate both parts of ground truth
    target1 = targets[:,0:int(SECTORS)] #OHE vector
    target2 = targets[:,int(SECTORS):] #Distributed weight vector
    
    epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    
    return - tf.math.reduce_sum((target1 * tf.math.log(output) + (1.0 - target2) * tf.math.log(1.0 - output)),
                           axis=1)

def 


#Generates a json file with the predicted categories in the output_path location
#Uses the instance ids from data json as keys and the predicted category as values
def detect(model_ckpt, output_path):
    model = load_model(model_ckpt, custom_objects = {'quality_loss': quality_loss})
    generator = preprocessor.getDetectionLoader(batchsize = PREDICT_BATCH_SIZE)
    steps = 0
    predictions = {}

    for generator_output in generator: # for each image id
        
        image_ids = generator_output[1]
        model_inputs = generator_output[0]
        
        batch_predictions = model.predict(model_inputs)
        steps += 1
        
        print("Step: " + str(steps))
        
        #Find the maximum value to use as the predicted category
        for i in range(PREDICT_BATCH_SIZE):
            predictions[image_ids[i]] = int(np.where(batch_predictions[i] == np.amax(batch_predictions[i]))[0][0])
        
        #Stop inferencing after passing through entire validation set
        if steps > (NUM_INSTANCES / 10 / PREDICT_BATCH_SIZE) + 1:
            break
    
    with open (output_path, 'w') as fp:
        json.dump([predictions], fp)
    

    

    
if __name__ =='__main__':
    '''Run the driver without arguments to train from scratch.
    If training from a checkpoint that used 8 sectors, pass in the checkpoint as
    the first and only argument'''
    
    if len(sys.argv) > 1:
        if sys.argv[1] != "detect":
            ckpt = sys.argv[1]
            train_from_8_ckpt(ckpt)
        else:
            if len(sys.argv) == 4:
                model_ckpt = sys.argv[2]
                output_path = sys.argv[3]
                detect(model_ckpt, output_path)
            else:
                print("Run inference using following format: python3 driver.py detect model_ckpt output_path")
    else:
        train()