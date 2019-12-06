import xmodel_dim as xmodel
import sys
import prep_dim as preprocessor
from tensorflow.keras import callbacks
import tensorflow.keras.backend as K
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D,Dense,Dropout,Flatten, BatchNormalization, Input, GlobalAveragePooling2D, concatenate, MaxPool2D, LeakyReLU, Lambda, add
import tensorflow.keras.layers as layers
import tensorflow as tf
import json
import numpy as np

K.set_floatx('float32')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
BATCH_SIZE = 4
_EPSILON = 1e-7
SECTORS = 32
PREDICT_BATCH_SIZE = 16
NUM_INSTANCES = 40570

def train():
    model = xmodel.getModel()
    generator = preprocessor.getLoader(batchsize=BATCH_SIZE)
    valgenerator = preprocessor.getLoader(mode='validation',batchsize=BATCH_SIZE)
    #sgd = optimizers.SGD(learning_rate=0.05, momentum=0.01, nesterov=True)
    nadam = optimizers.Nadam()# apparently according to keras, the default is the recommended value
    model.compile(optimizer='nadam', loss="mse", metrics = ['mae'])
    save_callback = callbacks.ModelCheckpoint('ckpts/weights.{epoch:d}-{loss:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5')# add validation
    tb = callbacks.TensorBoard(log_dir='./logs')
    model.fit_generator(generator, steps_per_epoch=int(36450/BATCH_SIZE), epochs=100, verbose=1, validation_freq=1, validation_data=valgenerator,validation_steps=int(4050/BATCH_SIZE), callbacks=[save_callback,tb])
    
    
def train_from_8_ckpt(ckpt, only_vehicles = True):
    backbone = load_model(ckpt)
    if only_vehicles:
        generator = preprocessor.getLoader(batchsize=BATCH_SIZE, discard_cls = [0,1,2,4,7])
        valgenerator = preprocessor.getLoader(mode='validation',batchsize=BATCH_SIZE, discard_cls = [0,1,2,4,7])
    else:
        generator = preprocessor.getLoader(batchsize=BATCH_SIZE)
        valgenerator = preprocessor.getLoader(mode='validation',batchsize=BATCH_SIZE)
    #sgd = optimizers.SGD(learning_rate=0.05, momentum=0.01, nesterov=True)
    nadam = optimizers.Nadam()
    backbone.compile(optimizer='nadam', loss="mse", metrics = ['mae'])
    save_callback = callbacks.ModelCheckpoint('ckpts/weights.{epoch:d}-{loss:.2f}-{val_loss:.2f}.hdf5')# add validation
    tb = callbacks.TensorBoard(log_dir='./logs')
    backbone.fit_generator(generator, steps_per_epoch=int(36450/BATCH_SIZE), epochs=100, verbose=1, validation_freq=1, validation_data=valgenerator,validation_steps=int(4050/BATCH_SIZE), callbacks=[save_callback,tb])

#Generates a json file with the predicted categories in the output_path location
#Uses the instance ids from data json as keys and the predicted category as values
def detect(model_ckpt, output_path):
    model = load_model(model_ckpt)
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
            predictions[image_ids[i]] = np.asarray(batch_predictions[i]).tolist()
        
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