import xmodel
import sys
import preprocessor
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
_EPSILON = 1e-7
SECTORS = 32

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
    model.compile(optimizer='nadam', loss=quality_loss)
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
    
#Loss function expects the ground truths to already be the distributed version, not the one-hot encoded vector. OHE vector is generated using tf.where
def quality_loss(targets, output):
    #Make output add up to 1
    output /= tf.reduce_sum(output,
                                axis=len(output.get_shape()) - 1,
                                keepdims=True)
    target1 = targets[:,0:int(SECTORS)]
    target2 = targets[:,int(SECTORS):]
    
    epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    
    return - tf.math.reduce_sum((target1 * tf.math.log(output) + (1.0 - target2) * tf.math.log(1.0 - output)),
                           axis=1)

    

    
if __name__ =='__main__':
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
        train_from_8_ckpt(ckpt)
    else:
        train()