import logging

import keras.backend as K
from keras import layers
from keras import regularizers
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.recurrent import GRU
from keras.layers.core import Lambda, Dense
from keras.layers.core import Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from ds_constants import *

layers_dict = dict()


def get(obj):
    layer_name = obj.name
    if layer_name not in layers_dict:
        logging.info("'-> Creating layer [{}]'.format(layer_name)")
        #create it
        layers_dict[layer_name] = obj
    else:
        logging.info("-> Using layer [{}]'.format(layer_name)")
    return layers_dict[layer_name]


def clipped_relu(inputs):
    return get(Lambda(lambda y: K.minimum(K.maximum(y, 0), 20), name='clipped'))(inputs)


def identity_block(input_tensor, kernel_size, filters, stage, block):
    conv_name_base = 'res{}_{}branch'.format(stage, block)

    x = get(Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=1))

def gru_model(batch_input_shape=(BATCH_NUM_TRIPLETS * NUM_FRAMES, 16, 16, 1),
              batch_size=BATCH_NUM_TRIPLETS, num_frames=NUM_FRAMES):

    inputs = Input(batch_input_shape=batch_input_shape)
    x = Conv2D(64,
               kernel_size=5,
               strides=2,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=0.0001))(inputs)
    x = GRU(1024,
            activation='tanh',
            recurrent_activation='hard_sigmoid',
            kernel_initializer="glorot_uniform",
            )(x)
    x = GRU(1024,
            activation='tanh',
            recurrent_activation='hard_sigmoid',
            kernel_initializer="glorot_uniform",
            )(x)
    x = GRU(1024,
            activation='tanh',
            recurrent_activation='hard_sigmoid',
            kernel_initializer="glorot_uniform",
            )(x)
    x = Lambda(lambda y: K.mean(y, axis=1), name="average")(x)
    x = Dense(512, name="affine")(x)
    x = Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln')(x)
    m = Model(inputs, x, name='GRU')
    return m

