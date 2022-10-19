from __future__ import division
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Conv2D, MaxPooling2D, concatenate, UpSampling2D, add, BatchNormalization

from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import os

img_w = 256
img_h = 256
# 有一个为背景
n_label = 1


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _bn_relu(input):
    # build a BN -> relu block
    norm = BatchNormalization(axis=3)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    # build a conv -> BN -> relu block
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _conv_bn(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return BatchNormalization(axis=3)(conv)

    return f


def _shortcut(input, residual):
    # Adds a shortcut between input and residual block and merges them with "sum"
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    shortcut = input
    # 1 X 1 conv if shape is different. 
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block(input, filters, init_strides=(1, 1)):
    # Builds a residual block
    conv1 = _conv_bn_relu(filters=filters, kernel_size=(3, 3), strides=init_strides)(input)
    residual = _conv_bn(filters=filters, kernel_size=(3, 3))(conv1)
    return _shortcut(input, residual)


def _Full_Resolution_Residual_block(input1, input2, filters, pool_size, size, init_strides=(1, 1)):
    y = input1
    z = input2
    _t = concatenate([y, MaxPooling2D(pool_size=pool_size, strides=pool_size)(z)], axis=3)
    conv1 = _conv_bn_relu(filters=filters, kernel_size=(3, 3), strides=init_strides)(_t)
    y_prime = _conv_bn_relu(filters=filters, kernel_size=(3, 3), strides=init_strides)(conv1)
    conv2 = Conv2D(filters=filters, kernel_size=(1, 1), strides=init_strides, padding='same', use_bias=False)(y_prime)
    up = UpSampling2D(size=size)(conv2)
    z_prime = _shortcut(up, z)
    return y_prime, z_prime


class frrnBuilder(object):
    #@staticmethod
    def build(self, input_shape, num_outputs):
     
        #_handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_rows, nb_cols, nb_channels,)")

        # Permute dimension order if necessary
        #if K.image_dim_ordering() == 'tf':
        #    input_shape = (input_shape[0], input_shape[1], input_shape[2])

        # Load function from str if needed.
        input = Input(shape=input_shape)
        # The First Conv
        conv1 = _conv_bn_relu(filters=48, kernel_size=(3, 3), strides=(1, 1))(input)
        # RU Layers
        pooling_stream = _residual_block(conv1, filters=24, init_strides=(1, 1))
        pooling_stream = _residual_block(pooling_stream, filters=24, init_strides=(1, 1))
        pooling_stream = _residual_block(pooling_stream, filters=24, init_strides=(1, 1))
        # Split Streams
        residual_stream = Conv2D(32, (1, 1), padding='same')(pooling_stream)
        # FFRU Layers (encoding)
        # /2
        pooling_stream = MaxPooling2D((2, 2), strides=(2, 2))(pooling_stream)
        pooling_stream,residual_stream = _Full_Resolution_Residual_block(pooling_stream, residual_stream, filters=48,
                                                                         pool_size=(2, 2), size=(2, 2), init_strides=(1, 1))
        pooling_stream,residual_stream = _Full_Resolution_Residual_block(pooling_stream,residual_stream, filters=48,
                                                                         pool_size=(2, 2), size=(2, 2), init_strides=(1, 1))
        pooling_stream,residual_stream = _Full_Resolution_Residual_block(pooling_stream,residual_stream, filters=48,
                                                                         pool_size=(2, 2), size=(2, 2), init_strides=(1, 1))
        # /4
        pooling_stream = MaxPooling2D((2, 2), strides=(2, 2))(pooling_stream)
        pooling_stream,residual_stream = _Full_Resolution_Residual_block(pooling_stream,residual_stream, filters=96,
                                                                         pool_size=(4, 4), size=(4, 4), init_strides=(1, 1))
        pooling_stream, residual_stream = _Full_Resolution_Residual_block(pooling_stream, residual_stream, filters=96,
                                                                          pool_size=(4, 4), size=(4, 4), init_strides=(1, 1))
        pooling_stream, residual_stream = _Full_Resolution_Residual_block(pooling_stream, residual_stream, filters=96,
                                                                          pool_size=(4, 4), size=(4, 4), init_strides=(1, 1))
        pooling_stream, residual_stream = _Full_Resolution_Residual_block(pooling_stream, residual_stream, filters=96,
                                                                          pool_size=(4, 4), size=(4, 4), init_strides=(1, 1))
        # /8
        pooling_stream = MaxPooling2D((2, 2), strides=(2, 2))(pooling_stream)
        pooling_stream, residual_stream = _Full_Resolution_Residual_block(pooling_stream, residual_stream, filters=192,
                                                                          pool_size=(8, 8), size=(8, 8), init_strides=(1, 1))
        pooling_stream, residual_stream = _Full_Resolution_Residual_block(pooling_stream, residual_stream, filters=192,
                                                                          pool_size=(8, 8), size=(8, 8), init_strides=(1, 1))
        # /16
        pooling_stream = MaxPooling2D((2, 2), strides=(2, 2))(pooling_stream)
        pooling_stream, residual_stream = _Full_Resolution_Residual_block(pooling_stream, residual_stream, filters=192,
                                                                          pool_size=(16, 16), size=(16, 16), init_strides=(1, 1))
        pooling_stream, residual_stream = _Full_Resolution_Residual_block(pooling_stream, residual_stream, filters=192,
                                                                          pool_size=(16, 16), size=(16, 16), init_strides=(1, 1))
        # FFRU Layers (Decoding)
        # /8
        pooling_stream = UpSampling2D((2, 2))(pooling_stream)
        pooling_stream, residual_stream = _Full_Resolution_Residual_block(pooling_stream, residual_stream, filters=96,
                                                                          pool_size=(8, 8), size=(8, 8),init_strides=(1, 1))
        pooling_stream, residual_stream = _Full_Resolution_Residual_block(pooling_stream, residual_stream, filters=96,
                                                                          pool_size=(8, 8), size=(8, 8),init_strides=(1, 1))
        # /4
        pooling_stream = UpSampling2D((2, 2))(pooling_stream)
        pooling_stream, residual_stream = _Full_Resolution_Residual_block(pooling_stream, residual_stream, filters=96,
                                                                          pool_size=(4, 4), size=(4, 4), init_strides=(1, 1))
        pooling_stream, residual_stream = _Full_Resolution_Residual_block(pooling_stream, residual_stream, filters=96,
                                                                          pool_size=(4, 4), size=(4, 4), init_strides=(1, 1))
        # /2
        pooling_stream = UpSampling2D((2, 2))(pooling_stream)
        pooling_stream, residual_stream = _Full_Resolution_Residual_block(pooling_stream, residual_stream, filters=48,
                                                                          pool_size=(2, 2), size=(2, 2),init_strides=(1, 1))
        pooling_stream, residual_stream = _Full_Resolution_Residual_block(pooling_stream, residual_stream, filters=48,
                                                                          pool_size=(2, 2), size=(2, 2),init_strides=(1, 1))
        # /1
        pooling_stream = UpSampling2D((2, 2))(pooling_stream)
        # Merge the two streams
        network = concatenate([pooling_stream,residual_stream], axis=3)
        # RU Layers
        network = _residual_block(network, filters=24, init_strides=(1, 1))
        network = _residual_block(network, filters=24, init_strides=(1, 1))
        network = _residual_block(network, filters=24, init_strides=(1, 1))
        # Classification layer
        classify = Conv2D(num_outputs, (1, 1), strides=(1, 1),padding='same',use_bias=False)(network)

        #classify = core.Reshape((img_w * img_h, n_label))(classify)
        
        classify = Activation('sigmoid', name='Classification')(classify)

        model = Model(input, classify)
        return model

    #@staticmethod
    def build_FRRN_A(self, input_shape, num_outputs):
        return self.build(input_shape, num_outputs)

def check_print():
    # Create a Keras Model
    #model=frrnBuilder.build_FRRN_A((128, 128, 3), 3)
    #model.summary()
    return

