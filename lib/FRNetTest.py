from tensorflow.keras import backend as k
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import UpSampling2D, add, concatenate, MaxPooling2D

pixelSize = 256

channels = [32, 64, 128, 256]

def conv3x3(x, out_filters, strides=(1, 1)):
    x = Conv2D(out_filters, 3, padding='same', strides=strides, use_bias=False, kernel_initializer='he_normal')(x)
    return x

def basic_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    x = conv3x3(input, out_filters, strides)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = conv3x3(x, out_filters)
    x = BatchNormalization(axis=3)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation('relu')(x)
    return x

def bottleneck_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    expansion = 4
    de_filters = int(out_filters / expansion)

    x = Conv2D(de_filters, 1, use_bias=False, kernel_initializer='he_normal')(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(de_filters, 3, strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(out_filters, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation('relu')(x)
    return x

def stem_net(input):

    x = Conv2D(channels[3], 3, strides=(1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # x = Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=3)(x)
    # x = Activation('relu')(x)

    # x = bottleneck_Block(x, pixelSize, with_conv_shortcut=True)
    # x = bottleneck_Block(x, pixelSize, with_conv_shortcut=False)
    # x = bottleneck_Block(x, pixelSize, with_conv_shortcut=False)
    # x = bottleneck_Block(x, pixelSize, with_conv_shortcut=False)

    return x

def transition_layerFR1(x, out_filters_list=channels):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x0 = BatchNormalization(axis=3)(x0)
    x0 = Activation('relu')(x0)

    x1 = Conv2D(out_filters_list[1], 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation('relu')(x1)

    return [x0, x1]

def make_branchFR1_0(x, out_filters=channels[0]):
    #x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x_down = toFlow(x, out_filters*2, downSize=2)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x, x_down

def make_branchFR1_1(x, out_filters=channels[1]):
    #x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x

def fuse_layerFR1(x):
    x0_0 = x[0]
    x0_1 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[1])
    x0_1 = BatchNormalization(axis=3)(x0_1)
    x0_1 = UpSampling2D(size=(2, 2))(x0_1)
    #x0 = add([x0_0, x0_1])

    # x1_0 = Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    # x1_0 = BatchNormalization(axis=3)(x1_0)
    # x1_1 = x[1]
    # x1 = add([x1_0, x1_1])
    #return [x0, x1]
    return [add([x0_0, x0_1]), x[1]]

def transition_layerFR2(x, out_filters_list=channels):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x0 = BatchNormalization(axis=3)(x0)
    x0 = Activation('relu')(x0)

    x1 = Conv2D(out_filters_list[1], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(out_filters_list[2], 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x2 = BatchNormalization(axis=3)(x2)
    x2 = Activation('relu')(x2)

    return [x0, x[1], x2]

def make_branchFR2_0(x, out_filters=channels[0]):
    #x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x_down = toFlow(x, out_filters*2, downSize=2)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x, x_down

def make_branchFR2_1(x, out_filters=channels[1]):
    #x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x_down = toFlow(x, out_filters*2, downSize=2)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x, x_down

def make_branchFR2_2(x, out_filters=channels[2]):
    #x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x

def fuse_layerFR2(x):
    x0_0 = x[0]
    x0_1 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[1])
    x0_1 = BatchNormalization(axis=3)(x0_1)
    x0_1 = UpSampling2D(size=(2, 2))(x0_1)
    x0_2 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[2])
    x0_2 = BatchNormalization(axis=3)(x0_2)
    x0_2 = UpSampling2D(size=(4, 4))(x0_2)
    x0 = add([x0_0, x0_1, x0_2])

    # x1_0 = Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    # x1_0 = BatchNormalization(axis=3)(x1_0)
    x1_1 = x[1]
    x1_2 = Conv2D(64, 1, use_bias=False, kernel_initializer='he_normal')(x[2])
    x1_2 = BatchNormalization(axis=3)(x1_2)
    x1_2 = UpSampling2D(size=(2, 2))(x1_2)
    #x1 = add([x1_0, x1_1, x1_2])
    x1 = add([x1_1, x1_2])

    # x2_0 = Conv2D(32, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    # x2_0 = BatchNormalization(axis=3)(x2_0)
    # x2_0 = Activation('relu')(x2_0)
    # x2_0 = Conv2D(128, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x2_0)
    # x2_0 = BatchNormalization(axis=3)(x2_0)
    # x2_1 = Conv2D(128, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    # x2_1 = BatchNormalization(axis=3)(x2_1)
    # x2_2 = x[2]
    # x2 = add([x2_0, x2_1, x2_2])
    #return [x0, x1, x2]
    return [x0, x1, x[2]]

def transition_layerFR3(x, out_filters_list=channels):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x0 = BatchNormalization(axis=3)(x0)
    x0 = Activation('relu')(x0)

    x1 = Conv2D(out_filters_list[1], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(out_filters_list[2], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[2])
    x2 = BatchNormalization(axis=3)(x2)
    x2 = Activation('relu')(x2)

    x3 = Conv2D(out_filters_list[3], 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[2])
    x3 = BatchNormalization(axis=3)(x3)
    x3 = Activation('relu')(x3)

    return [x0, x1, x2, x3]

def make_branchFR3_0(x, out_filters=channels[0]):
    #x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x_down = toFlow(x, out_filters*2, downSize=2)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x, x_down

def make_branchFR3_1(x, out_filters=channels[1]):
    #x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x_down = toFlow(x, out_filters*2, downSize=2)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x, x_down

def make_branchFR3_2(x, out_filters=channels[2]):
    #x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x_down = toFlow(x, out_filters*2, downSize=2)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x, x_down

def make_branchFR3_3(x, out_filters=channels[3]):
    #x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x

def fuse_layerFR3(x):
    x0_0 = x[0]
    x0_1 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[1])
    x0_1 = BatchNormalization(axis=3)(x0_1)
    x0_1 = UpSampling2D(size=(2, 2))(x0_1)
    x0_2 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[2])
    x0_2 = BatchNormalization(axis=3)(x0_2)
    x0_2 = UpSampling2D(size=(4, 4))(x0_2)
    x0_3 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[3])
    x0_3 = BatchNormalization(axis=3)(x0_3)
    x0_3 = UpSampling2D(size=(8, 8))(x0_3)
    x0 = concatenate([x0_0, x0_1, x0_2, x0_3], axis=-1)
    return x0

def final_layerFR(x, classes=1):
    #x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(classes, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('sigmoid', name='Classification')(x)
    return x

def FullConfusionUnit(input, channels, with_conv_shortcut=False):
    x = conv3x3(input, channels)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = conv3x3(x, channels)
    x = BatchNormalization(axis=3)(x)

    if with_conv_shortcut:
        residual = Conv2D(channels, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation('relu')(x)

    #x_half = MaxPooling2D(pool_size=(2, 2))(x)    # downsample
    x_half = Conv2D(channels, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x_half = BatchNormalization(axis=3)(x_half)
    x_half = Activation('relu')(x_half)

    return x, x_half

def BtNz(x):
    x = BatchNormalization(axis=3)(x)
    return Activation("relu")(x)

def toFlow(x, channels, upSize=None, downSize=None):
    x = Conv2D(channels, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = BtNz(x)

    if upSize is not None:
        x = UpSampling2D(size=(upSize, upSize))(x)

    if downSize is not None:        
        if downSize == 2:
            x = Conv2D(channels, 3, strides=(downSize, downSize), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
            x = BatchNormalization(axis=3)(x)
            x = Activation('relu')(x)
        else:
            x = MaxPooling2D(pool_size=(downSize, downSize))(x)

    return x


def seg_frnet_v1_test(batch_size, height, width, channel, classes):

    inputs = Input(batch_shape=(batch_size,) + (height, width, channel))

    # 32, 64, 128, pixelSize
    # l1, l2, l3, l4
    x = stem_net(inputs)

    s3_in = toFlow(x, 128, downSize=4)     #l3  add
    s4_in = toFlow(x, channels[3], downSize=8)    #l4  add

    x = transition_layerFR1(x)
    x0, x0_down = make_branchFR1_0(x[0])
    x1 = make_branchFR1_1(add([x[1], x0_down]))
    x = fuse_layerFR1([x0, x1])

    x = transition_layerFR2(x)
    x0, x0_down = make_branchFR2_0(x[0])
    x1, x0_down = make_branchFR2_1(add([x[1], x0_down]))
    #x2 = make_branch2_2(x[2])
    x2 = make_branchFR2_2(add([x[2], x0_down, s3_in]))     # add
    x = fuse_layerFR2([x0, x1, x2])

    x = transition_layerFR3(x)
    x0, x0_down = make_branchFR3_0(x[0])
    x1, x0_down = make_branchFR3_1(add([x[1], x0_down]))
    x2, x0_down = make_branchFR3_2(add([x[2], x0_down]))
    #x3 = make_branch3_3(x[3])
    x3 = make_branchFR3_3(add([x[3], x0_down, s4_in]))     # add
    x = fuse_layerFR3([x0, x1, x2, x3])

    out = final_layerFR(x0, classes=classes)

    model = Model(inputs=inputs, outputs=out)

    return model