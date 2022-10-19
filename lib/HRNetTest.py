from tensorflow.keras import backend as k
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import UpSampling2D, add, concatenate, MaxPooling2D

pixelSize = 256

channels = [16, 64, 128, 256]

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

def BtNz(x):
    x = BatchNormalization(axis=3)(x)
    return Activation("relu")(x)

def toFlow(x, channels, upSize=None, downSize=None, act=False):
    x = Conv2D(channels, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = BtNz(x)

    if upSize is not None:
        x = UpSampling2D(size=(upSize, upSize))(x)

    if downSize is not None:        
        if downSize == 2 and act:
            x = Conv2D(channels, 3, strides=(downSize, downSize), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
            x = BtNz(x)
        else:
            x = MaxPooling2D(pool_size=(downSize, downSize))(x)

    return x


def seg_frnet_v2_FulC(batch_size, height, width, channel, classes=1):
    inputs = Input(batch_shape=(batch_size,) + (height, width, channel))

    #pixelSize = 256 // 2
    s1_f = toFlow(inputs, channels[1])
    
    ## stage1, full feature layer
    s1_2f = MaxPooling2D(pool_size=(2, 2))(s1_f)    # downsample
    s1_4f = MaxPooling2D(pool_size=(4, 4))(s1_f)
    s1_8f = MaxPooling2D(pool_size=(8, 8))(s1_f)

    ## stage2, full & aspp layer 1
    s1_f = toFlow(s1_f, channels[0])
    s2_l1_f = basic_Block(s1_f, channels[0]) # L1
    s2_l1_f = basic_Block(s1_f, channels[0]) # L1
 
    s2_l2_f = toFlow(s1_2f, channels[1])
    s2_l2_f = basic_Block(s2_l2_f, channels[1]) # L2
    s2_l2_f = basic_Block(s2_l2_f, channels[1]) # L2

    ## stage 3
    s3_l1_f = add([s2_l1_f, toFlow(s2_l2_f, channels[0], upSize=2)])
    s3_l1_f = basic_Block(s3_l1_f, channels[0])      # L1
    s3_l1_f = basic_Block(s3_l1_f, channels[0])      # L1

    s3_l2_f = add([s2_l2_f, toFlow(s2_l1_f, channels[1], downSize=2)])
    s3_l2_f = basic_Block(s3_l2_f, channels[1])      # L2
    s3_l2_f = basic_Block(s3_l2_f, channels[1])      # L2

    #s3_l3_f = add([toFlow(s1_4f, channels[2]), toFlow(s2_l1_f, channels[2], downSize=4), toFlow(s2_l2_f, channels[2], downSize=2)])
    s3_l3_f = add([toFlow(s2_l1_f, channels[2], downSize=4), toFlow(s2_l2_f, channels[2], downSize=2)])
    s3_l3_f = basic_Block(s3_l3_f, channels[2]) #L3
    s3_l3_f = basic_Block(s3_l3_f, channels[2]) #L3


    ## stage 4
    s4_l1_f = add([s3_l1_f, toFlow(s3_l2_f, channels[0], upSize=2), toFlow(s3_l3_f, channels[0], upSize=4)])
    s4_l1_f = basic_Block(s4_l1_f, channels[0])      # L1
    s4_l1_f = basic_Block(s4_l1_f, channels[0])      # L1

    s4_l2_f = add([s3_l2_f, toFlow(s3_l1_f, channels[1], downSize=2), toFlow(s3_l3_f, channels[1], upSize=2)])
    s4_l2_f = basic_Block(s4_l2_f, channels[1])      # L2
    s4_l2_f = basic_Block(s4_l2_f, channels[1])      # L2

    s4_l3_f = add([s3_l3_f, toFlow(s3_l1_f, channels[2], downSize=4), toFlow(s3_l2_f, channels[2], downSize=2)])
    s4_l3_f = basic_Block(s4_l3_f, channels[2])      # L3
    s4_l3_f = basic_Block(s4_l3_f, channels[2])      # L3


    #s4_l4_f = add([toFlow(s1_8f, channels[3]), toFlow(s3_l1_f, channels[3], downSize=8), toFlow(s3_l2_f, channels[3], downSize=4), toFlow(s3_l3_f, channels[3], downSize=2)])
    s4_l4_f = add([toFlow(s3_l1_f, channels[3], downSize=8), toFlow(s3_l2_f, channels[3], downSize=4), toFlow(s3_l3_f, channels[3], downSize=2)])
    s4_l4_f = basic_Block(s4_l4_f, channels[3])      # L4
    s4_l4_f = basic_Block(s4_l4_f, channels[3])      # L4


    # upsampling
    s4_l2_f = UpSampling2D(size=(2, 2))(s4_l2_f)
    s4_l3_f = UpSampling2D(size=(4, 4))(s4_l3_f)
    s4_l4_f = UpSampling2D(size=(8, 8))(s4_l4_f)

    x = concatenate([s4_l1_f, s4_l2_f, s4_l3_f, s4_l4_f], axis=-1)

    if classes == 1:
        x = Conv2D(classes, 1, use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=3)(x)
        out = Activation('sigmoid', name='Classification')(x)
    else:
        x = Conv2D(classes, 1, use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=3)(x)
        out = Activation('softmax', name='Classification')(x)

    model = Model(inputs=inputs, outputs=out)

    return model
