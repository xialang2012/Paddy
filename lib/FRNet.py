from tensorflow.keras import backend as k
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import UpSampling2D, add, concatenate, MaxPooling2D

pixelSize = 256

channels = [16, 64, 128, 256]

def conv3x3(x, out_filters, strides=(1, 1)):
    x = Conv2D(out_filters, 3, padding='same', strides=strides, use_bias=False, kernel_initializer='he_normal')(x)
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

def composeStage(x):
    x = add(x)
    return BtNz(x)

def seg_frnet_v2(batch_size, height, width, channel, classes=1):
    inputs = Input(batch_shape=(batch_size,) + (height, width, channel))

    #pixelSize = 256 // 2
    x = Conv2D(channels[1], 3, strides=(1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(inputs)
    x = BatchNormalization(axis=3)(x)
    s1_f = Activation('relu')(x)
    
    # stage1, full feature layer
    # s1_f = bottleneck_Block(x, 64, with_conv_shortcut=True)
    # s1_f = bottleneck_Block(s1_f, 64, with_conv_shortcut=False)
    # s1_f = bottleneck_Block(s1_f, 64, with_conv_shortcut=False)
    # s1_f = bottleneck_Block(s1_f, 64, with_conv_shortcut=False)
    s1_2f = MaxPooling2D(pool_size=(2, 2))(s1_f)    # downsample
    s1_4f = MaxPooling2D(pool_size=(4, 4))(s1_f)
    s1_8f = MaxPooling2D(pool_size=(8, 8))(s1_f)

    # stage2, full & aspp layer 1
    s1_f = Conv2D(channels[0], 1, use_bias=False, kernel_initializer='he_normal')(s1_f)
    s1_f = BatchNormalization(axis=3)(s1_f)
    s1_f = Activation('relu')(s1_f)
    s2_l1_f = basic_Block(s1_f, channels[0])
    s2_l1_f = basic_Block(s2_l1_f, channels[0]) # L1
    #s2_l1_f = basic_Block(s2_l1_f, channels[0]) # L1

    #s2_l2_f = basic_Block(s1_2f, channels[1], with_conv_shortcut=True)
    s2_l2_f = Conv2D(channels[1], 1, use_bias=False, kernel_initializer='he_normal')(s1_2f)
    s2_l2_f = BatchNormalization(axis=3)(s2_l2_f)
    s2_l2_f = Activation('relu')(s2_l2_f)
    s2_l2_f = basic_Block(s2_l2_f, channels[1]) # L2
    #s2_l2_f = basic_Block(s2_l2_f, channels[1]) # L2

    # stage 3
    s3_l1_f = Conv2D(channels[0], 1, use_bias=False, kernel_initializer='he_normal')(s2_l2_f)
    s3_l1_f = BatchNormalization(axis=3)(s3_l1_f)
    s3_l1_f = Activation('relu')(s3_l1_f)
    s3_l1_f = UpSampling2D(size=(2, 2))(s3_l1_f)
    s3_l1_f = add([s2_l1_f, s3_l1_f])
    s3_l1_f, s3_l2_f = FullConfusionUnit(s3_l1_f, channels[0])
    s3_l1_f = basic_Block(s3_l1_f, channels[0])      # L1
    #s3_l1_f = basic_Block(s3_l1_f, channels[0])      # L1

    s3_l2_f = Conv2D(channels[1], 1, use_bias=False, kernel_initializer='he_normal')(s3_l2_f)
    s3_l2_f = BatchNormalization(axis=3)(s3_l2_f)
    s3_l2_f = Activation('relu')(s3_l2_f)
    s3_l2_f = add([s3_l2_f, s2_l2_f])
    s3_l2_f, s3_l3_f = FullConfusionUnit(s3_l2_f, channels[1])
    s3_l2_f = basic_Block(s3_l2_f, channels[1])      # L2
    #s3_l2_f = basic_Block(s3_l2_f, channels[1])      # L2

    s3_l3_f = Conv2D(channels[2], 1, use_bias=False, kernel_initializer='he_normal')(s3_l3_f)
    s3_l3_f = BatchNormalization(axis=3)(s3_l3_f)
    s3_l3_f = Activation('relu')(s3_l3_f)
    s2_l2_f_d = MaxPooling2D(pool_size=(2, 2))(s2_l2_f)    # downsample
    s2_l2_f_d = Conv2D(channels[2], 1, use_bias=False, kernel_initializer='he_normal')(s2_l2_f_d)
    s2_l2_f_d = BatchNormalization(axis=3)(s2_l2_f_d)
    s2_l2_f_d = Activation('relu')(s2_l2_f_d)
    s1_4f_c = Conv2D(channels[2], 1, use_bias=False, kernel_initializer='he_normal')(s1_4f)
    s1_4f_c = BatchNormalization(axis=3)(s1_4f_c)
    s1_4f_c = Activation('relu')(s1_4f_c)
    s3_l3_f = add([s3_l3_f, s2_l2_f_d, s1_4f_c])
    s3_l3_f = basic_Block(s3_l3_f, channels[2])      # L3 
    #s3_l3_f = basic_Block(s3_l3_f, channels[2])      # L3    

    # stage 4
    s4_l1_f_d = Conv2D(channels[0], 1, use_bias=False, kernel_initializer='he_normal')(s3_l2_f)
    s4_l1_f_d = BatchNormalization(axis=3)(s4_l1_f_d)
    s4_l1_f_d = Activation('relu')(s4_l1_f_d)
    s4_l1_f_d = UpSampling2D(size=(2, 2))(s4_l1_f_d)

    s4_l1_f_2up = Conv2D(channels[0], 1, use_bias=False, kernel_initializer='he_normal')(s3_l3_f)
    s4_l1_f_2up = BatchNormalization(axis=3)(s4_l1_f_2up)
    s4_l1_f_2up = Activation('relu')(s4_l1_f_2up)
    s4_l1_f_2up = UpSampling2D(size=(4, 4))(s4_l1_f_2up)

    s4_l1_f = add([s3_l1_f, s4_l1_f_d, s4_l1_f_2up])
    s4_l1_f, s4_l2_f = FullConfusionUnit(s4_l1_f, channels[0])
    s4_l1_f = basic_Block(s4_l1_f, channels[0])      # L1
    #s4_l1_f = basic_Block(s4_l1_f, channels[0])      # L1

    #s4_l2_f_d = MaxPooling2D(pool_size=(2, 2))(s4_l2_f)    # downsample
    s4_l2_f_d = Conv2D(channels[1], 1, use_bias=False, kernel_initializer='he_normal')(s4_l2_f)   # channels
    s4_l2_f_d = BatchNormalization(axis=3)(s4_l2_f_d)
    s4_l2_f_d = Activation('relu')(s4_l2_f_d)
    s4_l2_f_up = Conv2D(channels[1], 1, use_bias=False, kernel_initializer='he_normal')(s3_l3_f)
    s4_l2_f_up = BatchNormalization(axis=3)(s4_l2_f_up)
    s4_l2_f_up = Activation('relu')(s4_l2_f_up)
    s4_l2_f_up = UpSampling2D(size=(2, 2))(s4_l2_f_up)
    s4_l2_f = add([s4_l2_f_d, s4_l2_f_up, s3_l2_f])
    s4_l2_f, s4_l3_f = FullConfusionUnit(s4_l2_f, channels[1])
    s4_l2_f = basic_Block(s4_l2_f, channels[1])      # L2
    #s4_l2_f = basic_Block(s4_l2_f, channels[1])      # L2


    #s4_l3_f_d = MaxPooling2D(pool_size=(2, 2))(s4_l3_f)    # downsample
    s4_l3_f_d = Conv2D(channels[2], 1, use_bias=False, kernel_initializer='he_normal')(s4_l3_f)   # channels
    s4_l3_f_d = BatchNormalization(axis=3)(s4_l3_f_d)
    s4_l3_f_d = Activation('relu')(s4_l3_f_d)
    s4_l3_f = add([s4_l3_f_d, s3_l3_f])
    s4_l3_f, s4_l4_f = FullConfusionUnit(s4_l3_f, channels[2])
    s4_l3_f = basic_Block(s4_l3_f, channels[2])      # L3
    #s4_l3_f = basic_Block(s4_l3_f, channels[2])      # L3


    #s4_l4_f_d = MaxPooling2D(pool_size=(2, 2))(s4_l4_f)    
    s4_l4_f_d = Conv2D(channels[3], 1, use_bias=False, kernel_initializer='he_normal')(s4_l4_f)   # channels
    s4_l4_f_d = BatchNormalization(axis=3)(s4_l4_f_d)
    s4_l4_f_d = Activation('relu')(s4_l4_f_d)
    
    s3_l3_f_d = Conv2D(channels[3], 1, use_bias=False, kernel_initializer='he_normal')(s3_l3_f)   # channels
    s3_l3_f_d = BatchNormalization(axis=3)(s3_l3_f_d)
    s3_l3_f_d = Activation('relu')(s3_l3_f_d)
    s3_l3_f_d = MaxPooling2D(pool_size=(2, 2))(s3_l3_f_d)    # downsample

    s2_l2_f_2d = Conv2D(channels[3], 1, use_bias=False, kernel_initializer='he_normal')(s2_l2_f)   # channels
    s2_l2_f_2d = BatchNormalization(axis=3)(s2_l2_f_2d)
    s2_l2_f_2d = Activation('relu')(s2_l2_f_2d)
    s2_l2_f_2d = MaxPooling2D(pool_size=(4, 4))(s2_l2_f_2d)# downsample

    s1_8f = Conv2D(channels[3], 1, use_bias=False, kernel_initializer='he_normal')(s1_8f)   # channels
    s1_8f = BatchNormalization(axis=3)(s1_8f)
    s1_8f = Activation('relu')(s1_8f)

    s4_l4_f = add([s2_l2_f_2d, s4_l4_f_d, s1_8f, s3_l3_f_d])
    s4_l4_f, s4_l5_f = FullConfusionUnit(s4_l4_f, channels[3])
    s4_l4_f = basic_Block(s4_l4_f, channels[3])      # L4
    #s4_l4_f = basic_Block(s4_l4_f, channels[3])      # L4

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
