from tensorflow.keras import backend as k
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import UpSampling2D, add, concatenate

pixelSize = 384

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
    x = Conv2D(pixelSize, 3, strides=(1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # x = Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=3)(x)
    # x = Activation('relu')(x)

    x = bottleneck_Block(x, pixelSize, with_conv_shortcut=True)
    x = bottleneck_Block(x, pixelSize, with_conv_shortcut=False)
    x = bottleneck_Block(x, pixelSize, with_conv_shortcut=False)
    x = bottleneck_Block(x, pixelSize, with_conv_shortcut=False)

    return x


def transition_layer1(x, out_filters_list=[32, 64]):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x0 = BatchNormalization(axis=3)(x0)
    x0 = Activation('relu')(x0)

    x1 = Conv2D(out_filters_list[1], 3, strides=(2, 2),
                padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation('relu')(x1)

    return [x0, x1]


def make_branch1_0(x, out_filters=32):
    #x = basic_Block(x, out_filters, with_conv_shortcut=False)
    #x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def make_branch1_1(x, out_filters=64):
    #x = basic_Block(x, out_filters, with_conv_shortcut=False)
    #x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def fuse_layer1(x):
    x0_0 = x[0]
    x0_1 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[1])
    x0_1 = BatchNormalization(axis=3)(x0_1)
    x0_1 = UpSampling2D(size=(2, 2))(x0_1)
    x0 = add([x0_0, x0_1])

    x1_0 = Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x1_0 = BatchNormalization(axis=3)(x1_0)
    x1_1 = x[1]
    x1 = add([x1_0, x1_1])
    return [x0, x1]


def transition_layer2(x, out_filters_list=[32, 64, 128]):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x0 = BatchNormalization(axis=3)(x0)
    x0 = Activation('relu')(x0)

    x1 = Conv2D(out_filters_list[1], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(out_filters_list[2], 3, strides=(2, 2),
                padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x2 = BatchNormalization(axis=3)(x2)
    x2 = Activation('relu')(x2)

    return [x0, x1, x2]


def make_branch2_0(x, out_filters=32):
    #x = basic_Block(x, out_filters, with_conv_shortcut=False)
    #x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def make_branch2_1(x, out_filters=64):
    #x = basic_Block(x, out_filters, with_conv_shortcut=False)
    #x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def make_branch2_2(x, out_filters=128):
    #x = basic_Block(x, out_filters, with_conv_shortcut=False)
    #x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def fuse_layer2(x):
    x0_0 = x[0]
    x0_1 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[1])
    x0_1 = BatchNormalization(axis=3)(x0_1)
    x0_1 = UpSampling2D(size=(2, 2))(x0_1)
    x0_2 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[2])
    x0_2 = BatchNormalization(axis=3)(x0_2)
    x0_2 = UpSampling2D(size=(4, 4))(x0_2)
    x0 = add([x0_0, x0_1, x0_2])

    x1_0 = Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x1_0 = BatchNormalization(axis=3)(x1_0)
    x1_1 = x[1]
    x1_2 = Conv2D(64, 1, use_bias=False, kernel_initializer='he_normal')(x[2])
    x1_2 = BatchNormalization(axis=3)(x1_2)
    x1_2 = UpSampling2D(size=(2, 2))(x1_2)
    x1 = add([x1_0, x1_1, x1_2])

    x2_0 = Conv2D(32, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x2_0 = BatchNormalization(axis=3)(x2_0)
    x2_0 = Activation('relu')(x2_0)
    x2_0 = Conv2D(128, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x2_0)
    x2_0 = BatchNormalization(axis=3)(x2_0)
    x2_1 = Conv2D(128, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x2_1 = BatchNormalization(axis=3)(x2_1)
    x2_2 = x[2]
    x2 = add([x2_0, x2_1, x2_2])
    return [x0, x1, x2]


def transition_layer3(x, out_filters_list=[32, 64, 128, pixelSize]):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x0 = BatchNormalization(axis=3)(x0)
    x0 = Activation('relu')(x0)

    x1 = Conv2D(out_filters_list[1], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(out_filters_list[2], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[2])
    x2 = BatchNormalization(axis=3)(x2)
    x2 = Activation('relu')(x2)

    x3 = Conv2D(out_filters_list[3], 3, strides=(2, 2),
                padding='same', use_bias=False, kernel_initializer='he_normal')(x[2])
    x3 = BatchNormalization(axis=3)(x3)
    x3 = Activation('relu')(x3)

    return [x0, x1, x2, x3]


def make_branch3_0(x, out_filters=32):
    #x = basic_Block(x, out_filters, with_conv_shortcut=False)
    #x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def make_branch3_1(x, out_filters=64):
    #x = basic_Block(x, out_filters, with_conv_shortcut=False)
    #x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def make_branch3_2(x, out_filters=128):
    #x = basic_Block(x, out_filters, with_conv_shortcut=False)
    #x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def make_branch3_3(x, out_filters=pixelSize):
    #x = basic_Block(x, out_filters, with_conv_shortcut=False)
    #x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def fuse_layer3(x):
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


def final_layer(x, classes=1):
    #x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(classes, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('sigmoid', name='Classification')(x)
    return x


def final_layer(x, classes=1):
    #x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(classes, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('sigmoid', name='Classification')(x)
    return x

def seg_frnet(batch_size, height, width, channel, classes):
    inputs = Input(batch_shape=(batch_size,) + (height, width, channel))

    x = Conv2D(64, 3, strides=(1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(inputs)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    channels = [64, 128, 256]
    dilation_rates = [3, 6]

    # stage1, full feature layer
    s1_f = bottleneck_Block(x, pixelSize, with_conv_shortcut=True)
    s1_f = bottleneck_Block(s1_f, pixelSize, with_conv_shortcut=False)
    s1_f = bottleneck_Block(s1_f, pixelSize, with_conv_shortcut=False)
    s1_f = bottleneck_Block(s1_f, pixelSize, with_conv_shortcut=False)

    # stage2, full & aspp layer 1
    ## full layer
    s2_f = Conv2D(channels[0], 1, use_bias=False, kernel_initializer='he_normal')(s1_f)
    s2_f = BatchNormalization(axis=3)(s2_f)
    s2_f = Activation('relu')(s2_f)
    s2_f = basic_Block(s2_f, channels[0])
    s2_f = basic_Block(s2_f, channels[0])   # finish full layer
    ## aspp layer
    s2_a = Conv2D(channels[1], 1, use_bias=False, kernel_initializer='he_normal')(s1_f)   # up channels
    s2_a = BatchNormalization(axis=3)(s2_a)
    s2_a = Activation('relu')(s2_a)
    s2_a = Conv2D(channels[1], 3, padding='same', dilation_rate=dilation_rates[0], use_bias=False, kernel_initializer='he_normal')(s2_a) # finish aspp
    s2_a = BatchNormalization(axis=3)(s2_a)
    s2_a = Activation('relu')(s2_a)

    # stage3, full & aspp layer 2
    ## full layer
    s2_a_d = Conv2D(channels[0], 1, use_bias=False, kernel_initializer='he_normal')(s2_a)   # fusion aspp to full layer
    s2_a_d = BatchNormalization(axis=3)(s2_a_d)
    s2_a_d = Activation('relu')(s2_a_d)
    s3_f = add([s2_f, s2_a_d])
    s3_f = basic_Block(s3_f, channels[0])
    s3_f = basic_Block(s3_f, channels[0])   # finish full layer

    ## aspp layer 1
    s2_f_u = Conv2D(channels[1], 1, use_bias=False, kernel_initializer='he_normal')(s2_f)   # up channels
    s2_f_u = BatchNormalization(axis=3)(s2_f_u)
    s2_f_u = Activation('relu')(s2_f_u)
    x3_a_1 = add([s2_a, s2_f_u])
    x3_a_1 = Conv2D(channels[1], 3, padding='same', dilation_rate=dilation_rates[0], use_bias=False, kernel_initializer='he_normal')(x3_a_1) # finish aspp
    x3_a_1 = BatchNormalization(axis=3)(x3_a_1)
    x3_a_1 = Activation('relu')(x3_a_1)

    ## aspp layer 2
    s2_a_u = Conv2D(channels[2], 1, use_bias=False, kernel_initializer='he_normal')(s2_a)   # up channels from s2 a1
    s2_a_u = BatchNormalization(axis=3)(s2_a_u)
    s2_a_u = Activation('relu')(s2_a_u)

    s2_f_u = Conv2D(channels[2], 1, use_bias=False, kernel_initializer='he_normal')(s2_f)   # up channels from s2 full
    s2_f_u = BatchNormalization(axis=3)(s2_f_u)
    s2_f_u = Activation('relu')(s2_f_u)

    s1_u = Conv2D(channels[2], 1, use_bias=False, kernel_initializer='he_normal')(s1_f)   # up channels from s1 full
    s1_u = BatchNormalization(axis=3)(s1_u)
    s1_u = Activation('relu')(s1_u)

    x3_a_2 = add([s2_a_u, s2_f_u, s1_u])
    x3_a_2 = Conv2D(channels[2], 3, padding='same', dilation_rate=dilation_rates[1], use_bias=False, kernel_initializer='he_normal')(x3_a_2) # finish aspp
    x3_a_2 = BatchNormalization(axis=3)(x3_a_2)
    x3_a_2 = Activation('relu')(x3_a_2)

    x = concatenate([s3_f, x3_a_1, x3_a_2], axis=-1)
    #x = s3_f

    x = Conv2D(classes, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    out = Activation('sigmoid', name='Classification')(x)

    model = Model(inputs=inputs, outputs=out)

    return model

def seg_hrnet(batch_size, height, width, channel, classes):
    inputs = Input(batch_shape=(batch_size,) + (height, width, channel))

    x = stem_net(inputs)

    x = transition_layer1(x)
    x0 = make_branch1_0(x[0])
    x1 = make_branch1_1(x[1])
    x = fuse_layer1([x0, x1])

    x = transition_layer2(x)
    x0 = make_branch2_0(x[0])
    x1 = make_branch2_1(x[1])
    x2 = make_branch2_2(x[2])
    x = fuse_layer2([x0, x1, x2])

    x = transition_layer3(x)
    x0 = make_branch3_0(x[0])
    x1 = make_branch3_1(x[1])
    x2 = make_branch3_2(x[2])
    x3 = make_branch3_3(x[3])
    x = fuse_layer3([x0, x1, x2, x3])

    out = final_layer(x, classes=classes)

    model = Model(inputs=inputs, outputs=out)

    return model