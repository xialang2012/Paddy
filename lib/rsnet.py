import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout, Activation
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 256

def rsnet(input_size=(IMG_SIZE, IMG_SIZE, 3), num_class=1):
    # Note about BN and dropout: https://stackoverflow.com/questions/46316687/how-to-include-batch-normalization-in-non-sequential-keras-model
    
    inputs = Input(input_size)

    # -----------------------------------------------------------------------
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',
                    kernel_regularizer=regularizers.l2(1e-4))(inputs)
    conv1 = BatchNormalization(momentum=0.7)(conv1) if True else conv1
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',
                    kernel_regularizer=regularizers.l2(1e-4))(conv1)
    conv1 = BatchNormalization(momentum=0.7)(conv1) if True else conv1
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # -----------------------------------------------------------------------
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                    kernel_regularizer=regularizers.l2(1e-4))(pool1)
    conv2 = BatchNormalization(momentum=0.7)(conv2) if True else conv2
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                    kernel_regularizer=regularizers.l2(1e-4))(conv2)
    conv2 = BatchNormalization(momentum=0.7)(conv2) if True else conv2
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # -----------------------------------------------------------------------
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                    kernel_regularizer=regularizers.l2(1e-4))(pool2)
    conv3 = BatchNormalization(momentum=0.7)(conv3) if True else conv3
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                    kernel_regularizer=regularizers.l2(1e-4))(conv3)
    conv3 = BatchNormalization(momentum=0.7)(conv3) if True else conv3
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # -----------------------------------------------------------------------
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',
                    kernel_regularizer=regularizers.l2(1e-4))(pool3)
    conv4 = BatchNormalization(momentum=0.7)(conv4) if True else conv4
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',
                    kernel_regularizer=regularizers.l2(1e-4))(conv4)
    conv4 = BatchNormalization(momentum=0.7)(conv4) if True else conv4
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # -----------------------------------------------------------------------
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',
                    kernel_regularizer=regularizers.l2(1e-4))(pool4)
    conv5 = BatchNormalization(momentum=0.7)(conv5) if True else conv5
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',
                    kernel_regularizer=regularizers.l2(1e-4))(conv5)
    conv5 = BatchNormalization(momentum=0.7)(conv5) if True else conv5
    # -----------------------------------------------------------------------
    up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',
                    kernel_regularizer=regularizers.l2(1e-4))(up6)
    conv6 = Dropout(0.2)(conv6) if not True else conv6
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',
                    kernel_regularizer=regularizers.l2(1e-4))(conv6)
    conv6 = Dropout(0.2)(conv6) if not True else conv6
    # -----------------------------------------------------------------------
    up7 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',
                    kernel_regularizer=regularizers.l2(1e-4))(up7)
    conv7 = Dropout(0.2)(conv7) if not True else conv7
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',
                    kernel_regularizer=regularizers.l2(1e-4))(conv7)
    conv7 = Dropout(0.2)(conv7) if not True else conv7
    # -----------------------------------------------------------------------
    up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',
                    kernel_regularizer=regularizers.l2(1e-4))(up8)
    conv8 = Dropout(0.2)(conv8) if not True else conv8
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',
                    kernel_regularizer=regularizers.l2(1e-4))(conv8)
    conv8 = Dropout(0.2)(conv8) if not True else conv8
    # -----------------------------------------------------------------------
    up9 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',
                    kernel_regularizer=regularizers.l2(1e-4))(up9)
    conv9 = Dropout(0.2)(conv9) if not True else conv9
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',
                    kernel_regularizer=regularizers.l2(1e-4))(conv9)
    conv9 = Dropout(0.2)(conv9)
    # -----------------------------------------------------------------------
    #clip_pixels = np.int32(params.overlap / 2)  # Only used for input in Cropping2D function on next line
    #crop9 = Cropping2D(cropping=((clip_pixels, clip_pixels), (clip_pixels, clip_pixels)))(conv9)
    # -----------------------------------------------------------------------
    conv10 = Conv2D(num_class, (1, 1), activation='sigmoid')(conv9)
    # -----------------------------------------------------------------------
    model = Model(inputs, conv10)

    return model

if __name__=='__main__':

    
    pixelSize = 256
    bands = 3
    n_label = 1

    model = rsnet((pixelSize, pixelSize, bands), n_label)

    print(model.summary())