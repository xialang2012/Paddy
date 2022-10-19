from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

IMG_SIZE = 256

denseList = [64, 128, 256, 512, 1024]
#denseList = [16, 128, 512]


def unetPure_officail(input_size=(IMG_SIZE, IMG_SIZE, 3), num_class=1):
    
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv1)
    #conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same')(conv9)
    conv10 = Conv2D(num_class, 1, activation = 'sigmoid')(conv9)

    conv10 = BatchNormalization(axis=3)(conv10)
    out = Activation('sigmoid', name='Classification')(conv10)

    model = Model(inputs, conv10)

    return model
    
def unetPure(input_size=(IMG_SIZE, IMG_SIZE, 3), num_class=1):
    
    inputs = Input(input_size)

    conv1 = Conv2D(denseList[0], (3, 3), padding="same")(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    conv1 = Conv2D(denseList[0], (3, 3), padding="same")(conv1)
    #conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(denseList[1], (3, 3),  padding="same")(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)    
    conv2 = Conv2D(denseList[1], (3, 3), padding="same")(conv2)
    #conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(denseList[2], (3, 3), padding="same")(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3) 
    conv3 = Conv2D(denseList[2], (3, 3), padding="same")(conv3)
    #conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3) 
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(denseList[3], (3, 3),  padding="same")(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)     
    conv4 = Conv2D(denseList[3], (3, 3),  padding="same")(conv4)
    #conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)     
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(denseList[4], (3, 3), padding="same")(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)     
    conv5 = Conv2D(denseList[4], (3, 3),  padding="same")(conv5)
    #conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)   

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(denseList[3], (3, 3), padding="same")(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation("relu")(conv6)       
    conv6 = Conv2D(denseList[3], (3, 3), padding="same")(conv6)
    #conv6 = BatchNormalization()(conv6)
    conv6 = Activation("relu")(conv6) 

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(denseList[2], (3, 3), padding="same")(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7) 
    conv7 = Conv2D(denseList[2], (3, 3), padding="same")(conv7)
    #conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7) 

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(denseList[1], (3, 3),  padding="same")(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation("relu")(conv8)    
    conv8 = Conv2D(denseList[1], (3, 3), padding="same")(conv8)
    #conv8 = BatchNormalization()(conv8)
    conv8 = Activation("relu")(conv8) 

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(denseList[0], (3, 3),  padding="same")(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation("relu")(conv9)     
    conv9 = Conv2D(denseList[0], (3, 3), padding="same")(conv9)
    #conv9 = BatchNormalization()(conv9)
    conv9 = Activation("relu")(conv9)   
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same')(conv9)

    if num_class == 1:
        conv10 = Conv2D(num_class, (1, 1), activation="sigmoid")(conv9)
    else:
        conv10 = Conv2D(num_class, (1, 1), activation="softmax")(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    #model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=["binary_accuracy"])
    return model

def rsnetb(input_size=(IMG_SIZE, IMG_SIZE, 3), num_class=1):
    denseList = [32, 64, 128, 256, 512]
    inputs = Input(input_size)

    conv1 = Conv2D(denseList[0], (3, 3), padding="same")(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    conv1 = Conv2D(denseList[0], (3, 3), padding="same")(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(denseList[1], (3, 3),  padding="same")(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)    
    conv2 = Conv2D(denseList[1], (3, 3), padding="same")(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(denseList[2], (3, 3), padding="same")(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3) 
    conv3 = Conv2D(denseList[2], (3, 3), padding="same")(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3) 
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(denseList[3], (3, 3),  padding="same")(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)     
    conv4 = Conv2D(denseList[3], (3, 3),  padding="same")(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)     
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(denseList[4], (3, 3), padding="same")(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)     
    conv5 = Conv2D(denseList[4], (3, 3),  padding="same")(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)   

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(denseList[3], (3, 3), padding="same")(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation("relu")(conv6)       
    conv6 = Conv2D(denseList[3], (3, 3), padding="same")(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation("relu")(conv6) 

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(denseList[2], (3, 3), padding="same")(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7) 
    conv7 = Conv2D(denseList[2], (3, 3), padding="same")(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7) 

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(denseList[1], (3, 3),  padding="same")(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation("relu")(conv8)    
    conv8 = Conv2D(denseList[1], (3, 3), padding="same")(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation("relu")(conv8) 

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(denseList[0], (3, 3),  padding="same")(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation("relu")(conv9)     
    conv9 = Conv2D(denseList[0], (3, 3), padding="same")(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation("relu")(conv9)   

    if num_class == 1:
        conv10 = Conv2D(num_class, (1, 1), activation="sigmoid")(conv9)
    else:
        conv10 = Conv2D(num_class, (1, 1), activation="softmax")(conv9)

    #conv10 = Conv2D(num_class, (1, 1), activation="sigmoid")(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    #model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=["binary_accuracy"])
    return model

def unetPureDepth3(pretrained_weights=None, input_size=(IMG_SIZE, IMG_SIZE, 3), num_class=2):
    inputs = Input(input_size)
    conv1 = Conv2D(denseList[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(denseList[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(denseList[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(denseList[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(denseList[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(denseList[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(denseList[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(denseList[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    #drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(denseList[4], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(denseList[4], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(denseList[3], 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))

    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(denseList[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(denseList[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(denseList[2], 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(merge6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(denseList[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(denseList[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(denseList[1], 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(denseList[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(denseList[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(denseList[0], 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(denseList[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(denseList[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    #conv9 = Conv2D(num_class, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    if num_class == 2:
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        loss_function = 'binary_crossentropy'
    else:
        conv10 = Conv2D(num_class, 1, activation='softmax')(conv9)
        loss_function = 'categorical_crossentropy'
    model = Model(inputs, conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss=loss_function, metrics=["binary_accuracy"])
    #model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

def unetPure1(pretrained_weights=None, input_size=(IMG_SIZE, IMG_SIZE, 3), num_class=2):
    inputs = Input(input_size)
    conv1 = Conv2D(denseList[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(denseList[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(denseList[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(denseList[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(denseList[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(denseList[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(denseList[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(denseList[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(denseList[4], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(denseList[4], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = Conv2D(denseList[3], 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))

    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(denseList[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(denseList[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(denseList[2], 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(denseList[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(denseList[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(denseList[1], 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(denseList[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(denseList[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(denseList[0], 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(denseList[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(denseList[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    #conv9 = Conv2D(num_class, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)

    if num_class == 2:
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        loss_function = 'binary_crossentropy'
    else:
        conv10 = Conv2D(num_class, 1, activation='softmax')(conv9)
        loss_function = 'categorical_crossentropy'
    model = Model(inputs=inputs, outputs=conv10)

    #model.compile(optimizer=Adam(lr=1e-4), loss=loss_function, metrics=["binary_accuracy"])
    #model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

def conv_block(x, num_filters):
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def BuildUNet(input_size=(IMG_SIZE, IMG_SIZE, 3),):
    size = 256
    num_filters = [64, 128, 256, 512]
    inputs = Input(input_size)

    skip_x = []
    x = inputs

    ## Encoder
    for f in num_filters:
        x = conv_block(x, f)
        skip_x.append(x)
        x = MaxPool2D((2, 2))(x)

    ## Bridge
    x = conv_block(x, num_filters[-1])

    num_filters.reverse()
    skip_x.reverse()

    ## Decoder
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2))(x)
        xs = skip_x[i]
        x = Concatenate()([x, xs])
        x = conv_block(x, f)

    ## Output
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)

    return Model(inputs, x)