import os
import numpy as np
try:
    from osgeo import gdal, osr, ogr
except ImportError:
    import gdal, osr, ogr

import tensorflow as tf
from tensorflow.keras import backend as K
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
  try:
    for gpu in gpus: #Currently, memory growth needs to be the same across GPUs
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e) #Memory growth must be set before GPUs have been initialized

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
import random
from tensorflow.keras.models import load_model

from lib.FRNet import *

# number of class type
n_label = 1

# dimension of train samples
pixelSize = 256

# some metrics or Loss
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def dice_coef(y_true, y_pred):
    smooth = 1.
    #print(y_pred[0,0,0])
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    #print(y_pred_f[0])
    intersection = K.sum(y_true_f * y_pred_f)
    #print(intersection[0,:,0])
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
def dice_coef_loss(y_true, y_pred):
    
    #print(tf.round(y_pred))
    return 1 - dice_coef(y_true, y_pred)
    
class RiceTrain():

    def __init__(self) -> None:
        self.trainDataDict = {}

    # save raster to disk
    def SaveRaster(self, fileName, proj, geoTrans, data):

        # type
        if 'int8' in data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # check shape of array
        if len(data.shape) == 3:
            im_height, im_width, im_bands = data.shape
        else:
            im_bands, (im_height, im_width) = 1, data.shape 

        # create file
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(fileName, im_width, im_height, im_bands, datatype, options=['COMPRESS=LZW'])
        if len(geoTrans) == 6:
            dataset.SetGeoTransform(geoTrans)
        if len(proj) > 0:
            dataset.SetProjection(proj)

        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(data)
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i+1).WriteArray(data[:, :, i])

    # get information of the raster
    def LoadRasterInfo(self, rasterFile, bandNum=1):
        # Open the file:
        dataset = gdal.Open(rasterFile)
        band = dataset.GetRasterBand(bandNum).ReadAsArray()
        geoTrans = dataset.GetGeoTransform()
        proj = dataset.GetProjection()
        rows, cols = band.shape

        return proj, geoTrans, rows, cols

    # read raster data
    def LoadRaster(self, rasterFile, bandN=7, bandNum=1):           

        # Open the file:
        dataset = gdal.Open(rasterFile)
        band = dataset.GetRasterBand(bandNum).ReadAsArray()
        geoTrans = dataset.GetGeoTransform()
        proj = dataset.GetProjection()
        rows, cols = band.shape

        if dataset.RasterCount == 1:

            band[band != 1] = 0
            return band

        band = band.reshape((rows, cols, 1))

        for i in range(2, dataset.RasterCount+1):
            #if i == dataset.RasterCount-1: continue # band 6
            #if i == 2: continue #
            tmpBand = dataset.GetRasterBand(i).ReadAsArray().reshape((rows, cols, 1))
            band = np.concatenate([band, tmpBand], axis = 2)
    
        return band # cv2.resize(band/255, (256, 256), interpolation=cv2.INTER_NEAREST)

        for i in range(2, dataset.RasterCount+1):
            tmpBand = dataset.GetRasterBand(i).ReadAsArray().reshape((rows, cols, 1))
            band = np.concatenate([band, tmpBand], axis = 2)
        # if bandN == 4: band = band[:,:,1:-2] # no 1 6 7
        # if bandN == 5: band = band[:,:,1:-1] # no 1 7
        # if bandN == 6: band = band[:,:,1:] # no 1 
        return band[:,:,1:-1] * 2.0000E-05 - 0.1
        # return band * 2.0000E-05 - 0.1

    def get_train_val(self, filepath, val_rate = 0.3):
        train_url = []    
        train_set = []
        val_set  = []
        for pic in os.listdir(filepath + os.sep + 'image/'):
            train_url.append(pic)
        random.seed(43)
        random.shuffle(train_url)
        total_num = len(train_url)
        val_num = int(val_rate * total_num)
        for i in range(len(train_url)):
            if i < val_num:
                val_set.append(train_url[i]) 
            else:
                train_set.append(train_url[i])
        return train_set,val_set

    # data for training
    def generateData(self, filepath, batch_size, data=[],bandN=7):  
        #print 'generateData...'
        while True:  
            imgTrain = []  
            labelTrain = []
            flagB = True
            batch = 0  
            for i in (range(len(data))): 
                url = data[i]
                
                # if not os.path.exists(filepath + os.sep + 'mask/' + url): 
                    #print('*************',  url)
                    # continue
                batch += 1
                imgData = self.LoadRaster(filepath + os.sep + 'image/' + url, bandN=bandN)
                labelData = self.LoadRaster(filepath + os.sep + 'mask/' + url).astype('float32')

                #imgData = cv2.resize(imgData, (240,240), interpolation=cv2.INTER_NEAREST)
                #labelData = cv2.resize(labelData, (240,240), interpolation=cv2.INTER_NEAREST)

                #print(url)
                imgData = np.expand_dims(imgData, axis = 0)
                #labelData = tf.keras.utils.to_categorical(labelData, num_classes=n_label) 
                labelData = np.expand_dims(labelData, axis = 0)
                labelData = np.expand_dims(labelData, axis = 3)

                if flagB:
                    imgTrain = imgData
                    labelTrain = labelData
                    flagB = False
                    #print(url)
                else:
                    imgTrain = np.concatenate([imgTrain, imgData])
                    labelTrain = np.concatenate([labelTrain, labelData])
                    #print(url)

                if batch % batch_size==0: 
                    yield (imgTrain, labelTrain) 
                    imgTrain = []  
                    labelTrain = []
                    flagB = True

    # data for validation
    def generateValidData(self, filepath, batch_size,data=[],bandN=7):  
        while True:  
            imgTrain = []  
            labelTrain = []
            flagB = True
            batch = 0  
            for i in (range(len(data))): 
                url = data[i]
                # if not os.path.exists(filepath + os.sep + 'mask/' + url): 
                    #print('*************',  url)
                    # continue
                batch += 1
                imgData = self.LoadRaster(filepath + os.sep + 'image/' + url, bandN=bandN)
                labelData = self.LoadRaster(filepath + os.sep + 'mask/' + url).astype('float32')
                
                #imgData = cv2.resize(imgData, (240,240), interpolation=cv2.INTER_NEAREST)
                #labelData = cv2.resize(labelData, (240,240), interpolation=cv2.INTER_NEAREST)

                imgData = np.expand_dims(imgData, axis = 0)
                #labelData = tf.keras.utils.to_categorical(labelData, num_classes=n_label) 
                labelData = np.expand_dims(labelData, axis = 0)
                labelData = np.expand_dims(labelData, axis = 3)

                if flagB:
                    imgTrain = imgData
                    labelTrain = labelData
                    flagB = False
                else:
                    imgTrain = np.concatenate([imgTrain, imgData])
                    labelTrain = np.concatenate([labelTrain, labelData])

                if batch % batch_size==0: 
                    yield (imgTrain, labelTrain) 
                    imgTrain = []  
                    labelTrain = []
                    flagB = True

    # train the model
    def TrainPaddyModel(self, inPath, modelSavePath, modelName, epochs=50, batch_size=4, bands=6):

        #self.get_train_val(inPath, 0.3)
        train_set, val_set = self.get_train_val(inPath, 0.3, bands, batch_size)
        train_numb = len(train_set)
        valid_numb = len(val_set)

        # Prepare model model saving directory.
        model_name = modelName + '_%s_m.{epoch:03d}.h5' % 't'
        filepath = os.path.join(modelSavePath, model_name)
        checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True)

        # prepare model  
        if modelName == 'frnet_v2': model = seg_frnet_v2(batch_size, pixelSize, pixelSize, bands, n_label)
        
        #model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics=['binary_accuracy', f1_m, precision_m, recall_m])
        model.compile(optimizer = 'Adam', loss = dice_coef_loss, metrics=['binary_accuracy', f1_m, precision_m, recall_m])
        #model.compile(optimizer = 'Adam', loss = fl, metrics=['binary_accuracy', f1_m, precision_m, recall_m])
        #model.summary()
        #return

        history = model.fit_generator(generator=self.generateData(inPath, batch_size, train_set, bandN=bands),steps_per_epoch=train_numb//batch_size,epochs=epochs,
                        validation_data=self.generateValidData(inPath, batch_size, val_set, bandN=bands),validation_steps=valid_numb//batch_size, callbacks=[checkpoint])   #

    # used fo predict
    def PredictPaddy(self, modelFile, inFiles, resultPath):
        
        if not os.path.exists(os.path.dirname(resultPath)): os.mkdir(os.path.dirname(resultPath))
        if not os.path.exists(resultPath): os.mkdir(resultPath)
        if type(inFiles) is str :
            inFiles =[inFiles]

        if isinstance(modelFile, str) and os.path.exists(modelFile):
            model = load_model(modelFile, custom_objects={'f1_m':f1_m, 'precision_m':precision_m, 'recall_m':recall_m, 'dice_coef_loss':dice_coef_loss})
        else:
            model = modelFile
            modelFile = 'T'
        #print(model)
        bgTiem = time.time()
        for inFile in inFiles:
            proj, geoTrans, rows, cols = self.LoadRasterInfo(inFile)
            data = self.LoadRaster(inFile, bandN=7) * 2.0000E-05 - 0.1 # train data
            if data.shape[2] > 7:
                data = data[:, :, 0:-1]

            #resultFile = resultPath + os.sep + os.path.splitext(os.path.basename(modelFile))[0] + '-p-' + os.path.basename(inFile)
            resultFile = resultPath + os.sep + 'predict' + os.path.basename(inFile)

            rows = data.shape[0]
            cols = data.shape[1]
            result = np.zeros((rows, cols)).astype('uint8')
            for row in range(rows//pixelSize):
                for col in range(cols//pixelSize):
                    tmpData = data[row*pixelSize:(row+1)*pixelSize, col*pixelSize:(col+1)*pixelSize]
                    tmpData = np.expand_dims(tmpData, axis = 0)
                    yPredict = model.predict(tmpData)
                    yPredict = yPredict.reshape(pixelSize, pixelSize)
                    yPredict[yPredict > 0.5] = 255

                    result[row*pixelSize:(row+1)*pixelSize, col*pixelSize:(col+1)*pixelSize] = yPredict

            #cv2.imwrite(resultFile, result)
            self.SaveRaster(resultFile, proj, geoTrans, result.astype('uint8'))
            #print(resultFile)
        print('done in', modelFile, time.time()-bgTiem, (time.time()-bgTiem)/len(inFiles)*1000, 'ms')

if __name__ == '__main__':

    riceTrain = RiceTrain()

    # location of the train dataset where includes folders of image and mask
    inPath = r''

    # location where to save the trained model file
    modelSavePath = r''
    
    # some examples to train the model
    riceTrain.TrainPaddyModel(inPath, modelSavePath, 'frnet_v2', epochs=80, batch_size=8, bands=5)

    # an example to predict
    inFiles = [r'E:\shuidao\yangben\sd-data\stack\LC08_L1TP_114028_20140601_20170422_01_T1_Train.tif',
              r'E:\shuidao\yangben\sd-data\stack\LC08_L1TP_114028_20140703_20170421_01_T1_Train.tif',
              r'E:\shuidao\yangben\sd-data\stack\LC08_L1TP_114028_20140820_20170420_01_T1_Train.tif',
              r'E:\shuidao\yangben\sd-data\stack\LC08_L1TP_114028_20140921_20170419_01_T1_Train.tif']
    # model file
    modelFile = r''
    # path to save the results
    resultPath = r''
    # do the prediction
    riceTrain.PredictPaddy(modelFile, inFiles, resultPath)
    
