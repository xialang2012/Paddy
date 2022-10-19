import os
import numpy as np 
import json
import time, datetime
from dateutil.parser import parse
try:
    from osgeo import gdal, osr, ogr
except ImportError:
    import gdal, osr, ogr

import xml.dom.minidom
import pandas as pd
import subprocess

class IOProcess():

    def __init__(self):
        pass

    def MatchtoSameGeo(self, rowColsInfo, geoTrans, inMask, inTransform):

        colsT = np.linspace(0, rowColsInfo[1]-1, num=rowColsInfo[1], dtype='int32')
        colsT = np.tile(colsT, rowColsInfo[0])
        rowsT = np.linspace(0, rowColsInfo[0]-1, num=rowColsInfo[0], dtype='int32').repeat(rowColsInfo[1])

        lon = geoTrans[1] * colsT + geoTrans[0]
        lat = geoTrans[5] * rowsT + geoTrans[3]

        # convert to rors and cols at the level of mask
        cols, rows = ((-inTransform[0] + lon) / inTransform[1]).astype('int32'),  ((-inTransform[3] + lat) / inTransform[5]).astype('int32')

        index = np.logical_and(np.logical_and(cols > 0, cols < inMask.shape[1]), np.logical_and(rows > 0, rows < inMask.shape[0]))

        resultData = np.zeros(rowColsInfo)
        resultData[rowsT[index], colsT[index]] = inMask[rows[index], cols[index]]

        return resultData

    def world_to_pixel(self, geo_matrix, x, y):
        """
        Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
        the pixel location of a geospatial coordinate
        """
        ul_x= geo_matrix[0]
        ul_y = geo_matrix[3]
        x_dist = geo_matrix[1]
        y_dist = geo_matrix[5]
        pixel = (x - ul_x) / x_dist
        line = -(ul_y - y) / y_dist
        return pixel.astype('int32'), line.astype('int32')  # x, y

    def GetInfo(self, rasterFile):
        # Open the file:
        dataset = gdal.Open(rasterFile)
        cols, rows = dataset.RasterXSize, dataset.RasterYSize
        geoTrans = dataset.GetGeoTransform()
        proj = dataset.GetProjection()
        dataset = None
        return proj, geoTrans, cols, rows

    # load raster file to memory
    def LoadRaster(self, rasterFile, bandNum=1, xoff=0, yoff=0, xsize=0, ysize=0):           
  
        # Open the file:
        dataset = gdal.Open(rasterFile)        
        geoTrans = dataset.GetGeoTransform()
        proj = dataset.GetProjection()
        rows, cols = dataset.RasterYSize, dataset.RasterXSize

        if (xsize == 0 and ysize==0):
            xsize, ysize = cols, rows

        band = dataset.GetRasterBand(bandNum).ReadAsArray(xoff, yoff, xsize, ysize)

        if dataset.RasterCount == 1:
            return band, proj, geoTrans

        band = band.reshape((rows, cols, 1))
        for i in range(2, dataset.RasterCount+1):
            tmpBand = dataset.GetRasterBand(i).ReadAsArray(xoff, yoff, xsize, ysize)
            tmpBand = np.expand_dims(tmpBand, axis=2)
            band = np.concatenate([band, tmpBand], axis = 2)
        
        return band, proj, geoTrans

    # write raster
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
        # , 'BIGTIFF=YES'
        dataset = driver.Create(fileName, im_width, im_height, im_bands, datatype, options=['COMPRESS=LZW','BIGTIFF=YES'])
        if len(geoTrans) == 6:
            dataset.SetGeoTransform(geoTrans)
        if len(proj) > 0:
            dataset.SetProjection(proj)

        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(data)
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i+1).WriteArray(data[:, :, i])

    def ReprojectShp(self, inShp, outShp, inRaster):
        tif = gdal.Open(inRaster)

        #shapefile with the from projection
        driver = ogr.GetDriverByName("ESRI Shapefile")
        dataSource = driver.Open(inShp, 1)
        layer = dataSource.GetLayer()

        #set spatial reference and transformation
        sourceprj = layer.GetSpatialRef()
        targetprj = osr.SpatialReference(wkt = tif.GetProjection())
        transform = osr.CoordinateTransformation(sourceprj, targetprj)

        to_fill = ogr.GetDriverByName("Esri Shapefile")
        ds = to_fill.CreateDataSource(outShp)
        outlayer = ds.CreateLayer('', targetprj, ogr.wkbPolygon)
        outlayer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))

        #apply transformation
        i = 0

        for feature in layer:
            transformed = feature.GetGeometryRef()
            transformed.Transform(transform)

            geom = ogr.CreateGeometryFromWkb(transformed.ExportToWkb())
            defn = outlayer.GetLayerDefn()
            feat = ogr.Feature(defn)
            feat.SetField('id', i)
            feat.SetGeometry(geom)
            outlayer.CreateFeature(feat)
            i += 1
            feat = None

    # shp cut
    def ShpCut(self, shpFile, rasterFile, resultRaster):
        options = gdal.WarpOptions(options='-co COMPRESS=LZW')
        outBand = gdal.Warp(srcDSOrSrcDSTab=rasterFile, destNameOrDestDS=resultRaster, cutlineDSName=shpFile, \
            cropToCutline=True)
        outBand= None
    
    # resize img
    def Resize(self, rasterFile, resultTaster, xSize, ySize):
        kwargs = { 'options': '-co COMPRESS=LZW', 'xRes': str(res), 'yRes':str(res), 'dstNodata':0 , 'srcNodata':0}
        gdal.Warp(outFile, inFile, **kwargs, dstSRS = 'EPSG:4326')

    def Resize(self, rasterFile, resultTaster, res):
        kwargs = { 'xRes': str(res), 'yRes':str(res), 'dstNodata':0 , 'srcNodata':0}
        gdal.Warp(resultTaster, rasterFile, **kwargs, dstSRS = 'EPSG:4326')

    def Reprojection(self, inRaster, resultRaster, dstSRS, res=30, kwargs = None):
        options= gdal.WarpOptions(options='-co COMPRESS=LZW')        
        if kwargs is None:
            kwargs = {'xRes': str(res), 'yRes':str(res)}
        gdal.Warp(resultRaster, inRaster, dstSRS=dstSRS, **kwargs)

    def deleteDir(self, root):
        dirlist = os.listdir(root)   
        for f in dirlist:
            filepath = root + '\\' + f   
            print(filepath)
            if os.path.isdir(filepath):  
                self.deleteDir(filepath)      
                os.rmdir(filepath)        
            else:
                os.remove(filepath)      
        os.rmdir(root)  


    def GetFilesDirbySuff(self, inPath, suff='.txt'):
        results = []
        datanames = os.listdir(inPath)
        for dataname in datanames:
            if os.path.splitext(dataname)[1] == suff:
                results.append(dataname)

        return results

    def HandleDir(self, dir):
        if type(dir) is str:
            if (not os.path.exists(dir)):
                os.mkdir(dir)
        else:
            for d in dir:
                if (not os.path.exists(d)):
                    os.mkdir(d)

    def Remove(self, fileName):
        if os.path.exists(fileName):
            try:
                os.remove(fileName)
            except Exception as e:
                print('bad delete')            

    def RemoveFile(self, fileName):
        if isinstance(fileName, list):
            for ii in fileName:
                self.Remove(ii)
        else:
            self.Remove(fileName)
