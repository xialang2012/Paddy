import os
import gdal
import numpy as np 

from tools import IOProcess
class PaddyFloodMethod(IOProcess):

    def __init__(self):
        pass
    
    def CalNDVI(self, nir, red):
        return (nir*1.0 - red) / (nir*1.0  + red)

    def CalLSWI(self, nir, sw):
        return (nir*1.0  - sw) / (nir*1.0  + sw)

    def CalEVI(self, blue, red, nir):
        return 2.5*((nir*2.0E-05 - 0.1) - (red*2.0E-05 - 0.1)) / ((nir*2.0E-05 - 0.1) + 6*(red*2.0E-05 - 0.1) - 7.5*(blue*2.0E-05 - 0.1) + 1.0)

    def LoadData(self, inPath):
        fileName = os.path.basename(inPath)
        fileB2 = inPath + os.sep + fileName + '_B2.TIF'
        fileB4 = inPath + os.sep + fileName + '_B4.TIF'
        fileB5 = inPath + os.sep + fileName + '_B5.TIF'
        fileB6 = inPath + os.sep + fileName + '_B6.TIF'
        fileBQA = inPath + os.sep + fileName + '_BQA.TIF'
        
        band2, proj, geoTrans = self.LoadRaster(fileB2) # blue
        band4, proj, geoTrans = self.LoadRaster(fileB4) # red
        band5, proj, geoTrans = self.LoadRaster(fileB5) # nir
        band6, proj, geoTrans = self.LoadRaster(fileB6) # swir
        bqa, proj, geoTrans = self.LoadRaster(fileBQA) # bqa
        print(fileName, band4.shape[1], band4.shape[0], bqa.shape[1], bqa.shape[0])

        return band2, band4, band5, band6, bqa, proj, geoTrans, fileName

    def CalF(self, inPath, outPath):
        
        band2, band4, band5, band6, bqa, proj, transform, fileName = self.LoadData(inPath)
        if self.geoTran == '': self.geoTran = transform     # get geotrans projection and rows, cols info for the first swatch
        if self.proj == '': self.proj = proj
        if len(self.rowColsInfo) == 0:
            self.rowColsInfo = band2.shape

        evi = self.CalEVI(band2, band4, band5).astype('float32')
        ndvi = self.CalNDVI(band5, band4).astype('float32')
        lswi = self.CalLSWI(band5, band6).astype('float32')

        resultF = np.zeros(bqa.shape).astype('byte')
        resultF[lswi > ndvi] = 1

        clear1, clear2, clear3, clear4 = bqa == 2720, bqa == 2724, bqa == 2728, bqa == 2732  
        indexT = ~np.logical_or(np.logical_or(clear1, clear2), np.logical_or(clear3, clear4)) # get non-clear pixel
        resultF[indexT] = 0        
        fileNameF = outPath + os.sep + fileName + '_F.TIF'
        self.SaveRaster(fileNameF, proj, transform, resultF)        

        evi[indexT] = 0
        fileNameEVI = outPath + os.sep + fileName + '_EVI.TIF'
        #self.SaveRaster(fileNameEVI, proj, transform, evi) 

        if len(self.resultP) == 0:
            self.resultP = resultF
            self.eviMax = evi
        else:
            colsT = np.linspace(0, self.rowColsInfo[1]-1, num=self.rowColsInfo[1], dtype='int32')
            colsT = np.tile(colsT, self.rowColsInfo[0])
            rowsT = np.linspace(0, self.rowColsInfo[0]-1, num=self.rowColsInfo[0], dtype='int32').repeat(self.rowColsInfo[1])


            lon = self.geoTran[1] * colsT + self.geoTran[0]
            lat = self.geoTran[5] * rowsT + self.geoTran[3]

            cols, rows = ((-transform[0] + lon) / transform[1]).astype('int32'),  ((-transform[3] + lat) / transform[5]).astype('int32')

            index = np.logical_and(np.logical_and(cols > 0, cols < band2.shape[1]), np.logical_and(rows > 0, rows < band2.shape[0]))
            
            self.resultP[rowsT[index], colsT[index]] += resultF[rows[index], cols[index]]

            tmpEvi = np.zeros(self.rowColsInfo)
            tmpEvi[rowsT[index], colsT[index]] = evi[rows[index], cols[index]]
            indexTmp = self.eviMax < tmpEvi
            self.eviMax[indexTmp] = tmpEvi[indexTmp]

            # colShif = (self.geoTran[0]-transform[0])/30.0
            # rowShif = (self.geoTran[3]-transform[3])/30.0

            # rowsS, colsS = band2.shape[0], band2.shape[1]
            
            # rowBeg, rowEnd, colBeg, colEnd = self.CalInter(rowShif, colShif, rowsS, colsS)
            # rowBegO, rowEndO, colBegO, colEndO = 0, int(rowsS - rowShif), 0, int(colsS - colShif)

            # if rowEndO > self.rowColsInfo[0]:
            #     rowEndO = int(rowEndO - abs(self.rowColsInfo[0] - rowEndO))
            #     #rowEnd = int(rowEnd - abs(self.rowColsInfo[0] - rowEndO))
            
            # if colEndO > self.rowColsInfo[1]:
            #     colEndO = int(colEndO - abs(self.rowColsInfo[1] - colEndO))
            #     #colEnd = int(colEnd - abs(self.rowColsInfo[1] - colEndO))

            # if rowEndO > rowEnd:
            #     rowEndO = rowEnd
            # self.resultP[rowBegO:rowEndO, colBegO:colEndO] += resultF[rowBeg:rowEnd, colBeg:colEnd]

            #indexTmp = self.eviMax < evi
            #self.eviMax[indexTmp] = evi[indexTmp]
        
        #return proj, transform

    def CalPaddy(self, inPath, outPath):
        self.resultP = ''
        self.eviMax = ''
        self.geoTran = ''
        self.proj = ''
        self.rowColsInfo = []
        for p in os.listdir(inPath):
            if os.path.isdir(inPath + os.sep + p):
                self.CalF(inPath + os.sep + p, outPath)
        if len(self.resultP) != 0:
            self.SaveRaster(outPath + os.sep + 'max_evi.tif', self.proj, self.geoTran, self.eviMax)
            self.SaveRaster(outPath + os.sep + 'resultP.tif', self.proj, self.geoTran, self.resultP)

            paddyArea = np.zeros(self.resultP.shape).astype('byte')
            index = np.where((self.eviMax >= 0.55) & (self.resultP >= 1))
            paddyArea[index] = 1
            self.SaveRaster(outPath + os.sep + 'resultPaddyRice.tif', self.proj, self.geoTran, paddyArea)

pdMethod = PaddyFloodMethod()

inPath = r'D:\mask\2017-119-028-daqing'
outPath = inPath
pdMethod.CalPaddy(inPath, outPath)