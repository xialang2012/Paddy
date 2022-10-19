from tools import *

class CalVI(IOProcess):

    def CalVI1(self, inFile, outPath):
        data, porj, geotrans = self.LoadRaster(inFile)

        cols, rows, bands = data.shape[0], data.shape[1], data.shape[2]

        for i in range(bands):
            for j in range(bands):
                arr1 = data[:,:,i]
                arr2 = data[:,:,j]

                index1 = (arr1 - arr2) / (arr1 + arr2)
                index1 = np.squeeze(index1)
                outFile = outPath + os.sep + str(i ) + '_' + str(j) + '.tif'

                self.SaveRaster(outFile, '', '', index1)


calVIs = CalVI()

inFile = r'C:\Users\admin\Desktop\lt\HRRI-DAILI_FL1_2020-10-22_03-21-45-rect.dat'
outPath = r'C:\Users\admin\Desktop\lt\result'
calVIs.CalVI1(inFile, outPath)