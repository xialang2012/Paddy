# Implementation of 
[A full resolution deep learning network for paddy rice mapping using Landsat data](https://www.sciencedirect.com/science/article/pii/S0924271622002672)

    1. Requirements

We sugget to use the Conda to set the environment.  
a.Tensorflow > 2.6  
b.GDAL

    

    2.Run
```python shuidao.py```

    3.Dataset
Training dataset locates in the folder of dataset. For each image, 1 indicates paddy, 0 indicates non paddy, and 3 means background which should be masked when you generate the training samples, e.g. size of 256*256. 

For better visualization, ArcGIS is suggested to open the image, and Layer Properties -> Symbology -> Unique values should be employed. One demo is show below.

![demo image](images/demo.jpg)


