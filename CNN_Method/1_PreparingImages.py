import os, time, math, glob, random
random.seed(2)
import numpy as np
import pyrsgis.raster as raster

Img_Dir = r"D:\Project\CNN Of Project\CNN\ImageChips_128by128"
os.chdir(Img_Dir)

#Get the number of file (Count images)
nSample = len(glob.glob('*.tif'))

#Get and read information about imaage chips
ds, tempArr = raster.read(os.listdir(Img_Dir)[0])
nbands, X_axis, Y_axis = ds.RasterCount, ds.RasterXSize, ds.RasterYSize

#Create empty arrays to store data
Features_PKT = np.empty((nSample, nbands, X_axis, Y_axis))
Labels_PKT = np.empty((nSample))

#Loop read & stack file
for n, file in enumerate(glob.glob('*.tif')):
    ds, tempArr = raster.read(file)
    
    tempLabel = os.path.splitext(file)[0].split('_')[-1]
    
    Features_PKT[n, :, :, :] = tempArr
    Labels_PKT[n] = tempLabel
    
    if n % 6000 == 0:
        print('Sample read: %d 0f %d' % (n,nSample))

print(f'Feature Shape: {Features_PKT.shape}\nLabels Shape: {Labels_PKT.shape}')
print(f'Feature Value: Min {Features_PKT.min()} | Max {Features_PKT.max()}\nLabels Value: Min {Labels_PKT.min()} | Max {Labels_PKT.max()}')
