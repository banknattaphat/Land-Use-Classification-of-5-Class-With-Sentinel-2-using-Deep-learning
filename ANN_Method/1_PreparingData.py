import os
import pyrsgis.raster as raster
import numpy as np

# Change the directory
Current_work = r"D:\Project\CNN Of Project\Input\PrepareData\LDD_LU_PKT"
os.chdir(Current_work)

# Assign file names
Feature_Phuket = 'Feature_LU_PKT_2561_LDD.tif'
Label_Phuket = 'Label_LU_PKT_2561_LDD.tif'
Phuket_Sept = 'Feature_LU_PKT_2562_LDD.tif'

# Read the rasters as array
ds1, feat_Phuket = raster.read(Feature_Phuket, bands='all')
ds2, label_Phuket = raster.read(Label_Phuket, bands=1)
ds3, feature_Sept = raster.read(Phuket_Sept, bands='all')

# Clean the labelled data to replace NoData values by zero
label_Phuket = (label_Phuket).astype(int)

# Reshape the array to single dimensional array
feat_Phuket = changeDimension(feat_Phuket)
label_Phuket = changeDimension (label_Phuket)
feature_Sept = changeDimension(feature_Sept)
nBands = feat_Phuket.shape[1]

#After change dimension
print("Phuket 09/02/2018 image shape: ", feat_Phuket.shape)
print("Phuket Labels image shape: ", label_Phuket.shape)
print("Phuket 01/11/2018 image shape: ", feature_Sept.shape)

#Separate 5 class of labels data
Label_U = label_Phuket[label_Phuket==1]
Label_A = label_Phuket[label_Phuket==2]
Label_F = label_Phuket[label_Phuket==3]
Label_W = label_Phuket[label_Phuket==4]
Label_M = label_Phuket[label_Phuket==5]
Label_Zero = label_Phuket[label_Phuket==0]

#Display shape in each class
print(f'Count of number each pixel/class:\nUrban: {Label_U.shape[0]} | Agriculture: {Label_A.shape[0]} | Forest: {Label_F.shape[0]} | Water Area: {Label_W.shape[0]} | Bare Land: {Label_M.shape[0]}')
print(f'No Value data: {Label_Zero.shape[0]}\n','='*90)

#Check min & max value in each class (Before)
print(f'Urban: {Label_U.max()} | Agriculture: {Label_A.max()} | Forest: {Label_F.max()} | Water Area: {Label_W.max()} | Bare Land: {Label_M.max()}')
print(f'No Value: {Label_Zero.max()}')

#Save Features & Labels data store to array file
Output_dir = os.path.join(Current_work, "Output_data")
os.mkdir(Output_dir)

np.save('ANN_Features_PKT.npy', feat_Phuket)
np.save('ANN_Labels_PKT.npy', label_Phuket)
np.save('ANN_Feature_Predict_PKT.npy',feature_Sept)
print(f'Finished for save data & Location here >>> %s' % (os.getcwd()))
