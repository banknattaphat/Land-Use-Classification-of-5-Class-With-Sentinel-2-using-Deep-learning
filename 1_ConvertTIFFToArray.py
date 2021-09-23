import os
import pyrsgis.raster as raster
from pyrsgis.convert import changeDimension
import numpy as np

#Setting Work current directory
Input_dir = r"D:\Project\CNN Of Project\Input\PrepareData\NN_Test"
Output_dir = r"D:\Project\CNN Of Project\Input\PrepareData\NN_Test\Output"

os.chdir(Input_dir)

#Set File in each Varriable for using after
Feature_PKT = 'Feature_PKT_09022018_TCI.tif'
Labels_PKT = 'Label_LU_HKT_2561.tif'
Feature_predic = 'Feature_PKT_01112018_TCI.tif'

#Read all file to array
ds1, Feature_PKT = raster.read(Feature_PKT, bands='all')
ds2, Labels_PKT = raster.read(Labels_PKT, bands=1)
ds3, Feature_predic = raster.read(Feature_predic, bands='all')
print(f'Shape in other file\nFeature PKT: {Feature_PKT.shape}|Labels PKT: {Labels_PKT.shape}|Feature Predict: {Feature_predic.shape}')

#Change dimension for sum all every pixel
Feature_PKT = changeDimension(Feature_PKT)
Labels_PKT = changeDimension(Labels_PKT)
Feature_predic = changeDimension(Feature_predic)
print(f'New Shape in other file\nPhuket 09/02/2018 image shape: {Feature_PKT.shape}|Phuket Labels image shape: {Labels_PKT.shape}|Phuket 01/11/2018 image shape: {Feature_predic.shape}')

#If you have to save array file to work project in future, TO DO this code below
#Save array file to .npy file in output work
os.chdir(Output_dir)
np.save('Feature_NN_PKT.npy', Feature_PKT),np.save('Labels_NN_PKT.npy', Labels_PKT),np.save('Feature_Predicted_NN_PKT.npy', Feature_predic)
print('Save file to array here>> %s == Finished ==' % (os.getcwd()))
