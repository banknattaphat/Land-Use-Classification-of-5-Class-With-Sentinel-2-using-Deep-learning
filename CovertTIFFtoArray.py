import pyrsgis.raster as raster
import numpy as np
import os

#Identify TiF File from work dir
input_dir = r"D:\Project\CNN Of Project\Input\PrepareData\NN_Test"
output_dir = r"D:\Project\CNN Of Project\Input\Output"

#Change current wokr dir
"""Import input for using TIF file"""
os.chdir(input_dir)

#Identifity input data
Feature_PKT = 'Feature_PKT_09022018_TCI.tif'
Label_PKT = 'Label_LU_HKT_2561.tif'
TestModel_PKT = 'Feature_PKT_01112018_TCI.tif'

#Read TIF(raster) file as array
ds1, featurePKT = raster.read(Feature_PKT, band='all')
ds2, labelPKT = raster.read(Label_PKT, band='all')
ds3, featuretest_PKT = raster.read(TestModel_PKT, band='all')

#Check data process before to array size
print(f'Feature Shape: {featurePKT.shape}\nLabel Shape: {labelPKT.shape}\nTestFeature_PKT: {featuretest_PKT.shape}')
