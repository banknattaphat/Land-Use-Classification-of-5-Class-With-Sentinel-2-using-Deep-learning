import os
import pyrsgis.raster as raster
from pyrsgis.convert import changeDimension
import numpy as np

#Setting File in work directory
Labels_RF = r"D:\Project\CNN Of Project\ANN\ToAccuracy\Feauture_Landuse\HKT_RF_LU_2561.tif"
NDWI = r"D:\Project\CNN Of Project\ANN\Spectral Add on\NDWI_Edit.tif"
EVI = r"D:\Project\CNN Of Project\ANN\Spectral Add on\EVI_Edit.tif"

feature_PKT = r"D:\Project\CNN Of Project\ANN\Feature_PKT_09022018_TCI.tif"
predicted_PKT = r"D:\Project\CNN Of Project\ANN\Feature_PKT_01112018_TCI.tif"

#Read raster data (Feature data)
ds1, feature_pkt = raster.read(feature_PKT, bands='all')
ds2, predicted_pkt = raster.read(predicted_PKT, bands='all')

#Change Dimension (Feature Data)
Feature_PKT = changeDimension(feature_pkt)
Predicted_PKT = changeDimension(predicted_pkt)
nbands = Feature_PKT.shape[1]

#Read raster data (labels data)
ds3, label_rf = raster.read(Labels_RF, bands='all')
ds4, label_ndwi = raster.read(NDWI, bands=1)
ds5, label_evi = raster.read(EVI, bands=1)

#Change Dimension (labels data)
Labels_RF = changeDimension(label_rf)
Labels_NDWI = changeDimension(label_ndwi)
Labels_EVI = changeDimension(label_evi)

print(f'{label_rf.shape}, {label_ndwi.shape}, {label_evi.shape}')
print(f'Label shape >>> LU: {Labels_RF.shape} | NDWI: {Labels_NDWI.shape} | EVI: {Labels_EVI.shape}')
print(f'Feature shape >>> Feature PKT: {Feature_PKT.shape} | Predicted PKT: {Predicted_PKT.shape} | Bands: {nbands}')
