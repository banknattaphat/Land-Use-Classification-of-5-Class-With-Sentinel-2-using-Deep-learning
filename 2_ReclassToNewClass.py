#Separate each Class (Specific 2 Class more labels data)
# Bulit- up / Urban
RF_U = Labels_RF[Labels_RF==1]
#Agriculture / Farm
RF_A = Labels_RF[Labels_RF==2]
#Green Area / Forest
RF_F = Labels_RF[Labels_RF==3]
EVI_F = Labels_EVI[Labels_EVI==2]
#Water Area / Lake_River_Dam
RF_W = Labels_RF[Labels_RF==4]
NDWI_W = Labels_NDWI[Labels_NDWI==2]
#Bear Land / Emtry Land
RF_M = Labels_RF[Labels_RF==5]

#No Value & No interes Value
RF_No = Labels_RF[Labels_RF==0]
EVI_No = Labels_EVI[Labels_EVI==1]
NDWI_No = Labels_NDWI[Labels_NDWI==1]

print('Label 5 Class of all data shape:\n')
print(f'Built Up: {RF_U.shape[0]} | Agriculture: {RF_A.shape[0]} | Green Area: {RF_F.shape[0]} & {EVI_F.shape[0]}')
print(f'Water Area: {RF_W.shape[0]} & {NDWI_W.shape[0]} | Miscellaneous: {RF_M.shape[0]}\n',"="*90)
print(f'Labels Zero or cut off value:\nRF: {RF_No.shape[0]} | EVI: {EVI_No.shape[0]} | NDWI: {NDWI_No.shape[0]}')
