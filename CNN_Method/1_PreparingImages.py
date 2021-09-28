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

""" if you need to convert feature & label data as array file. Now, TODO below this code """
""" Before to Downsample & Separate data for feature & label dada """

#Setting output directory to Save array file
Output_dir = r"D:\Project\CNN Of Project\CNN\Output"
os.chdir(Output_dir)

np.save('CNN_Features_128by128_PKT.npy', Features_PKT)
np.save('CNN_labels_128by128_PKT.npy', Labels_PKT)
print('Finished save file here >> %s' % (os.getcwd()))
""" ================================================================ """

""" TODO Downsample & Separate data for feature & label dada """
#Separate & Balance for 5 class % No Value
#Built up / Urban
Features_U = Feature_PKT[Label_PKT==1]
Labels_U = Label_PKT[Label_PKT==1]
#Agriculture / Farm
Features_A = Feature_PKT[Label_PKT==2]
Labels_A = Label_PKT[Label_PKT==2]
#Green Area / Forest
Features_F = Feature_PKT[Label_PKT==3]
Labels_F = Label_PKT[Label_PKT==3]
#Water Area / Lake_River
Features_W = Feature_PKT[Label_PKT==4]
Labels_W = Label_PKT[Label_PKT==4]
#Miscelleaoues / Bare land
Features_M = Feature_PKT[Label_PKT==5]
Labels_M = Label_PKT[Label_PKT==5]

#No value % Ignore value
No_Feature = Feature_PKT[Label_PKT==0]
No_Label = Label_PKT[Label_PKT==0]

print('Labels each class shape:')
print(f'Urban: {Labels_U.shape[0]} |Agriculture: {Labels_A.shape[0]} |Green Area: {Labels_F.shape[0]} |Water Area: {Labels_W.shape[0]} |Bare Land: {Labels_M.shape[0]}')
print(f'No Value: {No_Label.shape[0]}')

#Downsample majority 5 class
#Feature data (Downsample)
No_Feat_U = resample(No_Feature, replace=False, n_samples=Features_U.shape[0], random_state=2)
No_Feat_A = resample(No_Feature, replace=False, n_samples=Features_A.shape[0], random_state=2)
No_Feat_F = resample(No_Feature, replace=False, n_samples=Features_F.shape[0], random_state=2)
No_Feat_W = resample(No_Feature, replace=False, n_samples=Features_W.shape[0], random_state=2)
No_Feat_M = resample(No_Feature, replace=False, n_samples=Features_M.shape[0], random_state=2)

#labels data (Downsample)
No_Label_U = resample(No_Label, replace=False, n_samples=Labels_U.shape[0], random_state=2)
No_Label_A = resample(No_Label, replace=False, n_samples=Labels_A.shape[0], random_state=2)
No_Label_F = resample(No_Label, replace=False, n_samples=Labels_F.shape[0], random_state=2)
No_Label_W = resample(No_Label, replace=False, n_samples=Labels_W.shape[0], random_state=2)
No_Label_M = resample(No_Label, replace=False, n_samples=Labels_M.shape[0], random_state=2)

print('Features Balance majority 5 Class:')
print(f'Urban: {No_Feat_U.shape[0]} |Agriculture: {No_Feat_A.shape[0]} |Green Area: {No_Feat_F.shape[0]} |Water Area: {No_Feat_W.shape[0]} |Bare Land: {No_Feat_M.shape[0]}')
print('Labels Balance majority 5 Class:')
print(f'Urban: {No_Label_U.shape[0]} |Agriculture: {No_Label_A.shape[0]} |Green Area: {No_Label_F.shape[0]} |Water Area: {No_Label_W.shape[0]} |Bare Land: {No_Label_M.shape[0]}')

#Combine data in the one (name: Promp is Ready file)
Feature_Promp = np.concatenate((Features_U,Features_A,Features_F,Features_W,Features_M,No_Feat_U,No_Feat_A,No_Feat_F,No_Feat_W,No_Feat_M), axis=0)
Label_Promp = np.concatenate((Labels_U,Labels_A,Labels_F,Labels_W,Labels_M,No_Label_U,No_Label_A,No_Label_F,No_Label_W,No_Label_M), axis=0)

print(f'Feature shape: {Feature_Promp.shape} | Label shape: {Label_Promp.shape}')

#Change to output folder to save array file
Output_dir = r"D:\Project\CNN Of Project\CNN\Output"
os.chdir(Output_dir)

#Load faeture & label data promp
np.save('CNN_Images_Features_128by128_PKT.npy', Feature_Promp)
np.load('CNN_Images_Labels_128by128_PKT.npy', Label_Promp)
print('Finished save file here >> %s' % (os.getcwd()))
