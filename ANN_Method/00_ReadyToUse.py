import os, math, glob, random
random.seed(2)
import numpy as np
import pyrsgis.raster as raster
from pyrsgis.convert import changeDimension
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

#pytorch module #ใช้สำหรับ Pytorch เท่านั้น
import torch
from torchvision import transforms, datasets, models
import torch.nn as NN

#tensorflow module #Use this one frist
import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

#set directory from folder
os.chdir(r'E:\project Test\DataSet\01_IMG_Preparing')
#Dir_IMG & Dir_Label
IMG_Input = '01_IMG_Feature_Dataset\HKT_DOS_09022018_Inner.tif'
IMG_Predict = 01_IMG_Feature_Dataset\HKT_DOS_15032021_Inner.tif'
Label_Input = '02_IMG_Label_Dataset\HKT_DOS_With_RF_09022021.tif'


#Read File
dsIMG, IMG_In = raster.read(IMG_Input, bands='all')
dsps, IMG_Pre = raster.read(IMG_Predict, bands='all')
dsLab, Label_In = raster.read(Label_Input)
print(f'Detail file : {IMG_In.shape} | {Label_In.shape} | Value IMG >> {IMG_In.min()} | {IMG_In.max()}')

# ใช้สำหรับข้อมูลภาพที่มีการลดคุณสมบัติของภาพ (Normailzied)
# IMG_8bit = 'HKT_DOS_09022021_8bit.tif'
# dsnor, IMG_eightbit = raster.read(IMG_8bit, bands='all')

#Resize the shpae
IMG_eightbit = changeDimension(IMG_eightbit)
IMG_HKT_Pre = changeDimension(IMG_Pre)
Label_RF = changeDimension(Label_In)
nBands = IMG_eightbit.shape[1]

#train & test split file (สำหรับข้อมูลที่ต้องการนำไป Training & Testing)
X_tain, X_test, Y_train, Y_Test = train_test_split(IMG_eightbit, Label_RF, train_size=0.6, test_size=0.4)
#Normalized Train Data
Xtrain , Xtest = X_tain / 255 ,X_test / 255

# Reshape the data
xTrain = Xtrain.reshape((Xtrain.shape[0], 1, Xtrain.shape[1]))
xTest = Xtest.reshape((Xtest.shape[0], 1, Xtest.shape[1]))
print(f'xTrain {xTrain.shape} | xTest {xTest.shape}')

# Define the parameters of the model (Update Model 01/12/2021)
model = ks.Sequential([
    ks.layers.Flatten(input_shape=(1, nBands)),
    ks.layers.Dense(600, activation='relu'),
    ks.layers.Dropout(0.5),
    ks.layers.Dense(120, activation='relu'),
    ks.layers.Dropout(0.2),
    ks.layers.Dense(24, activation='relu'),
    ks.layers.Dense(6, activation='softmax')])

# Define the accuracy metrics and parameters
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Run the model
model.fit(xTrain, Y_train, batch_size=10, epochs=10, verbose=1, validation_data=(xTest, Y_Test))

# Predict for test data 
yTestPredicted = model.predict(xTest)
yTestPredicted = np.argmax(yTestPredicted, axis=1)

# Calculate and display the error metrics
cMatrix = confusion_matrix(Y_Test, yTestPredicted)
pScore = precision_score(Y_Test, yTestPredicted, average='micro')
rScore = recall_score(Y_Test, yTestPredicted, average='micro')
f1Score = f1_score(Y_Test, yTestPredicted, average='micro')

print("Confusion matrix: for 24 last neural\n", cMatrix)
print("\nP-Score: %.3f, R-Score: %.3f, F1-Score: %.3f" % (pScore, rScore, f1Score))

#Save Model
model.save('New_Model')
Current_Model = ks.models.load_model('New_Model')

#Normailzied to Predicted Feature
predicted = Current_Model.predict(IMG_HKT_Pre)
predicted = np.argmax(predicted, axis=1)
# Predict new data and export the probability raster
prediction = np.reshape(predicted, (dsps.RasterYSize, dsps.RasterXSize))

#Set the outfile name
outFile = 'Feature_PKT_Predict_15_03_20201_ANN.tif'
raster.export(prediction, dsps, filename=outFile, dtype='float')
