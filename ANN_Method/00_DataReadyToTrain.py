import os
import pyrsgis.raster as raster
from sklearn.model_selection import train_test_split
from pyrsgis.convert import changeDimension
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import numpy as np
from tensorflow import keras

# Change the directory
os.chdir(r"C:\Users\cdhsn\miniconda3\envs\env_dl\Project Classification\Data_image\Dataset\LU_To_Label")

# Assign file names
Feature_PKT = "PKT_Extent_Clip_Inner_2018_09.tif"
Label_PKT = "Label_PKT_Extent_090261.tif"
Feature_Predic = "PKT_Extent_Clip_Inner_2018_01_11.tif"

# Read the rasters as array
ds1, Feature_PKT = raster.read(Feature_PKT, bands='all')
ds2, Label_PKT = raster.read(Label_PKT, bands=1)
ds3, Feature_Pre = raster.read(Feature_Predic, bands='all')

# Print the size of the arrays
print("Bangalore Multispectral image shape: ", Feature_PKT.shape)
print("Bangalore Binary built-up image shape: ", Label_PKT.shape)
print("Hyderabad Multispectral image shape: ", Feature_Pre.shape)

# # Clean the labelled data to replace NoData values by zero
# Label_PKT = (Label_PKT).astype(int)

# Reshape the array to single dimensional array
Feature_PKT = changeDimension(Feature_PKT)
Label_PKT = changeDimension (Label_PKT)
Feature_Pre = changeDimension(Feature_Pre)
nBands = Feature_PKT.shape[1]

print("Bangalore Multispectral image shape: ", Feature_PKT.shape)
print("Bangalore Binary built-up image shape: ", Label_PKT.shape)
print("Hyderabad Multispectral image shape: ", Feature_Pre.shape)

# Split testing and training datasets
xTrain, xTest, yTrain, yTest = train_test_split(Feature_PKT, Label_PKT, test_size=0.4, random_state=42)

print(xTrain.shape,yTrain.shape)
print(xTest.shape,yTest.shape)

# Normalise the data (Feature)
xTrain = xTrain / 255.0
xTest = xTest / 255.0

# # Normalise the data (Labels)
# yTrain = yTrain / 5
# yTest = yTest / 5

Feature_Pre = Feature_Pre / 255.0

# Reshape the data
xTrain = xTrain.reshape((xTrain.shape[0], 1, xTrain.shape[1]))
xTest = xTest.reshape((xTest.shape[0], 1, xTest.shape[1]))
Feature_Pre = Feature_Pre.reshape((Feature_Pre.shape[0], 1, Feature_Pre.shape[1]))

# Print the shape of reshaped data
print(xTrain.shape, xTest.shape, Feature_Pre.shape)

# Define the parameters of the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1, nBands)),
    keras.layers.Dense(18, activation='relu'),
    keras.layers.Dense(5, activation='softmax')])

# Define the accuracy metrics and parameters
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Run the model
model.fit(xTrain, yTrain, batch_size=5, epochs=25, verbose=1, validation_data=(xTest, yTest))
