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

# Change the directory
os.chdir(r"C:\Users\std\Desktop\Dataset_Project")

# Assign file names
Feature_PKT = "Phuket_GEE_2018_4Band.tif"
Label_PKT = "HKT_GEE_RF_2018_Sieve.tif"

# Read the rasters as array
ds1, Feature_PKT = raster.read(Feature_PKT, bands='all')
ds2, Label_PKT = raster.read(Label_PKT, bands=1)

# Print the size of the arrays
print("Phuket Multispectral image shape: ", Feature_PKT.shape)
print("Phuket Landuse image shape: ", Label_PKT.shape)

# # Clean the labelled data to replace NoData values by zero
# Label_PKT = (Label_PKT).astype(int)

# Reshape the array to single dimensional array
Feature_PKT = changeDimension(Feature_PKT)
Label_PKT = changeDimension (Label_PKT)
nBands = Feature_PKT.shape[1]

print("Phuket Multispectral New image shape: ", Feature_PKT.shape)
print("Phuket Laduse New image shape: ", Label_PKT.shape)

print("Bangalore Multispectral image shape: ", Feature_PKT.shape)
print("Bangalore Binary built-up image shape: ", Label_PKT.shape)
print("Hyderabad Multispectral image shape: ", Feature_Pre.shape)

# Split testing and training datasets
X_Train, X_Test, Y_Train, Y_Test = train_test_split(Feature_PKT, Label_PKT, test_size=0.4, random_state=42)

# Reshape the data
xTrain = X_Train.reshape((X_Train.shape[0], 1, X_Train.shape[1]))
xTest = X_Test.reshape((X_Test.shape[0], 1, X_Test.shape[1]))

# Print the shape of reshaped data
print(xTrain.shape, xTest.shape)

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
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(6, activation='softmax')])

# Define the accuracy metrics and parameters
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Run the model
model.fit(X_Train, Y_Train, batch_size=5, epochs=50, verbose=1, validation_data=(X_Test, Y_Test))

# Predict for test data 
yTestPredicted = model.predict(X_Test)
yTestPredicted = np.argmax(yTestPredicted, axis=1)

# Calculate and display the error metrics
cMatrix = confusion_matrix(Y_Test, yTestPredicted)
pScore = precision_score(Y_Test, yTestPredicted, average='micro')
rScore = recall_score(Y_Test, yTestPredicted, average='micro')
f1Score = f1_score(Y_Test, yTestPredicted, average='micro')

print("Confusion matrix: for 24 nodes\n", cMatrix)
print("\nP-Score: %.3f, R-Score: %.3f, F1-Score: %.3f" % (pScore, rScore, f1Score))

#Save Model Seguential
model.save('HKT_GEE_2018.h5')
New_Model = keras.models.load_model('HKT_GEE_2018.h5')

predicted = New_Model.predict(Feature_PKT)
predicted = np.argmax(predicted, axis=1)

# Predict new data and export the probability raster
prediction = np.reshape(predicted, (ds1.RasterYSize, ds1.RasterXSize))
outFile = 'Feature_PKT_Predict_GEE_2018.tif'
raster.export(prediction, ds1, filename=outFile, dtype='float')
