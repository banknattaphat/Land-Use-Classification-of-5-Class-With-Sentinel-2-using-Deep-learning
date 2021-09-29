import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

#Define the function split data
def train_test_spilt(Features, Labels, trainProp=0.6):
    datasize = Features.shape[0]
    sliceindex = int(datasize*trainProp)
    randindex = np.arange(datasize)
    random.shuffle(randindex)
    X_train = Features[[randindex[:sliceindex]], :, :, :][0]
    X_test = Features[[randindex[sliceindex:]], :, :, :][0]
    Y_train = Labels[randindex[:sliceindex]]
    Y_test = Labels[randindex[sliceindex:]]
    return (X_train, X_test, Y_train, Y_test)

#Spilt the data
X_train, X_test, Y_train, Y_test = train_test_spilt(Features_PKT, Labels_PKT)
print(f'Feature shape:\nX train: {X_train.shape} | X test: {X_test.shape}')
print(f'Lable shape:\nY train: {Y_train.shape} | Y test: {Y_test.shape}')
