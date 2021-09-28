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
