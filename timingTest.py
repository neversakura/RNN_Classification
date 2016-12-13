import h5py, time
import numpy as np
from utility import randomData
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed

def buildModel(trainXShape, trainYShape, numLayers, neuonPerLayer):
    '''
    This function trains a RNN with given parameters
    '''
    model=Sequential()
    model.add(LSTM(neuonPerLayer[0], input_dim=trainXShape[2], input_length=trainXShape[1], return_sequences=True))
    for i in range(1,numLayers+1):
        model.add(LSTM(neuonPerLayer[i-1],return_sequences=True))

    model.add(TimeDistributed(Dense(trainYShape[2])))
    model.add(Activation('linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def varyingNeurons(low, high, f, V=1):
    numSample=1
    numLayers=1

    trainX, trainY = randomData(f, numSample)
    resultY=np.array([])
    resultX=np.array([])

    for numNeuron in range(low, high):

        model=buildModel(trainX.shape, trainY.shape, numLayers, [numNeuron])
        start=time.time()
        model.fit(trainX, trainY, nb_epoch=1, batch_size=1, verbose=V)
        resultY=np.append(resultY,time.time()-start)
        resultX=np.append(resultX,numNeuron)

    finalResult=np.concatenate((resultX[:,np.newaxis],resultY[:,np.newaxis]),axis=1)
    np.save('./resultData/timeVsNumneuron',finalResult)

def varyingLayers(low, high, f, V=1):
    numSample=1

    trainX, trainY = randomData(f, numSample)
    resultY=np.array([])
    resultX=np.array([])

    for numLayer in range(low, high):
        numNeuron=[15]*numLayer

        model=buildModel(trainX.shape, trainY.shape, numLayer, numNeuron)

        start=time.time()
        model.fit(trainX, trainY, nb_epoch=1, batch_size=1, verbose=V)
        resultY=np.append(resultY,time.time()-start)
        resultX=np.append(resultX,numLayer)

    finalResult=np.concatenate((resultX[:,np.newaxis],resultY[:,np.newaxis]),axis=1)
    np.save('./resultData/timeVsNumLayer',finalResult)

def varyingSamples(low, high, f, V=1):
    numLayer=1
    numNeuron=[15]

    resultY=np.array([])
    resultX=np.array([])

    for numSample in range(low, high):
        trainX, trainY = randomData(f, numSample)
        model=buildModel(trainX.shape, trainY.shape, numLayer, numNeuron)

        start=time.time()
        model.fit(trainX, trainY, nb_epoch=1, batch_size=1, verbose=V)
        resultY=np.append(resultY,time.time()-start)
        resultX=np.append(resultX,numSample)

    finalResult=np.concatenate((resultX[:,np.newaxis],resultY[:,np.newaxis]),axis=1)
    np.save('./resultData/timeVsNumSample',finalResult)



if __name__=="__main__":

    f=h5py.File('processedData.hdf5','r')
    varyingNeurons(10,51,f)
    varyingLayers(1,11,f)
    varyingSamples(1,101,f)
