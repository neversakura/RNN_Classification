from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed

def buildModelOne(trainXShape, trainYShape, numNeuron, NumLayer):
    '''
    This function trains a RNN with both prediction and classfication task
    '''
    model=Sequential()
    model.add(LSTM(numNeuron, input_dim=trainXShape[2], input_length=trainXShape[1], return_sequences=True))
    model.add(Dropout(0.2))
    for i in range(NumLayer):
        model.add(LSTM(numNeuron,return_sequences=True))
        model.add(Dropout(0.2))

    model.add(TimeDistributed(Dense(trainYShape[2])))
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def buildModelTwo(trainXShape, trainYShape, numNeuron, NumLayer):
    '''
    This function trains a RNN with classfication task only
    '''
    model=Sequential()
    model.add(LSTM(numNeuron, input_dim=trainXShape[2], input_length=trainXShape[1], return_sequences=True))
    model.add(Dropout(0.2))
    for i in range(NumLayer-1):
        model.add(LSTM(numNeuron,return_sequences=True))
        model.add(Dropout(0.2))
    model.add(LSTM(numNeuron))
    model.add(Dropout(0.2))
    model.add(Dense(trainYShape[1]))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model
