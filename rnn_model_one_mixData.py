import h5py, time
import numpy as np
from utility import dataSplit
from utility import misClassificationRate
from modelList import buildModelOne



if __name__=="__main__":

    firstTag=True
    for numNeuron in range(10,15):
        for numLayer in range(1,3):

            result=np.array([numNeuron,numLayer])
            temp=np.array([])
            for i in range(10):
                trainX, trainY, testX, testY = dataSplit(0.8)
                model=buildModelOne(trainX.shape, trainY.shape, numNeuron, numLayer)
                model.fit(trainX, trainY, nb_epoch=10, batch_size=32, verbose=2)
                predictY=model.predict(testX)
                r=misClassificationRate(testY,predictY)
                temp=np.append(temp,r)

            result=np.append(result,np.mean(temp))
            result=np.append(result,np.std(temp))

            if firstTag==True:
                final=result[np.newaxis,:]
                firstTag=False
            else:
                final=np.concatenate((final,result[np.newaxis,:]),axis=0)

            np.save('./resultData/withPredictionMixmodel',final)
