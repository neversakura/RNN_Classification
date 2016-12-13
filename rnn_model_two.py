import h5py, time
import numpy as np
from utility import dataSplitGroup, misClassificationRate, convert2Classification, categoricalParse
from modelList import buildModelTwo


if __name__=="__main__":

    firstTag=True
    for numLayer in range(1,6):
        for numNeuron in range(10,21):

            result=np.array([numNeuron,numLayer])
            temp=np.array([])
            for i in range(20):
                trainX, trainY, testX, testY = dataSplitGroup(0.8)
                trainY=convert2Classification(trainY)
                trainY=trainY[:,0,:]
                model=buildModelTwo(trainX.shape, trainY.shape, numNeuron, numLayer)
                model.fit(trainX, trainY, nb_epoch=10, batch_size=32, verbose=2)
                predictY=model.predict(testX)
                testY=categoricalParse(testY)
                predictY=np.argmax(predictY,axis=1)
                r=np.sum(testY!=predictY)/predictY.size
                temp=np.append(temp,r)

            result=np.append(result,np.mean(temp))
            result=np.append(result,np.std(temp))

            if firstTag==True:
                final=result[np.newaxis,:]
                firstTag=False
            else:
                final=np.concatenate((final,result[np.newaxis,:]),axis=0)

            np.save('./resultData/withoutPrediction',final)
