import numpy as np
import h5py
import random

def numWindows(tot, deltaT):
    """ Evaluates the number of windows that will be used
    given the total time (tot) of a particular induction.
    """
    return int( (tot - deltaT)*60. )

def maxLengthCal(dataset, metadata):
    '''
    Calculate the maxium length of a 10 min window
    '''
    maxiumLength=0
    deltaT = 1./6.
    oneMin = 1./60.

    for preID in range(100):
        # number of windows to be used
        dataset_ = dataset[ dataset[:,0] == preID, 1: ]
        # restricting the dataset
        nwin = numWindows(metadata[preID][4], deltaT)
        for j in range(nwin):
            # evaluating indices
            IDX = np.logical_and(dataset_[:,0] >= j*oneMin, dataset_[:,0] < j*oneMin + 10*oneMin )
            maxiumLength=np.amax([dataset_[IDX, 1:].shape[0],maxiumLength])

    return maxiumLength

def classLabel(name):
    '''
    Encode the classLabel
    '''
    if name=='banana':
        return 1
    elif name=='wine':
        return 2
    elif name=='background':
        return 0

def vectorizeClass(classLabel):
    '''
    Translate the classLabel to vector for RNN regression
    '''
    if classLabel == 1:
        return np.array([-1,1,-1])
    elif classLabel == 2:
        return np.array([-1,-1,1])
    elif classLabel == 0:
        return np.array([1,-1,-1])

def randomData(file,numSample):
    '''
    This function randomly pick sample points from the processedData h5py object
    '''
    sample=random.sample(range(len(file)),k=numSample)
    try:
        shape=file['0'].shape
    except KeyError:
        return np.array([])

    trainX=np.zeros((numSample,shape[1],shape[2]-3))
    trainY=np.zeros((numSample,shape[1],shape[2]))
    dataset=np.zeros(shape)

    for index in range(numSample):
        file[str(sample[index])].read_direct(dataset)
        trainX[index,:,:]=dataset[0,:,:-3]
        trainY[index,:,:]=dataset[1,:,:]

    return trainX, trainY

def dataSplit(ratio):
    '''
    This function split the total data set into training and test sets with respect
    to a ratio.
    '''
    f=h5py.File('processedData.hdf5','r')
    perm=np.random.permutation(len(f))

    try:
        shape=f['0'].shape
    except KeyError:
        return np.array([])

    trainingSize=int(len(f)*ratio)
    dataset=np.zeros(shape)
    trainX=np.zeros((trainingSize, shape[1], shape[2]-3))
    trainY=np.zeros((trainingSize, shape[1], shape[2]))

    testX=np.zeros((len(f)-trainingSize, shape[1], shape[2]-3))
    testY=np.zeros((len(f)-trainingSize, shape[1], shape[2]))

    for i in range(trainingSize):
        f[str(perm[i])].read_direct(dataset)
        trainX[i,:,:]=dataset[0,:,:-3]
        trainY[i,:,:]=dataset[1,:,:]

    for i in range(trainingSize,len(f)):
        f[str(perm[i])].read_direct(dataset)
        testX[i-trainingSize,:,:]=dataset[0,:,:-3]
        testY[i-trainingSize,:,:]=dataset[1,:,:]

    return trainX, trainY, testX, testY

def categoricalParse(dataY):
    sampleCategory=np.sum(dataY[:,:,-3:],axis=1)
    return np.argmax(sampleCategory, axis=1)

def misClassificationRate(testY, predictY):
    testCategory=categoricalParse(testY)
    predictCategory=categoricalParse(predictY)
    return np.sum(testCategory!=predictCategory)/testCategory.size

def hdf2np(dataset):
    result=np.zeros((dataset.shape))
    dataset.read_direct(result)
    return result

def dataSplitGroup(ratio):
    f=h5py.File('processedDataGroup.hdf5','r')
    perm=np.random.permutation(len(f))
    trainingSize=int(len(f)*ratio)
    firstTag=True
    for i in range(trainingSize):
        try:
            tempCurrent=np.zeros(f[str(perm[i])]['current'].shape)
            tempFuture=np.zeros(f[str(perm[i])]['future'].shape)
        except KeyError:
            continue

        f[str(perm[i])]['current'].read_direct(tempCurrent)
        f[str(perm[i])]['future'].read_direct(tempFuture)
        if firstTag==True:
            trainX=np.copy(tempCurrent)
            trainY=np.copy(tempFuture)
            firstTag=False
        else:
             trainX=np.concatenate((trainX,tempCurrent),axis=0)
             trainY=np.concatenate((trainY,tempFuture),axis=0)

    firstTag=True
    for i in range(trainingSize,len(f)):
        try:
            tempCurrent=np.zeros(f[str(perm[i])]['current'].shape)
            tempFuture=np.zeros(f[str(perm[i])]['future'].shape)
        except KeyError:
            continue

        f[str(perm[i])]['current'].read_direct(tempCurrent)
        f[str(perm[i])]['future'].read_direct(tempFuture)
        if firstTag==True:
            testX=np.copy(tempCurrent)
            testY=np.copy(tempFuture)
            firstTag=False
        else:
            testX=np.concatenate((testX,tempCurrent),axis=0)
            testY=np.concatenate((testY,tempFuture),axis=0)

    return trainX, trainY, testX, testY

def convert2Classification(dataY):
    result=dataY[:,:,-3:]
    result[result==-1]=0
    return result
