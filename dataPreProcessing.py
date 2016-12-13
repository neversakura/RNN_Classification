import numpy as np
import h5py
from utility import vectorizeClass
import matplotlib.pyplot as plt

threshHold=541

## Read the windowed raw data
f=h5py.File('sensorData.hdf5','r')
fPre=h5py.File('processedData.hdf5','w')

windowSize=np.array([])
nameTag=0

for id, group in f.items():
    idVector=vectorizeClass(group['label'][()])
    idVector=np.repeat(idVector[np.newaxis,:],threshHold-1,axis=0)

    for window, dataset in group.items():
        if window!='label':
            windowSize=np.append(windowSize,dataset.shape[0])
            if dataset.shape[0]>=threshHold:
                temp=np.zeros(dataset.shape)
                dataset.read_direct(temp)
                current=temp[:threshHold-1,1:]
                future=temp[1:threshHold,1:]
                current=np.concatenate((current,idVector),axis=1)
                future=np.concatenate((future,idVector),axis=1)
                current=np.reshape(current,(1,current.shape[0],current.shape[1]))
                future=np.reshape(future,(1,future.shape[0],future.shape[1]))
                result=np.concatenate((current,future),axis=0)
                fPre.create_dataset(str(nameTag),data=result)
                nameTag=nameTag+1

fPre.close()
