import numpy as np
import h5py
from utility import vectorizeClass
import matplotlib.pyplot as plt

threshHold=541

## Read the windowed raw data
f=h5py.File('sensorData.hdf5','r')
fPre=h5py.File('processedDataGroup.hdf5','w')

windowSize=np.array([])

for id, group in f.items():
    idVector=vectorizeClass(group['label'][()])
    idVector=np.repeat(idVector[np.newaxis,:],threshHold-1,axis=0)
    grp=fPre.create_group(id)
    firstTag=True

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
                if firstTag==True:
                    grpCurrent=np.copy(current)
                    grpFuture=np.copy(future)
                    firstTag=False
                else:
                    grpCurrent=np.concatenate((grpCurrent,current),axis=0)
                    grpFuture=np.concatenate((grpFuture,future),axis=0)

    if firstTag==False:
        grp.create_dataset('current',data=grpCurrent)
        grp.create_dataset('future',data=grpFuture)

numValid=np.sum(windowSize>=threshHold)

print('The valid dataset number is ', numValid)
fig=plt.Figure()
fig.set_canvas(plt.gcf().canvas)
plt.hist(windowSize)
plt.xlabel('Number of sensor readings in one window')
plt.ylabel('Frequency')
fig.savefig("./DataSetImg/datasetWindowSize.pdf",format='pdf')
