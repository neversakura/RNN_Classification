import numpy as np
import h5py
from utility import numWindows
from utility import classLabel

## Extract 100 experimental presentations from the dataset
f=h5py.File('sensorData.hdf5','w')
## Define constants, use a moving windows with the size of 10 min
deltaT = 1./6.
oneMin = 1./60.
numSen=10

## Load metadata and sensor readings
dataType=np.dtype([('presentationID', 'u1'), ('date', 'S8'), ('class', 'S10'), ('t0', '<f8'), ('dt', '<f8')])
metadata = np.loadtxt('HT_Sensor_metadata.dat', skiprows=1, dtype=dataType)
dataset = np.loadtxt('HT_Sensor_dataset.dat', skiprows=1)

## Rescale the dataset
dataNoIndex=dataset[:, 2:]
dmax=np.amax(dataNoIndex,axis=0)
dmin=np.amin(dataNoIndex,axis=0)
print('Min value array is :')
print(dmin)
print('Max value array is :')
print(dmax)
shift=(dmax+dmin)/2.0
distance=(dmax-dmin)/2.0

for i in range(shift.size):
    dataset[:,i+2]=(dataset[:,i+2]-shift[i])/distance[i]

# Test if the scaling is successful
dmax=np.amax(dataset[:,2:],axis=0)
dmin=np.amin(dataset[:,2:],axis=0)
print('Min array after rescaling is :')
print(dmin)
print('Max array after rescaling is :')
print(dmax)


minLength=600
## Split each experimental presenations into 10 min window and store the result
## into a hdf5 file
for presentationID in range(100):
    # number of windows to be used
    dataset_ = dataset[ dataset[:,0] == presentationID, 1: ]
    # restricting the dataset
    nwin = numWindows(metadata[presentationID][4], deltaT)
    # caculating the class label for the presentation
    label_=classLabel(metadata[presentationID][2].decode('UTF-8'))
    # create a hdf5 group corresponding to the experimental presenation data
    groupName_=str(presentationID)
    grp=f.create_group(groupName_)
    dset=grp.create_dataset('label',data=label_,dtype='int')

    for j in range(nwin):
        # evaluating indices
        IDX = np.logical_and(dataset_[:,0] >= j*oneMin, dataset_[:,0] < j*oneMin + 10*oneMin )
        if np.flatnonzero(IDX).size==0:
            print('ID:'+groupName_+'; Window:'+str(j)+' is not a valid frame.')
        else:
            minLength=np.amin([np.flatnonzero(IDX).size,minLength])
            # Save the raw data to hdf5 file to /groupName_ with datasetName_
            datasetName_=str(j)
            dset=grp.create_dataset(datasetName_,data=dataset_[IDX, :])


print('Minimum frame length is:'+str(minLength))
f.close()
