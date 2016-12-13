# EE660 Final Project Code

This is the repository for EE 660 @USC for Fall 2016 class. The data set can be found at [UCI Machine Learning Archive]. Also, part of my data import codes is adapted form the author's [GitHub Repo].

## External packages
The code is written in python 3. Apart from the standard scipy stack packages, the codes also need several open source proejcts to run.
* [h5py]: A Pythonic interface to the HDF5 binary data format.
* [Theano]: A Python library that allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays efficiently.
* [Keras]: A high-level neural networks library, written in Python and capable of running on top of either TensorFlow or Theano.

## Instructions
1. Download the dataset file into the folder. There will be two files: HT_Sensor_dataset.dat and HT_Sensor_metadata.dat.
2. Run the script data2hdf.py will load all the data into a hdf5 file.
3. Run the script dataPreProcessingGroup.py will process the data into 10 minutes windows and store them into a hdf5 file 'processedDataGroup.hdf5' with respect to their presentations.
4. Run the script dataPreProcessing.py will process the data into 10 minutes windows and store them into a single hdf5 file 'processedData.hdf5' within a single group.
5. The structure and parameters of RNN is in the script modelList.py. And can be adjusted from the script.
6. Three scripts files: rnn_model_one.py, rnn_model_two.py and rnn_model_one_mixData.py correspond to three different running settings described in the report.
7. Visulization of the time series data is done in [Jupyter Notebook], which is located in Visulize folder.
8. timingTest.py contains the experiment for complexity analysis.
9. All other support functions are in utility.py file.

[UCI Machine Learning Archive]: <https://archive.ics.uci.edu/ml/datasets/Gas+sensors+for+home+activity+monitoring>
[GitHub Repo]: <https://github.com/thmosqueiro/ENose-Decorr_Humdt_Temp>
[h5py]: <http://www.h5py.org/>
[Theano]: <http://deeplearning.net/software/theano/>
[Keras]: <https://keras.io/>
[Jupyter Notebook]: <http://jupyter.org/>
