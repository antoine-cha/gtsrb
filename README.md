# GTSRB Challenge



The repository contains 5 files besides the README.md :
* data.lua : functions to interface with the unzipped version of the data
* getdata.lua : command line interface to data.lua
* learn.lua : training functions
* main.lua : to choose parameters and launch experiments
* test.lua : tests a trained network, displays the confusion matrix 

## data.lua

The processing is done by data.lua functions

#### shuffle
Shuffle a table in place

#### csvToTable
Transforms a CSV file into a table indexed by filenames

#### imageToTensorFiles
Extract images from ppm files to t7 files (one per class)

#### datasetFromClasses
Transform the per-class t7 files into a dataset object (dataset.data, 
dataset.labels) and writes it.

#### formatTestData
Perform the 2 previous operations on the test data (the folders have a different structure).



## getdata.lua
To be executed for data setup.
The origin dirs are hard coded.

    Gets the data in the right format
    -s, --set (default "test")  Set to be processed
    -e, --extraction  (default true) extraction from ppm files ?
    -c, --creation (default true) creation of dataset from t7 files ?
    -d, --destination (default nil) target directory to write

## learn.lua

#### getDataset
Returns the dataset from a given file, changes labels to 1,-1 if only 2 classes

#### createNetwork
Returns the (non-trained) 108 network from the paper, 
only takes the number of classes as input.

#### trainNetwork
Trains a network given as arg. 
* Training uses optim.sgd
* Can be done on CPU or GPU
* An example of the table of params is in main.ilua
* The network is saved every <params.freq_save> epochs

## main.lua 
Setup the experiment, and launch using learn.lua's functions.  
The models are saved at ./models/  
It will create a new file for every experiment.  
Can't retrain an existing network for now.


## test.lua
Test a network, the file has to be given in the script, and displays the confusion
matrix.  
**Warning :** Assumes 43 classes as output


