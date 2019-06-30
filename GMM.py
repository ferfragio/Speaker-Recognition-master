##############################################
##############################################
# Author: Fernando Fragío Sánchez
# Master Thesis: Speaker Recognition Based on Gaussian Mixture Models and 
# Neural Networks
# 
# Python code to train and test GMM with data from the Librispeech dataset
# using the MFCC of the audio samples.

import os
import sys
import librosa
import wave
import pickle
import tensorflow as tf
import librosa.display
import IPython.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
import numpy.matlib

import IPython.display as ipd

import os
import pandas as pd
import librosa
import glob
import librosa.display

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import csv
import os
from torch.utils.data.sampler import SubsetRandomSampler
import librosa
from sklearn import mixture
import IPython.display as ipd

######################################################################
# Check if a CUDA GPU is available and select our device. 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

######################################################################
# ----------------------
# Importing the Dataset
# ----------------------

# This part of the code can be used to create the dataset.csv file from 
# the SPEAKERS.TXT file.

#with open('./data/SPEAKERS.TXT', 'r') as in_file:
#    stripped = (line.strip() for line in in_file)
#    lines = (line.split("|") for line in stripped if line)
#    with open('./data/dataset.csv', 'w') as out_file:
#        writer = csv.writer(out_file)
#        writer.writerow(('ID', 'sex','subset','minutes','name'))
#        writer.writerows(lines)
#
#csvData = pd.read_csv('./data/dataset.csv')
#print(csvData.iloc[0, :])

# Number of Speakers we want to classify.
n_speakers = 100

class SRDataset(Dataset):
    # Arguments
    #  path to the csv file
    #  path to the audio files
    def __init__(self, csv_path, file_path):
        csvData = pd.read_csv(csv_path)
        #initialize lists to hold file names, labels, and folder numbers
        self.file_names = []
        self.labels = []
        self.folders = []
        #loop through the csv entries and only add entries from folders in the folder list
        label = 0
        i=0
        while label < n_speakers:
            if os.path.isdir('./data/audio/'+ csvData.iloc[i, 2].strip()+'/'+ str(csvData.iloc[i, 0])):
                allfiles = os.listdir('./data/audio/'+ csvData.iloc[i, 2].strip()+'/'+ str(csvData.iloc[i, 0])) # Lists all files in the current directory
                for item in allfiles:
                    if(item == '.DS_Store'):
                        allfiles.remove('.DS_Store')
                for item in allfiles: # iterate over all files in the current directory
                    list1 = os.listdir('./data/audio/'+ csvData.iloc[i, 2].strip()+'/'+ str(csvData.iloc[i, 0]) +'/'+ item)
                    for item2 in list1:
                        if item2.endswith('.flac'):
                            self.file_names.append(csvData.iloc[i, 2].strip()+'/'+ str(csvData.iloc[i, 0]) +'/'+ item +'/'+ item2)
                            self.labels.append(label)
                            self.folders.append(csvData.iloc[i, 2].strip())
                label = label+1
            i= i+1   
        self.file_path = file_path
        self.mixer = torchaudio.transforms.DownmixMono() 
        

    def __getitem__(self, index):
        #format the file path and load the file
        path = self.file_path + "/" + self.file_names[index]
        sound = torchaudio.load(path, out = None, normalization = True)
        #load returns a tensor with the sound data and the sampling frequency
        sr = sound[1]
        soundData = self.mixer(sound[0])
        soundData2 = sound[0].numpy()
        soundData3 = np.squeeze(soundData2)
        
        return soundData3, self.labels[index]

    def __len__(self):
        return len(self.file_names)

# Paths to the dataset.csv file and audio folder with the samples.
csv_path = './data/dataset.csv'
file_path = './data/audio'

dataset = SRDataset(csv_path, file_path)

class GenHelper(Dataset):
    def __init__(self, mother, length, mapping):
        # here is a mapping from this index to the mother ds index
        self.mapping=mapping
        self.length=length
        self.mother=mother

    def __getitem__(self, index):
        return self.mother[self.mapping[index]]

    def __len__(self):
        return self.length


def train_valid_split(ds, split_fold=10, random_seed=None):
    '''
    This is a pytorch generic function that takes a data.Dataset object and splits it to validation and training
    efficiently.
    :return:
    '''
    if random_seed!=None:
        np.random.seed(random_seed)

    dslen=len(ds)
    indices= list(range(dslen))
    valid_size=dslen//split_fold
    np.random.shuffle(indices)
    train_mapping=indices[valid_size:]
    valid_mapping=indices[:valid_size]
    train=GenHelper(ds, dslen - valid_size, train_mapping)
    valid=GenHelper(ds, valid_size, valid_mapping)

    return train, valid


train_set,test_set = train_valid_split(dataset, split_fold=10, random_seed=None)

print("Train set size: " + str(len(train_set)))
print("Test set size: " + str(len(test_set)))

kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {} #needed for using datasets on gpu

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 1, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1, **kwargs)

######################################################################
# --------------------------------
# Train and Test the GMM
# --------------------------------

# funciton to load the train data from a list given an speaker_id
def load_train_data(speaker_id):
    x =[]
    for batch_idx, (data, target) in enumerate(train_loader):
        if (target[0].numpy()==speaker_id):
            data = np.squeeze(data.numpy())
            x = np.concatenate((x, data), axis=0)
    return x

# Set the sampling rate to 16 kHz.
sr = 16000

# We fit the models of all the speakers

models = []

for i in range(0,n_speakers):

  X_train = load_train_data(i)
  mfcc_train = librosa.feature.mfcc(y=X_train, sr=sr, n_mfcc=20)
  clf = mixture.GaussianMixture(n_components=n_speakers, covariance_type='diag')
  models.append(clf.fit(mfcc_train.T))

# Function to load the test data and test it.

def test():
    
    Z=[]
    correct = 0
    acc = 0
    for batch_idx, (data, target) in enumerate(test_loader):

        X_test = np.squeeze(data.numpy())      
        mfcc_test = librosa.feature.mfcc(y=X_test, sr=sr, n_mfcc=20)

    
        for j in range(0,n_speakers):
            Z.append(models[j].score(mfcc_test.T))
      
        pred = np.argmax(Z)
        if(pred == target[0].numpy()):
            correct = correct+1
        Z=[]
    
    acc = correct/len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))        

# Test

test()