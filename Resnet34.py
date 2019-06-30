##############################################
##############################################
# Author: Fernando Fragío Sánchez
# Master Thesis: Speaker Recognition Based on Gaussian Mixture Models and 
# Neural Networks
# 
# Python code to train and test a Resnet34 with data from the Librispeech dataset
# using the Spectrogram of the audio samples.

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
import torchvision.models as models
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
        m_spect = librosa.feature.melspectrogram(y=soundData3, sr=sr)
        m_spect = torch.from_numpy(m_spect)
        tempData = torch.zeros([128,300])
        if m_spect.size()[1] < 300:
            tempData[:,:m_spect.size()[1]] = m_spect[:,:]
        else:
            tempData[:] = m_spect[:,:300]

        tempData.unsqueeze_(-1)
        tempData = tempData.expand(128,300,3)
        tempData = tempData.transpose(2, 0)
        
        return tempData, self.labels[index]

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

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 2, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 2, **kwargs)

######################################################################
# ------------------
# Define the Network
# ------------------

model = models.resnet34()
for param in model.parameters():
    param.requires_grad = False
    # Replace the last fully-connected layer
model.fc = nn.Linear(512, n_speakers) # last layer with n_speakers output
model.to(device)
print(model)

optimizer = optim.Adam(model.parameters(), lr = 0.01, weight_decay = 0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)

######################################################################
# --------------------------------
# Train and Test the Network
# --------------------------------

def train(model, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        try:
            optimizer.zero_grad()
        except:
            import ipdb
            ipdb.set_trace()
        data = data.to(device)
        target = target.to(device)
        data = data.requires_grad_() #set requires_grad to True for training
        output = model(data.float())
        loss = F.nll_loss(output, target) #the loss functions expects a batchSizex10 input
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0: #print training stats
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))

def test(model, epoch):
    model.eval()
    correct = 0
    for data, target in test_loader:
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = model(data.float())
        pred = output.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target).cpu().sum().item()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

log_interval = 20
for epoch in range(1, 40):
    if epoch == 31:
        print("First round of training complete. Setting learn rate to 0.001.")
    scheduler.step()
    model = model.float()
    train(model, epoch)
    test(model, epoch)


