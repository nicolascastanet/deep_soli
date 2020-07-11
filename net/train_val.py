# importing the libraries
import pandas as pd
import numpy as np
import os
import pickle
import random

# for reading and displaying images
# from skimage.io import imread
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split
# for evaluating the model
from sklearn.metrics import accuracy_score


# PyTorch libraries and modules
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.optim import *
#import h5py

# Import model
from net import CNNModel


# Data importation
print("loading pickle ...")
data = pickle.load( open( "data/data_numpy.p", "rb" ) )
frameLabels = pickle.load( open( "data/frameLabels_numpy.p", "rb" ) )
gestureLabels = pickle.load( open( "data/gestureLabels_numpy.p", "rb" ))

# Dim = (num_geste, num_frames, h, w, num_channels)
X = data.reshape(2750,40,32,32,4)
Y = gestureLabels.reshape(2750)

# Train and test data
sample_shape = (40,32,32,4)
idx_train = np.array(random.sample(range(len(Y)), int(0.8 * len(Y))))
idx_test = list(set(range(len(Y))) - set(idx_train))

X_train = X[idx_train]
Y_train = Y[idx_train]

X_test = X[idx_test]
Y_test = Y[idx_test]

# Numpy to tensor
train_x = torch.tensor(X_train, requires_grad = True).float()
train_y = torch.tensor(Y_train, requires_grad = True).long()
test_x = torch.tensor(X_test, requires_grad = True).float()
test_y = torch.tensor(Y_test, requires_grad = True).long()
batch_size = 50

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(train_x,train_y)
test = torch.utils.data.TensorDataset(test_x,test_y)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)
print("dump data loader")


num_classes = 11




# Hyperparameters
num_epochs = 50

# Create CNN
model = CNNModel(num_classes)
print(model)

# Cross Entropy Loss 
error = nn.CrossEntropyLoss()

# Adam Optimizer
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# Training loop
print("Training ...")

count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):

    print(f"epoch {epoch}")
    for i, (frames, labels) in enumerate(train_loader):
        train = frames.view(50,4,40,32,32)
        # Clear gradients
        optimizer.zero_grad()
        # Forward propagation
        #import ipdb; ipdb.set_trace()
        outputs = model(train)
        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)
        # Calculating gradients
        loss.backward()
        # Update parameters
        optimizer.step()
        count += 1
        if count % 50 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test data
            for frames, labels in test_loader:
                test = frames.view(50,4,40,32,32)
                # Forward propagation
                outputs = model(test)

                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                # Total number of labels
                total += len(labels)
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / float(total)

            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        if count % 500 == 0:
            # Print Loss

            out.write(f"Itération: {count} Loss: {loss.data} Accuracy: {accuracy}\n")
            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))

print("end training")

