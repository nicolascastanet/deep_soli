# importing the libraries
import time
import pandas as pd
import numpy as np
import os
import pickle
import random
from pathlib import Path
from datetime import date

p = os.getcwd()
today = date.today()
res_path = f"{p}/res/res_{today.day}_{today.month}_{today.year}"

try:
    os.mkdir(res_path)
except OSError:
    print(f"{res_path} already created")
else:
    print(f"{res_path} successfully created")

if bool(len(os.listdir(res_path)) == 0):
    path = f"{res_path}/res_0"
else:
    nb_res = 0
    for x in os.listdir(res_path):
        if int(x.split('_')[1]) > nb_res:
            nb_res = int(x.split('_')[1])
    path = f"{res_path}/res_{nb_res+1}"
print(f"{path} successfully created")
os.mkdir(path)

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
from net import CNNModel_1, CNNModel_2, CNN_RNN_model


# Data importation
print("loading pickle ...")
out = open(f"{path}/out.txt", "a")
out.write("loading pickle\n")
out.write("\n-------------------------")
out.write("\n-------------------------\n")
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

# Classes proportions in train and test data
train_prop, test_prop = np.zeros(10), np.zeros(10)
for i in range(10):
    train_prop[i] = len(np.where(Y_train == i)[0])
    test_prop[i] = len(np.where(Y_test == i)[0])
print("-"*25)
print("train classes proportions")
print(train_prop/train_prop.sum(0))

print("-"*25)
print("test classes proportions")
print(test_prop/test_prop.sum(0))
print("_"*25)
out.write("Train classes prop")
out.write(str(train_prop/train_prop.sum(0)))
out.write("\n-------------------------")
out.write("\n-------------------------")
out.write("Test classes prop")
out.write(str(test_prop/test_prop.sum(0)))
out.write("\n-------------------------")
# Numpy to tensor
train_x = torch.tensor(X_train).float()
train_y = torch.tensor(Y_train).long()
test_x = torch.tensor(X_test).float()
test_y = torch.tensor(Y_test).long()
batch_size = 50
out.write(f"batch size : {batch_size}\n")

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(train_x,train_y)
test = torch.utils.data.TensorDataset(test_x,test_y)
shuff = False
out.write(f"shuffle dataloader at each epoch: {shuff}\n")
# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = shuff)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = shuff)
print("dump data loader")

out.write("dump data loader\n")

torch.save(test_loader, f"{path}/test_loader")
torch.save(train_loader, f"{path}/train_loader")
num_classes = 11

# Hyperparameters
num_epochs = 50

# Create Model

CUDA = torch.cuda.is_available()
if CUDA:
    device = torch.device("cuda:0")
    print("running on GPU\n")
else:
    device = torch.device("cpu")
    print("running on CPU\n")

model = CNNModel_2(num_classes)
model.to(device)

print(model)
out.write(str(model))
# Cross Entropy Loss 
error = nn.CrossEntropyLoss()

# Adam Optimizer
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Training loop
print("Training ...")

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []
for epoch in range(num_epochs):

    start = time.time()
    correct = 0
    iterations = 0
    iter_loss = 0.0
    total = 0
    # Training
    model.train()

    for i, (frames, labels) in enumerate(train_loader):
        train = frames.view(50,4,40,32,32)
        total+= len(labels)
        train, labels = train.to(device), labels.to(device)
        # Clear gradients
        optimizer.zero_grad()
        # Forward propagation
        outputs = model(train)
        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)
        iter_loss += loss.data
        # Calculating gradients
        loss.backward()
        # Update parameters
        optimizer.step()
        # Record predictions for train data
        predicted = torch.max(outputs.data, 1)[1]
        correct += (predicted == labels).sum()
        iterations += 1

    # Record training loss
    train_loss.append(iter_loss/iterations)
    # Record training accuracy
    train_accuracy.append((100 * correct / float(total)))

    # Testing
    iter_loss = 0.0
    correct = 0
    iterations = 0
    total = 0
    model.eval()
    # Iterate through test data
    for i, (frames, labels) in enumerate(test_loader):

        test = frames.view(50,4,40,32,32)
        total+= len(labels)
        test, labels = test.to(device), labels.to(device)
        # Forward propagation
        with torch.no_grad():
            outputs = model(test)
        loss = error(outputs, labels)
        iter_loss += loss.data
        # Get predictions from the maximum value
        predicted = torch.max(outputs.data, 1)[1]
        # Total number of labels
        correct += (predicted == labels).sum()
        iterations += 1

    test_loss.append(iter_loss/iterations)
    test_accuracy.append((100 * correct / float(total)))
    stop = time.time()

    print('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Testing Loss: {:.3f}, Testing Acc: {:.3f}, Time: {}s'
               .format(epoch+1, num_epochs, train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[-1], stop-start))
    out.write('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Testing Loss: {:.3f}, Testing Acc: {:.3f}, Time: {}s'
                   .format(epoch+1, num_epochs, train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[-1], stop-start))

print("end training\n")

pickle.dump(test_loss, open( f"{path}/test_loss.p", "wb" ) )
pickle.dump(test_accuracy, open( f"{path}/test_acc.p", "wb" ) )
pickle.dump(train_loss, open( f"{path}/train_loss.p", "wb" ) )
pickle.dump(train_accuracy, open( f"{path}/train_acc.p", "wb" ) )


torch.save(model, f"{path}/trained_net")

print("loading confusion matr ...")

confusion_matrix = torch.zeros(num_classes, num_classes)

with torch.no_grad():
    for i,(inputs, classes) in enumerate(test_loader):
        print(f"batch {i}")
        inputs = inputs.view(batch_size,4,40,32,32)
        inputs, classes = inputs.to(device), classes.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs,1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()]+=1

torch.save(confusion_matrix, f"{path}/conf_matr")
print(confusion_matrix)
out.write(str(confusion_matrix))
out.close()
