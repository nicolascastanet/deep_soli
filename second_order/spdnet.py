import sys
from pathlib import Path
import os
import random
import time
import argparse
import numpy as np
import torch as th
import torch.nn as nn
from torch.utils import data

import torchspdnet.nn as spdnet
from torchspdnet.optimizers import MixOptimizer

# Definition of the SPDNet
class soliSpdNet(nn.Module):
    def __init__(self, bn=False):
        super(__class__,self).__init__()
        self._bn = bn
        dim=1024
        dim1=512; dim2=256; dim3=128;dim4=64
        classes=11
        self.cov = spdnet.CovPool()
        self.re=spdnet.ReEig()
        self.bimap1=spdnet.BiMap(1,1,dim,dim1)
        self.bimap2=spdnet.BiMap(1,1,dim1,dim2)
        self.bimap3=spdnet.BiMap(1,1,dim2,dim3)
        self.bimap4=spdnet.BiMap(1,1,dim3,dim4)
        self.logeig=spdnet.LogEig()
        self.linear=nn.Linear(dim4**2,classes)

    def forward(self,x):
        x=self.cov(x)
        x=self.bimap1(x)
        x=self.re(x)
        x=self.bimap2(x)
        x=self.re(x)
        x=self.bimap3(x)
        x=self.re(x)
        x=self.bimap4(x)
        x=self.logeig(x)
        x_vec=x.view(x.shape[0],-1)
        y=self.linear(x_vec)
        #import ipdb; ipdb.set_trace()
        return y

def soli(test_loader, train_loader):
    
    #main parameters
    lr=1e-2
    n=1024 #dimension of the data
    C=11 #number of classes
    batch_size=50 #batch size
    threshold_reeig=1e-4 #threshold for ReEig layer
    num_epochs=50

    CUDA = th.cuda.is_available()
    #import ipdb; ipdb.set_trace()
    if CUDA:
        device = th.device(f"cuda:{args.gpu}")
        print(f"running on GPU number {args.gpu}\n")
    else:
        device = th.device("cpu")
    print("running on CPU\n")

    #setup data and model
    model=soliSpdNet()
    model = model.double()

    #setup loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    opti = MixOptimizer(model.parameters(),lr=lr)

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
            train = frames.view(batch_size,32*32,40)
            total+= len(labels)
            train, labels = train.to(device), labels.to(device)
            # Clear gradients
            opti.zero_grad()
            # Forward propagation
            outputs = model(train)
            # Calculate softmax and ross entropy loss
            loss = loss_fn(outputs, labels)
            iter_loss += loss.data
            # Calculating gradients
            loss.backward()
            # Update parameters
            opti.step()
            # Record predictions for train data
            predicted = th.max(outputs.data, 1)[1]
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

            test = frames.view(batch_size,32*32,40)
            total+= len(labels)
            test, labels = test.to(device), labels.to(device)
            # Forward propagation
            with th.no_grad():
                outputs = model(test)
            loss = loss_fn(outputs, labels)
            iter_loss += loss.data
            # Get predictions from the maximum value
            predicted = th.max(outputs.data, 1)[1]
            # Total number of labels
            correct += (predicted == labels).sum()
            iterations += 1

        test_loss.append(iter_loss/iterations)
        test_accuracy.append((100 * correct / float(total)))
        stop = time.time()

        print('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Testing Loss: {:.3f}, Testing Acc: {:.3f}, Time: {}s'
                .format(epoch+1, num_epochs, train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[-1], stop-start))

        print("end training\n")
    return train_loss, test_loss, train_accuracy, test_accuracy

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate pickle")
    parser.add_argument('--train_fold', type=int)
    parser.add_argument('--shuffle', type=bool)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--folder', type=str)

    args = parser.parse_args()

    p = Path(os.getcwd())
    LABEL_PATH = f"{p}/data/gestureLabels.npy"
    DATA_PATH = f"{p}/data/data.npy"
    FOLDS_IDX_PATH = f"{p}/data/5_folds.npy"
    batch_size=50
    num_fold=0
    channel=0

    data = np.load(DATA_PATH)
    gestureLabels = np.load(LABEL_PATH)
    idx_folds = np.load(FOLDS_IDX_PATH)

    data = np.load(DATA_PATH)
    Y = np.load(LABEL_PATH).reshape(2750)
    # reshape to (n_trials, n_channels, n_samples) for pyriemann lib
    X = np.swapaxes(data.reshape(2750,40,32,32,4),1,4)
    X = X[:,channel,:,:,:].astype(float)

    idx_train = idx_folds[num_fold]
    idx_test = list(set(range(len(Y))) - set(idx_train))

    X_train = X[idx_train]
    Y_train = Y[idx_train]

    X_test = X[idx_test]
    Y_test = Y[idx_test]
    train_x = th.tensor(X_train).double()
    train_y = th.tensor(Y_train).long()
    test_x = th.tensor(X_test).double()
    test_y = th.tensor(Y_test).long()

    train = th.utils.data.TensorDataset(train_x,train_y)
    test = th.utils.data.TensorDataset(test_x,test_y)

    train_loader = th.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
    test_loader = th.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)


    #import ipdb;ipdb.set_trace()


    train_loss, test_loss, train_accuracy, test_accuracy = soli(test_loader,train_loader)
