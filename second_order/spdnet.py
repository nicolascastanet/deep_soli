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
sys.path.append('/home/castanet/Git/deep-soli')
import utils as ut

# Definition of the SPDNet
class soliSpdNet(nn.Module):
    def __init__(self, bn=False):
        super(__class__,self).__init__()
        self._bn = bn
        dim = 1024
        dim1 = 512; dim2 = 256; dim3 = 128; dim4 = 64
        classes = 11
        self.cov = spdnet.CovPool()
        self.re = spdnet.ReEig()
        self.bimap1 = spdnet.BiMap(1,1,dim,dim1)
        self.bimap2 = spdnet.BiMap(1,1,dim1,dim2)
        self.bimap3 = spdnet.BiMap(1,1,dim2,dim3)
        self.bimap4 = spdnet.BiMap(1,1,dim3,dim4)
        self.logeig = spdnet.LogEig()
        self.linear = nn.Linear(dim4**2,classes)

    def forward(self,x):
        x = self.cov(x)
        x = self.bimap1(x)
        x = self.re(x)
        x = self.bimap2(x)
        x = self.re(x)
        x = self.bimap3(x)
        x = self.re(x)
        x = self.bimap4(x)
        x = self.re(x)
        x = self.logeig(x)
        x_vec = x.view(x.shape[0],-1)
        y = self.linear(x_vec)
        #import ipdb; ipdb.set_trace()
        return y

def afew(test_loader, train_loader, out, device):
    
    #main parameters
    lr=1e-4
    n=1024 #dimension of the data
    C=11 #number of classes
    batch_size=50 #batch size
    threshold_reeig=1e-4 #threshold for ReEig layer
    num_epochs=50
    

    #setup data and model
    model = soliSpdNet()
    model = model.double()
    model = model.to(device)
    out.write("-"*25)
    out.write(str(model))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\ntrainable params : {total_params}")
    out.write(f"\ntrainable params : {total_params}")

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
        #import ipdb; ipdb.set_trace()

        start = time.time()
        correct = 0
        iterations = 0
        iter_loss = 0.0
        total = 0
        
        model.train()

        for i, (frames, labels) in enumerate(train_loader):
            train = frames.view(batch_size,32*32,40)
            total+= len(labels)
            train, labels = train.to(device), labels.to(device)
            opti.zero_grad()
            outputs = model(train)
            loss = loss_fn(outputs, labels)
            iter_loss += loss.data
            loss.backward()
            opti.step()
            predicted = th.max(outputs.data, 1)[1]
            correct += (predicted == labels).sum()
            iterations += 1

        # Record training loss
        train_loss.append(iter_loss/iterations)
        # Record training accuracy
        train_accuracy.append((100 * correct / float(total)))

        iter_loss = 0.0
        correct = 0
        iterations = 0
        total = 0
        model.eval()
       
        for i, (frames, labels) in enumerate(test_loader):
            test = frames.view(batch_size,32*32,40)
            total+= len(labels)
            test, labels = test.to(device), labels.to(device)
            with th.no_grad():
                outputs = model(test)
            loss = loss_fn(outputs, labels)
            iter_loss += loss.data
            predicted = th.max(outputs.data, 1)[1]
            correct += (predicted == labels).sum()
            iterations += 1

        test_loss.append(iter_loss/iterations)
        test_accuracy.append((100 * correct / float(total)))
        stop = time.time()

        print('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Testing Loss: {:.3f}, Testing Acc: {:.3f}, Time: {}s'
                .format(epoch+1, num_epochs, train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[-1], stop-start))

        out.write("\n---------------------------------------------\n")
        out.write('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Testing Loss: {:.3f}, Testing Acc: {:.3f}, Time: {}s'
                   .format(epoch+1, num_epochs, train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[-1], stop-start))
    print("end training\n")
    return train_loss, test_loss, train_accuracy, test_accuracy, model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate pickle")
    parser.add_argument('--train_fold', type=int)
    parser.add_argument('--shuffle', type=bool)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--folder', type=str)

    args = parser.parse_args()

    path = ut.folder_creation(args.folder)

    out = open(f"{path}/out.txt", "a")

    p = Path(os.getcwd())
    out.write("loading data ...\n")
    out.write("-"*25)
    LABEL_PATH = f"{p}/data/gestureLabels.npy"
    DATA_PATH = f"{p}/data/data.npy"
    FOLDS_IDX_PATH = f"{p}/data/5_folds.npy"
    num_classes = 11
    batch_size = 50
    num_fold = 0
    channel = 0

    

    data = np.load(DATA_PATH)
    gestureLabels = np.load(LABEL_PATH)
    idx_folds = np.load(FOLDS_IDX_PATH)

    data = np.load(DATA_PATH)
    Y = np.load(LABEL_PATH).reshape(2750)
    # reshape to (n_trials, n_channels, n_samples) for pyriemann lib

    print(f"\nSoli channel : {channel}\n")
    out.write(f"\nsoli channel : {channel}\n")
    X = np.swapaxes(data.reshape(2750,40,32,32,4),1,4)
    # dim = (nb_gesture, nb_channel, 32, 32, nb_frame)
    X = X[:,channel,:,:,:]

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

    # Pytorch train and test sets
    train = th.utils.data.TensorDataset(train_x,train_y)
    test = th.utils.data.TensorDataset(test_x,test_y)
    # data loader
    train_loader = th.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
    test_loader = th.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

    CUDA = th.cuda.is_available()

    if CUDA:
        device = th.device(f"cuda:{args.gpu}")
        print(f"running on GPU number {args.gpu}\n")
    else:
        device = th.device("cpu")
        print("running on CPU\n")

    train_loss, test_loss, train_accuracy, test_accuracy, model = afew(test_loader, train_loader, out, device)

    th.save(th.tensor(test_loss).cpu(),f"{path}/test_loss")
    th.save(th.tensor(test_accuracy).cpu(),f"{path}/test_acc")
    th.save(th.tensor(train_loss).cpu(),f"{path}/train_loss")
    th.save(th.tensor(train_accuracy).cpu(),f"{path}/train_acc")
    th.save(model, f"{path}/trained_net")

    print("loading confusion matr ...")

    confusion_matrix = th.zeros(num_classes, num_classes)

    with th.no_grad():
        for i,(inputs, classes) in enumerate(test_loader):
            print(f"batch {i}")
            inputs = inputs.view(batch_size,32*32,40)
            inputs, classes = inputs.to(device), classes.to(device)
            outputs = model(inputs)
            _, preds = th.max(outputs,1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()]+=1

    th.save(confusion_matrix, f"{path}/conf_matr")
    print(confusion_matrix)
    out.write(str(confusion_matrix))
    out.close()
