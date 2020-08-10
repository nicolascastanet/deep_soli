import time
import pandas as pd
import numpy as np
import os
import random
from pathlib import Path
from datetime import date
import torch as th


def folder_creation(folder):
    p = os.getcwd()
    today = date.today()
    if folder == None:
        res_path = f"{p}/res/res_{today.day}_{today.month}_{today.year}"
    else:
        res_path = f"{p}/res/res_{today.day}_{today.month}_{today.year}/{folder}"

    try:
        os.makedirs(res_path)
    except OSError:
        print(f"{res_path} already created")
    else:
        print(f"{res_path} successfully created")


    nb_res = 0
    for x in os.listdir(res_path):
        if x.split('_')[0] == 'res' and int(x.split('_')[1]) > nb_res:
            nb_res = int(x.split('_')[1])
    path = f"{res_path}/res_{nb_res+1}"

    os.mkdir(path)
    print(f"{path} successfully created")

    return path


def training_loop(device, model, loss_fn, num_epochs, opti, train_loader, test_loader, out):

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

            test = frames.view(50,4,40,32,32)
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
        out.write("\n---------------------------------------------\n")
        out.write('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Testing Loss: {:.3f}, Testing Acc: {:.3f}, Time: {}s'
                    .format(epoch+1, num_epochs, train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[-1], stop-start))

        print("end training\n")

    return train_loss, test_loss, train_accuracy, test_accuracy
    