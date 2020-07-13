import numpy as np
import os
import pickle
import random

data = pickle.load( open( "data/data_numpy.p", "rb" ) )
gestureLabels = pickle.load( open( "data/gestureLabels_numpy.p", "rb" ))

# Dim = (num_geste, num_frames, h, w, num_channels)
X = data.reshape(2750,40,32,32,4)
Y = gestureLabels.reshape(2750)

def k_folds(num_ex, num_folds):
    idx = np.array(random.sample(range(num_ex), num_ex))
    folds = np.split(idx, num_folds)
    return folds

folds = k_folds(len(Y),5)
idx1, idx2, idx3, idx4, idx5 = folds[0], folds[1], folds[2], folds[3], folds[4]
X1, X2, X3, X4, X5 = X[idx1], X[idx2], X[idx3], X[idx4], X[idx5]
Y1, Y2, Y3, Y4, Y5 = Y[idx1], Y[idx2], Y[idx3], Y[idx4], Y[idx5]

prop1, prop2, prop3, prop4, prop5 = np.zeros(10), np.zeros(10),np.zeros(10),np.zeros(10),np.zeros(10)
for i in range(10):
    prop1[i] = len(np.where(Y1 == i)[0])
    prop2[i] = len(np.where(Y2 == i)[0])
    prop3[i] = len(np.where(Y3 == i)[0])
    prop4[i] = len(np.where(Y4 == i)[0])
    prop5[i] = len(np.where(Y5 == i)[0])
print(np.concatenate((idx1,idx2)))
print(prop1)
print("----------------")
print(prop2)
print("----------------")
print(prop3)
print("----------------")
print(prop4)
print("----------------")
print(prop5)
print("----------------")

np.save('5_folds.npy', folds)