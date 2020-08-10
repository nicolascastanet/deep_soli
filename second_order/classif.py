import numpy as np
import time
import os
import sys
import pickle
import pyriemann
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from pathlib import Path
from covariance_pooling import Cov_soli
import argparse
sys.path.append('/home/castanet/Git/deep-soli')
import utils as ut

parser = argparse.ArgumentParser(description="cov_classif")
parser.add_argument('--train_fold', type=int)
parser.add_argument('--channel', type=int)
parser.add_argument('--model', type=str)
parser.add_argument('--folder', type=str)

args = parser.parse_args()

res_path = ut.folder_creation(args.folder)

import ipdb; ipdb.set_trace()


num_fold = args.train_fold
channel = args.channel

START = time.time()

p = Path(os.getcwd())
LABEL_PATH = f"{p.parent}/data/gestureLabels.npy"
DATA_PATH = f"{p.parent}/data/data.npy"
FOLDS_IDX_PATH = f"{p.parent}/data/5_folds.npy"

print(f"fold : {num_fold}, channel : {channel}\n")

print("loading data ...\n")
start = time.time()
data = np.load(DATA_PATH)
Y = np.load(LABEL_PATH).reshape(2750)
# reshape to (n_trials, n_channels, n_samples) for pyriemann lib
X = np.swapaxes(data.reshape(2750,40,32,32,4),1,4)
stop = time.time()
print(f"Done in : {stop-start}\n")
print("--------------------------\n")

idx_folds = np.load(FOLDS_IDX_PATH)

# train / test split indices
idx_train = idx_folds[num_fold]
idx_test = list(set(range(len(Y))) - set(idx_train))

# choose soli channel (!= channels for pyriemann) and nb_gestures
X = X[:,channel,:,:,:]

print("Loadind and shrinking covariance matrix ...\n")
start = time.time()
cov = pyriemann.estimation.Covariances().transform(X.reshape(2750,32*32,40))
cov = pyriemann.estimation.Shrinkage().transform(cov)
stop = time.time()
print(f"Done in {stop-start} sec\n")
print("--------------------------\n")

X_train = cov[idx_train]
Y_train = Y[idx_train]

X_test = cov[idx_test]
Y_test = Y[idx_test]

#import ipdb;ipdb.set_trace()

c = Cov_soli(0,2750)
c.set_cov_list(cov)
print("SPD test ...\n")
start = time.time()
res = c.spd_test()
print(f"results : {res}\n")
stop = time.time()
print(f"Done in {stop-start} sec")
print("--------------------------\n")

if res == False:
    print(f"regularization with factor {c.reg_fact} ...\n")
    c.reg_cov()
    print("new SPD test ...\n")
    print(f"results : {c.spd_test()}\n")

print("KNN with 5 neighbor algorithm\n")
model = pyriemann.classification.KNearestNeighbor(n_jobs = -1)

t1 = time.time()
print("fitting to data ...\n")
model.fit(X_train, Y_train)
t2 = time.time()
print(f"Done in {(t2-t1)} min")
print("--------------------------\n")

if args.mode == "prediction":
    print("get predictions ...")
    pred = model.predict(X_test)
    t3 = time.time()
    print(f"Done in : {(t3-t2)/60} min")
    np.save(f"{res_path}/predictions.npy", pred)
    

elif args.mode == "accuracy":
    print("computing accuracy ...\n")
    accuracy = model.score(X_test,Y_test)
    t3 = time.time()
    print(f"Done in : {(t3-t2)/60} min")
    print(f"Accuracy : {accuracy}\n")
    print("--------------------------\n")



STOP = time.time()
print(f"total time : {(STOP-START)/60} min")