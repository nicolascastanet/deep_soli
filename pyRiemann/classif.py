import numpy as np
import os
import time
import pickle
import pyriemann
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from pathlib import Path
from covariance_pooling import Cov_soli


p = Path(os.getcwd())
LABEL_PATH = f"{p.parent}/data/gestureLabels.npy"
DATA_PATH = f"{p.parent}/data/data.npy"
FOLDS_IDX_PATH = f"{p.parent}/data/5_folds.npy"

num_fold = 0
channel = 0

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
idx_train = idx_folds[num_fold]
idx_test = list(set(range(len(Y))) - set(idx_train))

# choose soli channel (!= channels for pyriemann) and nb_gestures
X = X[:,channel,:,:,:]

print("Loadind and shrinking covariance matrix ...\n")
start = time.time()
cov = pyriemann.estimation.Covariances().transform(X.reshape(2750,32*32,40))
cov = pyriemann.estimation.Shrinkage().transform(cov)
stop = time.time()
print(f"Done in {stop-start}\n")
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
print(f"Done in {stop-start}")
print("--------------------------\n")

if res == False:
    print(f"regularization with factor {c.reg_fact} ...\n")
    c.reg_cov()
    print("new SPD test ...\n")
    print(f"results : {c.spd_test()}\n")

mdm = pyriemann.classification.MDM(n_jobs = -1)

t1 = time.time()
print("fitting to data ...\n")
mdm.fit(X_train, Y_train)
t2 = time.time()
print(f"Done in {t2-t1}")
print("--------------------------\n")

print("predict ...\n")
mdm.predict(X_test)
t3 = time.time()
print(f"Done in {t3-t2}")

accuracy = mdm.score(X_test,Y_test)
print(f"Accuracy : {accuracy}")