import numpy as np
import os
import time
import pickle
import pyriemann
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from pathlib import Path
from covariance_pooling import Cov_soli
import argparse

p = Path(os.getcwd())
DATA_PATH = f"{p.parent}/data/data.npy"

def loader(path, channel, nb_geste = 2750, nb_frames = 40, reg_fact = 10e-16):

    c = Cov_soli(channel, nb_geste, nb_frames, reg_fact)
    print(f"loading soli_spectrograms for channel : {channel}\n")
    c.load_cov(path)
    print("SPD test\n")
    print("-"*20)
    if c.spd_test() == False:
        print(f"test result : {c.spd_test()}\n")
        print("-"*20)
        print(f"regularization with factor : {reg_fact}\n")
        c.reg_cov()
        print(f"new SPD test : {c.spd_test()}")
    
    else:
        print(f"test result : {c.spd_test()}\n")
        print("no regularization needed\n")

    cov_matr = c.get_cov_list()
    np.save(f"cov_SPD_ch{channel}.npy", cov_matr)


parser = argparse.ArgumentParser(description="cov pooling")
parser.add_argument('--soli_channel', type=int)
parser.add_argument('--nb_gestures', type=int)
parser.add_argument('--nb_frames', type=int)
parser.add_argument('--reg_fact', type=int)


args = parser.parse_args()

#loader(DATA_PATH, args.soli_channel, args.nb_gestures, args.nb_frames, args.reg_fact)
for c in range(4):
    loader(DATA_PATH, c)
