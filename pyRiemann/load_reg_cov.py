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

def loader(path, channel = 0, nb_geste = 100, nb_frames = 40, reg_fact = 10e-17):

    c = Cov_soli(channel, nb_geste, nb_frames, reg_fact)

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

    print("-"*20)
    print("\nSaving data ...")
    cov_matr = c.get_cov_list()
    np.save('cov_SPD.npy', cov_matr)


parser = argparse.ArgumentParser(description="cov pooling")
parser.add_argument('--soli_channel', type=int)
parser.add_argument('--nb_gestures', type=int)
parser.add_argument('--nb_frames', type=int)
parser.add_argument('--reg_fact', type=int)


args = parser.parse_args()

#loader(DATA_PATH, args.soli_channel, args.nb_gestures, args.nb_frames, args.reg_fact)
loader(DATA_PATH)