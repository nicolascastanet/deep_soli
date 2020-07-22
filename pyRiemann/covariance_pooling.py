import numpy as np
import os
import time
import pickle
import pyriemann
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from pathlib import Path


class Cov_soli():
    def __init__(self, channel, nb_geste = 100, nb_frames = 40, reg_fact = 10e-17):
        self.channel = channel
        self.nb_gestures = nb_geste
        self.nb_frames = nb_frames
        self.reg_fact = reg_fact
        

    def load_cov(self, path):
        # load soli spectrograms
        data = np.load(path)

        # reshape to (n_trials, n_channels, n_samples) for pyriemann lib
        X = np.swapaxes(data.reshape(2750,40,32,32,4),1,4)

        # choose soli channel (!= channels for pyriemann) and nb_gestures
        X = X[:self.nb_gestures,self.channel,:,:,:]

        # load covariance matrix list (one cov_matr per gesture)
        # pyriemann num_channel for one soli radar channel -> 32*32
        #import ipdb; ipdb.set_trace()
        self.cov_mat_list = pyriemann.estimation.Covariances().transform(X.reshape(self.nb_gestures,32*32,self.nb_frames))

    def spd_test_matr(self, matr):
        # test if all eigen values are > 0 <=> matr is SPD
        return True if np.all(np.linalg.eigvals(matr) > 0) else False

    def reg_matr(self, matr):
        # Regularize matrix with factor 'a'
        return matr + self.reg_fact*matr.trace()*np.identity(matr.shape[0])

    def spd_test(self):
        # test if all cov matr are SPD
        return np.all(list(map(self.spd_test_matr, self.cov_mat_list)))

    def reg_cov(self):
        # regularization with fact reg_fact
        self.cov_mat_list = np.array(list(map(self.reg_matr, self.cov_mat_list)))


    def set_cov_list(self, cov_list):
        self.cov_mat_list = cov_list

    def get_cov_list(self):
        return self.cov_mat_list


