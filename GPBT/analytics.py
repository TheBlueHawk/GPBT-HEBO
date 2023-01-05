import pandas as pd
from numpy import genfromtxt
import numpy as np
from scipy.interpolate import *
import matplotlib.pyplot as plt
import math
import os


DEFAULT_PATH = "./tmp/data"


def fonc(data):
    itera = data[:, 5:6]
    itera = itera[~np.isnan(itera)]
    dat = data[:, 0:2]
    dat = dat[~np.isnan(dat)]
    dat = dat.reshape(-1, 2)
    #  dat = dat[np.argsort(itera,axis=-1)]
    return dat


def maxof(a):
    ma = 0
    ta = 0
    for i in range(81):
        if a.shape[0] <= i:
            a = np.concatenate((a, (np.array([np.array([ma, ta])]))))
            print(a.shape)
        else:
            if ma < a[i, 0]:
                ta = a[i, 1]
            ma = max(ma, a[i, 0])
            a[i, 0] = ma
            a[i, 1] = ta
    return a[:80]


def iteration_corector(liste, num_config):
    for i in range(len(liste[:])):
        liste[i] = math.floor(i / num_config)


def getall(a):
    f = [b for b in a[:, 1::2].mean(1)]
    g = [b for b in a[:, 0::2].mean(1)]
    return (f, a[:, 1::2].std(1) / 2, g, a[:, 0::2].std(1) / 2)


def process(algo="GPBTHEBO", dataset="FMNIST", model="LeNet", num_iteration=10):
    results = []
    for iteration in num_iteration:
        filename = algo + "_" + dataset + "_" + model + "_" + str(iteration) + ".csv"
        filename = os.path.join(DEFAULT_PATH, filename)
        logs = genfromtxt(filename, delimiter=",")
        test_loss = logs[np.argsort(logs[:, -5], axis=-1, kind="stable")][:, -2:]
        test_loss = maxof(test_loss[:, ::])
        results.append(test_loss)
        plt.plot((np.concatenate((test_loss,), axis=1)[:, 0::2]))

