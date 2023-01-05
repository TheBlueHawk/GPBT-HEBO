import copy

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch

import math
import numpy as np
import random
import math

from hyperopt import hp, fmin,atpe ,tpe, Trials

from functools import *
import skopt
from skopt import gp_minimize
from skopt.plots import plot_objective


class GPGuesser():
    """Used to sample the hyperspace with the tools of `hyperopt`
    """

    def __init__(self, searchspace, verbose):
        self.searchspace = searchspace
        self.string = "aiteration"

        self.verbose = verbose
        self.algo = partial(tpe.suggest, n_startup_jobs=1)



    def repeat_good(self, trials, iteration, function, configuration):
        space = copy.deepcopy(self.searchspace)

     #   curr_eval = getattr(trials, '_ids')
     #   if curr_eval == set():
     #       curr_eval = 0
     #   else:
     #       curr_eval = max(curr_eval) + 1

       # space[self.string] = iteration #hp.quniform(self.string, -.5+iteration, .5+iteration, 1)
        space = []
        for k,v in configuration.items():
            space.append(v)
        loss=function(space)
        space.append((iteration))

        trials[0].append(space)
        trials[1] = np.append(trials[1],loss)


    #    print(curr_eval)
     #   coucou = skopt.gp_minimize(function,space,n_calls=1,x0=trials[0],y0=trials[1],n_initial_points=0)
        
        
      #  trials[0] = coucou.x_iters
      #  trials[1] = coucou.func_vals
      # # fmin(
       #     function,
       #     space, algo=self.algo,
       #     max_evals=curr_eval+nb_eval,
       #     trials=trials,
       #     verbose=self.verbose
       # )
       # _ = plot_objective(coucou)
        
        
        
        
        
    #    space = copy.deepcopy(configuration)

    #    for k, v in configuration.items():
    #        space[k] = hp.uniform(k, v - 1e-10, v + 1e-10)

    #    curr_eval = getattr(trials, '_ids')

    #    if curr_eval == set():
    #        curr_eval = 0
    #    else:
    #        curr_eval = max(curr_eval) + 1

     #   space[self.string] = iteration # hp.quniform(self.string, -.5+iteration, .5+iteration, 1)
        #print(space)
     #   fmin(
     #       function,
     #       space,
     ##       algo=self.algo,
      #      max_evals=curr_eval+1,
       #     trials=trials,
       #     verbose=self.verbose
       # )

    def compute_once(self, trials, iteration, function):
        space = copy.deepcopy(self.searchspace)

        curr_eval = getattr(trials, '_ids')
        if curr_eval == set():
            curr_eval = 0
        else:
            curr_eval = max(curr_eval) + 1

        space[self.string] = iteration # hp.quniform(self.string, -.5+iteration, .5+iteration, 1)
        fmin(
            function,
            space,
            algo=self.algo,
            max_evals=curr_eval+1,
            trials=trials,
            verbose=self.verbose
        )

    def compute_batch(self, trials, nb_eval, iteration, function):
        space = copy.deepcopy(self.searchspace)

     #   curr_eval = getattr(trials, '_ids')
     #   if curr_eval == set():
     #       curr_eval = 0
     #   else:
     #       curr_eval = max(curr_eval) + 1

       # space[self.string] = iteration #hp.quniform(self.string, -.5+iteration, .5+iteration, 1)
        space.append((0,iteration+1e-10))
      
    #    print(curr_eval)
        temp=0
        if isinstance(trials[0],type(None)):
            temp =2
        
        coucou = skopt.gp_minimize(function,space,n_calls=nb_eval,x0=trials[0],y0=trials[1],n_initial_points=temp)
        
        for i in coucou.x_iters:
            i[-1] = iteration
        trials[0] = coucou.x_iters
        
        trials[1] = coucou.func_vals
       # fmin(
       #     function,
       #     space, algo=self.algo,
       #     max_evals=curr_eval+nb_eval,
       #     trials=trials,
       #     verbose=self.verbose
       # )
        if(iteration==5):
            
            _ = plot_objective(coucou)

# ??? don't understand what it does
class Guesser():
    """Used to sample the hyperspace with the tools of `hyperopt`
    """

    def __init__(self, searchspace, verbose):
        self.searchspace = searchspace
        self.string = "aiteration"

        self.verbose = verbose
        self.algo = partial(tpe.suggest, n_startup_jobs=1)



    def repeat_good(self, trials, iteration, function, configuration):

        space = copy.deepcopy(configuration)

        for k, v in configuration.items():
            space[k] = hp.uniform(k, v - 1e-10, v + 1e-10)

        curr_eval = getattr(trials, '_ids')

        if curr_eval == set():
            curr_eval = 0
        else:
            curr_eval = max(curr_eval) + 1

        space[self.string] = iteration # hp.quniform(self.string, -.5+iteration, .5+iteration, 1)
        #print(space)
        fmin(
            function,
            space,
            algo=self.algo,
            max_evals=curr_eval+1,
            trials=trials,
            verbose=self.verbose
        )

    def compute_once(self, trials, iteration, function):
        space = copy.deepcopy(self.searchspace)

        curr_eval = getattr(trials, '_ids')
        if curr_eval == set():
            curr_eval = 0
        else:
            curr_eval = max(curr_eval) + 1

        space[self.string] = iteration # hp.quniform(self.string, -.5+iteration, .5+iteration, 1)
        fmin(
            function,
            space,
            algo=self.algo,
            max_evals=curr_eval+1,
            trials=trials,
            verbose=self.verbose
        )

    def compute_batch(self, trials, nb_eval, iteration, function):
        space = copy.deepcopy(self.searchspace)

        curr_eval = getattr(trials, '_ids')
        if curr_eval == set():
            curr_eval = 0
        else:
            curr_eval = max(curr_eval) + 1

        space[self.string] = iteration #hp.quniform(self.string, -.5+iteration, .5+iteration, 1)
        print(curr_eval)

        fmin(
            function,
            space, algo=self.algo,
            max_evals=curr_eval+nb_eval,
            trials=trials,
            verbose=self.verbose
        )



class BayesOpt():
    def __init__(self, searchspace):
        self.searchspace = searchspace

    def compute_Once(self, function):
        fmin(function, self.searchspace, algo=partial(tpe.suggest,
             n_startup_jobs=1), max_evals=EVALS, trials=Trials())
