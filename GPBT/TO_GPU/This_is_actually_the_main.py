
import models, datasets, schedulers, oracles, utils,main
import time
import torch
from hyperopt import hp, fmin, tpe, Trials
import os

n_configurations = 36 #Should be 24
total_iterations = 1  #Should be 50
epsilon = 1e-10
config = {
    "lr": hp.loguniform("lr",-8*2.3,-1
    ),
    "drp": hp.uniform("drp",0.05,.15
    ),
    "weight_decay": hp.loguniform("weight_decay",-4*2.3,-2.5*2
    ),
    "momentum" : 1-hp.loguniform("momentum",-2*2.3, -1
    ),
    "eps" : hp.loguniform("eps",-4*2.3, -2*2.3
   # ),
  #  "eps_arch" : hp.loguniform("eps_arch",-5*2.3,-5*2.3+epsilon#-6*2.3, -4*2.3
   # ),
    #"momentum_arch" : hp.loguniform("momentum_arch",-2.3,-2.3+epsilon#-2*2.3, -2
    )


}

timen_now = time.time()
model = main.TrainCIFAR
oracle = oracles.Guesser(config,0)
start_time = time.time()
scheduler = schedulers.Scheduler(model, total_iterations, n_configurations, oracle,1,True,1e100) #Should be 6
scheduler.initialisation()
scheduler.loop()
scheduler.close()
print("total exeuction time is " + str(time.time()-timen_now))
