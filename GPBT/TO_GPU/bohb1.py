# A random mnist from the internet to get a correct model to reason about
# IN THIS MODULE: IMPORTS, CNN, TRAIN, TEST, MNIS_FUNCTION, SPACE

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
#from ray.tune.suggest.skopt import SkOptSearch
from ray import tune
from ray.tune.suggest.bayesopt import BayesOptSearch
import time
import ray
from ray.tune.schedulers import AsyncHyperBandScheduler
import argparse
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.nevergrad import NevergradSearch
import json
from ray.tune import Trainable
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.dragonfly import DragonflySearch
from ray.tune.schedulers import AsyncHyperBandScheduler
import torch
from ray.tune.schedulers import PopulationBasedTraining

# IN THIS MODULE: IMPORTS, CNN, TRAIN, TEST, MNIS_FUNCTION, SPACE
import models, datasets, schedulers, utils,main

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn

import torch.nn.functional as F
from ray import tune
from ray.tune.schedulers import ASHAScheduler 
from ray.tune.schedulers import MedianStoppingRule

from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
#from ray.tune.suggest.skopt import SkOptSearch
from ray import tune
from ray.tune.suggest.bayesopt import BayesOptSearch
import time
import ray
from ray.tune.schedulers import AsyncHyperBandScheduler
import argparse
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.nevergrad import NevergradSearch
import json
import os
from ray.tune import Trainable
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
#from ray.tune.suggest.dragonfly import DragonflySearch
from ray.tune.schedulers import AsyncHyperBandScheduler
import torch
#import adabelief_pytorch
global_checkpoint_period=np.inf
from ray.tune.schedulers.pb2 import PB2

class train_mnist_pb2(tune.Trainable):
    def setup(self, config):
      self.obj = main.TrainCIFAR(config)
      self.config = config
    

    
    def reset_config(self, config):
        if "lr" in config:
            for param_group in self.obj.optimizer.param_groups:
                param_group["lr"] = config.get("lr")
        if "momentum" in config:
            for param_group in self.obj.optimizer.param_groups:
                param_group["row"] = 1-config.get("momentum")
                
        if "weight_decay" in config:
            for param_group in self.obj.optimizer.param_groups:
                param_group["weight_decay"] = config.get("weight_decay")
        
        if "eps" in config:
            for param_group in self.obj.optimizer.param_groups:
                param_group["eps"] = config.get("eps")
        
        
        self.obj.net.adapt(config.get("drp"))
        self.config = config
        return True

    def step(self):
        a,b=self.obj.train1()
        c,d = self.obj.val1()
      #  a,b,c,d= self.obj.step()
        e,f=self.obj.test1()
        return {'loss' :d, 'acc':c , 'test_loss' : f,
                'test_acc' :e}
    
    def save_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint")
        torch.save({
            "net": self.obj.net.state_dict(),
            "optim": self.obj.optimizer.state_dict(),

        }, path)

        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint")
        checkpoint = torch.load(path)
        self.obj.net.load_state_dict(checkpoint["net"])
        self.obj.optimizer.load_state_dict(checkpoint["optim"])




scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    perturbation_interval=1,
        hyperparam_mutations={
     "lr": [1e-8,.23] #tune.uniform(1e-4, 0.1 ),#,1e-4), #*10
,     "weight_decay":[1e-4,1e-2]#tune.uniform(1, 5)#,1e-4), #*10 et 0
,     "drp": [.05, .15]#,1e-4), #*10 et 0
 ,    "momentum": [.23, 1e-2] #,1e-4), #*10 et 0
 ,    "eps": [1e-4, 1e-2]#,1e-4), #*10 et 0
    }) 



class TestLogger(tune.logger.Logger):
    def _init(self):
        progress_file = os.path.join("", "last.csv") #aller jusqu'a 9
        self._continuing = os.path.exists(progress_file)
        self._file = open(progress_file, "a")
        self._csv_out = None
    def on_result(self, result):
        tmp = result.copy()
        result = flatten_dict(tmp, delimiter="/")
        if self._csv_out is None:
            self._csv_out = csv.DictWriter(self._file, result.keys())
            self._csv_out.writeheader()
        self._csv_out.writerow(
            {k: v
             for k, v in result.items() if k in self._csv_out.fieldnames})
        self._file.flush()
                   

from ray.tune.logger import *

from ray import tune


from ray.tune.suggest.hebo import HEBOSearch

for _ in range(1):
  ray.shutdown()
  ray.init()
  start_time = time.time()
  from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
  from ray.tune.suggest.bohb import *


  algo = TuneBOHB(metric="loss", mode="min")
  algo = ConcurrencyLimiter(algo, max_concurrent=36)

  bohb = HyperBandForBOHB(
        time_attr="training_iteration",
        metric="loss",
        mode="min",
        max_t=46)

  analysis = tune.run(
      
  train_mnist_pb2,
#  scheduler=scheduler,
  reuse_actors=True,
  search_alg=algo,
  verbose=2,
  checkpoint_at_end=True,
  num_samples=36,
  # export_formats=[ExportFormat.MODEL],
config= {
     "lr": tune.loguniform(1e-8,.36)
    , "drp": tune.uniform(.05,.15)
          ,   "weight_decay": tune.loguniform(1e-4,1e-2),
    "momentum" : tune.loguniform(1e-2, .36),
    "eps" : tune.loguniform(1e-4, 1e-2)
},      stop={
          "training_iteration": 46,
      },        metric="loss",
      mode="min"
,resources_per_trial={'cpu':2 ,'gpu': 1}
              ,     loggers=[TestLogger])
  print("time "+ str(time.time()- start_time))
