import math
from functools import partial
import numpy as np
import time
from Oracle import Guesser
import pandas as pd

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from hyperopt import hp, fmin, atpe, tpe, Trials
import math
from ray import tune
from FMNIST import train_mnist
import os
import csv


DEFAULT_PATH = "/tmp/data"
os.makedirs(DEFAULT_PATH, exist_ok=True)

class Parent():
    """Parent Class that handles the passage of Network Configurations from one step to the
    following
    """
    def __init__(self, point_hyperspace, configuration, model, loss):
        self.point_hyperspace = point_hyperspace
        self.configuration_list = [configuration]
        self.loss_list = [np.array(loss)]
        self.model = model
        self.is_replicated = False

    def update(self, configuration, loss, model):
        self.is_replicated = False
        self.configuration_list.append(configuration)
        self.loss_list=np.append(self.loss_list,loss)
        self.model = model

    def replication(self, n_children):
        self.is_replicated = True
      #  self.configuration_list.append(self.configuration_list[-1])
      #  self.loss_list=np.append(self.loss_list,self.loss_list[-1])
    #    replication_trials(self.point_hyperspace.trials, n_children)

    def get_last_conf(self):
        return self.configuration_list[-1]

    def get_point_hyperspace(self):
        return self.point_hyperspace

    def get_model(self):
        return self.model

    def get_loss(self):
        return self.loss_list


def translation(liste):
   # config = {}
  #  config["lr"] = liste[0]
    #config["droupout_prob"] = liste[1]
    #config["weight_decay"] = liste[2]
    #config["b1"] = liste[3]
    #config["b2"] = liste[4]
    return liste#config

def test_function(x,models,h,losses, parent_model,k_f,iteration):
    x= translation(x)

    if (isinstance(k_f,list)):
            k=k_f[0]
            Islist = True 
    else:
            k = k_f
            Islist = False
    print(k)
    if iteration == 0:
        models[k] = parent_model[k](x)
    else:      
        models[k] = parent_model.adapt(x)
    if(Islist):
        k_f[0] += 1
    
    #for key, value in x.items():
    #        print(key + " "+str(x[key]))

    h[k] = x
    #start_time = time.time()
    models[k].train1()
    loss = models[k].test1()
    test = models[k].val1()
    #print("--- %s seconds ---" % (time.time() - start_time))

    temp = dict(x)
    temp.update({'loss' : loss})
    temp.update({'test' : test})
    fsvnlogger.on_result(temp)

    losses[k] = -loss
    print("accuracy, " + str(loss) + "\n")
    return -loss


import copy


class Scheduler():
    def __init__(self, model, num_iteration, num_config,
                 oracle):
        #Oracle manages the Bayesian optimization
        self.oracle = oracle
        self.iteration = num_iteration
        self.num_config = num_config 
        self.sqrt_config = 2 # math.floor(math.sqrt(num_config)) # math.ceil(num_config/5) #
        self.n_parents = 1#self.sqrt_config
        #self.h is for the m "h" used at every loop, h is a configuration from the search space
        self.h = np.repeat({},num_config) 
        
        #self.out is for storing the result of the algorithm, ie all "h" from all iterations
        #from all sqrt(m) best models per iterations.
        self.out = np.zeros((num_iteration,self.sqrt_config))
        
        #self.hyperspaces is for storing the sqrt(m) hyperspaces used by the algorithm
        self.hyperspaces = np.zeros(self.sqrt_config)
        
        self.plot = np.zeros(num_iteration)

        
        #self.model is the m model that will explore new hyperspace points at every iterations
        self.models = np.repeat(model,num_config)
        
        #self.parents is the sqrt(m) best model from last iteration
        self.parents = np.repeat(model,self.sqrt_config)

        #self.losses remembers the performances of all m models at one iteration to decide which ones are the sqrt(m) best from self.models.
        self.losses = np.zeros(num_config)
        
        self.k = [0] # c'est pour avoir un pointeur sur k, c'est pas plus que O(sqrt)-paralélisable  pour le moment du coup.
    
    def initialisation(self):
        num_config = self.num_config
        extended_Hyperspace = Trials() #[None,None]
        fmin_objective = partial(test_function, models=self.models,h=self.h,losses=self.losses,parent_model=self.models, k_f = self.k,iteration = 0)
        self.oracle.compute_batch(extended_Hyperspace ,num_config , 0 ,fmin_objective)
            
        indexes = np.argsort(self.losses)     
        self.out[0] = (self.losses[indexes])[0:self.sqrt_config]
        self.hyperspaces = np.repeat(extended_Hyperspace,self.sqrt_config)    
        self.parents = np.array([Parent(copy.deepcopy(extended_Hyperspace),(self.h[indexes])[i], (self.models[indexes])[i],(self.losses[indexes])[i])  
                                 for i in range(self.sqrt_config) ])         
        self.plot[0] = self.losses[indexes][0]
        
    def loop(self):
        sqrt_config = self.sqrt_config
        iteration = self.iteration
        for i in range(1,iteration):
            
            self.k[0] = 0
            
            start_time = time.time()
            for j in range(self.n_parents):
                parent = self.parents[j]
                point_extended_hyperspace = parent.get_point_hyperspace()
                print("\n loss of parent " + str(parent.get_loss()[-1]) )
                print("\n loss " + str(parent.get_loss()))
                
                fmin_objective = partial(test_function, models=self.models,h=self.h,losses=self.losses,parent_model=parent.get_model(), k_f = self.k,iteration = len(parent.get_loss()))

                    
                if not parent.is_replicated:
                    print('not replicated')
                    self.oracle.repeat_good(
                        point_extended_hyperspace,
                        len(parent.get_loss()),
                        fmin_objective,
                        parent.configuration_list[-1]
                    )

                    # computes the new batch for each one of the parents for every iteration
                    self.oracle.compute_batch(
                        point_extended_hyperspace,
                        int(self.num_config/self.n_parents) - 1,
                        len(parent.get_loss()),
                        fmin_objective
                    )
                else:

                    print('replicated')
                    self.oracle.compute_batch(
                        point_extended_hyperspace,
                        int(self.num_config/self.n_parents),
                        len(parent.get_loss()),
                        fmin_objective
                    )
                    

            #self.oracle.Repeat_good(extended_Hyperspace ,i ,fmin_objective,parent.configuration_list[-1])
             #   self.oracle.compute_Batch(extended_Hyperspace ,int(self.num_config/sqrt_config) -1 , i ,fmin_objective)
           
            print("totalt time: " +  str(time.time() - start_time))


            combined_losses = np.concatenate(
                        (
                            self.losses,
                            
                                [self.parents[i].get_loss()[-1].item() for i in range(self.n_parents)]
                            
                        ),
                        0
                    )
            ixs_parents = np.argsort(combined_losses)
            parent_idx = ixs_parents[:self.n_parents]
            print(combined_losses)
            print(parent_idx)
            # ??? why saving it in a numpt array ?
            # It is creating the new Parent `array`
            temp_parents = [''] * self.n_parents

            for j, x in enumerate(parent_idx):
                # ??? why converting it to integer ?
                x = int(x)
                if x >= self.num_config:
                    temp_parents[j] = copy.deepcopy(self.parents[x - self.num_config])
                    temp_parents[j].replication(self.n_parents)
                else:
                    temp_parents[j] = copy.deepcopy(self.parents[math.floor(x/self.num_config * self.n_parents)])
                    temp_parents[j].update(self.h[x], self.losses[x], self.models[x])
               #     temp_parents[j].point_hyperspace = Trials()

            self.parents = temp_parents


def flatten_dict(d: dict, delimiter="/") -> dict:
    """
    >>> d = {'a': 1,
    ...     'c': {'a': 2, 'b': {'x': 5, 'y': 10}},
    ...     'd': [1, 2, 3]}

    >>> flatten_dict(d)
    {'a': 1, 'd': [1, 2, 3], 'c_a': 2, 'c_b_x': 5, 'c_b_y': 10}
    """
    df = pd.json_normalize(d, sep=delimiter)
    return df.to_dict(orient='records')[0]


class FSVNLogger(tune.logger.Logger):
    def _init(self):
        progress_file = os.path.join(DEFAULT_PATH, "FSNV_MNIST_GP.csv")
        self._continuing = os.path.exists(progress_file)
        self._file = open(progress_file, "a")
        self._csv_out = None
    def on_result(self, result):
        tmp = result.copy()
        result = flatten_dict(tmp, delimiter="/")
        if self._csv_out is None:
            self._csv_out = csv.DictWriter(self._file, result.keys())
            self._csv_out.writeheader()

            #if not self._continuing:
        self._csv_out.writerow(
            {k: v
             for k, v in result.items() if k in self._csv_out.fieldnames})
        self._file.flush()
                   

config= {
     "lr": hp.loguniform("lr",-10*2.3,0)

    #    , "droupout_prob": hp.uniform("droupout_prob",0,1)
#          ,   "weight_decay": hp.loguniform("weight_decay",-5*2.3,-1*2),
#    "b1" : 1-hp.loguniform("b1",-4*2.3, -1*2.3),
#    "b2" : 1-hp.loguniform("b2",-5*2.3, -2*2.3)
}


config = DesignSpace().parse([{'name' : 'lr', 'type' : 'pow', 'lb' : math.e**-10, 'ub' : 1},
                              {'name' : 'aiteration', 'type' : 'int', 'lb' : 0, 'ub' : 0}])

fsvnlogger = FSVNLogger(config,"")




    
CONFIGURATION = 4
ITERATIONS = 10

model = train_mnist
oracle = Guesser(config,0)


scheduler = Scheduler(
  model,
  ITERATIONS,
  CONFIGURATION,
  oracle) 

start_time = time.time()
scheduler.initialisation()     
scheduler.loop()     
print("totalt time: " +  str(time.time() - start_time))
