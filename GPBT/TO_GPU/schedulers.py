import copy
import math
import random
import os

import numpy as np
import time
import datetime
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

import time
# Trials is an object that contains all the informations about the different trials
from hyperopt import hp, fmin, tpe, Trials
from functools import *

# importing this
from sklearn import datasets
import pandas as pd

from sklearn.model_selection import train_test_split

# the x variable is the variable whit the model configurations

def translation(liste):
    config = {}
    config["lr"] = liste[0]
    config["droupout_prob"] = liste[1]
    config["weight_decay"] = liste[2]
    config["b1"] = liste[3]
    config["b2"] = liste[4]
    config["eps"] = liste[5]
    return config

def test_function(x, models, h, losses, losses_temp, parent_model, k_f, iteration, fsvnlogger, printer,old_time):
    """General function that trains the given model with the given hyperparameters and computes the
    result for the train and validation set. This function is called by the Scheduler from
    initialisation() and loop().

    Args:
        x ([type]): new hyperparameter values
        models ([type]): model
        h ([type]): list of model parameters
        losses ([type]): array with all the losses
        parent_model ([type]): [description]
        k_f ([type]): [description]
        iteration ([type]): [description]
        fsvnlogger ([type]): [description]
        printer ([type]): [description]
    """
 #   x= translation(x)
    if isinstance(k_f, list):
        k = k_f[0]
        is_list = True
    else:
        k = k_f
        is_list = False

    # checks if it's the first time and if so it initialises the model
    if iteration == 0:
        models[k] = parent_model[k](x)
    else:
        models[k] = parent_model.adapt(x)

    # here it increases the pointer to each one of the models
    if is_list:
        k_f[0] += 1

    # this changes the actual array in the Scheduler
    h[k] = x
    #models[k].train1()
    for i in range(39):
        train_acc, train_los,acc_val,loss_val = models[k].step()
        if printer:
            temp = dict(x)
            temp.update({'train_acc': train_acc})
            temp.update({'train_los': train_los})
            temp.update({'acc_val': acc_val})
            temp.update({'loss_val': loss_val})
            temp.update({'time':time.time()-old_time})
            fsvnlogger.writerow(temp)

    print(k)
    
    losses_temp[k] = loss_val

    print(np.sort(losses_temp[:k+1]))


    if iteration !=0 and (k==0 or (k==1 and loss_val<losses_temp[0]) or (k!=1 and loss_val<np.sort(losses_temp[:k])[math.floor(k/2)])):
        
        train_acc1, train_los1,acc_val1,loss_val1 = models[k].step()
        train_acc2, train_los2,acc_val2,loss_val2 = models[k].step()
        train_acc3, train_los3,acc_val3,loss_val3 = models[k].step()
        train_acc4, train_los4,acc_val4,loss_val4 = models[k].step()

        train_acc, train_los,acc_val,loss_val = ((train_acc+train_acc1+train_acc2+train_acc3+train_acc4)/5, (train_los+train_los1+train_los2+train_los3+train_los4)/5,
        (acc_val+acc_val1+acc_val2+acc_val3+acc_val4)/5,(loss_val+loss_val1+loss_val2+loss_val3+loss_val4)/5)
    
    if iteration ==0 or (k==0 or (k==1 and loss_val<losses_temp[0]) or (k!=1 and loss_val<np.sort(losses_temp[:k])[math.floor(k/2)])):
        acc_test,loss_test = models[k].test1()
    
        if printer:
            temp = dict(x)
            temp.update({'train_acc': train_acc})
            temp.update({'train_los': train_los})
            temp.update({'acc_val': acc_val})
            temp.update({'loss_val': loss_val})
            temp.update({'acc_test': acc_test})
            temp.update({'loss_test': loss_test})
            temp.update({'time':time.time()-old_time})
            fsvnlogger.writerow(temp)
    else:
        if printer:
            temp = dict(x)
            #acc_test,loss_test = models[k].test1()
            temp.update({'train_acc': train_acc})
            temp.update({'train_los': train_los})
            temp.update({'acc_val': acc_val})
            temp.update({'loss_val': loss_val})
            #temp.update({'acc_test': acc_test})
            #temp.update({'loss_test': loss_test})
            temp.update({'time':time.time()-old_time})
            fsvnlogger.writerow(temp)


    

        
#    print("# loss : {}\t test : {}".format(loss, test))

    losses[k] = loss_val

   # print("# Exiting test_function")

    return loss_val


def parent_idxs_choice(sorted_idxs, n_total, **opttional_args):
    """Returns the idexes of the new parents for the next iteration. Used in Scheduler.loop()

    Args:
        sorted_idxs (list): Indexes of the sorted loss for the previous iteration.
        n_total (int): Length of the returned array.

    Returns:
        list: List containing the sorted indexes of the new parents for the next iteration.
    """
    acceptance_probability = opttional_args.get("accept_prob", 0.9)
    new_idxs = [{}] * n_total

    i = 0
    j = 0
    while i < n_total and j < len(sorted_idxs) - (n_total - i):
        if random.uniform(0, 1) < acceptance_probability:
            new_idxs[i] = sorted_idxs[j]
            i += 1
        j += 1

    while i < n_total:
        new_idxs[i] = sorted_idxs[j]
        i += 1
        j += 1

    return new_idxs


class Parent():
    """Parent class that handles the passage of Network Configurations from one step to the
    following
    """

    def __init__(self, point_hyperspace, configuration, model, loss):
        # Trials function
        self.point_hyperspace = point_hyperspace
        # list of the different hyperpoints tried for this model
        self.configuration_list = [configuration]
        # list of all the previous losses of this function
        self.loss_list = [loss]
        self.model = model
        self.is_replicated = False

    def update(self, configuration, loss, model):
        self.is_replicated = False
        self.configuration_list.append(configuration)
        self.loss_list.append(loss)
        self.model = model

    def replication(self):
        self.is_replicated = True
       # self.configuration_list.append(self.configuration_list[-1])
       # self.loss_list.append(self.loss_list[-1])

       # ?? why it its working now ?
       # replication_trials(self.point_hyperspace.trials, n_children)

    def get_last_conf(self):
        return self.configuration_list[-1]

    def get_point_hyperspace(self):
        return self.point_hyperspace

    def get_model(self):
        return self.model

    def get_loss(self):
        return self.loss_list



class Scheduler():
    """Scheduler class that handles all the training process for our proposed algorithm (FSVN)
    """


    def __init__(self, model, num_iterations, num_config, oracle, naccache, printer,nott):
        self.oracle = oracle                        # Oracle manages the BO
        self.num_iterations = num_iterations        # total number of iterations
        self.num_config = num_config                # number of configurations at each step
        self.naccache = naccache                    # la constante de naccache
        self.n_parents = math.floor(math.sqrt(num_config/naccache))
        self.time = time.time()
        # self.h is for the num_config hyperparameters used at every loop, h is a configuration from the search space
        self.h = [{}] * self.num_config

        # it is a boolean to indicate if the trial should be stored in the .csv file
        self.print = printer

        # this will be called again in the initialization() method
      #  self.points_hyperspace = np.empty(num_config)     # self.hyperspaces is for storing the
                                                                # sqrt(m) hyperspaces used by the algorithm
        self.plot = np.empty(num_iterations)  # this will contain the value for the loss at each
        
        # ??? there is no need to save all the points in hyperspace
        #  self.points_hyperspace = np.empty(num_config)

        # self.parents is the sqrt(m) best model from last iteration that are the parents in the current iteration
        self.parents = [{}]*self.n_parents

        # all the models
        self.models = [model]*self.num_config

        # self.losses remembers the performances of all m models during the current iteration
        # sqrt(m) best from self.models
        self.losses = np.empty(num_config)
        self.losses_temp = np.empty(num_config)
        if not os.path.isdir('./log_data'):
            os.makedirs('./log_data')
        import csv


        temp = [k for k,v in oracle.searchspace.items()]
        temp.append('aiteration')

        temp.append('acc_val')
        temp.append('loss_val')
        temp.append('acc_test')
        temp.append('loss_test')
        temp.append('train_acc')
        temp.append('train_los')
        temp.append('time')
        self.file = open(os.path.join('./log_data', 'log_seed_{0}_{1}.csv'.format(1234, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))), 'a')
        self.logger = csv.DictWriter(self.file, delimiter=",", fieldnames=np.sort(np.array(temp)), extrasaction='ignore')
        self.logger.writeheader()
        self.file.flush()

        self.nott = nott
        
        # ??? didn't understand the use of k
        # c'est pour avoir un pointeur sur k, c'est pas plus que O(sqrt)-paralÃƒÆ’Ã‚Â©lisable  pour le moment du coup.
        self.k = [0]


    def initialisation(self):
        """It will initialise the search process creating all the relevat structures
        and it will also compute the first iteration of the algorithm.
        """
 #       print("> Entering Scheduler.initialisation()")

        # Database that will save all the evaluated points used by hyperopts
        point_extended_hyperspace = Trials()# [None,None]

        # defines the test function, partial sets all the different parameters
        fmin_objective = partial(
            test_function,
            models=self.models,
            h=self.h,
            losses=self.losses,
            losses_temp=self.losses_temp,
            parent_model=self.models,
            # train=self.train_loader,
            # val=self.val_loader,
            # test=self.test_loader,
            k_f=self.k,
            iteration=0,
            fsvnlogger=self.logger,
            printer=self.print,old_time = self.time
        )

        self.oracle.compute_batch(point_extended_hyperspace, self.num_config, 0, fmin_objective)

        # where are the losses computed? in the `test_function`
        indexes = np.argsort(self.losses)
        # self.out[0] = self.losses[indexes[0:self.n_parents]]

        # ??? NOT USED ANYMORE. WHY ?
        # self.points_hyperspace = [point_extended_hyperspace] * self.n_parents

        # we have in models all the models being trained, while we have that the models that
        # can generate all the other models are in self.parents
        # all the parents models have right now the informations about all the losses
        self.parents = [
            Parent(
                #copy.deepcopy(point_extended_hyperspace),   # Trials function
                Trials(),
                self.h[indexes[i]],                         # the hyperpoint is chosent during fmin
                self.models[indexes[i]],                    # saves the model
                self.losses[indexes[i]]                     # saves the loss
            )
            for i in range(self.n_parents)
        ]


        # for the plot it only saves the best loss, which is the one we are going to take in the end
        self.plot[0] = self.losses[indexes[0]]

    def close(self):
        for i in self.parents[0].configuration_list:
            self.logger.writerow(i)

        self.file.flush()

        
    def loop(self):
        """Function to do the training for a number of times defined by the variable self.num_iterations.
        """
   #     print("^ Entering Scheduler.loop()")

        for current_iter in range(1, self.num_iterations):

            # it reinitialises the value of k that will be modified in the `test_function`
            self.k[0] = 0

            threshold = 1.7976931348623157e+308
            if current_iter>1+self.nott:

                threshold = -self.plot[-1] + self.plot[-2] 

            stop = False
            self.losses = np.ones(self.num_config)*1.7976931348623157e+307
        
            for idx_parent, parent in enumerate(self.parents):
     
                if not(stop) and np.min(self.losses[-self.num_config:]) < - threshold + self.plot[-1] :
           #         print(np.min(self.losses[-self.num_config:]))
           #         print( self.plot[-1])
                    stop = True

                if stop:
   #                 print("stopped!")
  #                  print("saved trainings : " + str(self.num_config/self.n_parents))
                    break

                
 #               print("^^ current_iter : {:d} and idx_parent : {:d}".format(current_iter, idx_parent))

                point_extended_hyperspace = parent.get_point_hyperspace()

#                print("^^ last loss of parent : {:.4f}".format(parent.get_loss()[-1]))

                fmin_objective = partial(
                    test_function,
                    models=self.models,                 # ??? why it is calling all the self.models
                    h=self.h,                           # ??? why not using the configuration list in parent
                    losses=self.losses,
                    losses_temp=self.losses_temp,
                    parent_model=parent.get_model(),
                    k_f=self.k,
                    iteration=current_iter,
                    # train=self.train_loader,
                    # val=self.val_loader,
                    # test=self.test_loader,
                    fsvnlogger=self.logger,
                    printer=self.print,old_time=self.time
                )

                if not parent.is_replicated:
                #    point_extended_hyperspace = Trials() #TODO Delete
                #    parent.point_hyperspace = Trials() #TODO Delete
  #                  print("^^ parent has NOT been replicated")

                    self.oracle.repeat_good(
                        point_extended_hyperspace,      # trials fucntion for the current parent
                        len(parent.get_loss()),         # number of iterations of the parent
                        fmin_objective,
                        parent.configuration_list[-1]   # it is the last hyperpoint of the parent
                    )

                    

                    
                    
                    # define the number of models to train from this parent in this loop iteration
                    # only the best parent has childrens
                    if idx_parent == 0:
                        numb_training = self.num_config - (self.n_parents - 1) * \
                            math.floor(self.num_config/self.n_parents) - 1
                    else:
                        numb_training = math.floor(self.num_config/self.n_parents) - 1

                    if not(stop) and np.min(self.losses[-self.num_config:]) < - threshold + self.plot[-1] :
                        stop = True
                    if stop:
  #                      print("stopped!")
   #                     print("saved trainings : " + str(numb_training))

                        break      


                    # computes the new batch for each one of the parents for every iteration
                    # tehy are all going to be sons of this same parent since they have the same Trials func
                    self.oracle.compute_batch(
                        point_extended_hyperspace,
                        numb_training,
                        len(parent.get_loss()),
                        fmin_objective
                    )


                else:
#                    print("^^ parent has been replicated")

                    if idx_parent == 0:
                        numb_training = self.num_config - (self.n_parents - 1) * math.floor(self.num_config/self.n_parents)
                    else:
                        numb_training = math.floor(self.num_config/self.n_parents)

                    # replicated parent
                    self.oracle.compute_batch(
                        point_extended_hyperspace,
                        numb_training,
                        len(parent.get_loss()),
                        fmin_objective
                    )

            combined_losses = np.concatenate(
                (
                    self.losses,
                    [self.parents[i].get_loss()[-1] for i in range(self.n_parents)]
                ),
                0
            )

                
            combined_losses = self.losses
            #sorted_loss = np.sort(combined_losses)
            #sigma=sorted_loss[self.n_parents] - sorted_loss[1] 
            #best children has probability >99.5% to be taken
            #all sqrt(n) best children have probability >50% to be taken
            #noised_loss = [np.random.normal(i,sigma/5,) for i in combined_losses]
            #print(combined_losses)
            #print(np.argsort(noised_loss)[:self.n_parents])
            #print(np.argsort(combined_losses)[:self.n_parents])
            ixs_parents= np.argsort(combined_losses)
            #ixs_parents = np.argsort(noised_loss) #Amelioration proposed by Naccache: non deterministic selection of children
             
            
            parent_idx = ixs_parents[:self.n_parents]


            #parent_idx = parent_idxs_choice(ixs_parents, self.n_parents, accept_prob=0.95)

            temp_parents = [''] * self.n_parents

            for j, x in enumerate(parent_idx):
                x = int(x)
                if x >= self.num_config:
                    print("Do we go there? (replications)")
                    temp_parents[j] = copy.deepcopy(
                        self.parents[x - self.num_config])
                    temp_parents[j].replication()

                else:
                    
                    temp_parents[j] = copy.deepcopy(self.parents[math.floor(x/self.num_config * self.n_parents)])
                    temp_parents[j].update(self.h[x], self.losses[x], self.models[x])
                    temp_parents[j].point_hyperspace = Trials() #TODO Delete

            self.parents = temp_parents
            # updating the best loss for the plot
            self.plot[current_iter] = combined_losses[parent_idx[0]]
            print("iter" + str(current_iter))
            print("parent_idx" + str(combined_losses[parent_idx[0]]))
            self.file.flush()


    #    print("^ Exiting Scheduler.loop()")


