import copy
import math
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

from ray import tune
from ray.tune.logger import *

# Trials is an object that contains all the informations about the different trials
from hyperopt import hp, fmin, tpe, Trials
from functools import *

# importing this
from sklearn import datasets
import pandas as pd

from sklearn.model_selection import train_test_split

# the x variable is the variable whit the model configurations


def test_function(x, models, h, losses, parent_model, k_f, iteration, fsvnlogger, printer):
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
    print("# Entering test_function")
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

    if is_list:
        k_f[0] += 1

    h[k] = x
    models[k].train1()
    loss = models[k].test1()
    test = models[k].val1()

    if printer:
        temp = dict(x)
        temp.update({'loss': loss})
        temp.update({'test': test})
        fsvnlogger.on_result(temp)

    print("#§ k : {}\tloss : {}\ttest : {}".format(k, loss, test))
    losses[k] = -loss

    print("# Exiting test_function")

    return -loss


def parent_idxs_choice(sorted_idxs, n_total, **optional_args):
    """Returns the idexes of the new parents for the next iteration. Used in Scheduler.loop()

    Args:
        sorted_idxs (list): Indexes of the sorted loss for the previous iteration.
        n_total (int): Length of the returned array.

    Returns:
        list: List containing the sorted indexes of the new parents for the next iteration.
    """
    acceptance_probability = optional_args.get("accept_prob", 0.9)
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


def choice_of_c_constants(prev_c, **optional_args):
    """Function that returns the new constants that will be used in the iteration knowing the previous ones.
    The returned values for the two constants should be such that there are enough models to find enough parents and
    also should be bounded in ...

    Args:
        prev_c ([type]): [description]

    Returns:
        int, int: The two contants with the smaller one before and the bigger one after
    """
    c_1 = prev_c - 1 if not(prev_c == 1) else prev_c
    c_2 = prev_c + 1 if not(prev_c == 3) else prev_c
    return c_1, c_2


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

    def replication(self, n_children):
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

    def __init__(self, model, num_iterations, num_config, oracle, naccache, printer, learn_c=True):
        self.oracle = oracle                                        # Oracle manages the BO
        self.num_iterations = num_iterations                        # total number of iterations
        self.num_config = num_config                                # number of configurations at each step
        self.naccache = naccache                                    # la constante de naccache
        self.n_parents = math.floor(math.sqrt(num_config/naccache))
        self.c_1, self.c_2 = choice_of_c_constants(self.naccache)   # the next two constants

        self.n_first_half = math.floor((self.num_config - 1) / 2) + 1
        self.n_second_half = math.floor((self.num_config - 1) / 2)

        self.n_first_batch_parents = 0
        self.n_second_batch_parents = 0
        
        # self.h is for the num_config hyperparameters used at every loop, h is a configuration from the search space
        self.h = [{}] * self.num_config

        # it is a boolean to indicate if the trial should be stored in the .csv file
        self.print = printer

        # ??? there is no need to save all the points in hyperspace
        #  self.points_hyperspace = np.empty(num_config)

        # self.parents is the sqrt(m) best model from last iteration that are the parents in the current iteration
        self.parents = [{}]*self.n_parents

        # all the models
        self.models = [model]*self.num_config

        # self.losses remembers the performances of all m models during the current iteration
        # sqrt(m) best from self.models
        self.losses = np.empty(num_config)

        # Logger for the training results
        self.logger = FSVNLogger(oracle.searchspace, "")

        # c'est pour avoir un pointeur sur k, c'est pas plus que O(sqrt)-paralélisable  pour le moment du coup.
        self.k = [0]

    def initialisation(self):
        """It will initialise the search process creating all the relevat structures
        and it will also compute the first iteration of the algorithm.
        """
        print("> Entering Scheduler.initialisation()")

        # Database that will save all the evaluated points used by hyperopts
        point_extended_hyperspace = Trials()

        # defines the test function, partial sets all the different parameters
        fmin_objective = partial(
            test_function,
            models=self.models,
            h=self.h,
            losses=self.losses,
            parent_model=self.models,
            # train=self.train_loader,
            # val=self.val_loader,
            # test=self.test_loader,
            k_f=self.k,
            iteration=0,
            fsvnlogger=self.logger,
            printer=self.print
        )

        self.oracle.compute_batch(point_extended_hyperspace, self.num_config, 0, fmin_objective)

        # where are the losses computed? in the `test_function`
        indexes = np.argsort(self.losses)

        self.n_first_batch_parents = math.floor(math.sqrt(self.num_config / 2 * self.c_1))
        self.n_second_batch_parents = math.floor(math.sqrt(self.num_config / 2 * self.c_2))

        # we have in models all the models being trained, while we have that the models that
        # can generate all the other models are in self.parents
        # all the parents models have right now the informations about all the losses
        self.parents = [
            Parent(
                copy.deepcopy(point_extended_hyperspace),   # Trials function
                self.h[indexes[i]],                         # the hyperpoint is chosent during fmin
                self.models[indexes[i]],                    # saves the model
                self.losses[indexes[i]]                     # saves the loss
            )
            for i in range(self.n_first_batch_parents + self.n_second_batch_parents)
        ]

        self.n_parents = self.n_first_batch_parents + self.n_second_batch_parents

        print("> len of self.parents : {}".format(self.n_first_batch_parents + self.n_second_batch_parents))

        print("> Exiting Scheduler.initialisation()")

    def loop(self):
        """Function to do the training for a number of times defined by the variable self.num_iterations.
        """
        print("^ Entering Scheduler.loop()")

        for current_iter in range(1, self.num_iterations):

            # it reinitialises the value of k that will be modified in the `test_function`
            self.k[0] = 0

            parent_order = ["first_batch"] * self.n_first_batch_parents + ["second_batch"] * self.n_second_batch_parents
            random.shuffle(parent_order)

            flag_init_first_batch = False
            flag_init_second_batch = False

            trained_models_c = []
            pervious_parent_c = []

            print("^ len of parents : {} and c_1 : {} and c_2 : {}".format(len(self.parents), self.c_1, self.c_2))

            for idx_parent, parent in enumerate(self.parents):

                batch = parent_order[idx_parent]
                c = self.c_1 if batch == "first_batch" else self.c_2
                n_el_current = self.n_first_half if batch == "first_batch" else self.n_second_half
                n_el_curr_batch = self.n_first_batch_parents if batch == "first_batch" else self.n_second_batch_parents

                print("^^ current_iter : {:d} idx_parent : {:d} c : {:d} curr_el : {} elem_batch : {}".format(
                    current_iter, idx_parent, c, n_el_current, n_el_curr_batch))

                # this thing returns the Trials function for given parent
                point_extended_hyperspace = parent.get_point_hyperspace()

                fmin_objective = partial(
                    test_function,
                    models=self.models,                 # ??? why it is calling all the self.models
                    h=self.h,                           # ??? why not using the configuration list in parent
                    losses=self.losses,
                    parent_model=parent.get_model(),
                    k_f=self.k,
                    iteration=current_iter,
                    # train=self.train_loader,
                    # val=self.val_loader,
                    # test=self.test_loader,
                    fsvnlogger=self.logger,
                    printer=self.print
                )

                # ??? Is it repeating the configuration of the best one? HOW ?
                if not parent.is_replicated:
                    print("^^ parent has NOT been replicated")

                    # !!! gives a problem
                    self.oracle.repeat_good(
                        point_extended_hyperspace,      # trials fucntion for the current parent
                        len(parent.get_loss()),         # number of iterations of the parent
                        fmin_objective,
                        parent.configuration_list[-1]   # it is the last hyperpoint of the parent
                    )

                    # define the number of models to train from this parent in this loop iteration
                    # only the best parent has childrens
                    if not flag_init_first_batch if batch == "first_batch" else not flag_init_second_batch:
                        n_training = n_el_current - (n_el_curr_batch - 1) * math.floor(n_el_current / n_el_curr_batch) - 1

                        if batch == "first_batch":
                            flag_init_first_batch = True
                        else:
                            flag_init_second_batch = True

                    else:
                        n_training = math.floor(n_el_current / n_el_curr_batch) - 1

                    print("^^ n_training : {}".format(n_training))

                    # computes the new batch for each one of the parents for every iteration
                    # tehy are all going to be sons of this same parent since they have the same Trials func
                    self.oracle.compute_batch(
                        point_extended_hyperspace,
                        n_training,
                        len(parent.get_loss()),
                        fmin_objective
                    )

                    # used to remember if the model has been generate with which c
                    trained_models_c += [c] * (n_training)
                    pervious_parent_c += [c]

                    print("^^ trained_models_c : {}".format(trained_models_c))
                    print("^^ previous_parent_c : {}".format(pervious_parent_c))

                else:
                    print("^^ parent has been replicated")

                    if not flag_init_first_batch if batch == "first_batch" else not flag_init_second_batch:
                        n_training = n_el_current - (n_el_curr_batch - 1) * math.floor(n_el_current / n_el_curr_batch)

                        if batch == "first_batch":
                            flag_init_first_batch = True
                        else:
                            flag_init_second_batch = True

                    else:
                        n_training = math.floor(n_el_current / n_el_curr_batch)

                    print("^^ n_training : {}".format(n_training))

                    # replicated parent
                    self.oracle.compute_batch(
                        point_extended_hyperspace,
                        n_training,
                        len(parent.get_loss()),
                        fmin_objective
                    )

                    trained_models_c += [c] * n_training

                    print("^^ trained_models_c : {}".format(trained_models_c))

            combined_cs = trained_models_c + pervious_parent_c
            combined_losses = np.concatenate(
                (
                    self.losses,                        # contains the losses of the evolved parents
                    [self.parents[i].get_loss()[-1] for i in range(self.n_parents)]  # losses of prev. parents
                ),
                0
            )

            print("^ self.losses : {} parent.losses : {}".format(self.losses, [
                  self.parents[i].get_loss()[-1] for i in range(self.n_parents)]))

            ixs_parents = np.argsort(combined_losses)

            self.c_1, self.c_2 = choice_of_c_constants(combined_cs[ixs_parents[0]])

            print("^ new c : {}".format(combined_cs[ixs_parents[0]]))

            self.n_first_batch_parents = math.floor(math.sqrt(self.num_config / 2 * self.c_1))
            self.n_second_batch_parents = math.floor(math.sqrt(self.num_config / 2 * self.c_2))

            parent_idx = parent_idxs_choice(
                ixs_parents,
                self.n_first_batch_parents + self.n_second_batch_parents,
                accept_prob=0.95
            )

            temp_parents = [''] * self.n_parents

            for j, x in enumerate(parent_idx):
                x = int(x)
                if x >= self.num_config:
                    temp_parents[j] = copy.deepcopy(self.parents[x - self.num_config])
                    temp_parents[j].replication(self.n_parents)  # ??? should check why replication works like this
                else:
                    temp_parents[j] = copy.deepcopy(self.parents[math.floor(
                        x/self.num_config * (self.n_first_batch_parents + self.n_second_batch_parents)
                    )])
                    temp_parents[j].update(self.h[x], self.losses[x], self.models[x])

            self.parents = temp_parents
            self.n_parents = self.n_first_batch_parents + self.n_second_batch_parents

            print("^ Next loop")

        print("^ Exiting Scheduler.loop()")


class FSVNLogger(tune.logger.Logger):
    """Class for logging the result of tests in a .csv file
    """

    def _init(self):
        progress_file = os.path.join("", "FSVN_2_1-8.csv")
        self._continuing = os.path.exists(progress_file)
        self._file = open(progress_file, "a")
        self._csv_out = None

    def on_result(self, result):
        tmp = result.copy()
        result = flatten_dict(tmp, delimiter="/")
        if self._csv_out is None:
            self._csv_out = csv.DictWriter(self._file, result.keys())
            self._csv_out.writeheader()

            # if not self._continuing:
        self._csv_out.writerow(
            {k: v for k, v in result.items() if k in self._csv_out.fieldnames})
        self._file.flush()
