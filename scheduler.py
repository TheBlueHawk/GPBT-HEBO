import copy
import math
import time
import numpy as np
from functools import partial
from hyperopt import Trials


class Parent:
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
        self.loss_list = np.append(self.loss_list, loss)
        self.model = model

    def replication(self, n_children):
        self.is_replicated = True

    def get_last_conf(self):
        return self.configuration_list[-1]

    def get_point_hyperspace(self):
        return self.point_hyperspace

    def get_model(self):
        return self.model

    def get_loss(self):
        return self.loss_list

    def set_point_hyperspace(self, point_hyperspace):
        self.point_hyperspace = point_hyperspace


def test_function(x, models, h, losses, parent_model, k_f, iteration, logger):
    if isinstance(k_f, list):
        k = k_f[0]
        Islist = True
    else:
        k = k_f
        Islist = False
    print(k)
    if iteration == 0:
        models[k] = parent_model[k](x)
    else:
        models[k] = parent_model.adapt(x)
    if Islist:
        k_f[0] += 1

    h[k] = x
    models[k].train1()
    loss = models[k].test1()
    test = models[k].val1()

    temp = dict(x)
    temp.update({"loss": loss})
    temp.update({"test": test})
    logger.on_result(temp)

    losses[k] = -loss
    print("accuracy: " + str(loss) + "\n")
    return -loss


class Scheduler:
    def __init__(self, model, num_iteration, num_config, oracle, logger):
        # Oracle manages the Bayesian optimization
        self.oracle = oracle
        self.iteration = num_iteration
        self.num_config = num_config
        self.sqrt_config = math.ceil(num_config / 5)  #

        self.n_parents = self.sqrt_config
        # self.h is for the m "h" used at every loop, h is a configuration from the search space
        self.h = np.repeat({}, num_config)

        # self.out is for storing the result of the algorithm, ie all "h" from all iterations
        # from all sqrt(m) best models per iterations.
        self.out = np.zeros((num_iteration, self.sqrt_config))

        # self.hyperspaces is for storing the sqrt(m) hyperspaces used by the algorithm
        self.hyperspaces = np.zeros(self.sqrt_config)

        self.plot = np.zeros(num_iteration)

        # self.models are the m model that will explore new hyperspace points at every iterations
        self.models = np.repeat(model, num_config)

        # self.parents is the sqrt(m) best model from last iteration
        self.parents = np.repeat(model, self.sqrt_config)

        # self.losses remembers the performances of all m models at one iteration to decide which ones are the sqrt(m) best from self.models.
        self.losses = np.zeros(num_config)
        self.k = [
            0
        ]
        self.logger = logger

    def initialisation(self, model_type = ""):
        num_config = self.num_config
        extended_Hyperspace = [[], []]
        if model_type == "GPBT": extended_Hyperspace = Trials()
        fmin_objective = partial(
            test_function,
            models=self.models,
            h=self.h,
            losses=self.losses,
            parent_model=self.models,
            k_f=self.k,
            iteration=0,
            logger=self.logger,
        )
        self.oracle.compute_batch(extended_Hyperspace, num_config, 0, fmin_objective)

        indexes = np.argsort(self.losses)
        self.out[0] = (self.losses[indexes])[0 : self.sqrt_config]
        self.hyperspaces = [extended_Hyperspace] * self.sqrt_config
        if model_type == "GPBT": self.hyperspaces = np.repeat(extended_Hyperspace, self.sqrt_config)
        self.parents = np.array(
            [
                Parent(
                    copy.deepcopy(extended_Hyperspace),
                    (self.h[indexes])[i],
                    (self.models[indexes])[i],
                    (self.losses[indexes])[i],
                )
                for i in range(self.sqrt_config)
            ]
        )
        self.plot[0] = self.losses[indexes][0]

    def loop(self):
        iteration = self.iteration
        for i in range(1, iteration):

            self.k[0] = 0

            start_time = time.time()
            for j in range(self.n_parents):
                parent = self.parents[j]
                point_extended_hyperspace = parent.get_point_hyperspace()
                print("\n loss of parent " + str(parent.get_loss()[-1]))
                print("\n loss " + str(parent.get_loss()))

                fmin_objective = partial(
                    test_function,
                    models=self.models,
                    h=self.h,
                    losses=self.losses,
                    parent_model=parent.get_model(),
                    k_f=self.k,
                    iteration=len(parent.get_loss()),
                    logger=self.logger,
                )

                if not parent.is_replicated:
                    print("not replicated")
                    self.oracle.repeat_good(
                        point_extended_hyperspace,
                        len(parent.get_loss()),
                        fmin_objective,
                        parent.configuration_list[-1],
                    )

                    # computes the new batch for each one of the parents for every iteration
                    self.oracle.compute_batch(
                        point_extended_hyperspace,
                        int(self.num_config / self.n_parents) - 1,
                        len(parent.get_loss()),
                        fmin_objective,
                    )
                else:

                    print("replicated")
                    self.oracle.compute_batch(
                        point_extended_hyperspace,
                        int(self.num_config / self.n_parents),
                        len(parent.get_loss()),
                        fmin_objective,
                    )
                # Store new hyperspace points for parent, so that they get copied to new parent
                parent.set_point_hyperspace(point_extended_hyperspace)

            print("totalt time: " + str(time.time() - start_time))

            combined_losses = np.concatenate(
                (
                    self.losses,
                    [
                        self.parents[i].get_loss()[-1].item()
                        for i in range(self.n_parents)
                    ],
                ),
                0,
            )
            ixs_parents = np.argsort(combined_losses)
            parent_idx = ixs_parents[: self.n_parents]
            print(combined_losses)
            print(parent_idx)
            temp_parents = [""] * self.n_parents

            for j, x in enumerate(parent_idx):
                x = int(x)
                if x >= self.num_config:
                    temp_parents[j] = copy.deepcopy(self.parents[x - self.num_config])
                    temp_parents[j].replication(self.n_parents)
                else:
                    temp_parents[j] = copy.deepcopy(
                        self.parents[math.floor(x / self.num_config * self.n_parents)]
                    )
                    temp_parents[j].update(self.h[x], self.losses[x], self.models[x])
            self.parents = temp_parents
