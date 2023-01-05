import copy
from functools import partial
import numpy as np
from hebo.optimizers.hebo import HEBO
import pandas as pd
import hyperopt
from hyperopt import hp, fmin, tpe, Trials


def set_iteration(algo, iteration):
    algo.space.paras["iteration"].lb = iteration
    algo.space.paras["iteration"].ub = iteration


class GBPTOracle:
    """Used to sample the hyperspace with the tools of `hyperopt`"""

    def __init__(self, searchspace, search_algo, verbose):
        self.searchspace = searchspace
        self.verbose = verbose
        print(self.searchspace)
        self.search_algo = search_algo
        self.algo = search_algo(searchspace)

    def store_trials(self, trials: list):
        trials[0] = self.algo.X.to_dict("records")
        trials[1] = self.algo.y.tolist()

    # Function to copy trials to HEBO should not be needed
    def copy_trials(self, trials):
        if len(trials[0]) > 0:
            self.algo.observe(pd.DataFrame(trials[0]), np.asarray(trials[1]))

    def reset_HEBO(self):
        # TODO generalize
        self.algo = self.search_algo(self.searchspace)

    def repeat_good(self, trials, iteration, function, configuration):
        configuration = copy.deepcopy(configuration)
        configuration["iteration"] = iteration
        print(configuration)
        rec = pd.DataFrame(configuration, index=[0])
        res = np.array([np.array([function(configuration)])])
        self.algo.observe(rec, res)
        self.store_trials(trials)

    def compute_batch(
        self, trials: list, nb_eval, iteration, function
    ):  # hyperopt.base.Trials
        # print(trials.trials)
        # TODO: generalize
        self.reset_HEBO()
        set_iteration(self.algo, iteration)
        self.copy_trials(trials)
        for i in range(nb_eval):
            rec = self.algo.suggest(n_suggestions=1, fix_input={"iteration": iteration})
            rec1 = rec.to_dict()
            for key in rec1:
                rec1[key] = rec1[key][list(rec1[key].keys())[0]]
            print(rec1)
            res = np.array([np.array([function(rec1)])])
            self.algo.observe(rec, res)
        self.store_trials(trials)


class SimpleOracle:
    def __init__(self, searchspace, search_algo):
        self.search_algo = search_algo
        self.searchspace = searchspace

    def compute_Once(self, function, evals):
        fmin(
            function,
            self.searchspace,
            algo=self.search_algo,
            max_evals=evals,
            trials=Trials(),
        )


class BayesOpt:
    def __init__(self, searchspace):
        self.searchspace = searchspace

    def compute_Once(self, function, evals):
        fmin(
            function,
            self.searchspace,
            algo=partial(tpe.suggest, n_startup_jobs=1),
            max_evals=evals,
            trials=Trials(),
        )
