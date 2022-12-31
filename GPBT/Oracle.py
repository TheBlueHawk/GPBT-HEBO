import copy
import numpy as np
from hebo.optimizers.hebo import HEBO
import pandas as pd
import hyperopt


def set_iteration(algo,iteration):
  algo.space.paras["aiteration"].lb=iteration
  algo.space.paras["aiteration"].ub=iteration

class Guesser():
    """Used to sample the hyperspace with the tools of `hyperopt`
    """

    def __init__(self, searchspace, verbose):
        self.searchspace = searchspace
        self.string = "aiteration"

        self.verbose = verbose
        print(self.searchspace)
        self.algo = HEBO(searchspace)

    def store_trials(self, trials: list):
        trials[0] = (self.algo.X).to_dict("records")
        trials[1] = self.algo.y.tolist()

    #Function to copy trials to HEBO should not be needed
    """def copy_trials(self, trials):
        if len(trials[0]) > 0:
            df = pd.DataFrame(trials[0])
            self.algo.observe(df, np.asarray(trials[1]))"""

    def repeat_good(self, trials, iteration, function, configuration):
        configuration = copy.deepcopy(configuration)
        configuration["aiteration"] = iteration
        print(configuration)
        rec = pd.DataFrame(configuration,index=[0])   
        res = np.array([np.array([function(configuration)])])
        self.algo.observe(rec,res)
        self.store_trials(trials)

    def compute_batch(self, trials: list, nb_eval, iteration, function):#hyperopt.base.Trials
        #print(trials.trials)
        set_iteration(self.algo, iteration)
        #self.copy_trials(trials)
        for i in range(nb_eval):
         rec = self.algo.suggest(n_suggestions = 1,fix_input = {"aiteration":iteration})
         rec1 = rec.to_dict()
         for key in rec1:
          rec1[key] = rec1[key][list(rec1[key].keys())[0]] 
         res = np.array([np.array([function(rec1)])])
         self.algo.observe(rec,res)
        self.store_trials(trials)

