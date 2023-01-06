import copy
from functools import partial
import numpy as np
from hebo.optimizers.hebo import HEBO
import pandas as pd
import hyperopt
from hyperopt import hp, fmin, tpe, Trials
from general_model import general_model


def set_iteration(algo, iteration):
    algo.space.paras["iteration"].lb = iteration
    algo.space.paras["iteration"].ub = iteration


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


class GPBTOracle:
    def __init__(self, searchspace):
        # self.hyperspace is the original (input) searchspace
        self.searchspace = searchspace

    def repeat_good(self, trials, iteration, function, configuration):  # add space
        space = copy.deepcopy(configuration)
        for k, v in configuration.items():
            if not isinstance(v, str):
                space[k] = hp.uniform(k, -1e-10 + v, v + 1e-10)

        curr_eval = getattr(trials, "_ids")
        if curr_eval == set():
            curr_eval = 0
        else:
            curr_eval = max(curr_eval) + 1
        space["itération"] = hp.quniform(
            "itération", -0.5 + iteration, 0.5 + iteration, 1
        )
        fmin(
            function,
            space,
            algo=partial(tpe.suggest, n_startup_jobs=1),
            max_evals=curr_eval + 1,
            trials=trials,
        )

    def compute_once(self, trials, iteration, function):  # add space

        space = copy.deepcopy(self.searchspace)
        curr_eval = getattr(trials, "_ids")
        if curr_eval == set():
            curr_eval = 0
        else:
            curr_eval = max(curr_eval) + 1
        space["itération"] = hp.quniform(
            "itération", -0.5 + iteration, 0.5 + iteration, 1
        )
        fmin(
            function,
            space,
            algo=partial(tpe.suggest, n_startup_jobs=1),
            max_evals=curr_eval + 1,
            trials=trials,
        )

    def compute_batch(self, trials, nb_eval, iteration, function):  # add space

        space = copy.deepcopy(self.searchspace)
        curr_eval = getattr(trials, "_ids")
        if curr_eval == set():
            curr_eval = 0
        else:
            curr_eval = max(curr_eval) + 1

        space["itération"] = hp.quniform(
            "itération", -0.5 + iteration, 0.5 + iteration, 1
        )
        fmin(
            function,
            space,
            algo=partial(tpe.suggest, n_startup_jobs=1),
            max_evals=curr_eval + nb_eval,
            trials=trials,
        )


class GPBTHEBOracle:
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


class HEBOOralce:
    def __init__(self, searchspace):
        self.searchspace = searchspace
        self.algo = HEBO(searchspace)
        self.model = None

    def reset(self):
        self.algo = HEBO(self.searchspace)

    def compute_batch(self, num_config, iterations, logger):
        for i in range(iterations):
            print("iteration: ",i)
            set_iteration(self.algo, i)
            records = self.algo.suggest(n_suggestions=num_config, fix_input={"iteration": i})
            losses = []
            tests = []
            for idx, rec in records.iterrows():
                rec1 = rec.to_dict()

                self.model = general_model(rec1)
                # for j in range(iterations):
                self.model.train1(verbose=False)
                loss = self.model.test1()
                test = self.model.val1()
                print(f"accuracy sub model: {loss}")
                losses.append(loss)
                tests.append(test)
                temp = rec1
                temp.update({"iteration": i})
                temp.update({"loss": loss})
                temp.update({"test": test})
                logger.on_result(temp)
                # print("--- %s seconds ---" % (time.time() - start_time))

            best_idx = int(np.argsort(losses)[-1])
            """temp = records.iloc[[best_idx]].to_dict('records')[0]
            temp.update({"iteration": i})
            temp.update({"loss": losses[best_idx]})
            temp.update({"test": tests[best_idx]})
            logger.on_result(temp)"""
            print("accuracy: " + str(losses[best_idx]) + "\n")
            self.algo.observe(records, np.asarray([[l] for l in losses]))
