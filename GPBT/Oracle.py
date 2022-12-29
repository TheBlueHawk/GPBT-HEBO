def set_iteration(algo,iteration):
  algo.space.paras["aiteration"].lb=iteration
  algo.space.paras["aiteration"].ub=iteration

class Guesser():
    def __init__(self, searchspace, verbose):
        self.searchspace = searchspace
        self.verbose = verbose
        print(self.searchspace)
        self.algo = HEBO(searchspace)

    def repeat_good(self, trials, iteration, function, configuration):
        configuration = copy.deepcopy(configuration)
        configuration["aiteration"] = iteration
        print(configuration)
        rec = pd.DataFrame(configuration,index=[0])   
        res = np.array([np.array([function(configuration)])])
        self.algo.observe(rec,res)

    def compute_batch(self, trials, nb_eval, iteration, function):
        set_iteration(self.algo, iteration) 
        for i in range(nb_eval):
         rec = self.algo.suggest(n_suggestions = 1)
         rec1 = rec.to_dict()
         for key in rec1:
          rec1[key] = rec1[key][list(rec1[key].keys())[0]] 
         res = np.array([np.array([function(rec1)])])
         self.algo.observe(rec,res)

