import argparse
import ray
from datetime import datetime
from functools import partial
from hyperopt import hp, tpe
from hebo.optimizers.hebo import HEBO
from hebo.design_space.design_space import DesignSpace
from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers.pb2 import PB2
from ray.tune.schedulers.pbt import PopulationBasedTraining
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB

from oracles import GPBTHEBOracle, SimpleOracle, GPBTOracle, HEBOOralce
from scheduler import Scheduler
from general_model import general_model
from utils import Logger


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization")
    parser.add_argument(
        "--net",
        type=str,
        required=False,
        choices=["LeNet", "ConvNet", "ResNet50"],
        default="LeNet",
        help="Underlying neural network architecture",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        choices=["MNIST", "FMNIST", "CIFAR10"],
        default="FMNIST",
        help="Dataset used",
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=False,
        choices=["RAND", "BAYES", "BOHB", "PBT", "PB2", "GPBT", "HEBO", "GPBTHEBO"],
        default="GPBTHEBO",
        help="Choice of search_algo and scheduler",
    )
    parser.add_argument(
        "--num_configs",
        type=int,
        required=False,
        default=25,
        help="Number of configuration explored",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        required=False,
        default=10,
        help="Number of iteration per exploration",
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        required=False,
        default=10,
        help="Number of repetition for the current experiment",
    )
    args = parser.parse_args()

    NUM_CONFIGURATION = args.num_configs
    ITERATIONS = args.num_iterations
    NUM_EXPERIMENTS = args.num_experiments

    config = {
        "b1": hp.uniform("b1", 1e-4, 1e-1),
        "b2": hp.uniform("b2", 1e-5, 1e-2),
        "iteration": [0],
        "droupout_prob": hp.uniform("droupout_prob", 0, 1),
        "lr": hp.uniform("lr", 1e-5, 1),
        "weight_decay": hp.uniform("weight_decay", 0, 1),
        "net": args.net,
        "dataset": args.dataset,
    }

    if args.algo == "PBT" or args.algo == "PB2":
        hp_bounds = config = {
            "b1": [1e-4, 1e-1],
            "b2": [1e-5, 1e-2],
            "iteration": [0],
            "droupout_prob": [0, 1],
            "lr": [1e-5, 1],
            "weight_decay": [0, 1],
            "net": [args.net],
            "dataset": [args.dataset],
        }
        config = {
            "b1": tune.uniform(1e-4, 1e-1),
            "b2": tune.uniform(1e-5, 1e-2),
            "iteration": [0],
            "droupout_prob": tune.uniform(0, 1),
            "lr": tune.uniform(1e-5, 1),
            "weight_decay": tune.uniform(0, 1),
            "net": args.net,
            "dataset": args.dataset,
        }
    elif args.algo == "HEBO" or args.algo == "GPBTHEBO":
        config = DesignSpace().parse(
            [
                {"name": "b1", "type": "num", "lb": 1e-4, "ub": 1e-1},
                {"name": "b2", "type": "num", "lb": 1e-5, "ub": 1e-2},
                {"name": "droupout_prob", "type": "num", "lb": 0, "ub": 1},
                {"name": "iteration", "type": "int", "lb": 0, "ub": 0},
                {"name": "lr", "type": "num", "lb": 1e-5, "ub": 1},
                {"name": "weight_decay", "type": "num", "lb": 0, "ub": 1},
                {"name": "net", "type": "cat", "categories": [args.net]},
                {"name": "dataset", "type": "cat", "categories": [args.dataset]},
            ]
        )

    if args.algo == "RAND":
        search_algo = partial(tpe.rand.suggest)
    elif args.algo == "BAYES":
        search_algo = partial(tpe.suggest, n_startup_jobs=1)
    elif args.algo == "BOHB":
        search_algo = TuneBOHB(metric="loss", mode="max")
        scheduler = HyperBandForBOHB(
            time_attr="training_iteration",
            metric="loss",
            mode="max",
            max_t=ITERATIONS,
        )
    elif args.algo == "PB2":
        search_algo = ConcurrencyLimiter(
            HyperOptSearch(metric="loss", mode="max"), max_concurrent=25
        )
        scheduler = PB2(
            time_attr="training_iteration",
            perturbation_interval=2,
            hyperparam_bounds=hp_bounds,
        )
    elif args.algo == "PBT":
        search_algo = ConcurrencyLimiter(
            HyperOptSearch(metric="loss", mode="max"), max_concurrent=4
        )
        scheduler = PopulationBasedTraining(
            perturbation_interval=1,
            time_attr="training_iteration",
            hyperparam_mutations=hp_bounds,
        )
    elif args.algo == "GPBT":
        config.pop("iteration")
        oracle = GPBTOracle(searchspace=config)
    elif args.algo == "GPBTHEBO":
        search_algo = HEBO
        oracle = GPBTHEBOracle(
            searchspace=config,
            search_algo=search_algo,
            verbose=False,
        )
    elif args.algo == "HEBO":
        oracle = HEBOOralce(searchspace=config)


    # Main experiment loop
    for i in range(NUM_EXPERIMENTS):
        start_time = datetime.utcnow()
        logger = Logger(
            config,
            search_algo=args.algo,
            dataset=args.dataset,
            net=args.net,
            iteration=i,
        )
        if args.algo == "RAND" or args.algo == "BAYES":
            oracle = SimpleOracle(config, search_algo)
            fmin_objective = partial(basic_loop, iterations=ITERATIONS, logger=logger)
            oracle.compute_Once(fmin_objective, NUM_CONFIGURATION)
        elif args.algo == "BOHB" or args.algo == "PBT" or args.algo == "PB2":
            ray.shutdown()
            ray.init()
            analysis = tune.run(
                general_model,
                config=config,
                scheduler=scheduler,
                search_alg=search_algo,
                num_samples=NUM_CONFIGURATION,
                mode="max",
                metric="loss",
                loggers=[Logger],
                checkpoint_at_end=True,
                stop={"training_iteration": ITERATIONS},
                resources_per_trial={"cpu": 8, "gpu": 1},
                verbose=2,
            )
        elif args.algo == "GPBT" or args.algo == "GPBTHEBO":
            scheduler = Scheduler(
                general_model, ITERATIONS, NUM_CONFIGURATION, oracle, logger
            )
            scheduler.initialisation(args.algo)
            scheduler.loop()
        elif args.algo == "HEBO":
            oracle.compute_batch(ITERATIONS, logger)

        print("totalt time: " + str(datetime.utcnow() - start_time))


def basic_loop(x, iterations, logger):
    model = general_model(x)
    for _ in range(iterations):  # Iterations
        model.train1()
        loss = model.test1()
        test = model.val1()
        temp = dict(x)
        temp.update({"loss": loss})
        temp.update({"test": test})
        temp.update({"iteration": model.i})
        logger.on_result(result=temp)
    return -loss


if __name__ == "__main__":
    main()
