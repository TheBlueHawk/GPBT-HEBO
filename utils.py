import csv
import os
import pandas as pd
from datetime import datetime
from ray import tune

DEFAULT_PATH = "./tmp/data"
os.makedirs(DEFAULT_PATH, exist_ok=True)


def flatten_dict(d: dict, delimiter="/") -> dict:
    """
    >>> d = {'a': 1,
    ...     'c': {'a': 2, 'b': {'x': 5, 'y': 10}},
    ...     'd': [1, 2, 3]}

    >>> flatten_dict(d)
    {'a': 1, 'd': [1, 2, 3], 'c_a': 2, 'c_b_x': 5, 'c_b_y': 10}
    """
    df = pd.json_normalize(d, sep=delimiter)
    return df.to_dict(orient="records")[0]


class Logger(tune.logger.Logger):
    def __init__(
        self, config, search_algo="GPBTHEBO", dataset="FMNIST", net="LeNet", iteration=0
    ):
        self.config = config
        timestamp = datetime.utcnow().strftime("%H_%M_%d_%m_%Y")
        directory = os.path.join(DEFAULT_PATH, dataset, timestamp)
        os.makedirs(directory, exist_ok=True)
        filename = search_algo + "_" + net + "_" + str(iteration) + ".csv"
        progress_file = os.path.join(directory, filename)
        self.logdir = progress_file
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
            {k: v for k, v in result.items() if k in self._csv_out.fieldnames}
        )
        self._file.flush()
