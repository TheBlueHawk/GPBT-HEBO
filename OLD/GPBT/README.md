# GPBT

- `Experiments`: experiments environment and data processing. All experiments can be reproduced from jupyter notebook environments.
  - `Experiments/final_notebook.ipynb` and `Experiments/Boston_analytics.ipynb` for Boston
  - `Experiments/MNIST.ipynb` and `Experiments/MNIST_analytics.ipynb` for MNIST
  - `Experiments/FMNIST.ipynb` and `Experiments/FMNIST_analytics.ipynb` for FMNIST
  - `Experiments/CIFARanalytics.ipynb`for data processing of CIFAR

  - Raw data from such experiments are in `Experiments/data_brut` folder
  - Treated outputs are in `Experiments/data_result` folder. 

- `TO_GPU` is the folder for CIFAR that was sent to a GPU
- `TO_GPU1` is the folder for DCGAN experiments

- Run `TO_GPU/dependencies.sh` to run CIFAR-GPBT experiment.
  - There are files `TO_GPU/bohb1.py`, `TO_GPU/pbt.py` for other models as well. 
- `result_TO_GPU`: outputs of the GPU computations

