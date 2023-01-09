# Improving Adaptive Hyperparameter Optimization with Genealogical History and HEBO
This repository contains the code requried to reproduce the results of the paper submitted as part of the Deep Learning class project at ETHZ. 
We combined current state-of-the-art adaptive hyperparameter optimisation algorithms GPBT and combined it with the HEBO search algorithm that circumvents usually unmet assumption of others search algorithms (e.g. stationarity). Our results shows how GPBT-HEBO is more robust and outperforms the current state-of-the-art hyperparameter optimization algorithms. 

## Basic Installation and Quickstart
```
cd HEBO
pip install -e .
cd ../GPBT
pip install -r requirements.txt
```
### Protobuf Error
You may need to downgrade the Protobuf package.
```
pip install protobuf~=3.20.0
```

## Running experiments
Experiments are run using the ``runner.py`` file and passing the desired arguments. Available arguments are:
- ``--dataset``: choice of datasets is limited to ``MNIST`` and ``FMNIST``
- ``--algo``: desired combination of search algorithm and scheduler. Can take the values ``RAND``, ``BAYES``, ``BOBH``, ``PBT``, ``PB2``, ``HEBO``, ``GPBT``, ``GPBTHEBO``
- ``num_configs``, ``num_iterations``, ``num_experiments`` control the lenght of each part of the experiment. 
  
Results will be saved in a temporary directory.

## Result analysis and visualisation
Pleaes follow the instructions of ``analyitics.ipynb``

## Credits
We thanks the authors of GPBT for letting us reuse part of their code (``https://github.com/AJSVB/GPBT``). 
