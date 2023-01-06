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