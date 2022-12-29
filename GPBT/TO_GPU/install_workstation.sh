#!/bin/bash

# update repositories and upgrade packages
sudo apt update
sudo apt upgrade -y

export DEBIAN_FRONTEND=noninteractive

# install python
sudo apt install -y python3 python3-dev python3-distutils
wget https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py
sudo python3.6 /tmp/get-pip.py

# install tools
sudo apt install -y git nano screen wget zip unzip g++ htop software-properties-common pkg-config zlib1g-dev gdb cmake cmake-curses-gui autoconf gcc gcc-multilib g++-multilib libomp-dev


# download and install CUDA
VERSION="10.2"
SUB_VERSION="440"
SUB_SUB_VERSION="1"
CUDA_TAR_FILE="cuda-${VERSION}.${SUB_VERSION}-${SUB_SUB_VERSION}.deb"
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://lia.epfl.ch/dependencies/${CUDA_TAR_FILE} -O /tmp/${CUDA_TAR_FILE}
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
# sudo rm /etc/apt/sources.list.d/cuda*
# sudo apt remove nvidia-cuda-toolkit
# sudo apt remove nvidia-*
# sudo apt update
# sudo add-apt-repository ppa:graphics-drivers/ppa
# sudo apt-key adv --fetch-keys  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
# sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
# sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'
# sudo apt update
sudo dpkg -i /tmp/${CUDA_TAR_FILE}
sudo apt update
sudo apt install cuda -y

# download and install libcudnn
CUDNN_VERSION="7.6"
CUDNN_TAR_FILE="cudnn-${VERSION}-${CUDNN_VERSION}.tgz"
wget https://lia.epfl.ch/dependencies/${CUDNN_TAR_FILE} -O /tmp/${CUDNN_TAR_FILE}
tar -xzvf /tmp/${CUDNN_TAR_FILE}  -C /tmp/
sudo mkdir -p /usr/local/cuda-${VERSION}/lib64

sudo cp -P /tmp/cuda/include/cudnn.h /usr/local/cuda-${VERSION}/include
sudo cp -P /tmp/cuda/lib64/libcudnn* /usr/local/cuda-${VERSION}/lib64/
sudo chmod a+r /usr/local/cuda-${VERSION}/lib64/libcudnn*

# install python packages for machine learning
/usr/bin/yes | pip3.6 install --upgrade pip
/usr/bin/yes | pip3.6 install cython cmake mkl mkl-include dill pyyaml setuptools cffi typing mako pillow matplotlib mpmath klepto
/usr/bin/yes | pip3.6 install jupyter sklearn tensorflow keras spacy spacy_cld colored jupyterlab configparser gensim pymysql benepar tqdm wandb optuna bottleneck 
/usr/bin/yes | pip3.6 install selenium networkx bs4 fuzzywuzzy python-levenshtein pyldavis newspaper3k  wikipedia nltk py-rouge beautifultable tensor2tensor tensorboardX benepar adabelief-pytorch
/usr/bin/yes | pip3.6 install --ignore-installed PyYAML
/usr/bin/yes | pip3.6 install numpy==1.17.3
/usr/bin/yes | pip3.6 install pandas==1.0.5

sudo python3.6 -m spacy download en_core_web_lg
sudo python3.6 -c "import nltk; nltk.download('punkt')"
sudo python3.6 -c "import nltk; nltk.download('stopwords')"
sudo python3.6 -c "import benepar; benepar.download('benepar_en2')"
sudo python3.6 -c "import benepar; benepar.download('benepar_en2_large')"

# pytorch
#git clone --recursive https://github.com/pytorch/pytorch /tmp/pytorch
#cd /tmp/pytorch
#git checkout tags/v1.3.1 # 'hope this is 1.3.0a0+ee77ccb'
#git submodule sync 
#git submodule update --init --recursive
#sudo python3.6 setup.py install
/usr/bin/yes | pip3.6 install torch==1.6.0
#/usr/bin/yes | pip3.6 install torchvision
