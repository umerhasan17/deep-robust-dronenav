# deep-robust-robotnav


## Environment Setup

### Install required packages:
```
conda env create -f config/vis_env.yaml python=3.6.13
conda activate vis5
```

### Install habitat sim
```
conda install -c aihabitat -c conda-forge bullet=2.88 habitat-sim=0.1.6 headless withbullet
```

### Setup habitat lab folder

Build habitat lab from source using the specified commit hash (the folder is already in the repository, no need to git clone or download).

```
cd habitat-api
python setup.py develop --all
cd ..
```

### You may need to install the following libraries on Ubuntu if not already installed 

```
sudo apt install freeglut3 freeglut3-dev
```


### Get Gibson Dataset

```
wget https://dl.fbaipublicfiles.com/...
wget https://dl.fbaipublicfiles.com/...
```

### Run Train / Eval:

Remember to activate conda environment before running command.

```
python run.py --exp-config config/habitat_config.yaml --run-type train
```
