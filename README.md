# deep-robust-robotnav


## Environment Setup

### Install required packages:
```
conda env create -f config/vis_env.yaml
```

### Setup habitat lab folder

Build habitat lab from source using the specified commit hash.

```
cd habitat-api
python setup.py develop --all
cd ..
```

### Run Train / Eval:

Remember to activate conda environment before running command.

```
python run.py --exp-config config/habitat_config.yaml --run-type train
```