# deep-robust-robotnav


## Environment Setup

### Install habitat-sim:
```
conda install habitat-sim=0.1.6 withbullet -c conda-forge -c aihabitat; 
```

### Install habitat lab
```
sudo apt update;
sudo apt install pkg-config libhdf5-dev`;
git clone https://github.com/facebookresearch/habitat-lab.git`;
cd habitat-api; git checkout b5f2b00a25627ecb52b43b13ea96b05998d9a121;`
pip install -r requirements.txt;
python setup.py develop --all;
```

### Install Neural Slam:

1. Add configs/task/pointnav.yaml from https://github.com/facebookresearch/habitat-lab/blob/main/configs/tasks/pointnav.yaml 
2. Replace register_move_fn import: 
```
import habitat_sim.registry as r
register_move_fn = r.register_move_fn
```
3. Modify GPU memory requirements in `arguments.py`. Alernatively add the `--no_cuda` argument when running.
```
                    assert torch.cuda.get_device_properties(i).total_memory \
                            /1024/1024/1024 > 5.0, "Insufficient GPU memory"

            num_processes_per_gpu = 1
            num_processes_on_first_gpu = 1
```
