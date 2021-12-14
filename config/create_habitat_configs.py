import datetime
import os
from shutil import copyfile

import yaml

from config.config import CURRENT_POLICY, BATCHSIZE, MAP_DIMENSIONS

experiment_id_custom_details = dict(
    Baseline=dict(
        sensors=['RGB_SENSOR'],
        ppo_hidden_size=512,
    ),
    BaselineMidLevel=dict(
        sensors=['RGB_SENSOR', 'MIDLEVEL'],
        ppo_hidden_size=128,
    ),
    DRRN=dict(
        sensors=['RGB_SENSOR', 'MIDLEVEL', 'EGOMOTION'],
        ppo_hidden_size=MAP_DIMENSIONS[0] * MAP_DIMENSIONS[1] * MAP_DIMENSIONS[2],
    ),
    DRRNActualMap=dict(
        sensors=['MAP_SENSOR'],
        ppo_hidden_size=128,
    ),
    DRRNSupervisedMap=dict(
        sensors=['RGB_SENSOR', 'EGOMOTION'],
        ppo_hidden_size=128,
    ),
)


def create_habitat_config_for_experiment(experiment_id, results_base_dir):
    sensors = experiment_id_custom_details[experiment_id]['sensors']
    ckpt_folder = f"{results_base_dir}/checkpoints"

    return dict(
        BASE_TASK_CONFIG_PATH="config/habitat_pointnav_config.yaml",
        TRAINER_NAME="ppo",
        ENV_NAME="NavRLEnv",
        SIMULATOR_GPU_ID=0,
        TORCH_GPU_ID=0,
        VIDEO_OPTION=["disk", "tensorboard"],
        TENSORBOARD_DIR=f"{results_base_dir}/tb",
        VIDEO_DIR=f"{results_base_dir}/video_dir",
        # To evaluate on all episodes, set this to -1
        TEST_EPISODE_COUNT=-1,
        NUM_PROCESSES=BATCHSIZE,
        SENSORS=sensors,
        EVAL_CKPT_PATH_DIR=ckpt_folder,
        CHECKPOINT_FOLDER=ckpt_folder,
        NUM_UPDATES=5000,
        LOG_INTERVAL=10,
        LOG_FILE=f"{results_base_dir}/train.log",
        CHECKPOINT_INTERVAL=200,
        RL=dict(
            PPO=dict(
                # ppo params
                clip_param=0.1,
                ppo_epoch=4,
                num_mini_batch=BATCHSIZE,
                value_loss_coef=0.5,
                entropy_coef=0.01,
                lr=2.5e-4,
                eps=1e-5,
                max_grad_norm=0.5,
                num_steps=128,
                hidden_size=experiment_id_custom_details[experiment_id]['ppo_hidden_size'],
                use_gae=True,
                gamma=0.99,
                tau=0.95,
                use_linear_clip_decay=True,
                use_linear_lr_decay=True,
                reward_window_size=50,
            ),
        ),
    )


def create_habitat_pointnav_config_for_experiment(experiment_id):
    sensors = experiment_id_custom_details[experiment_id]['sensors']

    sensor_details = dict(
        RGB_SENSOR=dict(
            WIDTH=256,
            HEIGHT=256,
        ),
        MAP_SENSOR=dict(
            WIDTH=256,
            HEIGHT=256,
            HFOV=90,
            TYPE='MAP_SENSOR',
        ),
        EGOMOTION=dict(
            TYPE='EGOMOTION',
            HFOV=90,
        ),
        MIDLEVEL=dict(
            TYPE='MIDLEVEL',
            WIDTH=16,
            HEIGHT=16,
            HFOV=90,
        )
    )

    simulator_dict = dict(
        AGENT_0=dict(
            SENSORS=sensors
        ),
        HABITAT_SIM_V0=dict(
            GPU_DEVICE_ID=0
        ),
    )

    for sensor in sensors:
        simulator_dict[sensor] = sensor_details[sensor]

    return dict(
        ENVIRONMENT=dict(
            MAX_EPISODE_STEPS=500,
        ),
        SIMULATOR=simulator_dict,
        TASK=dict(
            TYPE='Nav-v0',
            SUCCESS_DISTANCE=0.2,
            SENSORS=['POINTGOAL_WITH_GPS_COMPASS_SENSOR'],
            POINTGOAL_WITH_GPS_COMPASS_SENSOR=dict(
                GOAL_FORMAT="POLAR",
                DIMENSIONALITY=2,
            ),
            GOAL_SENSOR_UUID="pointgoal_with_gps_compass",
            MEASUREMENTS=['DISTANCE_TO_GOAL', 'SUCCESS', 'SPL'],
            SUCCESS=dict(
                SUCCESS_DISTANCE=0.2
            ),
            POSSIBLE_ACTIONS=["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"],  # TODO possibly add stop action here
        ),
        DATASET=dict(
            TYPE='PointNav-v1',
            SPLIT='train',
            DATA_PATH='data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz',
        )
    )


def create_habitat_configs():
    experiment_id = CURRENT_POLICY
    cur_dt = datetime.datetime.now().strftime('%m_%d_%H_%M')
    results_base_dir = f"results/results_{experiment_id}_{cur_dt}"
    print(f'Creating configs for {experiment_id}')
    with open(f'config/habitat_config.yaml', 'w') as yaml_file:
        yaml.dump(create_habitat_config_for_experiment(experiment_id, results_base_dir), yaml_file)
    with open(f'config/habitat_pointnav_config.yaml', 'w') as yaml_file:
        yaml.dump(create_habitat_pointnav_config_for_experiment(experiment_id), yaml_file)
    os.makedirs(results_base_dir, exist_ok=True)
    copyfile('config/habitat_config.yaml', results_base_dir + '/habitat_config.yaml')
    copyfile('config/habitat_pointnav_config.yaml', results_base_dir + '/habitat_pointnav_config.yaml')
