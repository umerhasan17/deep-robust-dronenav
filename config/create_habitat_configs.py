import os

import yaml

import datetime

from config.config import CURRENT_POLICY

experiment_id_sensors = dict(
    Baseline=['RGB_SENSOR'],
    BaselineMidLevel=['RGB_SENSOR'],
    DRDN=['RGB_SENSOR', 'EGOMOTION'],
    DRDNActualMap=['RGB_SENSOR', 'MAP_SENSOR'],
    DRDNSupervisedMap=['RGB_SENSOR', 'EGOMOTION']
)


def create_habitat_config_for_experiment(experiment_id):
    cur_dt = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    ckpt_folder = f"models/checkpoints_{experiment_id}_{cur_dt}"
    sensors = experiment_id_sensors[experiment_id]

    return dict(
        BASE_TASK_CONFIG_PATH="config/habitat_pointnav_config.yaml",
        TRAINER_NAME="ppo",
        ENV_NAME="NavRLEnv",
        SIMULATOR_GPU_ID=0,
        TORCH_GPU_ID=0,
        VIDEO_OPTION=["disk", "tensorboard"],
        TENSORBOARD_DIR=f"results/tb_{experiment_id}_{cur_dt}",
        VIDEO_DIR="video_dir",
        # To evaluate on all episodes, set this to -1 # TODO create test validation datasets
        TEST_EPISODE_COUNT=2,
        EVAL_CKPT_PATH_DIR=ckpt_folder,
        NUM_PROCESSES=1,
        SENSORS=sensors,
        CHECKPOINT_FOLDER=ckpt_folder,
        NUM_UPDATES=20,  # TODO change this
        LOG_INTERVAL=1,
        CHECKPOINT_INTERVAL=10,
        RL=dict(
            PPO=dict(
                # ppo params
                clip_param=0.1,
                ppo_epoch=4,
                num_mini_batch=1,
                value_loss_coef=0.5,
                entropy_coef=0.01,
                lr=2.5e-4,
                eps=1e-5,
                max_grad_norm=0.5,
                num_steps=128,
                hidden_size=512,
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
    sensors = experiment_id_sensors[experiment_id]

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
            POSSIBLE_ACTIONS=["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"],
        ),
        DATASET=dict(
            TYPE='PointNav-v1',
            SPLIT='train',
            DATA_PATH='data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz',
        )
    )


def create_habitat_configs():
    experiment_id = CURRENT_POLICY
    print(f'Creating configs for {experiment_id}')
    with open(f'config/habitat_config.yaml', 'w') as yaml_file:
        yaml.dump(create_habitat_config_for_experiment(experiment_id), yaml_file)
    with open(f'config/habitat_pointnav_config.yaml', 'w') as yaml_file:
        yaml.dump(create_habitat_pointnav_config_for_experiment(experiment_id), yaml_file)
