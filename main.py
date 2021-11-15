""" 
main.py
----------------------------------------------------------------------------
     Authors : Yongtao Wu, Umer Hasan, Titouan Renard
     Last Update : Octobre 2021

     Main routine for our deep cognitive drone mapping agent,
     this file implements the architecture of our networks

----------------------------------------------------------------------------
Processing graph:
     


                     image
                       |
                       | variable name: img -- (BATCHSIZE x 3 x 256 x 256) tensor
                       v
          ------------done-----------
          |mid_level_representations| directly using the visualpriors pip install
          ---------------------------
                       |
                       | variable name: mid_level -- (BATCHSIZE x 3 x 256 x 256) tensor
                       v
          -----------done--------------
          |                           | 
          |  fc ->  UpSampleResNet    |  content : two functions, fully-connected layer "fc" and decoder resnet "decoder"
          |                           | 
          |                           |  (BATCHSIZE x 2048*REPRESENTATION_NUMBER) -(fc)->  (BATCHSIZE x 2*REPRESENTATION_NUMBER x 16 x 16) 
          -----------------------------  
                       |
                       | variable name: map_update  -- (BATCHSIZE x 2 x M x M) tensor // encodes confidence and free space channels
                       |
                       |                                  egomotion
                       |                                  | variable name : dx -- (3x1) numpy array, (x,y,theta)
                       v                                  v
                  -----todo----                   -------done------
 TODO: Implement  |  combine  |   <-------------  |   transform   | <------------- previous_map -- (BATCHSIZE x 2 x M x M) tensor
                  -------------                   -----------------
                       |
                       | variable name: map 
                       v
                ------todo-----
 TODO: Implement|   policy    |
                ---------------
                       |
                       |
                       v
                  velocities

  """

import pdb
import subprocess
from collections import deque

from networks.encoder_mid_level import mid_level_representations  # mid_level wrapper class
from networks.decoder_residual import UpResNet  # upsampling resnet
from networks.transform import egomotion_transform  # upsampling resnet
from networks.update import update_map  # upsampling resnet
from networks.fc import FC  # fully connected fc layer
from utils.storage import GlobalRolloutStorage
import torch
import gym
import time
from ppo import PPO

import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from model import RL_Policy

from config import REPRESENTATION_NAMES, BATCHSIZE, device, RESIDUAL_LAYERS_PER_BLOCK, RESIDUAL_NEURON_CHANNEL, STRIDES, \
    RESIDUAL_SIZE, IMG_DIMENSIONS


def forward(image, egomotion, prev_map, verbose=False):
    # ==========download image to debug==========

    img = TF.to_tensor(TF.resize(image, 256)) * 2 - 1
    img = img.unsqueeze(0)  # (1,3,256,256)
    activation = img.repeat(BATCHSIZE, 1, 1, 1)  # (BATCHSIZE x 3 x 256 x 256) tensor

    # ==========Mid level encoder==========
    print("Passing mid level encoder...")
    activation = mid_level_representations(activation,
                                           REPRESENTATION_NAMES)  #  (BATCHSIZE x REPRESENTATION_NUMBER*2 x 16 x 16) tensor

    # ==========FC==========
    print("Passing fully connected layer...")
    fc = FC()
    activation = activation.view(BATCHSIZE, -1)  # flatten all dimensions except batch,
    # --> tensor of the form (BATCHSIZE x 2048*REPRESENTATION_NUMBER)
    activation = fc(activation)  # pass through dense layer --> (BATCHSIZE x 2048*REPRESENTATION_NUMBER) tensor
    activation = activation.view(BATCHSIZE, 8 * len(REPRESENTATION_NAMES), 16,
                                 16)  # after fully connected layer, # (BATCHSIZE x REPRESENTATION_NUMBER*2 x 16 x 16) tensor

    # ==========Deconv==========
    print("Passing residual decoder...")
    decoder = UpResNet(layers=RESIDUAL_LAYERS_PER_BLOCK, channels=RESIDUAL_NEURON_CHANNEL, sizes=RESIDUAL_SIZE,
                       strides=STRIDES).to(device)
    map_update = decoder(activation)  # upsample to map object

    # ==========Transform and Update==========
    print("Passing transform and update steps...")
    prev_map = egomotion_transform(prev_map, egomotion)
    new_map = update_map(map_update, prev_map)
    print("Done!")
    return new_map


def get_map():
    NotImplemented()


def main():
    from env import make_vec_envs
    from utils.arguments import get_args

    torch.set_num_threads(1)
    args = get_args()
    args.device = device
    args.global_dowscaling = 3.75
    envs = make_vec_envs(args)
    obs, infos = envs.reset()

    # # Calculating full and local map sizes
    # map_size = args.map_size_cm // args.map_resolution
    # full_w, full_h = map_size, map_size
    # local_w, local_h = int(full_w / 3.75), int(full_h / 3.75) # TODO global downscaling 3.75

    # Global policy observation space
    g_observation_space = gym.spaces.Box(0, 1, (3, args.frame_width, args.frame_height), dtype='uint8')

    # Global policy action space
    g_action_space = gym.spaces.Discrete(3) #?
    # gym.spaces.Box(low=0.0, high=255.0,hape=(2,), dtype=np.float32)

    g_hidden_size = args.global_hidden_size

    g_policy = RL_Policy(
        obs_shape=g_observation_space.shape,
        action_space=g_action_space,
        base_kwargs={'recurrent': args.use_recurrent_global,
                     'hidden_size': g_hidden_size,
                     'downscaling': args.global_downscaling
                     }
    ).to(device)

    g_agent = PPO(
        actor_critic=g_policy,
        clip_param=args.clip_param,
        ppo_epoch=args.ppo_epoch,
        num_mini_batch=args.num_mini_batch,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        lr=args.global_lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
    )

    # Storage
    g_rollouts = GlobalRolloutStorage( # TODO these arguments look wrong
        num_steps=args.max_episode_length,
        num_processes=1,
        obs_shape=g_observation_space.shape,
        action_space=g_action_space,
        rec_state_size=g_policy.rec_state_size,
        extras_size=1
    ).to(device)

    # input of ppo
    global_input = obs
    g_rollouts.obs[0].copy_(global_input)


    print('Start PPO training')
    start = time.time()


    for j in range(args.num_episodes):
        episode_rewards = deque(maxlen=args.max_episode_length)

        for step in range(args.max_episode_length):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = g_policy.act(
                    g_rollouts.obs[step],
                    g_rollouts.rec_states[step],
                    g_rollouts.masks[step],
                    extras=g_rollouts.extras[step],
                    deterministic=False,
                )

            obs, reward, done, infos = envs.step(action)
            print(f'Episode {j}, Step {step}, Action {action}, Reward {reward}')
            episode_rewards.append(reward)

            # If done then clean the history of observations.

            masks = torch.FloatTensor([0.0] if done else [1.0])
            bad_masks = torch.FloatTensor([0.0] if 'bad_transition' in infos.keys() else [1.0])
            try:
                g_rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)
            except Exception as e:
                print('Failed')
        with torch.no_grad():
            next_value = g_policy.get_value(
                inputs=g_rollouts.obs[-1],
                rnn_hxs=g_rollouts.rec_states[-1],
                masks=g_rollouts.masks[-1],
                extras=g_rollouts.extras[-1]
            ).detach()

        # Note: changed gae_lambda to tau and use_proper_time_limits to False
        g_rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = g_agent.update(g_rollouts)

        g_rollouts.after_update()

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.max_episode_length
            end = time.time()
            message = "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n". \
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       len(episode_rewards), np.mean(episode_rewards),
                       np.median(episode_rewards), np.min(episode_rewards),
                       np.max(episode_rewards), dist_entropy, value_loss,
                       action_loss)
            print(message)


if __name__ == '__main__':
    main()
    # print("download image to debug...")
    # subprocess.call("curl -O https://raw.githubusercontent.com/StanfordVL/taskonomy/master/taskbank/assets/test.png", shell=True)
    # image = Image.open('test.png') #example image valuec
    # # prev_map_raw = Image.open('Bedroom.jpg') #example prevmap value
    # prev_map = TF.to_tensor(TF.resize(image, 256))[0:2] * 2 - 1
    # prev_map = prev_map.unsqueeze(0)                                                 # (1,3,256,256)
    # prev_map = prev_map.repeat(BATCHSIZE, 1, 1, 1)                            # (BATCHSIZE x 3 x 256 x 256) tensor
    #
    #
    # egomotion = np.array([.1,0.,1.4]) #example egomotion value
    # new_map=forward(image,egomotion,prev_map,verbose=True)

    # Starting environments
