import habitat

import numpy as np
import random

import matplotlib.pyplot as plt

from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb

config = habitat.get_config(config_paths='habitat_conf.yaml')

env = habitat.Env(config=config)
env.episodes = random.sample(env.episodes, 2)
max_steps = 4

action_mapping = {
    0: 'stop',
    1: 'move_forward',
    2: 'turn left',
    3: 'turn right'
}


def display_sample(rgb_obs, semantic_obs, depth_obs):
    rgb_img = Image.fromarray(rgb_obs, mode="RGB")

    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGBA")

    depth_img = Image.fromarray((depth_obs * 255).astype(np.uint8), mode="L")

    arr = [rgb_img, semantic_img, depth_img]

    titles = ['rgb', 'semantic', 'depth']
    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis('off')
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show()


if __name__ == '__main__':
    for i in range(len(env.episodes)):
        observations = env.reset()

        display_sample(observations['rgb'], observations['semantic'], np.squeeze(observations['depth']))

        count_steps = 0
        while count_steps < max_steps:
            action = random.choice(list(action_mapping.keys()))
            print(action_mapping[action])
            observations = env.step(action)
            display_sample(observations['rgb'], observations['semantic'], np.squeeze(observations['depth']))

            count_steps += 1
            if env.episode_over:
                break

    env.close()