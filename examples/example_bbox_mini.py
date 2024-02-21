import arcle
import gymnasium as gym
import time
import numpy as np
from arcle.loaders import MiniARCLoader
from arcle.wrappers import BBoxWrapper

env = gym.make('ARCLE/RawARCEnv-v0',render_mode='ansi',data_loader = MiniARCLoader(),max_grid_size=(5,5))
env = BBoxWrapper(env)
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()

    obs,reward,term,trunc,info = env.step(action)

    if term or trunc:
        obs, info = env.reset()
    time.sleep(0.3)

env.close()
