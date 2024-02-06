import arcle
import gymnasium as gym
import time
import numpy as np
from arcle.loaders import MiniARCLoader

env = gym.make('ARCLE/RawARCEnv-v0',render_mode='ansi',data_loader = MiniARCLoader(),max_grid_size=(5,5))

obs, info = env.reset()

for _ in range(1000):
    sel = np.zeros((5,5),dtype=np.int8)
    
    sel[np.random.randint(0,5),np.random.randint(0,5)] = 1
    op = env.action_space['operation'].sample()
    
    action = {'selection': sel, 'operation': op}
    
    obs,reward,term,trunc,info = env.step(action)

    if term or trunc:
        obs, info = env.reset()
    time.sleep(0.1)

env.close()
