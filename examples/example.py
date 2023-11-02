import arcle
import gymnasium as gym
import time
import numpy as np


env = gym.make('ARCLE/RawARCEnv-v0',render_mode='ansi')

obs, info = env.reset()

for _ in range(1000):
    sel = np.zeros((30,30),dtype=np.uint8)
    
    sel[np.random.randint(0,obs['grid_dim'][0]),np.random.randint(0,obs['grid_dim'][1])] = 1
    op = env.action_space['operation'].sample()
    
    action = {'selection': sel, 'operation': op}
    
    obs,reward,term,trunc,info = env.step(action)

    if term or trunc:
        obs, info = env.reset()
    time.sleep(0.3)

env.close()
