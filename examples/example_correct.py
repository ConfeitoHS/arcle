import arcle
import gymnasium as gym
import time
import numpy as np


env = gym.make('ARCLE/RawARCEnv-v0',render_mode='ansi')

sels = [np.zeros((30,30)) for _ in range(4)]
for i,(x,y) in enumerate([(3,2),(7,7)]):
    sels[i][x+1,y+1] = sels[i][x+1,y-1] = sels[i][x-1,y+1] = sels[i][x-1,y-1] = 1

sels[2][2+1,6] = sels[2][2,6-1] = sels[2][2-1,6] = sels[2][2,6+1] = 1
cols = [4,4,7,11]


option = { 'prob_index': 14, 'adaptation': False }
obs, info = env.reset(options=option)
ts = 0
for _ in range(1000):
    
    action = {'selection': sels[ts], 'operation': cols[ts]}
    
    obs,reward,term,trunc,info = env.step(action)
    ts +=1
    time.sleep(0.3)
    if term or trunc:
        obs, info = env.reset(options=option)
        ts = 0

env.close()
