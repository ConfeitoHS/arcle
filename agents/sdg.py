import gymnasium as gym
import numpy as np
from gymnasium import spaces
sp = spaces.Box(np.array([0,0]), np.array([5,6]), shape=(2,),dtype=np.int8)
for _ in range(10):
    print(sp.sample())
print(spaces.flatdim(sp))