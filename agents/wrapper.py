import gymnasium as gym
from gymnasium.core import Env
import numpy as np

def action(self,action):
    # 5-tuple: (x1, y1, x2, y2, op)
    x1, y1, x2, y2, op = action
    
    selection = np.zeros(self.env.unwrapped.max_grid_size, dtype=np.uint8)
    x1, x2 = min(x1,x2), max(x1,x2)
    y1, y2 = min(y1,y2), max(y1,y2)
    selection[x1:x2+1, y1:y2+1] = 1
    return {'selection': selection, 'operation': op}
