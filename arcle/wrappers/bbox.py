from typing import Any, Callable, List, SupportsInt, Tuple
import gymnasium as gym
from gymnasium.core import Env
import gymnasium.spaces as spaces
from arcle.envs import AbstractARCEnv
from arcle.envs import O2ARCv2Env
import numpy as np

class BBoxWrapper(gym.ActionWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.action_space = spaces.Tuple(
            (
                spaces.Discrete(self.H),
                spaces.Discrete(self.W),
                spaces.Discrete(self.H),
                spaces.Discrete(self.W),
                spaces.Discrete(len(self.operations)),
            )
        )
    
    def action(self,action: tuple):
        # 5-tuple: (x1, y1, x2, y2, op)
        x1, y1, x2, y2, op = action
        
        selection = np.zeros((self.H,self.W), dtype=np.int8)
        x1, x2 = min(x1,x2), max(x1,x2)
        y1, y2 = min(y1,y2), max(y1,y2)
        selection[x1:x2+1, y1:y2+1] = 1
        return {'selection': selection, 'operation': op}
    
class PointWrapper(gym.ActionWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.action_space = spaces.Tuple(
            (
                spaces.Discrete(self.H),
                spaces.Discrete(self.W),
                spaces.Discrete(len(self.operations)),
            )
        )
    
    def action(self,action: tuple):
        # 3-tuple: (x, y, op)
        x, y, op = action
        
        selection = np.zeros((self.H,self.W), dtype=np.int8)
        selection[x, y] = 1
        return {'selection': selection, 'operation': op}
    