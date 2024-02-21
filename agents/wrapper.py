from typing import Any, Callable, List, SupportsInt, Tuple
import gymnasium as gym
from gymnasium.core import Env
import gymnasium.spaces as spaces
from arcle.envs import AbstractARCEnv
from arcle.envs import O2ARCv2Env
import numpy as np

from arcle.loaders import ARCLoader, Loader

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
        
        selection = np.zeros(self.max_grid_size, dtype=np.int8)
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
        
        selection = np.zeros(self.env.unwrapped.max_grid_size, dtype=np.int8)
        selection[x, y] = 1
        return {'selection': selection, 'operation': op}
    
class O2ARCNoFillEnv(O2ARCv2Env):

    def create_operations(self) -> List[Callable[..., Any]]:
        ops= super().create_operations()
        return ops[0:10] + ops[20:] 

from gymnasium.envs.registration import register

register(
    id='ARCLE/O2ARCNoFillEnv',
    entry_point=__name__+':O2ARCNoFillEnv',
    max_episode_steps=300,
)
env = gym.make('ARCLE/O2ARCNoFillEnv')
env = PointWrapper(env)
obs, info = env.reset()
print(env.action_space)