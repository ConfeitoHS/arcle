
from typing import SupportsInt, Tuple
import ray
from ray import rllib
from ray.rllib.algorithms import ppo
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig


from arcle.envs import O2ARCv2Env
from arcle.loaders import ARCLoader, Loader

from gymnasium import spaces
from gymnasium.wrappers import FilterObservation

import numpy as np

class O2ARCBBoxEnv(O2ARCv2Env):
    def __init__(self, data_loader: Loader = ARCLoader(), max_grid_size: Tuple[SupportsInt, SupportsInt] = (30,30), colors: SupportsInt = 10, max_trial: SupportsInt = -1, render_mode: str = None, render_size: Tuple[SupportsInt, SupportsInt] = None) -> None:
        super().__init__(data_loader, max_grid_size, colors, max_trial, render_mode, render_size)
    def create_action_space(self, action_count):
        return spaces.Tuple(
            (
                spaces.Discrete(self.H),
                spaces.Discrete(self.W),
                spaces.Discrete(self.H),
                spaces.Discrete(self.W),
                spaces.Discrete(action_count),
            )
        )
    
    def step(self, action):
        x1, y1, x2, y2, op = action
    
        selection = np.zeros((self.unwrapped.H, self.unwrapped.W), dtype=np.uint8)
        x1, x2 = min(x1,x2), max(x1,x2)
        y1, y2 = min(y1,y2), max(y1,y2)
        selection[x1:x2+1, y1:y2+1] = 1
        return super().step({'selection': selection, 'operation': op})
    

ray.init()

def env_creator(config):
    return FilterObservation(O2ARCBBoxEnv(max_trial=3),["input", "input_dim", "grid", "grid_dim", "clip", "clip_dim"])
register_env('O2ARCBBoxEnv', env_creator)

algo = PPOConfig().experimental(_disable_preprocessor_api = True).environment('O2ARCBBoxEnv').build()

# TODO: Make Custom Policy and Run PPO
# TODO: Create Task Sampler
# TODO: Extend PPO to MAML+PPO
# TODO: E-MAML 

while True:
    algo.train()