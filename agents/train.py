from typing import List, SupportsInt, Tuple
import ray
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.tune.registry import register_env
from arcle.envs import O2ARCv2Env
from arcle.loaders import ARCLoader, Loader

from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from emaml import EMAML, EMAMLConfig
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

class O2ARCBBoxEnv(O2ARCv2Env, TaskSettableEnv):
    
    def __init__(self, data_loader: Loader = ARCLoader(), max_grid_size: Tuple[SupportsInt, SupportsInt] = (30,30), colors: SupportsInt = 10, max_trial: SupportsInt = -1, render_mode: str = None, render_size: Tuple[SupportsInt, SupportsInt] = None) -> None:
        super().__init__(data_loader, max_grid_size, colors, max_trial, render_mode, render_size)

        self.reset_options = {
            'adaptation': True, # Default is true (adaptation first!). To change this mode, call 'post_adaptation()'
            'prob_index': None
        }
        self.action_space= spaces.Tuple(
            (
                spaces.Discrete(self.H),
                spaces.Discrete(self.W),
                spaces.Discrete(self.H),
                spaces.Discrete(self.W),
                spaces.Discrete(len(self.operations)),
            )
        )
        grid_pixels = self.H*self.W
        boxsize=0
        boxsize += ( grid_pixels + 2 )       # input w/ dim
        boxsize += ( grid_pixels + 2 )       # grid w/ dim
        boxsize += grid_pixels              # selected
        boxsize += ( grid_pixels + 2 )       # clip w/ dim
        boxsize += 1                        # active
        boxsize += ( grid_pixels + 2 )      # object w/ dim 
        boxsize += grid_pixels              # object_sel 
        boxsize += 2                        # object_pos
        boxsize += grid_pixels              # background
        boxsize += 1                        # rotation_parity 
        self.boxsize = boxsize
        self.observation_space = spaces.Box(-100, max(self.H, self.W), (boxsize,) )

    def convert_obs(self,obs):
        o2s = obs["object_states"]
        for_concat = [
            obs["input"].reshape(-1),
            obs["input_dim"],
            obs["grid"].reshape(-1),
            obs["grid_dim"],
            obs["selected"].reshape(-1),
            obs["clip"].reshape(-1),
            obs["clip_dim"],
            [o2s["active"]],
            o2s["object"].reshape(-1),
            o2s["object_dim"],
            o2s["object_sel"].reshape(-1),
            o2s["object_pos"],
            o2s["background"].reshape(-1),
            [o2s["rotation_parity"]],
        ]
        return np.concatenate(for_concat, dtype=float,casting='unsafe')

    def step(self, action):
        x1, y1, x2, y2, op = action
    
        selection = np.zeros((self.unwrapped.H, self.unwrapped.W), dtype=np.uint8)
        x1, x2 = min(x1,x2), max(x1,x2)
        y1, y2 = min(y1,y2), max(y1,y2)
        selection[x1:x2+1, y1:y2+1] = 1
        obs, rew, term, trunc,info = super().step({'selection': selection, 'operation': op})
        return self.convert_obs(obs), rew, term, trunc, info
        
        
    
    # For Meta-RL Setting (problem settable env)
    def reset(self, seed = None, options= None):
        obs, info = super().reset(seed, self.reset_options)
        return self.convert_obs(obs), info

    #TaskSettableEnv API
    def sample_tasks(self, n_tasks: int) -> List[TaskType]:
        return np.random.choice(len(self.loader.data),n_tasks,replace=False)

    def get_task(self) -> TaskType:
        return super().get_task()
    
    def set_task(self, task: TaskType) -> None:
        self.reset_options = {
            'adaptation': True, # Default is true (adaptation first!). To change this mode, call 'post_adaptation()'
            'prob_index': task
        }
    
    def post_adaptation(self):
        self.reset_options['adaptation'] = False

ray.init()
def env_creator(config):
    return TimeLimit(O2ARCBBoxEnv(max_trial=100, ),100)
register_env("O2ARCBBoxEnv", env_creator)

config = EMAMLConfig().environment(env="O2ARCBBoxEnv").training()
algo = config.build()

print(algo.train())
