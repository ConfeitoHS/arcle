
from typing import List, SupportsInt, Tuple
import ray
from ray import rllib
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType

from ray.tune.registry import register_env



from arcle.envs import O2ARCv2Env
from arcle.loaders import ARCLoader, Loader

from gymnasium import spaces
from gymnasium.wrappers import FilterObservation

import numpy as np

class O2ARCBBoxEnv(O2ARCv2Env, TaskSettableEnv):
    
    def __init__(self, data_loader: Loader = ARCLoader(), max_grid_size: Tuple[SupportsInt, SupportsInt] = (30,30), colors: SupportsInt = 10, max_trial: SupportsInt = -1, render_mode: str = None, render_size: Tuple[SupportsInt, SupportsInt] = None) -> None:
        super(O2ARCv2Env).__init__(data_loader, max_grid_size, colors, max_trial, render_mode, render_size)
    
    # For BBox Wrapping
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
        return super(O2ARCv2Env).step({'selection': selection, 'operation': op})

    # For Meta-RL Setting (problem settable env)
    def reset(self, seed = None, options= None):
        return super(O2ARCv2Env).reset(seed, self.reset_options)

    #TaskSettableEnv API
    def sample_tasks(self, n_tasks: int) -> List[TaskType]:
        return np.random.choice(len(self.loader.data),n_tasks,replace=False)

    def get_task(self) -> TaskType:
        return super().get_task()
    
    def set_task(self, task: TaskType) -> None:
        self.reset_options = {
            'adaptation': True, # Default is true (adaptation first!). To change this mode, call 'post_adaptation()'
            'prob_index': task[1] 
        }
    
    

    def post_adaptation(self):
        self.reset_options['adaptation'] = False

# TODO: Create E-MAML with RLLib API
# TODO: Make Custom Policy and Run PPO

ray.init()

def env_creator(config):
    return FilterObservation(O2ARCBBoxEnv(max_trial=3),["input", "input_dim", "grid", "grid_dim", "clip", "clip_dim"])
register_env('O2ARCBBoxEnv', env_creator)

algo = PPOConfig().experimental(_disable_preprocessor_api = True).environment('O2ARCBBoxEnv').build()


while True:
    algo.train()