from typing import List, SupportsFloat, SupportsInt, Tuple
import ray
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.tune.registry import register_env
from arcle.envs import O2ARCv2Env
from arcle.loaders import ARCLoader, Loader
from arcle.wrappers import BBoxWrapper
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from gymnasium.wrappers.flatten_observation import FlattenObservation
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
    
    # For Meta-RL Setting (problem settable env)
    def reset(self, seed = None, options= None):
        obs, info = super().reset(seed, self.reset_options)
        return obs, info

    def reward(self, state) -> SupportsFloat:
        sparse_reward = super().reward(state)
        
        if self.adaptation:
            h, w  = state["grid_dim"].astype(int)
            H, W = self.answer.shape
            minh, minw = min(h,H) , min(w,W)
            total_size = minh*minw
            correct = np.sum(state["grid"][:minh,:minw]==self.answer[:minh,:minw])
            if (h <= H) == (w <= W):
                total_size += abs(H*W - h*w)
            else:
                total_size += abs(h-H)*minw + abs(w-W)*minh
            return -1+correct / total_size
        else:
            return sparse_reward
        
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
    env = O2ARCBBoxEnv(max_trial=127)
    env = BBoxWrapper(env)
    env = FlattenObservation(env)
    env = TimeLimit(env, max_episode_steps=100)
    return env
register_env("O2ARCBBoxEnv", env_creator)

config = EMAMLConfig().environment(env="O2ARCBBoxEnv").training(inner_adaptation_steps=5)
algo = config.build()

while True:
    print(algo.train())
