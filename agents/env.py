from typing import List, SupportsFloat, SupportsInt, Tuple
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from arcle.envs import O2ARCv2Env
from arcle.loaders import ARCLoader, Loader
from collections import OrderedDict
import gymnasium as gym
from gymnasium import spaces

import numpy as np

from ray.tune.logger import pretty_print


class CustomO2ARCEnv(O2ARCv2Env, TaskSettableEnv):
    
    def __init__(self, data_loader: Loader = ARCLoader(), max_grid_size: Tuple[SupportsInt, SupportsInt] = (30,30), colors: SupportsInt = 10, max_trial: SupportsInt = -1, render_mode: str = None, render_size: Tuple[SupportsInt, SupportsInt] = None) -> None:
        super().__init__(data_loader, max_grid_size, colors, max_trial, render_mode, render_size)

        self.reset_options = {
            'adaptation': True, # Default is true (adaptation first!). To change this mode, call 'post_adaptation()'
            'prob_index': None
        }
    def create_operations(self) :
        from arcle.actions.critical import crop_grid
        from arcle.actions.object import reset_sel
        ops = super().create_operations()
        ops[33] = reset_sel(crop_grid)
        return ops

    # For Meta-RL Setting (problem settable env)
    def reset(self, seed = None, options= None):
        obs, info = super().reset(seed, self.reset_options)
        rotate_k = np.random.randint(0,4)
        permute = np.random.permutation(10)
        f = lambda x: permute[int(x)]
        ffv = np.vectorize(f)
        # augment
        
        self.input_ = np.copy(np.rot90(ffv(self.input_),k=rotate_k).astype(np.int8))
        self.answer = np.copy(np.rot90(ffv(self.answer),k=rotate_k).astype(np.int8))
        self.init_state(self.input_.copy(),options)
        return obs, info

    def reward(self, state) -> SupportsFloat:
        #return super().reward(state)*100
        sparse_reward = super().reward(state) # is fully correct? 
        
        h, w  = state["grid_dim"].astype(int)
        H, W = self.answer.shape
        minh, minw = min(h,H) , min(w,W)
        total_size = minh*minw
        correct = np.sum(state["grid"][:minh,:minw]==self.answer[:minh,:minw])
        if (h <= H) == (w <= W):
            total_size += abs(H*W - h*w)
        else:
            total_size += abs(h-H)*minw + abs(w-W)*minh
    
        return sparse_reward*100 -1+correct / total_size
        if self.adaptation or self.last_action_op == len(self.operations)-1:
            return sparse_reward*100 -1+correct / total_size 
        else:
            return -1.0

        
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
        super().reset(options=self.reset_options)

    def init_adaptation(self):
        self.adaptation = True
        self.reset_options['adaptation'] = True
        super().reset(options=self.reset_options)
        
    def post_adaptation(self):
        self.adaptation = False
        self.reset_options['adaptation'] = False
        super().reset(options=self.reset_options)

class FilterO2ARC(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            "trials_remain": spaces.Box(-1, self.max_trial, shape=(1,), dtype=np.int8),

            "grid": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.int8),
            "grid_dim": spaces.Box(low=np.array([1,1]), high=np.array([self.H,self.W]), dtype=np.int8),

            "clip": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.int8),
            "clip_dim": spaces.Box(low=np.array([0,0]), high=np.array([self.H,self.W]), dtype=np.int8),

            "active": spaces.MultiBinary(1),
            "object": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.int8),
            "object_dim": spaces.Box(low=np.array([0,0]), high=np.array([self.H,self.W]), dtype=np.int8),
            "object_pos": spaces.Box(low=np.array([-128,-128]), high=np.array([127,127]), dtype=np.int8), 

            }
        )

    def observation(self, observation) :
        obs = observation
        o2s = obs["object_states"]
        return OrderedDict([
            
            ("trials_remain",obs["trials_remain"]),
            
            ("grid",obs["grid"]),
            ("grid_dim",obs["grid_dim"]),

            ("clip",obs["clip"]),
            ("clip_dim",obs["clip_dim"]),

            ("active",o2s["active"]),
            ("object",o2s["object"]),
            ("object_dim",o2s["object_dim"]),
            ("object_pos",o2s["object_pos"]), 
        ])

