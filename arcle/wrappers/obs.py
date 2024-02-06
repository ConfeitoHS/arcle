from typing import Any, Callable, List, SupportsInt, Tuple
import gymnasium as gym
from gymnasium.utils import RecordConstructorArgs
from gymnasium.core import Env
import gymnasium.spaces as spaces
from arcle.envs import AbstractARCEnv
from arcle.envs import O2ARCv2Env
import numpy as np

class FlattenDict(gym.ObservationWrapper):
    
    def __init__(self, env: Env, flatten_keys: List[str] = None):
        super().__init__(env)
        RecordConstructorArgs.__init__(self,flatten_keys=flatten_keys)
        spaces.flatten_space()
    
    def observation(self, observation: Any) -> Any:
        return super().observation(observation)
    
        


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
    return np.concatenate(for_concat, dtype=int,casting='unsafe')