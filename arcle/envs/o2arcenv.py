import numpy as np
import gymnasium as gym
import pygame as pg
from gymnasium import spaces,utils
from gymnasium.core import ObsType, ActType

from abc import abstractmethod, ABCMeta
from typing import Dict,Optional,Union,Callable,List,TypeAlias, Tuple, SupportsFloat, SupportsInt, SupportsIndex, Any
from numpy.typing import NDArray

from arcle.loaders import Loader
from ..loaders import MiniARCLoader, ARCLoader, Loader
from .arcenv import AbstractARCEnv
from ..actions.color import *
from ..actions.critical import *
from ..actions.object import *


class O2ARCv2Env(AbstractARCEnv):

    selected : NDArray | None = None
    clip: NDArray | None = None  
    clip_dim: Tuple[SupportsIndex,SupportsIndex] | None = None

    def __init__(self, render_mode: str | None = None, train:bool=True, render_size: Tuple[SupportsInt, SupportsInt] | None = None) -> None:
        super().__init__(ARCLoader(train=train), (30,30), 10, render_mode, render_size)

    def create_observation_space(self):
        return spaces.Dict(
            {
                "grid": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.uint8),
                "grid_dim": spaces.Tuple((spaces.Discrete(self.H,start=1),spaces.Discrete(self.W,start=1))),
                "selected": spaces.MultiBinary((self.H,self.W)),
                "clip": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.uint8),
                "clip_dim": spaces.Tuple((spaces.Discrete(self.H,start=1),spaces.Discrete(self.W,start=1))),
            }
        )
    
    def create_action_space(self) -> Any:
        return spaces.Dict(
            {
                "selection": spaces.MultiBinary((self.H,self.W)), # selection Mask
                "operation": spaces.Discrete(10+10+ 4+2+2 + 3 + 3+1)  # Color(10) + FloodFill(10) + Move(4) + Rot(2) + Flip(2) + (CopyI, CopyO, Paste)  +  (Reset, Copyfrominput, Resize) + submit
            }
        )
    
    def create_actions(self) -> List[Callable[..., Any]]:
        
        acts = []

        # color ops
        acts.extend([gen_color(i) for i in range(10)])
        acts.extend([gen_flood_fill(i) for i in range(10)])

        # obj ops
        acts.extend([gen_move(i) for i in range(4)])
        acts.append(gen_rotate(1))
        acts.append(gen_rotate(3))
        acts.append(FlipV)
        acts.append(FlipH)
        
        ## 여기에 카피 넣기

        acts.append(ResetGrid)
        acts.append(CopyFromInput)
        acts.append(ResizeGrid)

        return acts
    
    def init_observation(self, initial_grid: NDArray, options: Dict) -> None:
        super().init_observation(initial_grid, options)

        self.current_obs = self.get_observation()

    def get_observation(self) -> ObsType:
        return {
            "grid": self.grid,
            "grid_dim": self.grid_dim,
            "clip" : self.clip,
            "clip_dim" : self.clip_dim
        }
    
    def get_info(self) -> Dict:
        return {
            "steps": self.action_steps
        }

    def reward(self) -> SupportsFloat:
        if self.grid_dim == self.answer.shape:
            h,w = self.answer.shape
            if np.all(self.grid[0:h, 0:w] == self.answer):
                return 1
        return 0
    
    def step(self, action: ActType):

        selection = action['selection']
        operation = int(action['operation'])
        self.last_action_op = operation
        self.last_action = action

        # do action
        self.actions[operation](self,action)
        obs = self.get_observation()
        reward = self.reward()
        self.last_reward = reward
        info = self.get_info()
        self.action_steps+=1
        self.render()

        return obs, reward, self.terminated, self.truncated, info
