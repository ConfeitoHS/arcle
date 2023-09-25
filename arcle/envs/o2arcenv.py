import numpy as np
import gymnasium as gym
import pygame as pg
from gymnasium import spaces,utils
from gymnasium.core import ObsType, ActType

from abc import abstractmethod, ABCMeta
from typing import Dict,Optional,Union,Callable,List,TypeAlias, Tuple, SupportsFloat, SupportsInt, SupportsIndex, Any
from numpy.typing import NDArray

from .arcenv import AbstractARCEnv

class O2ARCv2Env(AbstractARCEnv):

    selected : NDArray | None = None
    clip: NDArray | None = None  
    clip_dim: Tuple[SupportsIndex,SupportsIndex] | None = None

    # O2Actions
    objsel_active: bool = False
    objsel: NDArray = np.zeros((30,30), dtype=np.uint8)
    objsel_area: NDArray = np.zeros((30,30), dtype=np.uint8)
    objsel_bg: NDArray = np.zeros((30,30), dtype=np.uint8)
    objsel_coord: Tuple = ()
    objsel_rot: SupportsInt = 0

    def create_observation_space(self):
        return spaces.Dict(
            {
                "grid": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.uint8),
                "grid_dim": spaces.Tuple((spaces.Discrete(self.H,start=1),spaces.Discrete(self.W,start=1))),
                "selected": spaces.MultiBinary((self.H,self.W)),
                "clip": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.uint8),
                "clip_dim": spaces.Tuple((spaces.Discrete(self.H,start=0),spaces.Discrete(self.W,start=0))),
            }
        )
    
    def create_action_space(self, action_count) -> Any:
        return spaces.Dict(
            {
                "selection": spaces.MultiBinary((self.H,self.W)), # selection Mask
                "operation": spaces.Discrete(action_count)
            }
        )
    
    def create_actions(self) -> List[Callable[..., Any]]:
        from ..actions.object import (
            reset_sel, keep_sel,
            gen_move, gen_rotate, gen_flip,
            gen_copy, Paste
        )
        from ..actions.color import (
            gen_color, gen_flood_fill
        )
        from ..actions.critical import (
            copy_from_input,reset_grid,resize_grid,crop_grid
        )
        acts = []

        # color ops (20)
        # Resets previous object selection for move/rot/flip.
        acts.extend([reset_sel(gen_color(i)) for i in range(10)])       # 0 ~ 9
        acts.extend([reset_sel(gen_flood_fill(i)) for i in range(10)])  # 10 ~ 19

        # obj ops (8)
        acts.extend([gen_move(i) for i in range(4)])                    # 20 ~ 23
        acts.append(gen_rotate(1))                                      # 24
        acts.append(gen_rotate(3))                                      # 25 
        acts.append(gen_flip("H"))                                      # 26
        acts.append(gen_flip("V"))                                      # 27
        
        # clipboard ops (3)
        acts.append(reset_sel(gen_copy("I"))) # reset selection since it is from input grid         # 28
        acts.append(reset_sel(gen_copy("O"))) # do not reset since it is selected from output grid   # 29
        acts.append(reset_sel(Paste))                                   # 30

        # critical ops (3)
        acts.append(reset_sel(copy_from_input))                           # 31 = -4
        acts.append(reset_sel(reset_grid))                               # 32 = -3
        acts.append(reset_sel(crop_grid))                              # 33 = -2

        # submit op (1)                                                 # 34 = -1

        #  20 + 8 + 3 + 3 + 1 = 35

        return acts
    
    def init_observation(self, initial_grid: NDArray, options: Dict) -> None:
        super().init_observation(initial_grid, options)
        self.selected = np.zeros((self.H,self.W), dtype=np.uint8)
        self.clip = np.zeros((self.H,self.W),dtype= np.uint8)
        self.clip_dim = (0,0)
        
        self.objsel_active = False
        self.objsel = np.zeros((self.H, self.W), dtype=np.uint8)
        self.objsel_area = np.zeros((self.H, self.W), dtype=np.uint8)
        self.objsel_bg= np.zeros((self.H,self.W), dtype=np.uint8)
        self.objsel_coord= ()
        self.objsel_rot = 0
        

        self.current_obs = self.get_observation()   
        

    def get_observation(self) -> ObsType:
        return {
            "selected": self.selected,
            "grid": self.grid,
            "grid_dim": self.grid_dim,
            "clip" : self.clip,
            "clip_dim" : self.clip_dim
        }
    
    def get_info(self) -> Dict:
        return {
            "steps": self.action_steps,
        }

    def reward(self) -> SupportsFloat:
        if not self.terminated:
            return 0
        if self.grid_dim == self.answer.shape:
            h,w = self.answer.shape
            if np.all(self.grid[0:h, 0:w] == self.answer):
                return 1
        return 0
    
    def step(self, action: ActType):

        selection = action['selection'].astype(np.bool_)
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

    def render_ansi(self):
        if self.rendering is None:
            self.rendering = True
            print('\033[2J',end='')
        
        print(f'\033[{self.H+3}A\033[K', end='')
        print('Problem Description:')
        print(self.description, '\033[K')

        grid = self.grid
        grid_dim = self.grid_dim
        sel = self.selected
        clip = self.clip
        clip_dim = self.clip_dim
        

        for i in range(self.H):
            for j in range(self.W):
                d = grid[i,j]
                st = "[]" if sel[i,j] else "  " 
                if i >= grid_dim[0] or j>= grid_dim[1]:
                    print(f'\033[47m{st}', end='')
                else:
                    print("\033[48;5;"+str(self.ansi256arc[d])+f"m{st}", end='')

            print("\033[0m  ",end='')
            for j in range(self.W):
                d = clip[i,j]
                
                if i >= clip_dim[0] or j>= clip_dim[1]:
                    print('\033[47m  ', end='')
                else:
                    print("\033[48;5;"+str(self.ansi256arc[d])+"m  ", end='')               
      
            print('\033[0m')

        print('Dimension : '+ str(grid_dim), end=' ')
        print('Action : ' + str(self.action_names[self.last_action_op] if self.last_action_op is not None else '') , end=' ')
        print(f'Selected : {True if self.last_action is not None and  np.any(self.last_action["selection"]) else False}', end=' ')
        print('Reward : ' + str(self.last_reward)+ '\033[K')
        