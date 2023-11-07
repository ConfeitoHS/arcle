import numpy as np
import gymnasium as gym
import pygame as pg
from gymnasium import spaces,utils
from gymnasium.core import ObsType, ActType

from typing import Dict,Optional,Union,Callable,List, Tuple, SupportsFloat, SupportsInt, SupportsIndex, Any
from numpy.typing import NDArray
from copy import deepcopy

from arcle.loaders import Loader, ARCLoader

from .base import AbstractARCEnv


class RawARCEnv(AbstractARCEnv):
    def __init__(self, data_loader: Loader =ARCLoader(), max_grid_size: Tuple[SupportsInt, SupportsInt]=(30,30), colors: SupportsInt=10, max_trial: SupportsInt = -1, render_mode: str =None, render_size: Tuple[SupportsInt, SupportsInt]= None) -> None:
        super().__init__(data_loader, max_grid_size, colors, max_trial, render_mode, render_size)

    def create_state_space(self):
        return super().create_state_space()
    
    def create_action_space(self, action_count) -> Any:
        return super().create_action_space(action_count)
    
    def create_operations(self) -> List[Callable[..., Any]]:
        ops = [None] * 12

        from ..actions.color import gen_color
        
        def resize_to_answer(state, action):
            h, w = self.answer.shape
            state['grid_dim'] = (h,w)
            state['grid'][h:,:] = 0
            state['grid'][:,w:] = 0
        
        ops[0:10] = [ gen_color(i)  for i in range(10) ]
        ops[10] = resize_to_answer
        ops[11] = self.submit

        return ops
    
    def init_state(self, initial_grid: NDArray, options: Dict) -> None:
        super().init_state(initial_grid, options)
    
    def get_info(self) -> Dict:
        return {
            "steps": self.action_steps
        }

    def reward(self, state) -> SupportsFloat:
        if not self.last_action_op == len(self.operations)-1:
            return 0
        if state['grid_dim'] == self.answer.shape:
            h,w = self.answer.shape
            if np.all(state['grid'][0:h, 0:w] == self.answer):
                return 1
        return 0

    def step(self, action: ActType):

        op = int(action['operation'])
        self.last_action_op = op
        self.last_action = action

        # do action
        state = self.current_state
        self.operations[op](state, action)
        
        reward = self.reward(state)
        self.last_reward = reward
        info = self.get_info()
        self.action_steps+=1
        self.render()

        return self.current_state, reward, bool(state["terminated"]) , self.truncated, info

class ARCEnv(AbstractARCEnv):
    def __init__(self, data_loader: Loader =ARCLoader(), max_grid_size: Tuple[SupportsInt, SupportsInt]=(30,30), colors: SupportsInt=10, max_trial: SupportsInt = 3, render_mode: str =None, render_size: Tuple[SupportsInt, SupportsInt]= None) -> None:
        super().__init__(data_loader, max_grid_size, colors, max_trial, render_mode, render_size)
    def init_state(self, initial_grid: NDArray, options: Dict) -> None:
        super().init_state(initial_grid, options)
        
        add_dict = {
            "clip" : np.zeros((self.H,self.W),dtype= np.uint8),
            "clip_dim" : (0, 0),
        }

        self.current_state.update(add_dict)
    
    def create_state_space(self):
        old_space = super().create_state_space()

        new_space_dict = {
                "clip": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.uint8),
                "clip_dim": spaces.Tuple((spaces.Discrete(self.H+1,start=0),spaces.Discrete(self.W+1,start=0))),
        }

        new_space_dict.update(old_space.spaces)
        return spaces.Dict(new_space_dict)
        
    def create_action_space(self, action_count) -> Any:
        return spaces.Dict(
            {
                "selection": spaces.MultiBinary((self.H,self.W)), # selection Mask
                "operation": spaces.Discrete(action_count)
            }
        )
    
    def create_operations(self) -> List[Callable[..., Any]]:
        from ..actions.object import (
            gen_copy, gen_paste
        )
        from ..actions.color import (
            gen_color, gen_flood_fill
        )
        from ..actions.critical import (
            copy_from_input,reset_grid,resize_grid
        )
        ops = [None] * 35

        # color ops (20)
        ops[0:10] = [gen_color(i) for i in range(10)]
        ops[10:20] = [gen_flood_fill(i) for i in range(10)]

        # clipboard ops (3)
        ops[20] = gen_copy("I")
        ops[21] = gen_copy("O")
        ops[22] = gen_paste(True)

        # critical ops (3)
        ops[23] = copy_from_input
        ops[24] = reset_grid
        ops[25] = resize_grid

        # submit op (1)
        ops[26] = self.submit
        return ops

    def get_info(self) -> Dict:
        return {
            "steps": self.action_steps,
            "submit_count": self.submit_count,
        }

    def reward(self, state) -> SupportsFloat:
        if not self.last_action_op == len(self.operations)-1:
            return 0
        if state['grid_dim'] == self.answer.shape:
            h,w = self.answer.shape
            if np.all(state['grid'][0:h, 0:w] == self.answer):
                return 1
        return 0
    
    def step(self, action: ActType):

        operation = int(action['operation'])

        self.transition(self.current_state, action)
        self.last_action_op = operation
        self.last_action = action

        # do action
        state = self.current_state
        reward = self.reward(state)
        self.last_reward = reward
        info = self.get_info()
        self.action_steps+=1
        self.render()

        return self.current_state, reward, bool(state["terminated"]), self.truncated, info

    def transition(self, state: ObsType, action: ActType) -> None:
        op = int(action['operation'])
        self.operations[op](state,action)

    def render_ansi(self):
        if self.rendering is None:
            self.rendering = True
            print('\033[2J',end='')
        
        print(f'\033[{self.H+3}A\033[K', end='')
        print('Problem Description:')
        print(self.description, '\033[K')

        grid = self.current_state['grid']
        grid_dim = self.current_state['grid_dim']
        sel = self.current_state['selected']
        clip = self.current_state['clip']
        clip_dim = self.current_state['clip_dim']
        

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
        print('Action : ' + str(self.op_names[self.last_action_op] if self.last_action_op is not None else '') , end=' ')
        print(f'Selected : {True if self.last_action is not None and  np.any(self.last_action["selection"]) else False}', end=' ')
        print('Reward : ' + str(self.last_reward)+ '\033[K')
        