import numpy as np
import gymnasium as gym
import pygame as pg
from gymnasium import spaces,utils
from gymnasium.core import ObsType, ActType

from typing import Dict,Optional,Union,Callable,List, Tuple, SupportsFloat, SupportsInt, SupportsIndex, Any
from numpy.typing import NDArray
from ..loaders import ARCLoader, Loader

from .base import AbstractARCEnv

class O2ARCv2Env(AbstractARCEnv):
    def __init__(self, data_loader: Loader =ARCLoader(), max_grid_size: Tuple[SupportsInt, SupportsInt]=(30,30), colors: SupportsInt=10, max_trial: SupportsInt = -1, render_mode: str =None, render_size: Tuple[SupportsInt, SupportsInt]= None) -> None:
        super().__init__(data_loader, max_grid_size, colors, max_trial, render_mode, render_size)
    def init_state(self, initial_grid: NDArray, options: Dict) -> None:
        super().init_state(initial_grid, options)
        
        add_dict = {
            "selected": np.zeros((self.H,self.W), dtype=np.int8),
            "clip" : np.zeros((self.H,self.W), dtype= np.int8),
            "clip_dim" : np.zeros((2,), dtype=np.int8),
            "object_states": {
                "active": np.zeros((1,),dtype=np.int8), 
                "object": np.zeros((self.H, self.W), dtype=np.int8),
                "object_sel": np.zeros((self.H, self.W), dtype=np.int8),
                "object_dim": np.zeros((2,), dtype=np.int8),
                "object_pos": np.zeros((2,), dtype=np.int8), 
                "background": np.zeros((self.H, self.W), dtype=np.int8), 
                "rotation_parity": np.zeros((1,),dtype=np.int8),
            }
        }

        self.current_state.update(add_dict)
    
    def create_state_space(self):
        old_space = super().create_state_space()

        '''
        active: is object selection mode enabled?
        object: original data of object shapes and colors
        object_sel: original shape of selection area, same-shaped to object_dim
        object_pos: position of object
        background: background separated to object, same-shaped with grid_dim
        rotation_parity: rotation parity to keep rotation center
        '''

        new_space_dict = {
                "selected": spaces.Box(0,1,(self.H,self.W),dtype=np.int8),
                "clip": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.int8),
                "clip_dim": spaces.Box(low=np.array([0,0]), high=np.array([self.H,self.W]), dtype=np.int8),

                "object_states":spaces.Dict({
                    "active": spaces.MultiBinary(1),
                    "object": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.int8),
                    "object_sel":  spaces.Box(0,1,(self.H,self.W),dtype=np.int8),
                    "object_dim": spaces.Box(low=np.array([0,0]), high=np.array([self.H,self.W]), dtype=np.int8),
                    "object_pos": spaces.Box(low=np.array([-128,-128]), high=np.array([127,127]), dtype=np.int8), 

                    "background": spaces.Box(0, self.colors, (self.H,self.W),dtype=np.int8),
                    "rotation_parity": spaces.MultiBinary(1),
                })
        }

        new_space_dict.update(old_space.spaces)
        return spaces.Dict(new_space_dict)
        
    def create_action_space(self, action_count) -> Any:
        return spaces.Dict(
            {
                "selection": spaces.Box(0,1,(self.H,self.W),dtype=np.int8), # selection Mask
                "operation": spaces.Discrete(action_count)
            }
        )
    
    def create_operations(self) -> List[Callable[..., Any]]:
        from ..actions.object import (
            reset_sel, keep_sel,
            gen_move, gen_rotate, gen_flip,
            gen_copy, gen_paste
        )
        from ..actions.color import (
            gen_color, gen_flood_fill
        )
        from ..actions.critical import (
            copy_from_input,reset_grid,resize_grid,crop_grid
        )
        ops = [None] * 35

        # color ops (20)
        ops[0:10] = [reset_sel(gen_color(i)) for i in range(10)]
        ops[10:20] = [reset_sel(gen_flood_fill(i)) for i in range(10)]

        # obj ops (8)
        ops[20:24] = [gen_move(i) for i in range(4)]
        ops[24] = gen_rotate(1)
        ops[25] = gen_rotate(3)
        ops[26] = gen_flip("H")
        ops[27] = gen_flip("V")
        
        # clipboard ops (3)
        ops[28] = reset_sel(gen_copy("I"))
        ops[29] = reset_sel(gen_copy("O"))  
        ops[30] = reset_sel(gen_paste(paste_blank=True))

        # critical ops (3)
        ops[31] = reset_sel(copy_from_input)
        ops[32] = reset_sel(reset_grid)
        ops[33] = reset_sel(resize_grid)

        # submit op (1)
        ops[34] = self.submit
        return ops

    def init_info(self) -> Dict:
        info = super().init_info()
        info["steps"] = 0
        info["submit_count"] = 0
        return info

    def reward(self, state) -> SupportsFloat:
        if not self.last_action_op == len(self.operations)-1:
            return 0
        if tuple(state['grid_dim']) == self.answer.shape:
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
        self.action_steps+=1
        self.info['steps'] = self.action_steps
        self.info['submit_count'] = self.submit_count
        self.render()

        return self.current_state, reward, bool(state["terminated"][0]), self.truncated, self.info

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