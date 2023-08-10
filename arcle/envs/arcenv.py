import numpy as np
import gymnasium as gym
import pygame as pg
from gymnasium import spaces,utils
from ..arcs.parser import ARCParser
from typing import Dict,Optional,Union, Callable,List
import pygame as pg

class ArcEnv(gym.Env):
    """
    Default ARC Environment with selection & color actions.
    """
    
    ansi256arc = [0,12,9,10,11,8,13,208,14,52]
    metadata = { "render_modes": ["ansi","human"] , "render_fps": 5 }

    def __init__(self, render_mode = None, parser=None, train=True, grid_size = (30,30), colors = 10, window_size = (1024,512)):

        self.parser = parser
        self.window_size = window_size
        self.grid_size = grid_size 
        self.colors = colors
        self.terminated = False

        if parser is None:
            self.parser = ARCParser()
            self.parser.load_ARC(train=train)

        self.observation_space = spaces.Dict(
            {
                "output": spaces.Box(0,colors,grid_size,dtype=np.uint8),
                "output_dim": spaces.Tuple((spaces.Discrete(30,start=1),spaces.Discrete(30,start=1)))
            }
        )
        self.action_space = spaces.Dict(
            {
                "selection": spaces.MultiBinary(grid_size), # selection Mask
                "operation": spaces.Discrete(colors + 1 + 1)  # Color(10) + ResizeToAnswer + Submit
            }
        )
        self.action_names = ['Color0', 'Color1','Color2','Color3','Color4','Color5','Color6','Color7','Color8','Color9', 'ResizeAns', 'Submit']
        
        def color_grid(color):
            def colors(selection) :
                self._output = np.ma.array(self._output,mask=selection).filled(fill_value=color)
            return colors
        
        def resize_to_answer(selection):
            self._output_dim = self._answer.shape
            
        
        def submit(selection):
            self.terminated = True
            

        self.actions: List[Callable] = [ color_grid(c) for c in range(10) ]
        self.actions.append( resize_to_answer )
        self.actions.append( submit )
        

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        
        self.arc_window = None
        self.clock = None
        self.ansi_init = False

    def get_obs(self):
        return {"output": self._output, "output_dim": self._output_dim}

    def get_info(self):
        return {
            "distance": self.distance(self._output),
            "steps": self._action_steps
        }
    
    def distance(self, prediction: np.ndarray):
        return 0
    
    def reward(self):
        
        if self._output_dim == self._answer.shape:
            h,w = self._answer.shape
            if np.all(self._output[0:h, 0:w] == self._answer):
                return 1
        return 0
    

    def reset(self, seed=None, options=None ):
        super().reset(seed=seed)
        self._last_action = None
        self.terminated = False
        if options is None:
            prob_id = -1
            sub_id = -1
            adaptation = True
        else:
            prob_id = options['prob_id']
            sub_id = options['sub_id']
            adaptation = options['adaptation']
        self.adaptation = adaptation

        exi,exo,ti,to = self.parser.pick_ARC(data_index=prob_id)

        if adaptation:
            subTaskIndex = np.random.randint(0,len(exi)) if sub_id <0 else sub_id
            isize = exi[subTaskIndex].shape
            osize = exo[subTaskIndex].shape

            self._input = np.pad(exi[subTaskIndex].copy(), [(0, 30-isize[0]),(0, 30-isize[1])],constant_values=0)
            self._output = np.pad(exi[subTaskIndex].copy(), [(0, 30-isize[0]),(0, 30-isize[1])],constant_values=0)
            self._output_dim = exi[subTaskIndex].shape
            self._answer = exo[subTaskIndex].copy()
        else:
            subTaskIndex = np.random.randint(0,len(ti)) if sub_id <0 else sub_id
            isize = ti[subTaskIndex].shape
            osize = to[subTaskIndex].shape

            self._input = np.pad(ti[subTaskIndex].copy(), [(0, 30-isize[0]),(0, 30-isize[1])],constant_values=0)
            self._output = np.pad(ti[subTaskIndex].copy(), [(0, 30-isize[0]),(0, 30-isize[1])],constant_values=0)
            self._output_dim = ti[subTaskIndex].shape
            self._answer = to[subTaskIndex].copy()
        
        self._action_steps = 0

        obs = self.get_obs()
        info = self.get_info()

        if self.render_mode =='ansi':
            self._render_step()

        return obs, info

    def step(self, action: Dict[str,Union[int,np.ndarray]]):
        
        selection = action['selection']
        operation = int(action['operation'])
        self._last_action = action
        # do action
        self.actions[operation](selection)
        
        obs = self.get_obs()
        reward = self.reward()
        terminated = self.terminated
        info = self.get_info()

        if self.render_mode =='ansi':
            self._render_step()

        return obs, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_step()
            

    def _render_step(self):
        if self.arc_window is None and self.render_mode == "human":
            pg.init()
            pg.display.init()
        if self.clock is None and self.render_mode == "human":
            self.clock = pg.time.clock()

        if self.render_mode == "ansi" and self.ansi_init ==False:
            self.ansi_init = True
            print('\033[2J',end='')

        if self.render_mode == "ansi":
            print('\033[32A',end='')
            for i,dd in enumerate(self._output):
                for j,d in enumerate(dd):
                    if i >= self._output_dim[0] or j>= self._output_dim[1]:
                        print('\033[47m  ',end='')
                    else:
                        print("\033[48;5;"+str(self.ansi256arc[d])+"m  ",end='')
                print('\033[0m')
            print('Dimension : '+ str(self._output_dim),'      ')
            print('Action : ' + str(self.action_names[self._last_action['operation']] if self._last_action is not None else '') + '      ')

