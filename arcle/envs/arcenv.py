import numpy as np
import gymnasium as gym
import pygame as pg
from gymnasium import spaces,utils
from arcs import ARCParser
from typing import Dict,Optional,Union

class ArcEnv(gym.Env):
    metadata: {
        "render_modes": ["human","ansi"]
    }

    def __init__(self, render_mode = None, grid_size = (30,30), colors = 10, window_size = (1024,512), actions: Optional[spaces.Space]=None, parser=ARCParser()):

        self.parser = parser
        self.window_size = window_size
        self.grid_size = grid_size 
        self.colors = colors

        self.parser.load_ARC()

        self.observation_space = spaces.Dict(
            {
                "clipboard": spaces.Box(0,colors-1,grid_size,dtype=np.int8),
                "clipboard_size": spaces.MultiDiscrete((2,),dtype=np.int8,start=1),
                "output": spaces.Box(0,colors-1,grid_size,dtype=np.int8),
                "output_size": spaces.MultiDiscrete((2,),dtype=np.int8,start=1),
            }
        )
        if actions is None:
            self.action_space = spaces.Dict(
                {
                    "source": spaces.Discrete(2),   # from input(0) / output grid(1)
                    "selection": spaces.MultiBinary(grid_size), # selection Mask
                    "action": spaces.Discrete(10+3+3+4+1+2)  # Color(10) + Rotate(3) + Flip(3) + Move(4) + Crop(1) + Copy/Paste(2)
                }
            )
        else:
            self.action_space = spaces.Dict(
                {
                    "source": spaces.Discrete(2), 
                    "selection": spaces.MultiBinary(grid_size),
                    "action": actions
                }
            )

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        
        self.arc_window = None
        self.clock = None

    def _get_obs(self):
        return {"clipboard": self._clipboard,"clipboard_size": self._clipboard_size, "output": self._output, "output_size": self._output_size}

    def _get_info(self):
        return {"distance": self.distance(self._output,self._test_answer)}
    
    def distance(self,A: np.ndarray,B: np.ndarray):
        return 0

    def reset(self, seed=None, options=None, arc_id=-1):
        super().reset(seed=seed)

        self._clipboard = np.zeros(self.grid_size,dtype=np.int8)
        self._clipboard_size = (1,1)
        exi,exo,ti,to = self.parser.pick_ARC(data_index=arc_id)

        self._example_input = exi
        self._example_output = exo
        self._test_answer = to[0]
        self._output = ti[0]
        self._output_size = ti[0].shape

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action: Dict[str,Union[int,np.ndarray]]):
        pass