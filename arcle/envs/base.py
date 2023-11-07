import numpy as np
import gymnasium as gym
import pygame as pg
from gymnasium import spaces,utils
from gymnasium.core import ObsType, ActType

from abc import abstractmethod, ABCMeta
from typing import Dict,Optional,Union,Callable,List, Tuple, SupportsFloat, SupportsInt, SupportsIndex, Any
from numpy.typing import NDArray

from arcle.loaders import Loader

from ..loaders import MiniARCLoader, ARCLoader, Loader

class AbstractARCEnv(gym.Env, metaclass=ABCMeta):
    """
    Abstract ARC Environment
    """

    ansi256arc = [0,12,9,10,11,8,13,208,14,52] # ANSI Color Code
    metadata = { "render_modes": ["ansi","human"] , "render_fps": 5 }
    
    # Observation(state)
    current_state: ObsType
    
    # Action Histories
    last_action: ActType = None
    last_action_op: SupportsIndex = None
    last_reward: SupportsFloat = 0
    action_steps: SupportsInt = 0
    
    # problem
    input_ : NDArray = None
    answer : NDArray = None
    description: Dict = None

    def __init__(self, 
                 data_loader: Loader,  
                 max_grid_size: Tuple[SupportsInt,SupportsInt], 
                 colors : SupportsInt,
                 max_trial: SupportsInt = -1,
                 render_mode: Optional[str] = None, 
                 render_size: Optional[Tuple[SupportsInt,SupportsInt]] = None) -> None:
        
        assert render_mode is None or render_mode in self.metadata['render_modes']

        self.loader = data_loader

        self.H, self.W = max_grid_size
        self.colors = colors
        self.max_trial = max_trial
        self._action_steps = 0

        # Render Related
        self.render_mode = render_mode
        self.render_size = render_size # Only for render_mode='human'
        self.clock = None # Only for render_mode='human'
        self.rendering = None  # Current rendering obj: PyGame window when render_mode='human', or True when render_mode='ansi'
        
        # Assign action functions / names
        self.operations = self.create_operations()

        # Create obs / action spaces
        self.observation_space = self.create_state_space()
        self.action_space = self.create_action_space(len(self.operations))
        self.op_names = [ ''.join(map(str.capitalize, op.__name__.split('_')))  for op in self.operations]

    
    def reset(self, seed = None, options: Optional[Dict] = None):
        super().reset(seed=seed,options=options)

        # Reset Internal States
        self.truncated = False
        self.submit_count = 0
        self.last_action: ActType  = None
        self.last_action_op : SupportsIndex  = None
        self.last_reward: SupportsFloat = 0
        self.action_steps: SupportsInt = 0
        
        # env option
        self.prob_index = None
        self.subprob_index = None
        self.adaptation = True
        self.reset_on_submit = False
        self.options = options

        if options is not None:
            self.prob_index = options.get('prob_index')
            self.subprob_index = options.get('subprob_index')
            _ad = options.get('adaptation')
            self.adaptation = True if _ad is None else bool(_ad)
            _ros = options.get('reset_on_submit')
            self.reset_on_submit = False if _ros is None else _ros
        
        ex_in, ex_out, tt_in, tt_out, desc = self.loader.pick(data_index=self.prob_index)

        
        if self.adaptation:
            self.subprob_index = np.random.randint(0,len(ex_in)) if self.subprob_index is None else self.subprob_index
            self.input_ = ex_in[self.subprob_index]
            self.answer = ex_out[self.subprob_index]

        else:
            self.subprob_index = np.random.randint(0,len(tt_in)) if self.subprob_index is None else self.subprob_index
            self.input_ = tt_in[self.subprob_index]
            self.answer = tt_out[self.subprob_index]

        self.init_state(self.input_.copy(),options)

        self.description = desc

        if self.render_mode:
            self.render()

        obs = self.current_state
        info = self.get_info()

        return obs, info
    
    @abstractmethod
    def create_state_space(self) -> spaces.Dict:
        return spaces.Dict({
            "trials_remain": spaces.Discrete(self.max_trial+2, start=-1),
            "terminated": spaces.Discrete(2, start=0),

            "input": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.uint8),
            "input_dim": spaces.Tuple((spaces.Discrete(self.H,start=1),spaces.Discrete(self.W,start=1))),

            "grid": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.uint8),
            "grid_dim": spaces.Tuple((spaces.Discrete(self.H,start=1),spaces.Discrete(self.W,start=1))),
        })
    
    @abstractmethod
    def create_action_space(self, action_count) -> spaces.Dict: 
        return spaces.Dict({
                "selection": spaces.MultiBinary((self.H,self.W)),
                "operation": spaces.Discrete(action_count)
        })

    @abstractmethod
    def create_operations(self) -> List[Callable]:
        pass

    @abstractmethod
    def get_info(self) -> Dict:
        pass

    @abstractmethod
    def init_state(self, initial_grid: NDArray, options: Dict) -> None:
        isize = initial_grid.shape
        self.current_state = {
            "trials_remain": self.max_trial,
            "terminated": 0,
            "input": np.pad(self.input_, [(0, self.H-isize[0]),(0, self.W-isize[1])], constant_values=0),
            "input_dim": self.input_.shape,

            "grid": np.pad(initial_grid, [(0, self.H-isize[0]),(0, self.W-isize[1])],constant_values=0),
            "grid_dim": isize
        }

    @abstractmethod 
    def reward(self) -> SupportsFloat:
        return 0

    def submit(self, state, action) -> None:
        if state["trials_remain"] > 0:
            state["trials_remain"] -=1
            self.submit_count +=1
            h,w = state["grid_dim"]
            if self.answer.shape == (h,w) and np.all(self.answer==state["grid"][:h,:w]):
                state["terminated"] = 1 # correct
            if self.reset_on_submit:
                self.init_state(self.input_, options=self.options)

        if state["trials_remain"] == 0:
            state["terminated"] = 1 # end 

    def render(self):
        if self.rendering is None and self.render_mode == "human":
            pg.init()
            pg.display.init()
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pg.time.clock()

        if self.render_mode == "ansi":
            self.render_ansi()

    def render_human(self):
        raise NotImplementedError
    
    def render_ansi(self):
        if self.rendering is None:
            self.rendering = True
            print('\033[2J',end='')

        print(f'\033[{self.H+3}A\033[K', end='')
        print('Problem Description:')
        print(self.description, '\033[K')

        state = self.current_state
        grid = state['grid']
        grid_dim = state['grid_dim']

        for i,dd in enumerate(grid):
            for j,d in enumerate(dd):
                
                if i >= grid_dim[0] or j>= grid_dim[1]:
                    print('\033[47m  ', end='')
                else:
                    print("\033[48;5;"+str(self.ansi256arc[d])+"m  ", end='')

            print('\033[0m')

        print('Dimension : '+ str(grid_dim), end=' ')
        print('Action : ' + str(self.op_names[self.last_action_op] if self.last_action_op is not None else '') , end=' ')
        print('Reward : ' + str(self.last_reward)+ '\033[K')