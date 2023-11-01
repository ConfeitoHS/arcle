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
                 max_trial: SupportsInt = 3,
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
        self.terminated = False
        self.truncated = False
        self.last_action: ActType  = None
        self.last_action_op : SupportsIndex  = None
        self.last_reward: SupportsFloat = 0
        self.action_steps: SupportsInt = 0
        
        prob_index = None
        subprob_index = None
        adaptation = True

        if options is not None:
            prob_index = options.get('prob_index')
            subprob_index = options.get('subprob_index')
            _ad = options.get('adaptation')
            adaptation = adaptation if _ad is None else bool(_ad)
        
        ex_in, ex_out, tt_in, tt_out, desc = self.loader.pick(data_index=prob_index)

        
        if adaptation:
            subprob_index = np.random.randint(0,len(ex_in)) if subprob_index is None else subprob_index
            self.input_ = ex_in[subprob_index]
            self.answer = ex_out[subprob_index]

        else:
            subprob_index = np.random.randint(0,len(tt_in)) if subprob_index is None else subprob_index
            self.input_ = tt_in[subprob_index]
            self.answer = tt_out[subprob_index]

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
            "trials_remain": spaces.Discrete(self.max_trial+1, start=0),

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

            "input": np.pad(self.input_, [(0, self.H-isize[0]),(0, self.W-isize[1])], constant_values=0),
            "input_dim": self.input_.shape,

            "grid": np.pad(initial_grid, [(0, self.H-isize[0]),(0, self.W-isize[1])],constant_values=0),
            "grid_dim": isize
        }

    @abstractmethod 
    def reward(self) -> SupportsFloat:
        return 0

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

        grid = self.grid
        grid_dim = self.grid_dim

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

class ARCEnv(AbstractARCEnv):

    def create_state_space(self):
        return super().create_state_space()
    
    def create_action_space(self, action_count) -> Any:
        return super().create_action_space
    
    def create_operations(self) -> List[Callable[..., Any]]:
        ops = [None] * 12

        from ..actions.color import gen_color
        from ..actions.critical import submit
        
        def resize_to_answer(state, action):
            h, w = self.answer.shape
            state['grid_dim'] = (h,w)
            state['grid'][h:,:] = 0
            state['grid'][:,w:] = 0
        
        ops[0:10] = [ gen_color(i)  for i in range(10) ]
        ops[10] = resize_to_answer
        ops[11] = submit

        return ops
    
    def init_state(self, initial_grid: NDArray, options: Dict) -> None:
        super().init_state(initial_grid, options)
    
    def get_info(self) -> Dict:
        return {
            "steps": self.action_steps
        }

    def reward(self, state) -> SupportsFloat:
        if not self.last_action_op == 12:
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

        return self.current_state, reward, self.terminated, self.truncated, info