import numpy as np
import gymnasium as gym
import pygame as pg
from gymnasium import spaces,utils
from gymnasium.core import ObsType, ActType

from abc import abstractmethod, ABCMeta
from typing import Dict,Optional,Union,Callable,List,TypeAlias, Tuple, SupportsFloat, SupportsInt, SupportsIndex, Any
from numpy.typing import NDArray

from ..loaders import MiniARCLoader, ARCLoader, Loader

class AbstractARCEnv(gym.Env, metaclass=ABCMeta):
    """
    Abstract ARC Environment
    """

    ansi256arc = [0,12,9,10,11,8,13,208,14,52] # ANSI Color Code
    metadata = { "render_modes": ["ansi","human"] , "render_fps": 5 }

    terminated: bool = False
    truncated: bool = False

    # internal states
    grid: NDArray | None = None  
    ''' You can freely add more observable internal states, but do not remove `grid` and `grid_dim`.'''
    grid_dim: Tuple[SupportsIndex,SupportsIndex] | None = None
    '''You can freely add more observable internal states, but do not remove `grid` and `grid_dim`.'''
    current_obs: ObsType | None = None
    # Action Histories
    last_action: ActType | None = None
    last_action_op : SupportsIndex | None = None  # action index of DSLs
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
                 render_mode: Optional[str] = None, 
                 render_size: Optional[Tuple[SupportsInt,SupportsInt]] = None) -> None:
        
        assert render_mode is None or render_mode in self.metadata['render_modes']

        self.loader = data_loader

        self.H, self.W = max_grid_size
        self.colors = colors
        self._action_steps = 0

        # Render Related
        self.render_mode = render_mode
        self.render_size = render_size # Only for render_mode='human'
        self.clock = None # Only for render_mode='human'
        self.rendering = None  # Current rendering obj: PyGame window when render_mode='human', or True when render_mode='ansi'
        
        # Assign action functions / names
        self.actions = self.create_actions()
        actcnt = len(self.actions)

        # Create obs / action spaces
        self.observation_space = self.create_observation_space()
        self.action_space = self.create_action_space(actcnt+1)

        def submit(cls: AbstractARCEnv, action, *args):
            cls.terminated = True
        
        self.actions.append(submit)
        self.action_names = [ ''.join(map(str.capitalize,ac.__name__.split('_')))  for ac in self.actions]

    
    def reset(self, seed = None, options: Optional[Dict] = None):
        super().reset(seed=seed,options=options)

        # Reset Internal States
        self.terminated = False
        self.truncated = False
        self.last_action: ActType | None = None
        self.last_action_op : SupportsIndex | None = None
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

        self.init_observation(self.input_.copy(),options)

        self.description = desc

        if self.render_mode:
            self.render()

        obs = self.get_observation()
        info = self.get_info()

        return obs, info
        
    @abstractmethod
    def create_observation_space(self) -> spaces.Space:
        pass

    @abstractmethod
    def create_action_space(self, action_count) -> spaces.Space:
        pass

    @abstractmethod
    def create_actions(self) -> List[Callable]:
        pass

    @abstractmethod
    def get_observation(self) -> ObsType:
        pass

    @abstractmethod
    def get_info(self) -> Dict:
        pass

    @abstractmethod
    def init_observation(self, initial_grid: NDArray, options: Dict) -> None:
        isize = initial_grid.shape
        self.grid = np.pad(initial_grid, [(0, self.H-isize[0]),(0, self.W-isize[1])],constant_values=0)
        self.grid_dim = isize
        self.current_obs = None

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
        print('Action : ' + str(self.action_names[self.last_action_op] if self.last_action_op is not None else '') , end=' ')
        print('Reward : ' + str(self.last_reward)+ '\033[K')

class ARCEnv(AbstractARCEnv):
    def __init__(self, render_mode: str | None = None, train:bool=True, render_size: Tuple[SupportsInt, SupportsInt] | None = None) -> None:
        super().__init__(ARCLoader(train=train), (30,30), 10, render_mode, render_size)

    def create_observation_space(self):
        return spaces.Dict(
            {
                "grid": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.uint8),
                "grid_dim": spaces.Tuple((spaces.Discrete(self.H,start=1),spaces.Discrete(self.W,start=1)))
            }
        )
    
    def create_action_space(self, action_count) -> Any:
        return spaces.Dict(
            {
                "selection": spaces.MultiBinary((self.H,self.W)), # selection Mask
                "operation": spaces.Discrete(action_count)  # Color(10) + ResizeToAnswer + Submit
            }
        )
    
    def create_actions(self) -> List[Callable[..., Any]]:
        
        acts = []

        def color_grid(color):
            def colors(cls: AbstractARCEnv, action) :
                cls.grid = np.ma.array(cls.grid, mask=action['selection']).filled(fill_value=color)
            colors.__name__ = 'color_'+str(color)
            return colors
        
        def resize_to_answer(cls: AbstractARCEnv, action):
            cls.grid_dim = cls.answer.shape
        
        acts = [ color_grid(i)  for i in range(10)  ]
        acts.append(resize_to_answer)

        return acts
    
    def init_observation(self, initial_grid: NDArray, options: Dict) -> None:
        super().init_observation(initial_grid, options)

        self.current_obs = self.get_observation()

    def get_observation(self) -> ObsType:
        return {
            "grid": self.grid,
            "grid_dim": self.grid_dim
        }
    
    def get_info(self) -> Dict:
        return {
            "steps": self.action_steps
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

class MiniARCEnv(AbstractARCEnv):
    def __init__(self, render_mode: str | None = None, render_size: Tuple[SupportsInt, SupportsInt] | None = None) -> None:
        super().__init__(MiniARCLoader(), (5,5), 10, render_mode, render_size)

    def create_observation_space(self):
        return spaces.Box(0,self.colors,(self.H,self.W),dtype=np.uint8)
    
    def create_action_space(self, action_count) -> Any:
        return spaces.Dict(
            {
                "selection": spaces.MultiBinary((self.H,self.W)), # selection Mask
                "operation": spaces.Discrete(action_count)  # Color(10) + Submit
            }
        )
    
    def create_actions(self) -> List[Callable[..., Any]]:
        
        acts = []

        def color_grid(color):
            def colors(cls: AbstractARCEnv, action) :
                cls.grid = np.ma.array(cls.grid, mask=action['selection']).filled(fill_value=color)
            colors.__name__ = 'color_'+str(color)
            return colors
        
        acts = [ color_grid(i)  for i in range(10)  ]

        return acts
    
    def init_observation(self, initial_grid: NDArray, options: Dict) -> None:
        super().init_observation(initial_grid, options)

        self.current_obs = self.get_observation()

    def get_observation(self) -> ObsType:
        return self.grid
    
    def get_info(self) -> Dict:
        return {
            "steps": self.action_steps
        }

    def reward(self) -> SupportsFloat:
        if not self.terminated:
            return 0
        if np.all(self.grid == self.answer):
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
