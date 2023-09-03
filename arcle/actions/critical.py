import numpy as np
from numpy import ma
from ..envs import AbstractARCEnv as AAE
from numpy.typing import NDArray
from typing import SupportsInt,Callable,Tuple
from .object import _get_bbox

def ResetGrid(cls: AAE, action):
    '''
    ResetGrid function that resets grid.
    
    Action Space Requirements (key: type) : None

    Class State Requirements (key: type) : (`grid`: NDArray)
    '''
    
    cls.grid = np.zeros((cls.H, cls.W),dtype=np.uint8)

def CopyFromInput(cls: AAE, action):
    '''
    Copy input grid and puts into output grid.
    
    Action Space Requirements (key: type) : None

    Class State Requirements (key: type) : (`grid`: NDArray), (`grid_dim`: Tuple), 
    '''
    cpi = np.copy(cls.input_).astype(np.uint8)
    inp_shape = cpi.shape
    cls.grid_dim = inp_shape
    cls.grid[:, :] = 0
    cls.grid[:inp_shape[0], :inp_shape[1]] = cpi
    
def ResizeGrid(cls: AAE, action):
    '''
    Resize Grid and Reset.
    
    Action Space Requirements (key: type) : (`selection`: NDArray)

    Class State Requirements (key: type) : (`grid`: NDArray), (`grid_dim`: Tuple), 
    '''
    xmin,xmax,ymin,ymax = _get_bbox(action['selection'])
    H = xmax-xmin+1
    W = ymax-ymin+1
    cls.grid[:, :] = 0
    cls.grid_dim = (H,W)
    