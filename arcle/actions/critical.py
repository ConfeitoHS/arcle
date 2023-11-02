import numpy as np
from numpy import ma
from ..envs import AbstractARCEnv as AAE
from numpy.typing import NDArray
from typing import SupportsInt,Callable,Tuple
from .object import _get_bbox

def reset_grid(state, action):
    '''
    ResetGrid function that resets grid.
    
    Action Space Requirements (key: type) : None

    Class State Requirements (key: type) : (`grid`: NDArray)
    '''
    
    state['grid'][:, :] = 0

def copy_from_input(state, action):
    '''
    Copy input grid and puts into output grid.
    
    Action Space Requirements (key: type) : None

    Class State Requirements (key: type) : (`grid`: NDArray), (`grid_dim`: Tuple), 
    '''

    state['grid_dim'] = state['input_dim']
    state['grid'][:, :] = state['input']
    
def resize_grid(state, action):
    '''
    Resize Grid and Reset. Use bounding box of 'selection' as a target bbox size.
    
    Action Space Requirements (key: type) : (`selection`: NDArray)

    Class State Requirements (key: type) : (`grid`: NDArray), (`grid_dim`: Tuple), 
    '''
    if not np.any(action['selection']):
        return
    
    xmin, xmax, ymin, ymax = _get_bbox(action['selection'])
    h = xmax-xmin+1
    w = ymax-ymin+1
    state['grid'][:, :] = 0
    state['grid_dim'] = (h,w)
    
def crop_grid(state, action):
    '''
    Crop Grid by selection.
    
    Action Space Requirements (key: type) : (`selection`: NDArray)

    Class State Requirements (key: type) : (`grid`: NDArray), (`grid_dim`: Tuple), 
    '''
    if not np.any(action['selection']):
        return
    
    xmin, xmax, ymin, ymax = _get_bbox(action['selection'])
    H = xmax-xmin+1
    W = ymax-ymin+1
    patch = np.zeros((H,W),dtype=np.uint8)
    np.copyto(dst=patch, src=state['grid'][xmin:xmax+1, ymin:ymax+1], where= np.logical_and(action['selection'][xmin:xmax+1, ymin:ymax+1],state['grid'][xmin:xmax+1, ymin:ymax+1]))
    state['grid'][:,:]=0
    state['grid'][0:H, 0:W] = patch # comment when test
    state['grid_dim'] = (H,W)
    