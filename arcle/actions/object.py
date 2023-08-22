
import numpy as np
from numpy import ma
from ..envs import AbstractARCEnv as AAE
from numpy.typing import NDArray
from typing import SupportsInt,Callable,Tuple

def _get_bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def gen_rotate(k=1):
    '''
    Generates Rotate90 / Rotate180 / Rotate270 actions counterclockwise.

    Action Space Requirements (key: type) : (`selection`: NDArray)  

    Class State Requirements (key: type) : (`grid`: NDArray), (`grid_dim`: NDArray), (`objsel`: NDArray), (`objsel_bg`: NDArray), (`objsel_ts`: Integer), (`objsel_coord`, Tuple)
    '''
    assert 0<k<4

    def Rotate(cls: AAE, action):
        sel = action['selection']
        xmin, xmax, ymin, ymax = _get_bbox(sel)
        

        if cls.objsel_ts != cls.action_steps-1: # last action was not an object operation -> objsel invalid
            # then separate & backup
            cls.objsel = np.copy(cls.grid[xmin:xmax+1, ymin:ymax+1])
            cls.objsel_bg = np.copy(cls.grid)
            cls.objsel_bg[xmin:xmax+1, ymin:ymax+1] = 0
            cls.objsel_coord = (xmin, ymin) 
            cls.objsel_ts = cls.action_steps
        
        if k==2: 
            cls.objsel = np.rot90(cls.objsel,k=2)

        elif cls.objsel.shape[0]%2 == cls.objsel.shape[1]%2:
            cx = (xmax + xmin) *0.5
            cy = (ymax + ymin) *0.5
            x,y = cls.objsel_coord
            cls.objsel_coord = ( np.round(cx-cy+y), np.round(cy-cx+x)) #left-top corner will be diagonally swapped
            
        else: # ill-posed rotation
            