
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

def _init_objsel(cls, sel):
    
    something_selected = np.any(sel>0)

    # if    last action was not an object operation
    #       or something newly selected
    # then  previous object selection will be wiped
    if cls.objsel_ts != cls.action_steps-1 or something_selected: 
        # separate
        xmin, xmax, ymin, ymax = _get_bbox(sel)
        # backup
        cls.objsel = np.copy(cls.grid[xmin:xmax+1, ymin:ymax+1])
        cls.objsel_bg = np.copy(cls.grid)
        cls.objsel_bg[xmin:xmax+1, ymin:ymax+1] = 0
        cls.objsel_coord = (xmin, ymin) 
        cls.objsel_ts = cls.action_steps
        cls.objsel_rot = 0
        return xmin, xmax, ymin, ymax
    
    else:
        cls.objsel_ts = cls.action_steps
        x, y = cls.objsel_coord
        h, w = cls.objsel.shape
        return x, x+h-1, y, y+w-1

def _apply_patch(cls):
    img = cls.objsel_bg
    p = cls.objsel
    coord = cls.objsel_coord
    dim = cls.grid_dim
    
    patch = np.zeros((cls.H,cls.W), dtype=np.uint8)
    x, y = coord
    h, w = p.shape
    gh, gw = dim
    
    if x+h<=0 or x>=gh or y+w<=0 or y>=gw: # Perfectly out of bound
        pass
    else:

        stx = max(0,x)
        edx = min(gh,x+h)
        
        sty = max(0,y)
        edy = min(gw,y+w)

        if x<0:
            p = p[-x: , :]
        elif x+h>gh:
            p = p[:gh-x-h, :]
        if y<0:
            p = p[:, -y:]
        elif y+w>gw:
            p = p[:, :gw-y-w]

        patch[stx:edx, sty:edy] = p

    #patch[gh:, :] = 0
    #patch[:, gw:] = 0
    np.copyto(img, patch, where= (patch!=0))
    return np.copy(img)

def gen_rotate(k=1):
    '''
    Generates Rotate90 / Rotate180 / Rotate270 actions counterclockwise.

    Action Space Requirements (key: type) : (`selection`: NDArray)  

    Class State Requirements (key: type) : (`grid`: NDArray), (`grid_dim`: NDArray), (`objsel`: NDArray), (`objsel_bg`: NDArray), (`objsel_ts`: Integer), (`objsel_coord`: Tuple), (`objsel_rot`: Integer)
    '''
    assert 0<k<4

    def Rotate(cls: AAE, action):
        sel = action['selection']
        xmin, xmax, ymin, ymax = _init_objsel(cls,sel)
        
        if k==2:
            cls.objsel = np.rot90(cls.objsel,k=2)

        elif cls.objsel.shape[0]%2 == cls.objsel.shape[1]%2:
            cx = (xmax + xmin) *0.5
            cy = (ymax + ymin) *0.5
            x,y = cls.objsel_coord
            cls.objsel_coord = ( round(cx-cy+y), round(cy-cx+x)) #left-top corner will be diagonally swapped

        else: # ill-posed rotation. Manually setted
            cls.objsel_rot +=1
            sig = (k+2)%4-2
            mod = cls.objsel_rot%2
            mx = min(  cx+sig*(cy-ymin) , cx+sig*(cy-ymax) )-mod
            my = min(  cy-sig*(cx-xmin) , cy-sig*(cx-xmax) )-mod
            cls.objsel_coord = (mx,my)
        
        cls.objsel = np.rot90(cls.objsel, k=k)
        cls.grid = _apply_patch(cls)

    Rotate.__name__ = f"Rotate{90*k}"    
    return Rotate

def gen_move(d=0):
    '''
    Generates Move[U,D,R,L] actions. d=0 means move up, d=1 is down, d=2 is right, d=3 is left.

    Action Space Requirements (key: type) : (`selection`: NDArray)  

    Class State Requirements (key: type) : (`grid`: NDArray), (`grid_dim`: NDArray), (`objsel`: NDArray), (`objsel_bg`: NDArray), (`objsel_ts`: Integer), (`objsel_coord`: Tuple)
    '''
    assert 0<= d <4
    dirX = [-1, +1, 0, 0]
    dirY = [0, 0, +1, -1]
    def Move(cls: AAE, action):
        sel = action['selection']
        _,_,_,_ = _init_objsel(cls,sel)
        x, y = cls.objsel_coord
        cls.objsel_coord = (x + dirX[d], y + dirY[d])
        cls.grid = _apply_patch(cls)
        
    Move.__name__ = f"Move{'UDRL'[d]}"
    return Move

def FlipH(cls: AAE, action ):
    '''
    Flip Horizontally.

    Action Space Requirements (key: type) : (`selection`: NDArray)  

    Class State Requirements (key: type) : (`grid`: NDArray), (`grid_dim`: NDArray), (`objsel`: NDArray), (`objsel_bg`: NDArray), (`objsel_ts`: Integer), (`objsel_coord`: Tuple)
    '''
    sel = action['selection']
    _,_,_,_ = _init_objsel(cls,sel)
    cls.objsel = np.fliplr(cls.objsel)
    cls.grid = _apply_patch(cls)

def FlipV(cls: AAE, action ):
    '''
    Flip Vertically.

    Action Space Requirements (key: type) : (`selection`: NDArray)  

    Class State Requirements (key: type) : (`grid`: NDArray), (`grid_dim`: NDArray), (`objsel`: NDArray), (`objsel_bg`: NDArray), (`objsel_ts`: Integer), (`objsel_coord`: Tuple)
    '''
    sel = action['selection']
    _,_,_,_ = _init_objsel(cls,sel)
    cls.objsel = np.flipud(cls.objsel)
    cls.grid = _apply_patch(cls)

def FlipD0(cls: AAE, action ):
    '''
    Flip by Main Diagonal Axis (Transpose, or FlipH + Rot90).

    Action Space Requirements (key: type) : (`selection`: NDArray)  

    Class State Requirements (key: type) : (`grid`: NDArray), (`grid_dim`: NDArray), (`objsel`: NDArray), (`objsel_bg`: NDArray), (`objsel_ts`: Integer), (`objsel_coord`: Tuple)
    '''
    sel = action['selection']
    _,_,_,_ = _init_objsel(cls,sel)
    cls.objsel = np.rot90(np.fliplr(cls.objsel))
    cls.grid = _apply_patch(cls)

def FlipD1(cls: AAE, action ):
    '''
    Flip by Minor Diagonal Axis (Rot90 + FlipH).

    Action Space Requirements (key: type) : (`selection`: NDArray)  

    Class State Requirements (key: type) : (`grid`: NDArray), (`grid_dim`: NDArray), (`objsel`: NDArray), (`objsel_bg`: NDArray), (`objsel_ts`: Integer), (`objsel_coord`: Tuple)
    '''
    sel = action['selection']
    _,_,_,_ = _init_objsel(cls,sel)
    cls.objsel = np.fliplr(np.rot90(cls.objsel))
    cls.grid = _apply_patch(cls)

