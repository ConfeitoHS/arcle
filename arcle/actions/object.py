
import numpy as np
from numpy import ma
#from ..envs import AbstractARCEnv as AAE
from ..envs import O2ARCv2Env as O2E
from numpy.typing import NDArray
from typing import SupportsInt,Callable,Tuple
from functools import wraps

def reset_sel(function):
    '''
    Wrapper for Non-O2ARC actions. This wrapper resets `selected` of obs space and resets object-operation states.

    It does this before calling function.
    ```
        cls.selected = np.zeros((cls.H,cls.W), dtype=np.uint8)
        cls.objsel_active = False
    ```
    '''
    @wraps(function)
    def wrapper(cls: O2E, action, **kwargs):
        cls.selected = np.zeros((cls.H,cls.W), dtype=np.uint8)
        cls.objsel_active = False
        
        return function(cls, action, **kwargs)
    return wrapper

def keep_sel(function):
    '''
    Wrapper for Non-O2ARC actions. This wrapper keeps `selection` of action in `selected` of obs space.

    It does this before calling function.
    ```
        cls.selected = np.copy(action["selection"])
    ```
    '''
    @wraps(function)
    def wrapper(cls: O2E, action, **kwargs):
        cls.selected = np.copy(action["selection"])
        return function(cls, action, **kwargs)
    return wrapper

def _get_bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def _init_objsel(cls: O2E, sel: NDArray):

    # if    something newly selected
    # then  previous object selection will be wiped
    if np.any(sel>0): 
        # separate
        xmin, xmax, ymin, ymax = _get_bbox(sel)
        # backup
        H = xmax-xmin+1
        W = ymax-ymin+1

        cls.objsel = np.zeros((H,W), dtype=np.uint8)
        np.copyto(dst=cls.objsel, src=cls.grid[xmin:xmax+1, ymin:ymax+1], where=sel[xmin:xmax+1, ymin:ymax+1])

        cls.objsel_area = np.copy(sel[xmin:xmax+1, ymin:ymax+1])

        cls.objsel_bg = np.copy(cls.grid)
        np.copyto(dst=cls.objsel_bg, src=0, where=sel)

        cls.objsel_coord = (int(xmin), int(ymin)) 
        cls.objsel_active = True
        cls.objsel_rot = 0

        cls.selected = np.copy(sel)

        return xmin, xmax, ymin, ymax
    
    # if    objsel active and no selection
    # then  continue with prev objsel
    elif cls.objsel_active: 
        x, y = cls.objsel_coord
        h, w = cls.objsel.shape
        return x, x+h-1, y, y+w-1
    
    # if    objsel inactive and no selection
    # then  we ignore this action
    else:
        return None, None, None, None

def _apply_patch(cls: O2E):
    p = cls.objsel

    patch = np.zeros((cls.H,cls.W), dtype=np.uint8)
    x, y = cls.objsel_coord
    h, w = p.shape
    gh, gw = cls.grid_dim
    
    if x+h<=0 or x>=gh or y+w<=0 or y>=gw: # Perfectly out of bound
        pass
    else:
        
        stx = max(0,x)
        edx = min(gh,x+h)
        
        sty = max(0,y)
        edy = min(gw,y+w)

        if x<0 and x+h>gh:
            p = p[-x:gh-x-h, :]
        elif x<0:
            p = p[-x: , :]
        elif x+h>gh:
            p = p[:gh-x-h, :]
        
        if y<0 and y+w>gw:
            p = p[:, -y:gw-y-w]
        elif y<0:
            p = p[:, -y:]
        elif y+w>gw:
            p = p[:, :gw-y-w]

        patch[stx:edx, sty:edy] = p

    img = np.copy(cls.objsel_bg)
    np.copyto(img, patch, where=(patch!=0))
    return img

def _apply_sel(cls: O2E):
    p = cls.objsel_area
    
    patch = np.zeros((cls.H,cls.W), dtype=np.bool_)
    x, y = cls.objsel_coord
    h, w = p.shape
    gh, gw = cls.grid_dim

    if x+h<=0 or x>=gh or y+w<=0 or y>=gw: # Perfectly out of bound
        pass
    else:
        
        stx = max(0,x)
        edx = min(gh,x+h)
        
        sty = max(0,y)
        edy = min(gw,y+w)

        if x<0 and x+h>gh:
            p = p[-x:gh-x-h, :]
        elif x<0:
            p = p[-x: , :]
        elif x+h>gh:
            p = p[:gh-x-h, :]
        
        if y<0 and y+w>gw:
            p = p[:, -y:gw-y-w]
        elif y<0:
            p = p[:, -y:]
        elif y+w>gw:
            p = p[:, :gw-y-w]

        patch[stx:edx, sty:edy] = p
    
    return patch

def gen_rotate(k=1):
    '''
    Generates Rotate90 / Rotate180 / Rotate270 actions counterclockwise.

    Action Space Requirements (key: type) : (`selection`: NDArray)  

    Class State Requirements (key: type) : (`grid`: NDArray), (`grid_dim`: NDArray), (`selected`: NDArray), (`objsel`: NDArray), (`objsel_bg`: NDArray), (`objsel_active`: Boolean), (`objsel_coord`: Tuple), (`objsel_rot`: Integer)
    '''
    assert 0<k<4

    def Rotate(cls: O2E, action):
        sel = action['selection']
        xmin, xmax, ymin, ymax = _init_objsel(cls,sel)
        if xmin is None:
            return

        if k==2:
            cls.objsel = np.rot90(cls.objsel,k=2)

        elif cls.objsel.shape[0]%2 == cls.objsel.shape[1]%2:
            cx = (xmax + xmin) *0.5
            cy = (ymax + ymin) *0.5
            x,y = cls.objsel_coord
            cls.objsel_coord = ( int(np.floor(cx-cy+y)), int(np.floor(cy-cx+x))) #left-top corner will be diagonally swapped

        else: # ill-posed rotation. Manually setted
            cx = (xmax + xmin) *0.5
            cy = (ymax + ymin) *0.5
            cls.objsel_rot +=k
            sig = (k+2)%4-2
            mod = 1-cls.objsel_rot%2
            mx = min(  cx+sig*(cy-ymin) , cx+sig*(cy-ymax) )+mod
            my = min(  cy-sig*(cx-xmin) , cy-sig*(cx-xmax) )+mod
            cls.objsel_coord = (int(np.floor(mx)),int(np.floor(my)))
        
        cls.objsel = np.rot90(cls.objsel, k=k)
        cls.objsel_area = np.rot90(cls.objsel_area, k=k)
        cls.grid = _apply_patch(cls)
        cls.selected = _apply_sel(cls)

    Rotate.__name__ = f"Rotate_{90*k}"    
    return Rotate

def gen_move(d=0):
    '''
    Generates Move[U,D,R,L] actions. d=0 means move up, d=1 is down, d=2 is right, d=3 is left.

    Action Space Requirements (key: type) : (`selection`: NDArray)  

    Class State Requirements (key: type) : (`grid`: NDArray), (`grid_dim`: NDArray), (`selected`: NDArray), (`objsel`: NDArray), (`objsel_bg`: NDArray), (`objsel_active`: Boolean), (`objsel_coord`: Tuple)
    '''
    assert 0<= d <4
    dirX = [-1, +1, 0, 0]
    dirY = [0, 0, +1, -1]
    def Move(cls: O2E, action):
        sel = action['selection']
        xmin, xmax, ymin, ymax = _init_objsel(cls,sel)

        if xmin is None:
            return

        x, y = cls.objsel_coord
        cls.objsel_coord = (int(x + dirX[d]), int(y + dirY[d]))
        cls.grid = _apply_patch(cls)
        cls.selected = _apply_sel(cls)
        
    Move.__name__ = f"Move_{'UDRL'[d]}"
    return Move

def gen_flip(axis:str = "H"):
    '''
    Generates Flip[H, V, D0, D1] actions. H=Horizontal, V=Vertical, D0=Major diagonal(transpose), D1=Minor diagonal 

    Action Space Requirements (key: type) : (`selection`: NDArray)  

    Class State Requirements (key: type) : (`grid`: NDArray), (`grid_dim`: NDArray), (`selected`: NDArray), (`objsel`: NDArray), (`objsel_bg`: NDArray), (`objsel_active`: Boolean), (`objsel_coord`: Tuple)
    '''
    
    
    flips = {
        "H": lambda x: np.fliplr(x), 
        "V": lambda x: np.flipud(x), 
        "D0": lambda x: np.rot90(np.fliplr(x)), 
        "D1": lambda x: np.fliplr(np.rot90(x))
        }
    assert axis in flips,  "Invalid Axis"
    
    flipfunc = flips[axis]

    def Flip(cls: O2E, action ):
        sel = action['selection']
        xmin, xmax, ymin, ymax = _init_objsel(cls,sel)
        if xmin is None:
            return
        
        cls.objsel = flipfunc(cls.objsel)
        cls.objsel_area = flipfunc(cls.objsel_area)
        cls.grid = _apply_patch(cls)
        cls.selected = _apply_sel(cls)
    
    Flip.__name__ = f"Flip_{axis}"
    return Flip

def gen_copy(source="I"):
    '''
    Generates Copy[I,O] actions. Source is input grid when "I", otherwise "O". It is for O2ARCv2Env. If you want to use generic Copy/Paste, please wait further updates.

    Action Space Requirements (key: type) : (`selection`: NDArray)

    Class State Requirements (key: type) : (`grid`: NDArray), (`grid_dim`: NDArray), (`clip`: NDArray), (`clip_dim`: Tuple)
    '''
    assert source in ["I", "O"], "Invalid Source grid"
    def Copy(cls: O2E, action ):
        sel = action["selection"]
        
        if not np.any(sel>0): #nothing to copy
            return
        
        xmin, xmax, ymin, ymax = _get_bbox(sel)
        H = xmax-xmin+1
        W = ymax-ymin+1
        cls.clip = np.zeros((cls.H,cls.W),dtype=np.uint8)
        cls.clip_dim = (H,W)

        if source == "I":
            np.copyto(cls.clip[:H, :W], cls.input_[xmin:xmin+H, ymin:ymin+W], where=np.logical_and(cls.input_[xmin:xmin+H, ymin:ymin+W]>0,sel[xmin:xmin+H, ymin:ymin+W] ))
        elif source == "O":
            np.copyto(cls.clip[:H, :W], cls.grid[xmin:xmin+H, ymin:ymin+W], where=np.logical_and(cls.grid[xmin:xmin+H, ymin:ymin+W]>0, sel[xmin:xmin+H, ymin:ymin+W]))
        
    Copy.__name__ = f"Copy_{source}"
    return Copy

def Paste(cls: O2E, action):
    '''
    Paste action. If you want to use generic Copy/Paste, please wait further updates.

    Action Space Requirements (key: type) : (`selection`: NDArray)

    Class State Requirements (key: type) :  (`grid`: NDArray), (`grid_dim`: NDArray), (`clip`: NDArray), (`clip_dim`: Tuple)
    '''
    sel = action["selection"]
    if not np.any(sel>0) : # no location specified
        return
    xmin, _, ymin, _ = _get_bbox(sel)

    if xmin >= cls.H or ymin >= cls.W: # out of bound
        return
    
    H,W = cls.clip_dim

    if H==0 or W==0: # No selection
        return 
    
    edx = min(xmin+H, cls.H)
    edy = min(ymin+W, cls.W)
    np.copyto(cls.grid[xmin:edx, ymin:edy], cls.clip[:edx-xmin, :edy-ymin], where=cls.clip[:edx-xmin, :edy-ymin]>0)