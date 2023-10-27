
import numpy as np
from numpy import ma
#from ..envs import AbstractARCEnv as AAE
from ..envs import O2ARCv2Env as O2E
from numpy.typing import NDArray
from typing import SupportsInt,Callable,Tuple
from functools import wraps

def _pad(source: NDArray, H, W):
    h, w = source.shape
    return np.pad(source, ((0,H-h),(0,W-w)),constant_values=0)

def reset_sel(function):
    '''
    Wrapper for Non-O2ARC actions. This wrapper resets `selected` of obs space and resets object-operation states.

    It does this before calling function.
    ```
        state['selected'] = np.zeros((H,W), dtype=np.uint8)
        state['object_states']['active'] = False
    ```
    '''
    @wraps(function)
    def wrapper(state, action, **kwargs):
        state['selected'] = np.zeros(state['input'].shape, dtype=np.uint8)
        state['object_states']['active'] = False
        
        return function(state, action, **kwargs)
    return wrapper

def keep_sel(function):
    '''
    Wrapper for Non-O2ARC actions. This wrapper keeps `selection` of action in `selected` of obs space.

    It does this before calling function.
    ```
        state['selected'] = np.copy(action["selection"])
    ```
    '''
    @wraps(function)
    def wrapper(state, action, **kwargs):
        state['selected'] = np.copy(action["selection"])
        return function(state, action, **kwargs)
    return wrapper

def _get_bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def _init_objsel(state, sel: NDArray):

    objdict = state['object_states']
    H, W = state['input'].shape
    # if    something newly selected
    # then  previous object selection will be wiped
    if np.any(sel>0): 
        # separate
        xmin, xmax, ymin, ymax = _get_bbox(sel)
        # backup
        h = xmax-xmin+1
        w = ymax-ymin+1

        objdict['object'][:, :] = 0
        np.copyto(objdict['object'][:h,:w], state['grid'][xmin:xmax+1, ymin:ymax+1] * sel[xmin:xmax+1, ymin:ymax+1])
        objdict['object_dim'] = (h, w)
        objdict['object_sel'][:,:] = 0
        np.copyto(objdict['object_sel'][:h,:w], sel[xmin:xmax+1, ymin:ymax+1])

        objdict['background'] = np.copy(state['grid'])
        np.copyto(dst=objdict['background'], src=0, where=sel)

        objdict['object_pos'] = (int(xmin), int(ymin)) 
        objdict['active'] = 1
        objdict['rotation_parity'] = 0

        state['selected'] = np.copy(sel)

        return xmin, xmax, ymin, ymax
    
    # if    objsel active and no selection
    # then  continue with prev objsel
    elif objdict['active']: 
        x, y = objdict['object_pos']
        h, w = objdict['object_dim']
        return x, x+h-1, y, y+w-1
    
    # if    objsel inactive and no selection
    # then  we ignore this action
    else:
        return None, None, None, None

def _apply_patch(state):
    objdict = state['object_states']
    p = objdict['object']
    H, W = state['input'].shape

    patch = np.zeros((H, W), dtype=np.uint8)
    x, y = objdict['object_pos']
    h, w = objdict['object_dim']
    gh, gw = state['grid_dim']
    p = p[:h, :w]
    
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

    img = np.copy(objdict['background'])
    np.copyto(img, patch, where=(patch!=0))
    return img

def _apply_sel(state):
    objdict = state['object_states']
    p = objdict['object_sel']
    H, W = state['input'].shape
    patch = np.zeros((H,W), dtype=np.bool_)
    x, y = objdict['object_pos']
    h, w = objdict['object_dim']
    gh, gw = state['grid_dim']
    p = p[:h, :w]
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

    def Rotate(state, action):
        
        sel = action['selection']
        xmin, xmax, ymin, ymax = _init_objsel(state,sel)
        if xmin is None:
            return
        objdict = state['object_states']
        h, w = objdict['object_dim']
        H, W = state['input'].shape

        if k%2 ==0:
            pass

        elif h%2 == w%2:
            cx = (xmax + xmin) *0.5
            cy = (ymax + ymin) *0.5
            x,y = objdict['object_pos']
            objdict['object_pos'] = ( int(np.floor(cx-cy+y)), int(np.floor(cy-cx+x))) #left-top corner will be diagonally swapped
            objdict['object_dim'] = (w,h)
            

        else: # ill-posed rotation. Manually setted
            cx = (xmax + xmin) *0.5
            cy = (ymax + ymin) *0.5
            objdict['rotation_parity'] +=k
            objdict['rotation_parity'] %=2
            sig = (k+2)%4-2
            mod = 1-objdict['rotation_parity']
            mx = min(  cx+sig*(cy-ymin) , cx+sig*(cy-ymax) )+mod
            my = min(  cy-sig*(cx-xmin) , cy-sig*(cx-xmax) )+mod
            objdict['object_pos'] = (int(np.floor(mx)),int(np.floor(my)))
            objdict['object_dim'] = (w,h)
            
        
        objdict['object'] = _pad(np.rot90(objdict['object'][:h,:w],k=k), H, W)
        objdict['object_sel'] = _pad(np.rot90(objdict['object_sel'][:h,:w],k=k), H, W)
        state['grid'] = _apply_patch(state)
        state['selected'] = _apply_sel(state)

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
    def Move(state, action):
        sel = action['selection']
        xmin, xmax, ymin, ymax = _init_objsel(state,sel)

        if xmin is None:
            return

        x, y = state['object_states']['object_pos']
        state['object_states']['object_pos'] = (int(x + dirX[d]), int(y + dirY[d]))
        state['grid'] = _apply_patch(state)
        state['selected'] = _apply_sel(state)
        
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

    def Flip(state, action ):
        sel = action['selection']
        xmin, xmax, ymin, ymax = _init_objsel(state,sel)
        if xmin is None:
            return
        objdict = state['object_states']
        h, w = objdict['object_dim']
        H, W = state['input'].shape
        objdict['object'] = _pad(flipfunc(objdict['object'][:h,:w]), H, W)
        objdict['object_sel'] = _pad(flipfunc(objdict['object_sel'][:h,:w]), H, W)
        state['grid'] = _apply_patch(state)
        state['selected'] = _apply_sel(state)
    
    Flip.__name__ = f"Flip_{axis}"
    return Flip

def gen_copy(source="I"):
    '''
    Generates Copy[I,O] actions. Source is input grid when "I", otherwise "O". It is for O2ARCv2Env. If you want to use generic Copy/Paste, please wait further updates.

    Action Space Requirements (key: type) : (`selection`: NDArray)

    Class State Requirements (key: type) : (`grid`: NDArray), (`grid_dim`: NDArray), (`clip`: NDArray), (`clip_dim`: Tuple)
    '''
    assert source in ["I", "O"], "Invalid Source grid"
    def Copy(state, action ):
        sel = action["selection"]
        
        if not np.any(sel>0): #nothing to copy
            return
        
        xmin, xmax, ymin, ymax = _get_bbox(sel)
        h = xmax-xmin+1
        w = ymax-ymin+1
        H, W = state['input'].shape
        state['clip'] = np.zeros((H,W),dtype=np.uint8)
        state['clip_dim'] = (h,w)

        if source == "I":
            np.copyto(state['clip'][:h, :w], state['input'][xmin:xmin+h, ymin:ymin+w], where=np.logical_and(state['input'][xmin:xmin+h, ymin:ymin+w]>0,sel[xmin:xmin+h, ymin:ymin+w] ))
        elif source == "O":
            np.copyto(state['clip'][:h, :w], state['grid'][xmin:xmin+h, ymin:ymin+w], where=np.logical_and(state['grid'][xmin:xmin+h, ymin:ymin+w]>0, sel[xmin:xmin+h, ymin:ymin+w]))
        
    Copy.__name__ = f"Copy_{source}"
    return Copy

def Paste(state, action):
    '''
    Paste action. If you want to use generic Copy/Paste, please wait further updates.

    Action Space Requirements (key: type) : (`selection`: NDArray)

    Class State Requirements (key: type) :  (`grid`: NDArray), (`grid_dim`: NDArray), (`clip`: NDArray), (`clip_dim`: Tuple)
    '''
    sel = action["selection"]
    if not np.any(sel>0) : # no location specified
        return
    xmin, _, ymin, _ = _get_bbox(sel)
    H, W = state['input'].shape
    if xmin >= H or ymin >= W: # out of bound
        return
    
    h,w = state['clip_dim']

    if h==0 or w==0: # No selection
        return 
    
    edx = min(xmin+h, H)
    edy = min(ymin+w, W)
    #np.copyto(state['grid'][xmin:edx, ymin:edy], state['clip'][:edx-xmin, :edy-ymin], where=state['clip'][:edx-xmin, :edy-ymin]>0)
    np.copyto(state['grid'][xmin:edx, ymin:edy], state['clip'][:edx-xmin, :edy-ymin]) # Comment if testing