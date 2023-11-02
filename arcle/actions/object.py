
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

def _pad_assign(dst: NDArray, src: NDArray):
    h, w = src.shape
    dst[:h, :w] = src
    dst[h:,:] = 0
    dst[:, w:] = 0
    
def _get_bbox(img: NDArray) -> Tuple[int,int,int,int]:
    '''
    Receives NDArray, returns bounding box of its truthy values.
    '''

    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def _init_objsel(state: dict, selection: NDArray) -> Tuple[int,int,int,int]:
    '''
    Initialize object selection states for smooth object-oriented actions
    '''

    objdict = state['object_states']
    sel = selection
    # when something newly selected, previous object selection will be wiped
    if np.any(sel): 
        
        xmin, xmax, ymin, ymax = _get_bbox(sel) # bounding box of selection
        
        h = xmax-xmin+1
        w = ymax-ymin+1

        
        # wipe object and set a new object from selected pixels
        objdict['object_dim'] = (h, w)
        selected_part = sel[xmin:xmax+1, ymin:ymax+1]

        objdict['object'][:, :] = 0 
        np.copyto(objdict['object'][0:h,0:w], state['grid'][xmin:xmax+1, ymin:ymax+1], where=selected_part)
        
        objdict['object_sel'][:,:] = 0
        np.copyto(objdict['object_sel'][0:h,0:w], selected_part, where=selected_part)
        
        # background backup
        np.copyto(objdict['background'], state['grid'])
        np.copyto(objdict['background'], 0, where=sel)

        # position, active, parity initialize
        objdict['object_pos'] = (int(xmin), int(ymin)) 
        objdict['active'] = 1
        objdict['rotation_parity'] = 0

        # copy selection into selected obs
        np.copyto(state['selected'], np.copy(sel))

        # return bounding box of selection
        return xmin, xmax, ymin, ymax
    

    # when object selection was active without new selection, continue with prev objsel
    elif objdict['active']: 
        # gives previous bounding pox
        x, y = objdict['object_pos']
        h, w = objdict['object_dim']
        return x, x+h-1, y, y+w-1
    
    # when objsel inactive and no selection, we ignore this action
    else:
        return None, None, None, None

def _apply_patch(state: dict):
    '''
    Combine 'background' and 'object' at 'object_pos', and put it into 'grid'.
    '''
    objdict = state['object_states']
    p: NDArray = objdict['object']

    x, y = objdict['object_pos']
    h, w = objdict['object_dim']
    gh, gw = state['grid_dim']
    p = p[:h, :w]

    # copy background
    np.copyto(state['grid'], objdict['background'])
    if   x+h>0 and x<gh and y+w>0 and y<gw  :
        # if patch is inside of the grid
        
        # patch paste bounding box
        stx = max(0,x)
        edx = min(gh,x+h)
        sty = max(0,y)
        edy = min(gw,y+w)
        
        # truncate patch
        p = p[ stx-x : (gh-x-h if x+h>gh else gh), sty-y : (gw-y-w if y+w>gw else gw) ]
        np.copyto(state['grid'][stx:edx, sty:edy], p, where=(p>0))

def _apply_sel(state):
    '''
    Place the 'object_sel' into 'selected', at 'object_pos'
    '''
    objdict = state['object_states']
    p: NDArray = objdict['object_sel']

    x, y = objdict['object_pos']
    h, w = objdict['object_dim']
    gh, gw = state['grid_dim']
    p = p[:h, :w]

    # copy background
    state['selected'][:,:] = 0
    if   x+h>0 and x<gh and y+w>0 and y<gw  :
        # if patch is inside of the grid
        
        # patch paste bounding box
        stx = max(0,x)
        edx = min(gh,x+h)
        sty = max(0,y)
        edy = min(gw,y+w)
        
        # truncate patch
        p = p[ stx-x : (gh-x-h if x+h>gh else gh), sty-y : (gw-y-w if y+w>gw else gw) ]
        np.copyto(state['selected'][stx:edx, sty:edy], p)

def gen_rotate(k=1):
    '''
    Generates Rotate90 / Rotate180 / Rotate270 actions counterclockwise.

    Action Space Requirements (key: type) : (`selection`: NDArray)  

    State Requirements (key: type) : (`grid`: NDArray), (`grid_dim`: NDArray), (`selected`: NDArray), (`objsel`: NDArray), (`objsel_bg`: NDArray), (`objsel_active`: Boolean), (`objsel_coord`: Tuple), (`objsel_rot`: Integer)
    '''
    assert 0<k<4

    def Rotate(state, action):

        xmin, xmax, ymin, ymax = _init_objsel(state,action['selection'])
        if xmin is None:
            return
        
        objdict = state['object_states']
        h, w = objdict['object_dim']

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
            
        
        _pad_assign(objdict['object'], np.rot90(objdict['object'][:h,:w],k=k))
        _pad_assign(objdict['object_sel'], np.rot90(objdict['object_sel'][:h,:w],k=k))
        _apply_patch(state)
        _apply_sel(state)

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
        par,_,_,_ = _init_objsel(state,sel)

        if par is None:
            return

        x, y = state['object_states']['object_pos']
        state['object_states']['object_pos'] = (int(x + dirX[d]), int(y + dirY[d]))
        _apply_patch(state)
        _apply_sel(state)
        
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

    def Flip(state, action):
        sel = action['selection']
        valid,_,_,_ = _init_objsel(state,sel)
        if valid is None:
            return
        objdict = state['object_states']
        h, w = objdict['object_dim']

        _pad_assign(objdict['object'],flipfunc(objdict['object'][:h,:w]))
        _pad_assign(objdict['object_sel'],flipfunc(objdict['object_sel'][:h,:w]))
        _apply_patch(state)
        _apply_sel(state)
    
    Flip.__name__ = f"Flip_{axis}"
    return Flip

def gen_copy(source="I"):
    '''
    Generates Copy[I,O] actions. Source is input grid when "I", otherwise "O". It is for O2ARCv2Env. If you want to use generic Copy/Paste, please wait further updates.

    Action Space Requirements (key: type) : (`selection`: NDArray)

    Class State Requirements (key: type) : (`grid`: NDArray), (`grid_dim`: NDArray), (`clip`: NDArray), (`clip_dim`: Tuple)
    '''
    assert source in ["I", "O"], "Invalid Source grid"
    srckey = 'input' if source=="I" else 'grid'
    def Copy(state, action):
        sel = action["selection"]
        
        if not np.any(sel>0): #nothing to copy
            return
        
        xmin, xmax, ymin, ymax = _get_bbox(sel)

        ss_h, ss_w = state[srckey+'_dim']

        if xmax>=ss_h or ymax>ss_w: # out of bound
            return
        
        h = xmax-xmin+1
        w = ymax-ymin+1

        state['clip'][:,:] = 0
        state['clip_dim'] = (h,w)

        src_grid = state[srckey][xmin:xmax+1, ymin:ymax+1]
        np.copyto(state['clip'][:h, :w], src_grid, \
                      where=np.logical_and(src_grid, sel[xmin:xmax+1, ymin:ymax+1] ))
    Copy.__name__ = f"Copy_{source}"
    return Copy

def gen_paste(paste_blank = False):
    def Paste(state, action):
        '''
        Paste action.

        Action Space Requirements (key: type) : (`selection`: NDArray)

        Class State Requirements (key: type) :  (`grid`: NDArray), (`grid_dim`: NDArray), (`clip`: NDArray), (`clip_dim`: Tuple)
        '''
        sel = action["selection"]
        if not np.any(sel>0) : # no location specified
            return
        
        xmin, _, ymin, _ = _get_bbox(sel)

        H, W = state['input'].shape
        h, w = state['clip_dim']

        if xmin >= H or ymin >= W or h==0 or w==0: # out of bound or no selection
            return
        
        patch = state['clip'][:h, :w]
        
        # truncate patch
        edx = min(xmin+h, H)
        edy = min(ymin+w, W)
        patch = patch[:edx-xmin, :edy-ymin]
        
        # paste
        if paste_blank:
            np.copyto(state['grid'][xmin:edx, ymin:edy], patch) # for debug'
        else:
            np.copyto(state['grid'][xmin:edx, ymin:edy], patch, where=(patch>0))
    return Paste