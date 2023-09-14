
import numpy as np
from numpy import ma
from ..envs import AbstractARCEnv as AAE
from numpy.typing import NDArray
from typing import SupportsInt,Callable,Tuple

def dfs(grid: NDArray, grid_dim: Tuple, point: Tuple) -> NDArray:
    
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    xmax, ymax = grid_dim
    visit_mask = np.zeros_like(grid)
    col = grid[point]
    
    def _dfs(x, y):
        if not visit_mask[x,y]:
            visit_mask[x,y] = 1
            
            for i in range(4):
                xnew, ynew = x+dx[i], y+dy[i]
                if ( 0 <= xnew < xmax ) and (0 <= ynew < ymax) and (grid[xnew,ynew]==col) and (not visit_mask[xnew,ynew]):
                    _dfs(xnew,ynew)
                else:
                    continue

        else:
            return
    _dfs(*point)
    return visit_mask

def dfs_area(grid: NDArray, grid_dim: Tuple, selection: NDArray) -> NDArray:
    
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    xmax, ymax = grid_dim
    visit_mask = np.zeros_like(grid)
    
    
    def _dfs(x, y, c):
        if not visit_mask[x,y]:
            visit_mask[x,y] = 1
            
            for i in range(4):
                xnew, ynew = x+dx[i], y+dy[i]
                if ( 0 <= xnew < xmax ) and (0 <= ynew < ymax) and (grid[xnew,ynew]==c) and (not visit_mask[xnew,ynew]):
                    _dfs(xnew,ynew,c)
                else:
                    continue
        else:
            return

    for i in range(selection.shape[0]):
        for j in range(selection.shape[1]):
            if selection[i,j]:
                col = grid[i,j]
                _dfs(i,j,col)

    return visit_mask


def gen_color(color: SupportsInt) -> Callable:
    '''
    Generates Color0 ~ Color9 functions that color multiple pixels within selection.
    
    Action Space Requirements (key: type) : (`selection`: NDArray)

    Class State Requirements (key: type) : (`grid`: NDArray)
    '''
    def colorf(cls: AAE, action) -> None:
        sel = action['selection']
        if not np.any(sel):
            return
        cls.grid = ma.array(cls.grid, mask=sel).filled(fill_value=color)
    
    colorf.__name__ = f"Color{color}"
    return colorf

def gen_flood_fill(color: SupportsInt) -> Callable:
    '''
    Generates FloodFill0 ~ FloodFill9 operations. Shortcut of `dfs_area` + `Color0`~`Color9`.
    
    Action Space Requirements (key: type) : (`selection`: NDArray)  

    Class State Requirements (key: type) : (`grid`: NDArray), (`grid_dim`: NDArray)
    '''

    def floodfillf(cls: AAE, action):
        sel = action['selection']
        if np.sum(sel)>1:
            return # NOOP if two or more pixel selected
        
        x,y = np.unravel_index(np.argmax(sel),shape=sel.shape)

        if x>=cls.grid_dim[0] or y>=cls.grid_dim[1]:
            return # NOOP outofbound
        
        sel = dfs(cls.grid,cls.grid_dim,(x,y))
        cls.grid = ma.array(cls.grid, mask=sel).filled(fill_value=color)

    floodfillf.__name__ = f"FloodFill{color}"
    return floodfillf