import json
import numpy as np
import glob,os
from typing import (Union, Any,Tuple,List)

class ARCParser:
    
    data_path = None
    _internal_path = None
    _pathlist = []

    def __init__(self, arc_data : str = 'arc') -> None:
        
        match arc_data:
            case 'arc':
                self.data_path = os.path.join(os.path.dirname(__file__),'ARC/data')
            case _:
                raise ValueError(f'No dataset named {arc_data}.')
    
    def load_ARC(self, train: bool = True) -> None:
        path = self.data_path
        
        if train:
            path = os.path.join(path,'training')
        else:
            path = os.path.join(path,'evaluation')
        path = os.path.join(path,'*.json')
        self._internal_path = path 
        self._pathlist = glob.glob(path)
        self._pathlist.sort()

        train_input = []
        train_output = []
        test_input = []
        test_output = []

        for p in self._pathlist:
            with open(p) as fp:
                problem = json.load(fp)

                ti = []
                to = []
                for d in problem['train']:
                    ti.append(np.array(d['input'],dtype=np.uint8))
                    to.append(np.array(d['output'],dtype=np.uint8))
                train_input.append(ti)
                train_output.append(to)

                ti = []
                to = []
                for d in problem['test']:
                    ti.append(np.array(d['input'],dtype=np.uint8))
                    to.append(np.array(d['output'],dtype=np.uint8))
                test_input.append(ti)
                test_output.append(to)
        
        self._train_input = train_input 
        self._train_output = train_output
        self._test_input = test_input 
        self._test_output = test_output 

    def pick_ARC(self, data_index: int = -1) -> Tuple[List[np.ndarray],List[np.ndarray],List[np.ndarray],List[np.ndarray]]:

        assert self._train_input is not None, 'Call load_ARC first.'

        sel = data_index
        max_index = len(self._pathlist)
        if data_index < 0:
            sel = np.random.randint(0,max_index)

        assert 0 <= sel < max_index, f'Problem indices should be in [0, {max_index}).'
        
        return self._train_input[sel],self._train_output[sel], self._test_input[sel], self._test_output[sel]
