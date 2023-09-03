import json
import numpy as np
import glob,os
from abc import ABCMeta,abstractmethod
from typing import (Union, Any, Tuple, List, Dict, TypeAlias)
from numpy.typing import NDArray

class Loader(metaclass=ABCMeta):
    '''
    Abstract Class of ARC-like problem loader.

    __init__ calls self.get_path() and self.parse(). You should implement these functions to work properly.

    '''
    data_path = None
    _pathlist = []

    def __init__(self, **kwargs) -> None:

        self._pathlist = self.get_path(**kwargs)
        self.data = self.parse(**kwargs)
        #self._train_input, self._train_output, self._test_input, self._test_output, self._problem_info = 

    @abstractmethod
    def get_path(self, **kwargs) -> List[str]:
        '''
        Returns list of paths of ARC-like dataset.
        '''
        pass
        

    @abstractmethod
    def parse(self, **kwargs) -> List[Tuple[List[NDArray],List[NDArray],List[NDArray],List[NDArray],Dict]]:
        '''
        Parses dataset from loaded pathlist by get_path.

        Returns list of 5-tuple of lists, (train_inputs, train_outputs, test_inputs, test_outputs, problem_description).
        '''
        pass
        

    def pick(self, data_index: int | None = None, **kwargs) -> Tuple[List[NDArray],List[NDArray],List[NDArray],List[NDArray],Dict]:
        '''
        Pick one problem from loaded data.
        '''
        assert self.data is not None and len(self.data)>0, "Dataset wasn't loaded properly"

        sel = data_index
        max_index = len(self.data)
        if data_index is None:
            sel = np.random.randint(0,max_index)

        assert 0 <= sel < max_index, f'Problem indices should be in [0, {max_index}).'
        
        return self.data[sel]


class ARCLoader(Loader):
    '''
    Original ARC Loader.
    '''


    def __init__(self, train=True) -> None:
        '''
        train: load ARC problems from data/training folder. If false, it loads from data/evaluation folder.
        '''
        super().__init__(train=train)
    
    def get_path(self, **kwargs):

        path = os.path.join(os.path.dirname(__file__),'../arcs/ARC/data')
        train = kwargs['train']
        self.train = train

        if train:
            path = os.path.join(path,'training')
        else:
            path = os.path.join(path,'evaluation')

        self.data_path = path

        path = os.path.join(path,'*.json')
        pathlist = glob.glob(path)
        
        pathlist.sort()
        return pathlist

    def parse(self, **kwargs):
        
        dat = []

        for p in self._pathlist:
            with open(p) as fp:
                problem = json.load(fp)

                ti: List[NDArray] = []
                to: List[NDArray] = []
                ei: List[NDArray] = []
                eo: List[NDArray] = []

                for d in problem['train']:
                    ti.append(np.array(d['input'],dtype=np.uint8))
                    to.append(np.array(d['output'],dtype=np.uint8))

                for d in problem['test']:
                    ei.append(np.array(d['input'],dtype=np.uint8))
                    eo.append(np.array(d['output'],dtype=np.uint8))

                desc = {'id': os.path.basename(fp.name).split('.')[0]}
                dat.append((ti,to,ei,eo,desc))
                
        return dat
    

class MiniARCLoader(Loader):

    def __init__(self) -> None:
        super().__init__()
    
    def get_path(self, **kwargs):

        path = os.path.join(os.path.dirname(__file__),'../arcs/Mini-ARC/data/MiniARC')

        self.data_path = path

        path = os.path.join(path,'*.json')
        pathlist = glob.glob(path)
        pathlist.sort(key=lambda fn : fn.split('_')[-1])

        return pathlist

    def parse(self, **kwargs):
        
        dat = []

        for p in self._pathlist:
            with open(p) as fp:
                fpdata = fp.read().replace('null', '"0"')
                problem = json.loads(fpdata)

                ti: List[NDArray] = []
                to: List[NDArray] = []
                ei: List[NDArray] = []
                eo: List[NDArray] = []

                for d in problem['train']:
                    ti.append(np.array(d['input'],dtype=np.uint8))
                    to.append(np.array(d['output'],dtype=np.uint8))
                
                for d in problem['test']:
                    ei.append(np.array(d['input'],dtype=np.uint8))
                    eo.append(np.array(d['output'],dtype=np.uint8))

                fns = os.path.basename(fp.name).split('_')
                desc = {'id': fns[-1].split('.')[-2], 'description': ' '.join(fns[0:-1]).strip() }

                dat.append((ti,to,ei,eo,desc))
                
        return dat