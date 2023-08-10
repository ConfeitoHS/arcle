import json
import numpy as np
import glob,os
from abc import ABCMeta,abstractmethod
from typing import (Union, Any, Tuple, List)

class Loader(metaclass=ABCMeta):
    
    data_path = None
    _pathlist = []

    def __init__(self, **kwargs) -> None:

        self._pathlist = self.get_path(**kwargs)
        self._train_input, self._train_output, self._test_input, self._test_output, self._problem_info = self.parse(**kwargs)

    @abstractmethod
    def get_path(self, **kwargs) -> List[str]:
        pass
        

    @abstractmethod
    def parse(self, **kwargs) -> Tuple[List,List,List,List,List]:
        pass
        

    def pick(self, data_index: int = -1) -> Tuple[List[np.ndarray],List[np.ndarray],List[np.ndarray],List[np.ndarray]]:

        assert self._train_input is not None and len(self._train_input)>0, "Dataset wasn't loaded properly"

        sel = data_index
        max_index = len(self._pathlist)
        if data_index < 0:
            sel = np.random.randint(0,max_index)

        assert 0 <= sel < max_index, f'Problem indices should be in [0, {max_index}).'
        
        return self._train_input[sel],self._train_output[sel], self._test_input[sel], self._test_output[sel], self._problem_info[sel]


class ARCLoader(Loader):

    def __init__(self, train=True) -> None:
        super().__init__(train=train)
    
    def get_path(self, **kwargs) -> List[str]:

        path = os.path.join(os.path.dirname(__file__),'ARC/data')
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

    def parse(self, **kwargs) -> Tuple[List, List, List, List, List]:
        
        train_input = []
        train_output = []
        test_input = []
        test_output = []
        prob_data = []

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

                prob_data.append({'id': os.path.basename(fp.name).split('.')[0]})
                
        return train_input, train_output, test_input, test_output, prob_data
    

class MiniARCLoader(Loader):

    def __init__(self) -> None:
        super().__init__()
    
    def get_path(self, **kwargs) -> List[str]:

        path = os.path.join(os.path.dirname(__file__),'Mini-ARC/data/MiniARC')

        self.data_path = path

        path = os.path.join(path,'*.json')
        pathlist = glob.glob(path)
        pathlist.sort(key=lambda fn : fn.split('_')[-1])

        return pathlist

    def parse(self, **kwargs) -> Tuple[List, List, List, List, List]:
        
        train_input = []
        train_output = []
        test_input = []
        test_output = []
        prob_data = []

        for p in self._pathlist:
            with open(p) as fp:
                fpdata = fp.read()
                fpdata = fpdata.replace('null', '"0"')
                problem = json.loads(fpdata)

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
                fns = os.path.basename(fp.name).split('_')
                prob_data.append({'id': fns[-1].split('.')[-1], 'description': ' '.join(fns[0:-1]).strip() })
                
        return train_input, train_output, test_input, test_output, prob_data