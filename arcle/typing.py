from typing import List, Union, Tuple, Callable, OrderedDict, Any, Dict
from numpy.typing import NDArray
import numpy as np


GridType = NDArray[np.int8]
DimType = Union[NDArray[np.uint8], Tuple[int, int]]

ActionType = Dict[str, Any]
StateType = Dict[str, Any]

TaskType = Tuple[
    List[GridType], List[GridType], List[GridType], List[GridType], Dict[str, Any]
]


Operation = Callable[[StateType, ActionType], None]
