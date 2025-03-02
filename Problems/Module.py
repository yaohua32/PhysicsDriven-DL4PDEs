# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-07-18 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-07-18
#  */
import numpy as np
import torch

class Problem():

    def __init__(self, dataType:np.dtype=np.float32):
        self._dtype = dataType

    @property
    def name(self):
        return 'Problem_Module'
    
    @property
    def _lb(self):
        '''The lower bound 
        '''
        raise NotImplementedError
    
    @property
    def _ub(self):
        '''The upper bound
        '''
        raise NotImplementedError