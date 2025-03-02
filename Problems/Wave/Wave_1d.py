# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-09-30 00:00:38 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-09-30 00:00:38 
#  */
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
#
import numpy as np
import torch 
#
from Problems.Module import Problem
from Utils.PlotFigure import Plot

class Problem(Problem):

    def __init__(self, freq:float=np.pi, 
                 dataType=torch.float32):
        '''u_tt - u_xx = 0
                   u(t=0, x) = f(x)
                   u(t=T, x) = 0
            The truth is:
             u(x,t) = 0.5 * (f(x+t) + f(x-t))
        '''
        self._dtype = dataType
        # Parameters in the PDE
        self._freq = freq
        # The defined domain
        self.lb = [-1.]
        self.ub = [1.]

    @property
    def name(self):
        return 'wave_1d'

    @property
    def _lb(self):
        '''The lower bound '''
        return self.lb 
    
    @property
    def _ub(self):
        '''The upper bound '''
        return self.ub

    def _f_init(self, x:torch.tensor):
        '''The initial function f(x)'''
        f = torch.sin(self._freq * x)

        return f

    def _u_star(self, xt:torch.tensor):
        '''The truth solution of u'''
        x, t = xt[...,0:1], xt[...,1:2]
        u = 0.5 * (self._f_init(x+t)+self._f_init(x-t))

        return u

    def _test(self):
        '''Check the setups of the problem'''
        from Utils.GenPoints_Time import Point1D
        pointGen = Point1D(x_lb=self._lb, x_ub=self._ub, random_seed=1234)
        ########### Check the solution u
        xt_in = pointGen.inner_point(nx=20, nt=50, method='mesh', tT=[5.])
        u = self._u_star(xt_in)
        Plot.show_1dt(xt_in, u, title='u True', tT=5., lb=-1., ub=1.)
        ########### Check the solution u
        x_mesh = pointGen.integral_grid(n_mesh_or_grid=100)
        f = self._f_init(x_mesh)
        Plot.show_1d_list(x_mesh, [f], ['f'])   # show v vs. dv vs. Lv

if __name__=='__main__':
    demo = Problem()
    demo._test()
