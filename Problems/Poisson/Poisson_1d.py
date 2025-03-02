# /*
#  * @Author: yaohua.zang 
#  * @Date: 2025-01-31 23:12:51 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2025-01-31 23:12:51 
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

    def __init__(self, freq:float, dataType=torch.float32):
        self._dtype = dataType
        # Parameters in the PDE
        self.freq = freq
        # The defined domain
        self.lb = [-1.]
        self.ub = [1.]

    @property
    def name(self):
        return 'poisson_1d'

    @property
    def _lb(self):
        '''The lower bound '''
        return self.lb 
    
    @property
    def _ub(self):
        '''The upper bound '''
        return self.ub

    def _u_star(self, x:torch.tensor):
        '''The truth solution of u '''
        u = torch.sin(self.freq * x[:,0:1])

        return u
     
    def fun_f(self, x:torch.tensor)->torch.tensor:
        '''
        The right hand side of PDE (or the source term)
        Input:  
            x: size(?,d)
        Output: 
            f: size(?,1)
        '''
        f = self.freq**2 * torch.sin(self.freq*x)

        return f

    def _test(self):
        '''Check the setups of the problem'''
        from Utils.GenPoints import Point1D
        pointGen = Point1D(x_lb=self._lb, x_ub=self._ub)
        #
        x_in = pointGen.inner_point(num_sample_or_mesh=100, method='mesh')
        ############ Check the fun_f
        f = self.fun_f(x_in)
        u = self._u_star(x_in)
        Plot.show_1d_list([x_in]*2, [f,u], label_list=['f', 'u'])
    
    def _check_strong(self):
        '''check the strong form'''
        from torch.autograd import Variable, grad
        from Utils.GenPoints import Point1D
        pointGen = Point1D(x_lb=self._lb, x_ub=self._ub)
        #
        x_in = pointGen.inner_point(num_sample_or_mesh=5, method='uniform')
        x = Variable(x_in, requires_grad=True)
        #
        u = self._u_star(x)
        f = self.fun_f(x)
        #
        du = grad(inputs=x, outputs=u, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        ddu = grad(inputs=x, outputs=du, grad_outputs=torch.ones_like(du), create_graph=True)[0]
        #
        left = - ddu
        right = f 
        # print(left)
        # print(right)
        print(torch.abs(left-right))
    
    def _check_weak(self):
        '''check the weak form'''
        from torch.autograd import Variable, grad
        from Utils.GenPoints import Point1D
        from Utils.TestFun_ParticleWNN import TestFun_ParticleWNN
        pointGen = Point1D(x_lb=self._lb, x_ub=self._ub)
        int_grid, phi, dphi_dr, Lphi_dr = TestFun_ParticleWNN(
            fun_type='Wendland', dim=1, n_mesh_or_grid=25, 
            dataType=torch.float32, retrun_Lv=True).get_testFun()
        n_grid = int_grid.shape[0]
        #
        xc, R = pointGen.weight_centers(n_center=5, R_max=1e-4, R_min=1e-4)
        x = xc + int_grid*R
        x = Variable(x.reshape(-1, 1), requires_grad=True)
        ## size(n_grid, 1) -> (nc, n_grid, 1) -> (nc*n_grid, 1)
        phi = phi.repeat((5,1,1)).reshape(-1,1)
        # size(nc, n_grid, 1) -> (nc*n_grid, 1)
        dphi = (dphi_dr / R).reshape(-1, 1)
        # #
        u = self._u_star(x)
        f = self.fun_f(x)
        du = grad(inputs=x, outputs=u, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        # # The first weak form: <kdu, dv> = <f, v>
        left = torch.sum(du*dphi, dim=-1).reshape(5, n_grid)
        left = torch.mean(left, dim=-1)
        right = (f*phi).reshape(5, n_grid)
        right = torch.mean(right, dim=-1)
        print(right)
        print(torch.abs(left-right))
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        # # The second weak form: - <u, ddv> = <f, v>
        Lphi = (Lphi_dr / R).reshape(-1, 1)
        left = - torch.sum(u*Lphi, dim=-1).reshape(5, n_grid) 
        left = torch.mean(left, dim=-1)
        right = (f*phi).reshape(5, n_grid)
        right = torch.mean(right, dim=-1)
        print('####')
        print(right)
        print(left)
        print(torch.abs(left-right))

if __name__=='__main__':
    demo = Problem(freq=np.pi)
    # demo._test()
    # demo._check_strong()
    # demo._check_weak()