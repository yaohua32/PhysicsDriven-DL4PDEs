# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-08-22 14:17:35 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-08-22 14:17:35 
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
        self._k_constant = 0.1
        self._k_scale = 1.
        self._k_center = np.array([[0.2, 0.2]])
        self._k_sigma = np.array([[0.04, 0.25]])
        # The defined domain
        self.lb = [-1., -1.]
        self.ub = [1., 1.]

    @property
    def name(self):
        return 'darcyflow_2d'

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
        u = torch.sin(self.freq * x[:,0:1]) * torch.sin(self.freq * x[:,1:])

        return u
    
    def fun_k(self, x:torch.tensor):
        '''The coefficient in PDE'''
        exp_term = 0.
        for i in range(2):
            exp_term += (x[:,i:i+1]-self._k_center[0,i])**2 / self._k_sigma[0,i]
        
        return self._k_constant + self._k_scale * torch.exp( - exp_term)

    def fun_f(self, x:torch.tensor)->torch.tensor:
        '''
        The right hand side of PDE (or the source term)
        Input:  
            x: size(?,d)
        Output: 
            f: size(?,1)
        '''
        #*********************************************************
        # The value of f(x)
        k = self.fun_k(x)
        u = self._u_star(x)
        part1 = - 2*self.freq**2 * k * u 
        #
        du_x = self.freq * torch.cos(self.freq * x[:,0:1]) * torch.sin(self.freq * x[:,1:])
        du_y = self.freq * torch.sin(self.freq * x[:,0:1]) * torch.cos(self.freq * x[:,1:])
        dk_x = - (k-self._k_constant) * 2. * (x[:,0:1]-self._k_center[0,0]) / self._k_sigma[0,0]
        dk_y = - (k-self._k_constant) * 2. * (x[:,1:2]-self._k_center[0,1]) / self._k_sigma[0,1]
        part2 = du_x * dk_x + du_y * dk_y
        
        return - part1 -part2

    def _test(self):
        '''Check the setups of the problem '''
        from Utils.GenPoints import Point2D
        pointGen = Point2D(x_lb=self._lb, x_ub=self._ub)
        #
        x_in = pointGen.inner_point(num_sample_or_mesh=100, method='mesh')
        ############ Check the fun_f
        k = self.fun_k(x_in)
        f = self.fun_f(x_in)
        u = self._u_star(x_in)
        Plot.show_3d_list(x_in, [f,u], label_list=['f', 'u'])
        ########### Check the solution u
        # Plot.show_2d(x_in, f, title='f')
        # Plot.show_2d(x_in, u, title='u')
    
    def _check_strong(self):
        '''check the strong form '''
        from torch.autograd import Variable, grad
        from Utils.GenPoints import Point2D
        pointGen = Point2D(x_lb=self._lb, x_ub=self._ub)
        #
        x_in = pointGen.inner_point(num_sample_or_mesh=5, method='uniform')
        x = Variable(x_in, requires_grad=True)
        #
        u = self._u_star(x)
        k = self.fun_k(x)
        f = self.fun_f(x)
        #
        du = grad(inputs=x, outputs=u, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        dkdux = grad(inputs=x, outputs=k*du[...,0:1], grad_outputs=torch.ones_like(k), create_graph=True)[0]
        dkduy = grad(inputs=x, outputs=k*du[...,1:2], grad_outputs=torch.ones_like(k), create_graph=True)[0]
        #
        left = - (dkdux[...,0:1] + dkduy[...,1:2])
        right = f 
        print(torch.abs(left-right))
    
    def _check_weak(self):
        '''check the weak form '''
        from torch.autograd import Variable, grad
        from Utils.GenPoints import Point2D
        from Utils.TestFun_ParticleWNN import TestFun_ParticleWNN
        pointGen = Point2D(x_lb=self._lb, x_ub=self._ub)
        int_grid, phi, dphi_dr, Lphi_dr = TestFun_ParticleWNN(
            fun_type='Wendland', dim=2, n_mesh_or_grid=23, 
            dataType=torch.float32, retrun_Lv=True).get_testFun()
        n_grid = int_grid.shape[0]
        print(int_grid.shape, phi.shape, dphi_dr.shape, Lphi_dr.shape)
        #
        xc, R = pointGen.weight_centers(n_center=5, R_max=1e-4, R_min=1e-4)
        x = xc + int_grid*R
        print(xc.shape, R.shape, x.shape)
        x = Variable(x.reshape(-1, 2), requires_grad=True)
        ## size(n_grid, 1) -> (nc, n_grid, 1) -> (nc*n_grid, 1)
        phi = phi.repeat((5,1,1)).reshape(-1,1)
        # size(nc, n_grid, 1) -> (nc*n_grid, 1)
        dphi = (dphi_dr / R).reshape(-1, 2)
        # #
        u = self._u_star(x)
        k = self.fun_k(x)
        f = self.fun_f(x)
        du = grad(inputs=x, outputs=u, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        # # The first weak form: <kdu, dv> = <f, v>
        left = torch.sum(k * du*dphi, dim=-1).reshape(5, n_grid)
        left = torch.mean(left, dim=-1)
        right = (f*phi).reshape(5, n_grid)
        right = torch.mean(right, dim=-1)
        print(torch.abs(left-right))

if __name__=='__main__':
    demo = Problem(freq=np.pi)
    # demo._test()
    # demo._check_strong()
    # demo._check_weak()

    
