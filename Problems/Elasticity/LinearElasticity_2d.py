# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-07-18
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-07-18
#  */
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
#
import numpy as np
import math
import torch 
#
from Problems.Module import Problem
from Utils.PlotFigure import Plot

class Problem(Problem):

    def __init__(self, lamda=1., mu=0.5, Q=4., 
                 dataType=torch.float32):
        self._dtype = dataType
        # Parameters in the PDE
        self.lamda = lamda
        self.mu = mu
        self.Q = Q
        # The defined domain
        self.lb = [0., 0.]
        self.ub = [1., 1.]

    @property
    def name(self):
        return 'linear_elasticity_2d'

    @property
    def _lb(self):
        '''The lower bound '''
        return self.lb 
    @property
    def _ub(self):
        '''The upper bound'''
        return self.ub

    def _lamda(self, x:torch.tensor=None):
        '''The 1st lame's parameter'''
        if x:
            return NotImplementedError
        else:
            return self.lamda

    def _mu(self, x:torch.tensor=None):
        '''The 2ed lame's parameter'''
        if x:
            return NotImplementedError
        else:
            return self.mu

    def _ux_star(self, x:torch.tensor):
        '''The truth solution of u'''
        ux = torch.cos(2 * math.pi * x[:,0:1]) * torch.sin(math.pi * x[:,1:])

        return ux

    def _uy_star(self, x:torch.tensor):
        '''The truth solution of u'''
        uy = torch.sin(math.pi * x[:,0:1]) * self.Q * x[:,1:]**4 / 4

        return uy  
    
    def _sxx_star(self, x:torch.tensor):
        ''' '''
        sxx = (self.lamda*(self.Q* torch.sin(math.pi * x[:,0:1]) * x[:,1:]**3 
                          - 2 * math.pi * torch.sin(2*math.pi*x[:,0:1]) * torch.sin(math.pi * x[:,1:]) ) 
                          - 4 * self.mu * math.pi * torch.sin(2*math.pi*x[:,0:1]) * torch.sin(math.pi * x[:,1:]))
        return sxx

    def _syy_star(self, x:torch.tensor):
        ''' '''
        syy = (self.lamda*(self.Q* torch.sin(math.pi * x[:,0:1]) * x[:,1:]**3 
                          - 2 * math.pi * torch.sin(2*math.pi*x[:,0:1]) * torch.sin(math.pi * x[:,1:]) ) 
                          + 2 * self.mu * self.Q * torch.sin(math.pi*x[:,0:1]) * x[:,1:]**3 )
        return syy

    def _sxy_star(self, x:torch.tensor):
        ''' '''
        sxy = self.mu*(torch.cos(math.pi * x[:,0:1]) * x[:,1:]**4 * math.pi * self.Q / 4
                        + math.pi * torch.cos(2*math.pi*x[:,0:1]) * torch.cos(math.pi * x[:,1:]) )
        return sxy
    
    def fun_f(self, x:torch.tensor):
        '''The body force
        Input:
            x: size(?,2)
        Output:
            fx: size(?,1)
            fy: size(?,1)
        '''
        fx = (self._lamda() * 
              ( 4 * math.pi**2 * torch.cos(2*math.pi * x[:,0:1]) * torch.sin(math.pi * x[:,1:]) 
               - math.pi * torch.cos(math.pi * x[:,0:1]) * self.Q * x[:,1:]**3 ) + 
               self._mu() * 
               (9 * math.pi**2 * torch.cos(2*math.pi * x[:,0:1]) * torch.sin(math.pi * x[:,1:]) 
                - math.pi * torch.cos(math.pi * x[:,0:1]) * self.Q * x[:,1:]**3 ) ) 
        fy = (self._lamda() * 
              ( -3 * torch.sin(math.pi * x[:,0:1]) * self.Q * x[:,1:]**2
               + 2 * math.pi**2 * torch.sin(2*math.pi * x[:,0:1]) * torch.cos(math.pi * x[:,1:]) )
               + self._mu() * 
               (- 6 * torch.sin(math.pi * x[:,0:1]) * self.Q * x[:,1:]**2
                + 2 * math.pi**2 * torch.sin(2*math.pi * x[:,0:1]) * torch.cos(math.pi * x[:,1:]) 
                + math.pi**2 * torch.sin(math.pi * x[:,0:1]) * self.Q * x[:,1:]**4 / 4 ) )
        
        return fx, fy

    def _test(self):
        '''Check the setups of the problem'''
        from Utils.GenPoints import Point2D
        pointGen = Point2D(x_lb=self._lb, x_ub=self._ub)
        #
        x_in = pointGen.inner_point(num_sample_or_mesh=100, method='mesh')
        ############ Check the fun_f
        fx, fy = self.fun_f(x_in)
        # Plot.show_2d(x_in, fx, title='fx')
        # Plot.show_2d(x_in, fy, title='fy')
        ########### Check the solution u
        ux = self._ux_star(x_in)
        uy = self._uy_star(x_in)
        # Plot.show_2d(x_in, ux, title='ux')
        # Plot.show_2d(x_in, uy, title='uy')
        ########### Check the solution S
        sxx = self._sxx_star(x_in)
        syy = self._syy_star(x_in)
        sxy = self._sxy_star(x_in)
        # Plot.show_2d(x_in, sxx, title='sxx')
        # Plot.show_2d(x_in, syy, title='syy')
        # Plot.show_2d(x_in, sxy, title='sxy')
        

if __name__=='__main__':
    demo = Problem()
    demo._test()

    
