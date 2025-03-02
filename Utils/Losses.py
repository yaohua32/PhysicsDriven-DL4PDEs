# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-08-22 14:18:09 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-08-22 14:18:09 
#  */
import torch 

class MyLoss(object):

    def __init__(self, p=2, size_average=True):
        super(MyLoss, self).__init__()
        self.p = p
        self.size_average = size_average
        self.eps = 1e-8

    def lp_abs(self, y_pred:torch.tensor, y_true:torch.tensor):
        ''' The lp loss w/o relative (not lp error)
            loss = sqrt( sum(yi, yi_true)**p ) / sqrt( n_batch )
        Input:
            y_pred: size(n_batch, 1)
            y_true: size(n_batch, 1)
        '''
        assert y_true.shape==y_pred.shape
        batch_size = y_pred.shape[0]
        # size(1,)
        diff_norm = torch.norm(
            y_true.reshape(batch_size,-1) - y_pred.reshape(batch_size,-1), 
            self.p, dim=0)
        # Divided by (batch_size)**(1/p)
        if self.size_average:
            return diff_norm/batch_size**(1/self.p)
        
        return diff_norm

    def lp_rel(self, y_pred:torch.tensor, y_true:torch.tensor):
        ''' The lp loss w relative (not lp error)
            loss = sqrt( sum( (yi-yi_true)**p ) ) / sqrt( sum( yi_true**p ) )
        Input:
            y_pred: size(n_batch, 1)
            y_true: size(n_batch, 1)
        '''
        assert y_true.shape==y_pred.shape
        batch_size = y_pred.shape[0]
        # size(1,)
        diff_norms = torch.norm(
            y_true.reshape(batch_size,-1) - y_pred.reshape(batch_size,-1), 
            self.p, dim=0)
        # size(1,)
        y_norms = torch.norm(y_true.reshape(batch_size,-1), 
                             self.p, dim=0) + self.eps
        # Divided by y_norms
        return diff_norms/y_norms

    def __call__(self, y_pred, y_true):
        '''The default called Error'''
        return self.lp_abs(y_pred, y_true)