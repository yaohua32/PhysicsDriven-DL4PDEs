# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-08-22 14:17:50 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-08-22 14:17:50 
#  */
import torch
from torch.autograd import grad
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

#########################
class MyDataset(Dataset):

    def __init__(self, x:torch.tensor):
        '''
        Input:
            x: size(N, )
        '''
        self.x = x

    def __getitem__(self, index):
        return self.x[index]
    
    def __len__(self):
        return self.x.shape[0]

#############################
class MyIndex(Dataset):

    def __init__(self, index:torch.tensor):
        '''
        Input:
            index: size(N,)
        '''
        self.index = index
    
    def __getitem__(self, idx):
        return self.index[idx]

    def __len__(self):
        return self.index.shape[0]

#################
class Solver():

    def __init__(self, device, dtype):
        '''DNN-based PDE Solver
        '''
        self.device = device
        self.dtype = dtype

    def saveModel(self):
        '''Save the model
        '''
        raise NotImplementedError

    def loadModel(self):
        '''Load the model
        '''
        raise NotImplementedError

    def dataloader(self, x:torch.tensor, batch_size:int=100, shuffle=True):
        '''Prepare the data_loader for training
        Input:
            x: size(N, d)
            batch_size: int
        Output:
            train_loader
        '''
        return DataLoader(MyDataset(x), 
                          batch_size= batch_size, 
                          shuffle=shuffle)

    def indexloader(self, N:int, batch_size:int=100, shuffle=True):
        '''Prepare the index_loader for training
        '''
        index = torch.tensor([i for i in range(N)], dtype=torch.int32)
        return DataLoader(MyIndex(index),
                          batch_size=batch_size,
                          shuffle=shuffle)

    def grad(self, y:torch.tensor, x_list:list[Variable]):
        '''The grad of y
        Input:  
            x_list: [size(?,1)]*d
            y: size(?,1)
        Output: 
            dy_list: [size(?,1)]*d
        '''
        if type(x_list).__name__!='list':
            x_list = [x_list]
        #
        dy_list = []
        for x in x_list:
            dy_list.append(grad(inputs=x, outputs=y, grad_outputs=torch.ones_like(y), 
                                create_graph=True)[0])
            
        return dy_list

    def div(self, y_list:list[torch.tensor], x_list:list[Variable]):
        '''The divergence of y
        Input: 
            x_list: [ size(?,1) ]*d
            y_list: [ size(?,1) ]*d
        Output: 
            div_y: size(?,1)
        '''
        assert len(x_list)==len(y_list)
        #
        div_y = torch.zeros_like(y_list[0])
        for y, x in zip(y_list, x_list):
            div_y += grad(inputs=x, outputs=y, grad_outputs=torch.ones_like(y), 
                          create_graph=True)[0]
        return div_y

    def train(self):
        '''Train the model
        '''
        raise NotImplementedError
    
    def predict(self):
        '''Make prediction with the traned model
        '''
        raise NotImplementedError

################
class LossClass(object):

    def __init__(self, solver:Solver, **kwrds):
        self.solver = solver

    def Loss_pde(self):
        '''The loss of pde
        '''
        return torch.tensor(0., device=self.solver.device, 
                            dtype=self.solver.dtype)
    
    def Loss_bdic(self):
        '''The loss of boundary/initial conditions
        '''
        return torch.tensor(0., device=self.solver.device, 
                            dtype=self.solver.dtype)
    
    def Error(self):
        '''The errors
        '''
        return torch.tensor(0., device=self.solver.device, 
                            dtype=self.solver.dtype)