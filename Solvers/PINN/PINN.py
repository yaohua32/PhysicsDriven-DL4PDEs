# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-08-22 14:16:40 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-08-22 14:16:40 
#  */
import torch
import time
import os
import scipy.io
from tqdm import trange
#
from Solvers import Module
from Networks.FCNet import FCNet
from Networks.ResNet import ResNet
#
from Utils.Losses import MyLoss

class Solver(Module.Solver):

    def __init__(self, device='cuda:0',
                 dtype=torch.float32):
        '''The PINN method
        '''
        self.device = device
        self.dtype = dtype
        #
        self.iter = 0
        self.time_list = []
        self.loss_bd_list = []
        self.loss_eq_list = []
        self.loss_list = []
        self.err_list =[]
        #
        self.error_setup()
        self.loss_setup()
    
    def loadModel(self, path:str, name:str):
        '''Load trained model
        '''
        return torch.load(path+f'{name}.pth', map_location=self.device)

    def saveModel(self, path:str, name:str, model_dict:dict):
        '''Save trained model (the whole model)
        '''
        if not os.path.exists(path):
            os.makedirs(path)
        #
        torch.save(model_dict, path+f'{name}.pth')
    
    def loadLoss(self, path:str, name:str):
        '''Load saved losses
        '''
        loss_dict = scipy.io.loadmat(path+f'{name}.mat')

        return loss_dict

    def saveLoss(self, path:str, name:str):
        '''Save losses
        '''
        dict_loss = {}
        dict_loss['loss'] = self.loss_list
        dict_loss['loss_pde'] = self.loss_eq_list
        dict_loss['loss_bd'] = self.loss_bd_list
        dict_loss['time'] = self.time_list
        dict_loss['error'] = self.err_list
        scipy.io.savemat(path+f'{name}.mat', dict_loss)

    def callBack(self, loss_eq, loss_bd, loss_val, errors, t_start):
        '''call back
        '''
        self.loss_eq_list.append(loss_eq.item())
        self.loss_bd_list.append(loss_bd.item())
        self.loss_list.append(loss_val.item())
        self.time_list.append(time.time()-t_start)
        if isinstance(errors, list):
            errs = [err.item() for err in errors]
            self.err_list.append(errs)
        else:
            self.err_list.append(errors.item())
    
    def error_setup(self, err_type:str='lp_rel', p:int=2, 
                    size_average=True):
        ''' setups of error
        Input:
            err_type: from {'lp_rel', 'lp_abs'}
        '''
        Error = MyLoss(p, size_average)
        #
        if err_type=='lp_rel':
            self.getError = Error.lp_rel
        elif err_type=='lp_abs':
            self.getError = Error.lp_abs
        else:
            raise NotImplementedError(f'{err_type} has not defined.')

    def loss_setup(self, loss_type:str='lp_abs', p:int=2, 
                   size_average=True):
        '''setups of loss
        Input:
            loss_type: from {'lp_abs', 'lp_rel'}
        '''
        Loss = MyLoss(p, size_average)
        #
        if loss_type=='lp_abs':
            self.getLoss = Loss.lp_abs
        elif loss_type=='lp_rel':
            self.getLoss = Loss.lp_rel
        else:
            raise NotImplementedError(f'{loss_type} has not defined.')

    def getModel(self, layers_list, activation:str='Tanh', 
                 netType:str='FCNet', **kwrds):
        '''Set the models
        '''
        self.netType = netType
        if netType=='FCNet':
            model = FCNet(layers_list, activation, self.dtype)
        elif netType=='ResNet':
            model = ResNet(layers_list, activation, self.dtype)
        else:
            raise NotImplementedError
        
        return model.to(self.device)
    
    def train_setup(self, model_dict:dict, lr:float=1e-3, 
                    optimizer='Adam', scheduler_type='StepLR',
                    lbfgs=False, lr_lbfgs=1., max_iter=1000, history_size=10,
                    step_size=200, gamma=1/2, patience=20, factor=1/2, 
                    **kwrds):
        '''Setups for training
        '''
        self.model_dict = model_dict
        ###### The models' parameters
        param_list = []
        for model in model_dict.values():
            param_list += list(model.parameters())
        ########### Set the optimizer
        if optimizer=='Adam':
            self.optimizer = torch.optim.Adam(params=param_list, lr=lr, weight_decay=1e-4)
        elif optimizer=='AdamW':
            self.optimizer = torch.optim.AdamW(params=param_list, lr=lr, weight_decay=1e-4)
        else:
            raise NotImplementedError
        ########### Set the LBFGS Optimizer
        if lbfgs:
            self.optimizer_LBFGS = torch.optim.LBFGS(
                params=param_list, lr=lr_lbfgs, 
                history_size=history_size, 
                max_iter=max_iter, line_search_fn=None)
        else:
            self.optimizer_LBFGS = None
        # ####### Set the scheduler
        if scheduler_type=='StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, 
                gamma=gamma, last_epoch=-1)
        elif scheduler_type=='Plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=factor, patience=patience)
        self.scheduler_type = scheduler_type
        #
        self.t_start = time.time()
        self.best_loss_val = 1e10
    
    def train(self, LossClass:Module.LossClass,
              x_in_train, batch_size_in:int, 
              epochs:int=1, w_pde=1., w_bd=1., 
              epoch_show:int=100, **kwrds):
        '''Train the model
        Input:
            LossClass: class for calculating losses
        '''
        ############### The training data
        index_loader = self.indexloader(x_in_train.shape[0], batch_size_in, shuffle=True)
        ############### The training process
        for epoch in trange(epochs):
            for inx in index_loader:
                lossClass = LossClass(self)
                ############# Calculate losses
                loss_in = lossClass.Loss_pde(x_in_train[inx].to(self.device))
                loss_bd = lossClass.Loss_bd()
                loss_train = w_pde*loss_in + w_bd*loss_bd
                ############# Calculate errors
                errors = lossClass.Error()
                #
                self.optimizer.zero_grad()
                loss_train.backward()
                self.optimizer.step()
                self.iter += 1
                #
                self.callBack(loss_in, loss_bd, loss_train, errors, self.t_start)
            ####################### The validation loss
            lossClass = LossClass(self)
            try:
                loss_in = lossClass.Loss_pde(kwrds['x_in_val'].to(self.device))
            except:
                loss_in = lossClass.Loss_pde(x_in_train.to(self.device))
            loss_bd = lossClass.Loss_bd()
            #
            loss_val = w_pde*loss_in + w_bd*loss_bd
            if loss_val.item()<self.best_loss_val:
                self.best_loss_val = loss_val.item()
                self.saveModel(kwrds['save_path'], 'model_pinn_bestloss', self.model_dict)
            #######################  The scheduler
            if self.scheduler_type is None:
                pass
            elif self.scheduler_type=='Plateau':
                self.scheduler.step(loss_val.item())
            else:
                self.scheduler.step()
            ###################
            if (epoch+1)%epoch_show==0:
                print(f'Epoch:{epoch+1} Time:{time.time()-self.t_start:.4f}, loss_in:{loss_in.item():.6f}, loss_bd:{loss_bd.item():.6f}')
                for para in self.optimizer.param_groups:
                    print('          lr:', para['lr'], 'l2_err', self.err_list[-1])
        ########################
        self.saveModel(kwrds['save_path'], name='model_pinn_final', 
                    model_dict=self.model_dict)
        self.saveLoss(kwrds['save_path'], name='loss_pinn')
        print(f'The total training time is {time.time()-self.t_start:.4f}')

    def train_lbfgs(self, LossClass:Module.LossClass, 
                    x_in_train, epochs:int=1, 
                    w_pde=1., w_bd=1., 
                    epoch_show:int=5, **kwrds):
        '''Final trining with LBFGS
        '''
        print('****************** The training with LBFGS optimizer ***********')
        for epoch in trange(epochs):
            def closure():
                ''' The closure function
                '''
                self.optimizer_LBFGS.zero_grad()
                lossClass = LossClass(self)
                ############# Calculate losses
                loss_in = lossClass.Loss_pde(x_in_train.to(self.device))
                loss_bd = lossClass.Loss_bd()
                loss_train = w_pde*loss_in + w_bd*loss_bd
                ############# Calculate errors
                errors = lossClass.Error()
                self.callBack(loss_in, loss_bd, loss_train, errors, self.t_start)
                #
                loss_train.backward()
                return loss_train
            #
            self.optimizer_LBFGS.step(closure)
            self.iter += 1
            ####################### The validation loss
            lossClass = LossClass(self)
            try:
                loss_in = lossClass.Loss_pde(kwrds['x_in_val'].to(self.device))
            except:
                loss_in = lossClass.Loss_pde(x_in_train.to(self.device))
            loss_bd = lossClass.Loss_bd()
            #
            loss_val = w_pde*loss_in + w_bd*loss_bd
            if loss_val.item()<self.best_loss_val:
                self.best_loss_val = loss_val.item()
                self.saveModel(kwrds['save_path'], 'model_pinn_bestloss', self.model_dict)
            ###################
            if (epoch+1)%epoch_show==0:
                print(f'Epoch:{epoch+1} Time:{time.time()-self.t_start:.4f}, loss_in:{loss_in.item():.6f}, loss_bd:{loss_bd.item():.6f}')
                for para in self.optimizer_LBFGS.param_groups:
                    print('          lr_lbfgs:', para['lr'], 'l2_err', self.err_list[-1])
        ########################
        self.saveModel(kwrds['save_path'], name='model_pinn_final', 
                    model_dict=self.model_dict)
        self.saveLoss(kwrds['save_path'], name='loss_pinn')
        print(f'The total training time is {time.time()-self.t_start:.4f}')
