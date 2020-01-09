# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:55:56 2019

@author: wyzhang
"""

import numpy as np
#import matplotlib.pyplot as plt
#import scipy.io as scio
#from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
#from torch.autograd import Variable
#from torch.nn import Parameter
from torch.distributions.multivariate_normal import MultivariateNormal
#import torch.utils.data as Data
#import Regularization as Re
#from tensorboardX import SummaryWriter
#import os
#import time

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        
        #assert hidden_channels % 2 == 0
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        #self.num_features = 4
        
        self.padding = int((kernel_size-1)/2)
        
        self.Wxi = nn.Conv2d( self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias = True)
        self.Whi = nn.Conv2d( self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias = False)
        self.Wxf = nn.Conv2d( self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias = True)
        self.Whf = nn.Conv2d( self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias = False)
        self.Wxc = nn.Conv2d( self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias = True)
        self.Whc = nn.Conv2d( self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias = False)
        self.Wxo = nn.Conv2d( self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias = True)
        self.Who = nn.Conv2d( self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias = False)
       
        self.Wci = None
        self.Wcf = None
        self.Wco = None
        
        
    def forward(self, x, h, c):
        #print(x)
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c*self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c*self.Wcf)
        cc = cf*c + ci*torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc*self.Wco)
        ch = co*torch.tanh(cc)
        return ch, cc
    
    
    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = torch.zeros(1, hidden, shape[0], shape[1])
            self.Wcf = torch.zeros(1, hidden, shape[0], shape[1])
            self.Wco = torch.zeros(1, hidden, shape[0], shape[1])
        
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        
        return (torch.zeros(batch_size, hidden, shape[0], shape[1]), 
               torch.zeros(batch_size, hidden, shape[0], shape[1]))
    
        

class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step = 1, effective_step = [1]):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        self.linear_in = nn.Linear(64, 64)
        self.linear_out = nn.Linear(64, 64)
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)
            
    
    def forward(self, input):
        internal_state = []
        outputs = []
        assert input.shape[1] == self.step, 'Input Step Mismatched!'
        xNorm = input
        for step in range(self.step):       # 在每一个时步进行前向运算
            x = xNorm[:,step,:,:].view(-1,1,8,8)
            #print(x)
            x = x.view(-1,64)
            x = self.linear_in(x)
            x = x.view(-1,1,8,8)
            for i in range(self.num_layers):        # 对多层convLSTM中的每一层convLSTMCell，依次进行前向运算
                # all cells are initialized in the first step
                
                name = 'cell{}'.format(i)
                if step == 0:       # 如果是在第一个时步，则需要调用init_hidden进行convLSTMCell的初始化
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))
                    #print(h.shape,c.shape)
                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)     #调用convLSTMCell的forward进行前向运算
                internal_state[i] = (x, new_c)
            # only record effective steps
            x = x.view(-1, 64)
            x = self.linear_out(x)
            x = x.view(-1,1,8,8)



            
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)



def Kernel(sigma, gamma, moneyness, tau):
    length = 64 #N1*N2
    width = 64 #N1*N2
    cov = torch.zeros(length, width)
    x1, x2 = np.meshgrid(tau,moneyness)
    x1 = torch.from_numpy(x1).float()
    x2 = torch.from_numpy(x2).float()
    x1 = x1.view(length, )
    x2 = x2.view(width, )
    #print(x1, x2)
    for i in range(length):
        for j in range(width):
            sample1 = torch.Tensor([x1[i], x2[i]])
            sample2 = torch.Tensor([x1[j], x2[j]])
            cov[i, j] = sigma**2*torch.exp(-torch.dist(sample1, sample2)/gamma**2)
    #print(cov)
    #print(cov.shape)
    return cov
    

def nMarginalloglikelihood(sigma, gamma, moneyness, tau, y):
    #print(y)
    y = y.view(-1, 64) # N1*N2)
    sig = Kernel(sigma, gamma, moneyness, tau)
    mu = torch.zeros((1, y.shape[1]))
    logllhood = MultivariateNormal(mu, sig).log_prob(y)
    ret = -torch.mean(logllhood)
    return ret     


def mlog_prob(inv_kernel, y):
    y = y.view(-1, 64) #(N, 64)
    M = []
    for i in range(y.shape[0]):
      z = y[i].view(1, 64)
      c = (z.mm(inv_kernel)).mm(z.t())
      M.append( c)
    M = torch.Tensor(M)
    M = torch.mean(M)










      
if __name__ == '__main__':
    # gradient check
    torch.manual_seed(1)
    # 定义一个5层的convLSTM
    convlstm = ConvLSTM(input_channels=1, hidden_channels=[1], kernel_size=5, step=4,
                        effective_step=[2,3])
    loss_fn = torch.nn.MSELoss()

    input = torch.randn(1, 4, 8, 8)
    target = torch.randn(1, 1, 8, 8)

    output, _ = convlstm(input)
    print(output)
    output = output[-1]
    print(output.shape)
    #res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    #print(res)
    print()
        
        
        
        
        
        
        
        
        


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    