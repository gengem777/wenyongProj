# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:03:31 2020

@author: wyzhang
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
#from torch.autograd import Variable
#from torch.nn import Parameter
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.utils.data as Data
import Regularization as Re
#from tensorboardX import SummaryWriter
#import os
import time

torch.manual_seed(1)

#hyperparameters
TIME_LAG = 10

EPOCH = 1000
Kernel_EPOCH = 100
BATCH_SIZE = 200
INPUT_SIZE = 128
HIDDEN_SIZE = 128
#OUTPUT_SIZE = 64

LRsigma = 1e-2
LRgamma = 1e-2
LRnet = 1e-2

tau = [0.0833, 0.1786, 0.2738, 0.3690, 0.4643, 0.5595, 0.6548, 0.7500]
moneyness = [0.7000, 0.8000, 0.9000, 1, 1.1000, 1.2000, 1.3000, 1.4000]

class MeanNet(nn.Module):   #lstm 64 to 128, 128 to 64
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE):
        super(MeanNet, self).__init__()
        
        self.linear_in = nn.Linear(64, INPUT_SIZE)
        
        self.rnn = nn.GRU(
                input_size = INPUT_SIZE,
                hidden_size = HIDDEN_SIZE,
                num_layers = 1,
                batch_first = True
                )
        
        self.linear_out = nn.Linear(HIDDEN_SIZE, 64)
        
    def forward(self, x):#, hstate):
        z = torch.zeros(x.shape[0], x.shape[1], INPUT_SIZE)
        for i in range(TIME_LAG):
            z[:, i, :] = self.linear_in(x[:, i, :])
        r_out, _ = self.rnn(z, None)
        out = self.linear_out(r_out[:, -1, :])
        return out
'''
net = MeanNet(INPUT_SIZE, HIDDEN_SIZE)
inputX = torch.randn(BATCH_SIZE, TIME_LAG, 64)
print(net.forward(inputX).shape)
'''


def Kernel(sigma, gamma, moneyness, tau):  #calculate the Gaussian kernel matrix of disturbance
    length = 64
    width = 64
    cov = torch.zeros(length, width)
    x1, x2 = np.meshgrid(tau,moneyness)
    x1 = torch.from_numpy(x1).float()
    x2 = torch.from_numpy(x2).float()
    x1 = x1.view(length, )
    x2 = x2.view(width, )
    for i in range(length):
        for j in range(width):
            sample1 = torch.Tensor([x1[i], x2[i]])
            sample2 = torch.Tensor([x1[j], x2[j]])
            cov[i, j] = sigma**2*torch.exp(-torch.dist(sample1, sample2)/gamma**2)
    return cov

def nMarginalloglikelihood(sigma, gamma, moneyness, tau, y): #calculate the negative log-likelihood
    sig = Kernel(sigma, gamma, moneyness, tau)
    mu = torch.zeros((1, y.shape[1]))
    logllhood = MultivariateNormal(mu, sig).log_prob(y)
    ret = -torch.mean(logllhood)
    return ret

#==============================================================================
IVS_realmarket = np.loadtxt('IVS')
#get log
logIVSreal = np.log(IVS_realmarket)
#split time series to train, validate, test.
logIVSreal_train = logIVSreal[:1029]
logIVSreal_val = logIVSreal[1029:1372]
logIVSreal_test = logIVSreal[1372:]
#do the first ordered differentiation
difflogtrain = np.diff(logIVSreal[:1029], axis = 0)
difflogval = np.diff(logIVSreal[1029:1372], axis = 0)
difflogtest =  np.diff(logIVSreal[1372:], axis = 0)
difflogtrain = torch.from_numpy(difflogtrain).float()
difflogval = torch.from_numpy(difflogval).float()
difflogtest = torch.from_numpy(difflogtest).float()


lnIVS_train = difflogtrain
lnIVS_val = difflogval
T_train = lnIVS_train.shape[0]
T_val = lnIVS_val.shape[0]

# reshape the input to (batch, time_lag, 64) and the target to (batch, 64)
y_train = lnIVS_train[TIME_LAG:T_train,:]
X_train = torch.zeros([T_train-TIME_LAG,TIME_LAG, 64])
for i in range(int(T_train - TIME_LAG)):
  for j in range(int(TIME_LAG)):
         X_train[i,j,:] = lnIVS_train[i+TIME_LAG-1-j, :].view(1,1,64)
#y_train = y_train.view(-1, 8, 8)
#X_train = X_train.view(-1, TIME_LAG, 8, 8)

y_val = lnIVS_val[TIME_LAG:T_val,:]     
X_val = torch.zeros([T_val-TIME_LAG,TIME_LAG, 64])
for i in range(int(T_val - TIME_LAG)):
  for j in range(int(TIME_LAG)):
         X_val[i,j,:] = lnIVS_val[i+TIME_LAG-1-j, :].view(1,1,64)
#y_val = y_val.view(-1, 8, 8)
#X_val = X_val.view(-1, TIME_LAG, 8, 8)

#===============================================================================
         
torch_trainingset = Data.TensorDataset(X_train, y_train)
trainloader = Data.DataLoader(dataset=torch_trainingset, batch_size=BATCH_SIZE, shuffle = False, num_workers = 0)

torch_valset = Data.TensorDataset(X_val, y_val)
valloader = Data.DataLoader(dataset = torch_valset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)


regressor = MeanNet(INPUT_SIZE, HIDDEN_SIZE)

sigma = nn.Parameter(torch.Tensor([0.02]), requires_grad = True)
gamma = nn.Parameter(torch.Tensor([3.33]), requires_grad = True)
optimizer = optim.Adam([{'params': sigma, 'lr': LRsigma}, {'params': regressor.parameters(), 'lr': LRnet}, {'params': gamma, 'lr': LRgamma}], amsgrad = True)

trainlosses = []
testlosses = []
RMSEtrain = []
RMSEtest = []
MAPEtrain = []
MAPEtest = []

start = time.clock()
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer,  step_size=50, gamma=0.5)
for epoch in range(EPOCH):
    regressor.train()
    batch_nll = []
    for batch_idx, (batch_x, batch_y) in enumerate(trainloader):
        predict = regressor.forward(batch_x)
        diff = batch_y - predict
        trainloss = nMarginalloglikelihood(sigma, gamma, moneyness, tau, diff)
        optimizer.zero_grad()
        trainloss.backward()
        optimizer.step()
        rmse = torch.sqrt(torch.mean(diff**2))
        mape = torch.abs(torch.mean(torch.abs(diff)/batch_y))
        batch_nll.append(trainloss.item())
    trainlosses.append(np.mean(batch_nll))
    RMSEtrain.append(rmse.item())
    MAPEtrain.append(mape.item())
    if epoch % 10 == 0:
        print("----------------------------------------------------------------------------------------")
        print(epoch, "trainingloss is:", np.mean(batch_nll))
        print(epoch, "RMSE trainingloss is:", rmse.item())
        print(epoch, "percentage trainingloss is:", (mape*100).item(),'%')
    with torch.no_grad():
     regressor.eval()
     batch_nll = []
     for batch_idx, (batch_x, batch_y) in enumerate(valloader):
         predict = regressor.forward(batch_x)
         diff = batch_y - predict
         testloss = nMarginalloglikelihood(sigma, gamma, moneyness, tau, diff)
         rmse = torch.sqrt(torch.mean(diff**2))
         mape = torch.abs(torch.mean(torch.abs(diff)/batch_y))
         batch_nll.append(testloss.item())
    testlosses.append(np.mean(batch_nll))
    RMSEtest.append(rmse.item())
    MAPEtest.append(mape.item())
    if epoch % 10 == 0:
        #print(epoch, "sigma is:", torch.sqrt(sigma**2).item(), "gamma is:", gamma.item())
        print(epoch, "testloss is:", np.mean(batch_nll))
        print(epoch, "RMSE testingloss is:", rmse.item())
        print(epoch, "percentage testloss is:", (mape*100).item(),'%')
        print("----------------------------------------------------------------------------------------")
    #writer.add_scalars('loss/NLL', {'train NLL loss/test': np.float64(trainloss.item()), 'test NLL loss': np.float64(testloss.item())}, epoch)
    #writer.add_scalars('loss/NLL', {'train RMSE loss': trainloss.item(), 'test RMSE loss': testloss.item()}, epoch)
    #writer.add_scalars('loss/NLL', {'train MAPE loss': trainloss.item(), 'test MAPE loss': testloss.item()}, epoch)
    scheduler1.step()
    
elapsed = (time.clock() - start)
print("Time used:",elapsed)

print('LAG'+str(TIME_LAG)+' model: ' 'sigma is:',  torch.sqrt(sigma**2).item(), 'gamma is:', gamma.item())        
        
torch.save(regressor, 'MeanNet' + str(TIME_LAG) + '.pkl')
torch.save(regressor.state_dict(), 'param MeanNet'+ str(TIME_LAG) +'.pkl')


plt.figure()
plt.title('Summary for lag '+str(TIME_LAG)+'model')
plt.plot(trainlosses, alpha = 0.4, color = 'red', label = 'training NLL')
plt.plot(testlosses, alpha = 0.5, color = 'blue', label = 'validating NLL')
plt.xlabel('epochs')
plt.ylabel('negative loglikelihood')
plt.title('Negative log likelihood')
plt.legend()
plt.show()

plt.figure()
plt.plot(RMSEtrain, alpha = 0.4, color = 'red', label = 'RMSE for training')
plt.plot(RMSEtest, alpha = 0.6, color = 'blue', label = 'RMSE for validating')
plt.xlabel('epochs')
plt.ylabel('RMSE')
plt.title('RMSE of the conditional mean')
plt.legend()
plt.show()

plt.figure()
plt.plot(MAPEtrain, alpha = 0.4, color = 'red', label = 'MAPE for training')
plt.plot(MAPEtest, alpha = 0.6, color = 'blue', label = 'MAPE for validating')
plt.xlabel('epochs')
plt.ylabel('MAPE(%)')
plt.title('MAPE of the conditional mean(%)')
plt.legend()


print('Summary sheet for lag '+str(TIME_LAG)+'model')
print('--------------------Negative Loglikelihood of lag '+str(TIME_LAG)+'--------------')
print(np.mean(trainlosses[-20:-1]))
print(np.mean(testlosses[-20:-1]))
print('-------------------RMSE of conditional mean with lag '+str(TIME_LAG)+'------------')
print(np.mean(RMSEtrain[-20:-1]))
print(np.mean(RMSEtest[-20:-1]))
print('--------------------MAPE of conditional mean with lag '+str(TIME_LAG)+'------------)#')
print(np.mean(MAPEtrain[-20:-1])*100,'%')
print(np.mean(MAPEtest[-20:-1])*100,'%')





















