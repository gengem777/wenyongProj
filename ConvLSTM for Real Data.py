# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:43:40 2019

@author: wyzhang
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
#from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torch.autograd import Variable
#from torch.nn import Parameter
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.utils.data as Data
import Regularization as Re
#from tensorboardX import SummaryWriter
import os
import time

from ConvLSTM import ConvLSTMCell, ConvLSTM, Kernel, nMarginalloglikelihood

#全局参数设定
torch.manual_seed(1)

TIME_LAG = 10
steps = 10
EPOCH = 80
Kernel_EPOCH = 150
BATCH_SIZE = 512
INPUT_SIZE = 64
HIDDEN_SIZE = 12
OUTPUT_SIZE = 64
#TIME_LAG = 3
LRsigma = 1e-3
LRgamma = 1e-3
LRnet = 1e-3

tau = [0.0833, 0.1786, 0.2738, 0.3690, 0.4643, 0.5595, 0.6548, 0.7500]
moneyness = [0.7000, 0.8000, 0.9000, 1, 1.1000, 1.2000, 1.3000, 1.4000]



#数据导入及预处理
#------------------------------------------------------------------------------
IVS_realmarket = np.loadtxt('IVS')
logIVSreal = np.log(IVS_realmarket)
logIVSreal_train = logIVSreal[:1029]
logIVSreal_val = logIVSreal[1029:1372]
logIVSreal_test = logIVSreal[1372:]
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

print('Lag '+str(TIME_LAG)+' model')
y_train = lnIVS_train[TIME_LAG:T_train,:]
X_train = torch.zeros([T_train-TIME_LAG,TIME_LAG, INPUT_SIZE])
for i in range(int(T_train - TIME_LAG)):
  for j in range(int(TIME_LAG)):
         X_train[i,j,:] = lnIVS_train[i+TIME_LAG-1-j, :].view(1,1,INPUT_SIZE)
y_train = y_train.view(-1, 8, 8)
X_train = X_train.view(-1, TIME_LAG, 8, 8)

y_val = lnIVS_val[TIME_LAG:T_val,:]     
X_val = torch.zeros([T_val-TIME_LAG,TIME_LAG, INPUT_SIZE])
for i in range(int(T_val - TIME_LAG)):
  for j in range(int(TIME_LAG)):
         X_val[i,j,:] = lnIVS_val[i+TIME_LAG-1-j, :].view(1,1,INPUT_SIZE)
y_val = y_val.view(-1, 8, 8)
X_val = X_val.view(-1, TIME_LAG, 8, 8)
#------------------------------------------------------------------------------
         



#加载成dataloader
#------------------------------------------------------------------------------
torch_trainingset = Data.TensorDataset(X_train, y_train)
trainloader = Data.DataLoader(dataset=torch_trainingset, batch_size=BATCH_SIZE, shuffle = False, num_workers = 0)

torch_testset = Data.TensorDataset(X_val, y_val)
testloader = Data.DataLoader(dataset = torch_testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
#------------------------------------------------------------------------------





#加载网络对象以及初始化优化器参数
#------------------------------------------------------------------------------
regressor = ConvLSTM(input_channels=1, hidden_channels=[2, 1], kernel_size=5, step = TIME_LAG, effective_step=[3])


sigma = nn.Parameter(torch.Tensor([1.0]), requires_grad = True)
gamma = nn.Parameter(torch.Tensor([1.0]), requires_grad = True)

loss_fn = nn.MSELoss(reduction='mean')
reg_loss=Re.Regularization(regressor, weight_decay = 10, p=1) 


trainlosses = []
testlosses = []
RMSEtrain = []
RMSEtest = []
MAPEtrain = []
MAPEtest = []
#writer = SummaryWriter(log_dir = 'loss',)
start = time.clock()
#if __name__ == "__main__":
for step in range(steps):
    
    
    net_opt = optim.Adam(regressor.parameters(), lr = LRnet, amsgrad = True)
    kernel_opt = optim.Adam([{'params': sigma, 'lr': LRsigma}, {'params': gamma, 'lr': LRgamma}], amsgrad = True)
    scheduler1 = torch.optim.lr_scheduler.StepLR(net_opt,  step_size=40, gamma=0.5)
    for epoch in range(EPOCH):
        #print("The", epoch,"st", "epoch")
        #trainloss, rmse, mape = train()
        regressor.train()
        batch_nll = []
        for batch_idx, (batch_x, batch_y) in enumerate(trainloader):
           predict, _ = regressor.forward(batch_x)
           predict = predict[0].view(-1,8,8)
           diff = batch_y - predict
           trainloss = nMarginalloglikelihood(sigma, gamma, moneyness, tau, diff) #+ reg_loss.forward(regressor)
           
           
           #trainloss = loss_fn(predict, batch_y) + reg_loss.forward(regressor)
           #optimizer.zero_grad()
           #loss.backward()
           #optimizer.step()
           
           net_opt.zero_grad()
           trainloss.backward()
           net_opt.step()
           rmse = torch.sqrt(torch.mean(diff**2))
           mape = torch.abs(torch.sum(torch.abs(diff))/torch.sum(torch.abs(batch_y)))
           batch_nll.append(trainloss.item())
           #rmse = torch.sqrt(loss)
          # mape = torch.abs(torch.mean(torch.abs(diff)/batch_y))
          #batch_nll.append(loss.item())
        trainlosses.append(np.mean(batch_nll))
        RMSEtrain.append(rmse.item())
       # RMSE1train.append((torch.sqrt(torch.sum(diff**2))).item())
        MAPEtrain.append(mape.item())
        if epoch % 2 == 0:
            print("----------------------------------------------------------------------------------------")
            print(epoch, "trainingloss is:", np.mean(batch_nll))
            print(epoch, "RMSE trainingloss is:", rmse.item())
            print(epoch, "percentage trainingloss is:", (mape*100).item(),'%')
       
        with torch.no_grad():
            #testloss, rmse, mape = test()
            regressor.eval()
            batch_nll = []
            for batch_idx, (batch_x, batch_y) in enumerate(testloader):
               predict, _ = regressor.forward(batch_x)
               predict = predict[0].view(-1,8,8)
           #hstate = hstate.data
               diff = batch_y - predict
               testloss = nMarginalloglikelihood(sigma, gamma, moneyness, tau, diff) #+ reg_loss.forward(regressor)
               #testloss = loss_fn(predict, batch_y) + reg_loss.forward(regressor)
               rmse = torch.sqrt(torch.mean(diff**2))
               #rmse = torch.sqrt(loss)
               mape = torch.abs(torch.sum(torch.abs(diff))/torch.sum(torch.abs(batch_y)))
               batch_nll.append(testloss.item())
        testlosses.append(np.mean(batch_nll))
        RMSEtest.append(rmse.item())
        MAPEtest.append(mape.item())
        if epoch % 2 == 0:
            #print(epoch, "sigma is:", torch.sqrt(sigma**2).item(), "gamma is:", gamma.item())
            print(epoch, "testloss is:", np.mean(batch_nll))
            print(epoch, "RMSE testingloss is:", rmse.item())
            print(epoch, "percentage testloss is:", (mape*100).item(),'%')
            print("----------------------------------------------------------------------------------------")
        
        scheduler1.step()
    
    #writer.close()
    
    #store the network
    torch.save(regressor, 'ConvlstmReal' + str(TIME_LAG) + '.pkl')
    torch.save(regressor.state_dict(), 'param lstmReal'+ str(TIME_LAG) +'.pkl')
    
    #print(sigma, gamma)
       #with open('param.txt', 'wt') as f1:
    reghat = torch.load('ConvlstmReal' + str(TIME_LAG) + '.pkl')
    scheduler2 = torch.optim.lr_scheduler.StepLR(kernel_opt,  step_size=40, gamma=0.5)
    trainloss2 = []
    for epoch in range(Kernel_EPOCH):
        g,_ = reghat.forward(X_train)
        g = g[0].view(-1,8,8)
        d = y_train - g
        loss2 = nMarginalloglikelihood(sigma, gamma, moneyness, tau, d)
        kernel_opt.zero_grad()
        loss2.backward()
        kernel_opt.step()
        trainloss2.append(loss2)
        if epoch % 10 == 0:
            print(epoch, "sigma is:", torch.sqrt(sigma**2).item(), "gamma is:", gamma.item())
        scheduler2.step()
    print(str(TIME_LAG) +'TIME_LAG'+' model: ' 'sigma is:',  torch.sqrt(sigma**2).item(), 'gamma is:', gamma.item())#, file = f1) # sigma 0.1810, gamma 1.6471
    regressor = reghat

torch.save(regressor, 'ConvlstmReal' + str(TIME_LAG) + '.pkl')

plt.figure()
plt.title('Summary for lag '+str(TIME_LAG)+'model')
plt.subplot(221)
plt.plot(trainlosses, alpha = 0.4, color = 'red', label = 'training NLL')
plt.plot(testlosses, alpha = 0.5, color = 'blue', label = 'validating NLL')
plt.xlabel('epochs')
   # plt.ylim(ymin = -300, ymax = 400)
plt.ylabel('negative loglikelihood')
plt.title('Negative log likelihood')
plt.legend()
plt.show()

plt.subplot(222)
plt.plot(RMSEtrain, alpha = 0.4, color = 'red', label = 'RMSE for training')
plt.plot(RMSEtest, alpha = 0.6, color = 'blue', label = 'RMSE for validating')
plt.xlabel('epochs')
plt.ylabel('RMSE')
plt.title('RMSE of the conditional mean')
plt.legend()
plt.show()
'''
plt.figure()
plt.plot(RMSE1train, alpha = 0.4, color = 'red', label = 'RMSE for training')
plt.plot(RMSE1test, alpha = 0.6, color = 'blue', label = 'RMSE for validating')
plt.xlabel('epochs')
plt.ylabel('RMSE')
plt.title('RMSE of the conditional mean')
plt.legend()
plt.show()
'''
plt.subplot(223)
plt.plot(MAPEtrain, alpha = 0.4, color = 'red', label = 'MAPE for training')
plt.plot(MAPEtest, alpha = 0.6, color = 'blue', label = 'MAPE for validating')
plt.xlabel('epochs')
plt.ylabel('MAPE(%)')
plt.title('MAPE of the conditional mean(%)')
plt.legend()


plt.subplot(224)
plt.plot(trainloss2, alpha = 0.5, color = 'green', label = 'loss2')
plt.xlabel('kernel epochs')
plt.show()
'''
plt.subplot(224)
plt.hist(testlosses[-200:-1], bins = 50, density = 0, facecolor = 'blue', edgecolor = 'black', alpha = 0.7)
plt.show()
'''

elapsed = (time.clock() - start)
print("Time used:",elapsed)
#with open('summary.txt', 'wt') as f2:
print('Summary sheet for lag '+str(TIME_LAG)+'model')
print('--------------------Negative Loglikelihood of lag '+str(TIME_LAG)+'--------------')#, file = f2)
print(np.mean(trainlosses[-20:-1]))#, file = f2)
print(np.mean(testlosses[-20:-1]))#, file = f2)
print('-------------------RMSE of conditional mean with lag '+str(TIME_LAG)+'------------')# , file = f2)
print(np.mean(RMSEtrain[-20:-1]))#, file = f2)
print(np.mean(RMSEtest[-20:-1]))#, file = f2)
#print('-------------------RMSE of sum--------------------------')
#print(np.mean(RMSE1train[400:600]), np.std(RMSEtrain[400:600]))
#print(np.mean(RMSE1test[400:600]), np.std(RMSEtest[400:600]))
print('--------------------MAPE of conditional mean with lag '+str(TIME_LAG)+'------------)#')#, file = f2)
print(np.mean(MAPEtrain[-20:-1])*100,'%')#, file = f2)
print(np.mean(MAPEtest[-20:-1])*100,'%')#, file = f2)








