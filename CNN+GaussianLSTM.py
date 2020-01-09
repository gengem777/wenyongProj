# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 19:19:48 2019

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
#import os
import time

#rnn = nn.LSTM(10, 20, 2)  #(input_size,hidden_size,num_layers)
#input = torch.randn(128, 3, 10)
#h0 = torch.randn(2, 3, 20)
#c0 = torch.randn(2, 3, 20)
#output, (hn, cn) = rnn(input, None)
torch.manual_seed(1)

TIME_LAG = 8


EPOCH = 1200
BATCH_SIZE = 128
INPUT_SIZE = 64
HIDDEN_SIZE = 12
OUTPUT_SIZE = 64
#TIME_LAG = 3
LRsigma = 1e-2
LRgamma = 1e-2
LRnet = 5*1e-2

tau = [0.0833, 0.1786, 0.2738, 0.3690, 0.4643, 0.5595, 0.6548, 0.7500]
moneyness = [0.7000, 0.8000, 0.9000, 1, 1.1000, 1.2000, 1.3000, 1.4000]


class Decoder(nn.Module):
    def __init__(self,hiddensize, outputsize):
        super(Decoder, self).__init__()
        
        self.bn0 = nn.BatchNorm1d(hiddensize)
        self.fc1 = nn.Linear(hiddensize, 12)
        self.bn1 = nn.BatchNorm1d(12)
        self.fc2 = nn.Linear(12,  outputsize)
        
    def forward(self, x):
        x = self.bn0(x)
        x = self.bn1(torch.tanh(self.fc1(x)))
        x = self.fc2(x)
        return x 

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder,self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(
                        in_channels = 1,
                        out_channels = 6,
                        kernel_size = 5,
                        stride = 1,
                        padding = 2,
                ),
                nn.ReLU(),
                nn.MaxPool2d(2),
        )
        self.out = nn.Linear(6*4*4, 12) 
        
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.shape[0], 6*4*4)
        output = self.out(x)
        return output
    

class CNNLSTM(nn.Module):
    def __init__(self,):
        super(CNNLSTM, self).__init__()
        self.lstm = nn.GRU(
                input_size = 12,
                hidden_size = HIDDEN_SIZE,
                num_layers = 1,
                batch_first = True
                )
        
        self.cnn = CNNEncoder()
        
        self.out = Decoder(HIDDEN_SIZE, OUTPUT_SIZE)
        
    def forward(self, x):#, hstate):
        #print(x.shape)
        z = torch.zeros(x.shape[0], x.shape[1], 12)
        #print(z)
        for i in range(TIME_LAG):
            z[:, i, :] = self.cnn.forward(x[:, i, :, :].view(x.shape[0], 1, 8, 8))
        r_out, _ = self.lstm(z, None)
       # print(z.shape)
        out = self.out(r_out[:, -1, :] + z[:, -1, :]) #residual net
        #print(r_out[:, -1, :].shape)
        out = out.view(-1, 8, 8)
        #out = F.softplus(out)
        return out
    

    

    

def Kernel(sigma, gamma, moneyness, tau):
    length = N1*N2
    width = N1*N2
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
    

def nMarginalloglikelihood(sigma, gamma, moneyness, tau, y):
    #print(y)
    y = y.view(-1, N1*N2)
    sig = Kernel(sigma, gamma, moneyness, tau)
    mu = torch.zeros((1, y.shape[1]))
    logllhood = MultivariateNormal(mu, sig).log_prob(y)
    ret = -torch.mean(logllhood)
    return ret


dataFile = 'IVS1260AllData.mat'
data = scio.loadmat(dataFile)
IVS = data['implyvoloutofsample']
IVS = torch.log(torch.from_numpy(IVS).float())
#log_IVS = IVS
N1 = IVS.shape[0]
N2 = IVS.shape[1]
T = IVS.shape[2]
IVS = IVS.view(N1*N2, T).t()


IVS_train = IVS[:756]
IVS_val = IVS[756:1008]
IVS_test = IVS[1008:]

T_train = IVS_train.shape[0]
T_val = IVS_val.shape[0]
T_test = IVS_test.shape[0]
    
    
y_train = IVS_train[TIME_LAG:T_train,:]
X_train = torch.zeros([T_train-TIME_LAG,TIME_LAG, INPUT_SIZE])
for i in range(int(T_train - TIME_LAG)):
  for j in range(int(TIME_LAG)):
         X_train[i,j,:] = IVS_train[i+TIME_LAG-1-j, :].view(1,1,INPUT_SIZE)

'''
  
dataFile = 'IVS88126outofsample.mat'
data = scio.loadmat(dataFile)
IVSoutofsample = data['implyvoloutofsample']

IVSoutofsample = torch.from_numpy(IVSoutofsample).float()
log_IVS = torch.log(IVSoutofsample)

N1 = log_IVS.shape[0]
N2 = log_IVS.shape[1]
T = log_IVS.shape[2]
log_IVS = log_IVS.view(N1*N2, T).t()
'''
y_val = IVS_val[TIME_LAG:T_val,:]     
X_val = torch.zeros([T_val-TIME_LAG,TIME_LAG, INPUT_SIZE])
for i in range(int(T_val - TIME_LAG)):
  for j in range(int(TIME_LAG)):
         X_val[i,j,:] = IVS_val[i+TIME_LAG-1-j, :].view(1,1,INPUT_SIZE)      


X_train = X_train.view(-1, TIME_LAG, N1, N2)    
y_train = y_train.view(-1, N1, N2)

X_val = X_val.view(-1, TIME_LAG, N1, N2)    
y_val = y_val.view(-1, N1, N2)
    
torch_trainingset = Data.TensorDataset(X_train, y_train)
trainloader = Data.DataLoader(dataset=torch_trainingset, batch_size=BATCH_SIZE, shuffle = False, num_workers = 0)

torch_testset = Data.TensorDataset(X_val, y_val)
testloader = Data.DataLoader(dataset = torch_testset, batch_size = 126, shuffle = False, num_workers = 0)

regressor = CNNLSTM()

sigma = nn.Parameter(torch.Tensor([0.45]), requires_grad = True)
gamma = nn.Parameter(torch.Tensor([1.05]), requires_grad = True)
optimizer = optim.Adam([{'params': sigma, 'lr': LRsigma}, {'params': regressor.parameters(), 'lr': LRnet}, {'params': gamma, 'lr': LRgamma}], amsgrad = True)

#hstate = None
loss_fn = nn.MSELoss(reduce = True, size_average = True)
reg_loss=Re.Regularization(regressor, weight_decay = 10, p=1) 
traininglosses = []
testlosses = []
RMSEtrain = []
RMSEtest = []
MAPEtrain = []
MAPEtest = []
#writer = SummaryWriter(log_dir = 'loss',)
start = time.clock()
#if __name__ == "__main__":
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,  step_size=50, gamma=0.3)
for epoch in range(EPOCH):
    print("The", epoch,"st", "epoch")
    #trainloss, rmse, mape = train()
    regressor.train()
    batch_nll = []
    for batch_idx, (batch_x, batch_y) in enumerate(trainloader):
       predict = regressor.forward(batch_x)
       #print(predict)
       diff = batch_y - predict
       trainloss = nMarginalloglikelihood(sigma, gamma, moneyness, tau, diff)      
       #loss = loss_fn(predict, batch_y)
       #optimizer.zero_grad()
       #loss.backward()
       #optimizer.step(
       
       #loss = trainloss + reg_loss.forward(regressor)
       optimizer.zero_grad()
       trainloss.backward()
       optimizer.step()
       rmse = torch.sqrt(torch.mean(diff**2))
       mape = torch.abs(torch.mean(torch.abs(diff)/batch_y))
       batch_nll.append(loss_fn(batch_y, predict).item())
       #rmse = torch.sqrt(loss)
      # mape = torch.abs(torch.mean(torch.abs(diff)/batch_y))
      #batch_nll.append(loss.item())
    traininglosses.append(np.mean(batch_nll))
    RMSEtrain.append(rmse.item())
    # RMSE1train.append((torch.sqrt(torch.sum(diff**2))).item())
    MAPEtrain.append(mape.item())
    print(epoch, "percentage trainingloss is:", (mape*100).item(),'%')
    if epoch % 20 == 0:
        print("----------------------------------------------------------------------------------------")
        print(epoch, "trainingloss is:", np.mean(batch_nll))
        print(epoch, "RMSE trainingloss is:", rmse.item())
        print(epoch, "percentage trainingloss is:", (mape*100).item(),'%')
    #print("----------------------------------------------")
    with torch.no_grad():
        #testloss, rmse, mape = test()
        regressor.eval()
        batch_nll = []
        for batch_idx, (batch_x, batch_y) in enumerate(testloader):
           predict = regressor.forward(batch_x)
       #hstate = hstate.data
           diff = batch_y - predict
           #testloss = nMarginalloglikelihood(sigma, gamma, moneyness, tau, diff)
           testloss = nMarginalloglikelihood(sigma, gamma, moneyness, tau, diff) + reg_loss.forward(regressor)
           rmse = torch.sqrt(torch.mean(diff**2))
           #rmse = torch.sqrt(loss)
           mape = torch.abs(torch.mean(torch.abs(diff)/batch_y))
           batch_nll.append(loss_fn(batch_y, predict).item())
    testlosses.append(np.mean(batch_nll))
    RMSEtest.append(rmse.item())
    MAPEtest.append(mape.item())
    print(epoch, "percentage testloss is:", (mape*100).item(),'%')
    if epoch % 20 == 0:
        print(epoch, "sigma is:", torch.sqrt(sigma**2).item(), "gamma is:", gamma.item())
        print(epoch, "testloss is:", np.mean(batch_nll))
        print(epoch, "RMSE testingloss is:", rmse.item())
        print(epoch, "percentage testloss is:", (mape*100).item(),'%')
        print("----------------------------------------------------------------------------------------")
    #writer.add_scalars('loss/NLL', {'train NLL loss/test': np.float64(trainloss.item()), 'test NLL loss': np.float64(testloss.item())}, epoch)
    #writer.add_scalars('loss/NLL', {'train RMSE loss': trainloss.item(), 'test RMSE loss': testloss.item()}, epoch)
    #writer.add_scalars('loss/NLL', {'train MAPE loss': trainloss.item(), 'test MAPE loss': testloss.item()}, epoch)
    scheduler.step()

#writer.close()

#store the network
#torch.save(regressor, 'CNNLSTM' + str(TIME_LAG) + '.pkl')
#torch.save(regressor.state_dict(), 'param CNNLSTM'+ str(TIME_LAG) +'.pkl')

#print(sigma, gamma)
   #with open('param.txt', 'wt') as f1:

print('LAG '+'TIME_LAG'+' model: ' 'sigma is:',  torch.sqrt(sigma**2).item(), 'gamma is:', gamma.item())#, file = f1) # sigma 0.1810, gamma 1.6471



plt.figure()
plt.title('Summary for lag '+str(TIME_LAG)+'model')
plt.subplot(221)
plt.plot(traininglosses, alpha = 0.4, color = 'red', label = 'training NLL')
plt.plot(testlosses, alpha = 0.5, color = 'blue', label = 'validating NLL')
plt.xlabel('epochs')
#plt.ylim(ymin = -300, ymax = 400)
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
plt.hist(testlosses[-200:-1], bins = 50, density = 0, facecolor = 'blue', edgecolor = 'black', alpha = 0.7)
plt.show()

elapsed = (time.clock() - start)
print("Time used:",elapsed)
#with open('summary.txt', 'wt') as f2:
print('Summary sheet for lag '+str(TIME_LAG)+'model')
print('--------------------Negative Loglikelihood of lag '+str(TIME_LAG)+'--------------')#, file = f2)
print(np.mean(traininglosses[-20:-1]))#, file = f2)
print(np.mean(testlosses[-20:-1]))#, file = f2)
print('-------------------RMSE of conditional mean with lag '+str(TIME_LAG)+'------------')# , file = f2)
print(np.mean(RMSEtrain[-20:-1]))#, file = f2)
print(np.mean(RMSEtest[-20:-1]))#, file = f2)
print('--------------------MAPE of conditional mean with lag '+str(TIME_LAG)+'------------)#')#, file = f2)
print(np.mean(MAPEtrain[-20:-1])*100,'%')#, file = f2)
print(np.mean(MAPEtest[-20:-1])*100,'%')#, file = f2)

    
    
    
    