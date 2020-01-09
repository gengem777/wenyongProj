# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 20:41:41 2019

@author: wyzhang
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:40:23 2019

@author: wyzhang
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
#from sklearn.model_selection import train_test_split

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

#rnn = nn.LSTM(10, 20, 2)  #(input_size,hidden_size,num_layers)
#input = torch.randn(128, 3, 10)
#h0 = torch.randn(2, 3, 20)
#c0 = torch.randn(2, 3, 20)
#output, (hn, cn) = rnn(input, None)
torch.manual_seed(1)

TIME_LAG = 10


EPOCH = 1000
Kernel_EPOCH = 100
BATCH_SIZE = 64
INPUT_SIZE = 64
HIDDEN_SIZE = 12
OUTPUT_SIZE = 64
#TIME_LAG = 3
LRsigma = 1e-2
LRgamma = 1e-2
LRnet = 1e-2

tau = [0.0833, 0.1786, 0.2738, 0.3690, 0.4643, 0.5595, 0.6548, 0.7500]
moneyness = [0.7000, 0.8000, 0.9000, 1, 1.1000, 1.2000, 1.3000, 1.4000]


class subnet(nn.Module):
    def __init__(self,hiddensize, outputsize):
        super(subnet, self).__init__()
  
        self.bn0 = nn.BatchNorm1d(hiddensize)
        self.fc1 = nn.Linear(hiddensize, 6)       
        self.bn1 = nn.BatchNorm1d(6)
        self.fc2 = nn.Linear(6,  outputsize)
      
    def forward(self, x):
        x = self.bn0(x)
        x = self.bn1(torch.tanh(self.fc1(x)))
        x = self.fc2(x)
        return x
    

class RNN(nn.Module):
    def __init__(self, INPUT_SIZE, OUTPUT_SIZE):
        super(RNN, self).__init__()
        
        self.rnn = nn.LSTM(
                input_size = INPUT_SIZE,
                hidden_size = HIDDEN_SIZE,
                num_layers = 1,
                batch_first = True
                )
        
        self.out = subnet(HIDDEN_SIZE, OUTPUT_SIZE)
        
    def forward(self, x):#, hstate):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])        
        return out

           
def Kernel(sigma, gamma, moneyness, tau):
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
    

def nMarginalloglikelihood(sigma, gamma, moneyness, tau, y):
    sig = Kernel(sigma, gamma, moneyness, tau)
    mu = torch.zeros((1, y.shape[1]))
    logllhood = MultivariateNormal(mu, sig).log_prob(y)
    ret = -torch.mean(logllhood)
    return ret

loss_fn = nn.MSELoss()

def train():
    regressor.train()
    for batch_idx, (batch_x, batch_y) in enumerate(trainloader):
        predict = regressor.forward(batch_x)
        diff = batch_y - predict
        trainloss = nMarginalloglikelihood(sigma, gamma, moneyness, tau, diff)
        optimizer.zero_grad()
        trainloss.backward()
        optimizer.step()
        rmse = torch.sqrt(torch.mean(diff**2))
        mape = abs(torch.mean(diff/batch_y))
    return trainloss, rmse, mape


def test():
    regressor.eval()
    for batch_idx, (batch_x, batch_y) in enumerate(testloader):
           predict = regressor.forward(batch_x)
           #hstate = hstate.data
           diff = batch_y - predict
           testloss = nMarginalloglikelihood(sigma, gamma, moneyness, tau, diff) 
           rmse = torch.sqrt(torch.mean(diff**2))
           mape = abs(torch.mean(diff/batch_y))           
    return testloss, rmse, mape

def init_net(net):
    for mod in net:
        if isinstance(mod, nn.Linear):
            nn.init.xavier_normal_(mod.weight)
            nn.init.normal_(mod.bias,0,1)


#==============================================================================
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
#y_train = y_train.view(-1, 8, 8)
#X_train = X_train.view(-1, TIME_LAG, 8, 8)

y_val = lnIVS_val[TIME_LAG:T_val,:]     
X_val = torch.zeros([T_val-TIME_LAG,TIME_LAG, INPUT_SIZE])
for i in range(int(T_val - TIME_LAG)):
  for j in range(int(TIME_LAG)):
         X_val[i,j,:] = lnIVS_val[i+TIME_LAG-1-j, :].view(1,1,INPUT_SIZE)
#y_val = y_val.view(-1, 8, 8)
#X_val = X_val.view(-1, TIME_LAG, 8, 8)

#===============================================================================
'''

dataFile = 'IVS88756.mat'
data = scio.loadmat(dataFile)
IVS = data['implyvolsurfaces']
IVS = torch.from_numpy(IVS).float()
log_IVS = torch.log(IVS)
N1 = IVS.shape[0]
N2 = IVS.shape[1]
T = IVS.shape[2]
log_IVS = log_IVS.view(N1*N2, T).t()


dataFile = 'IVS1260AllData.mat'
data = scio.loadmat(dataFile)
IVS = data['implyvoloutofsample']
IVS = torch.from_numpy(IVS).float()
log_IVS = torch.log(IVS)
N1 = IVS.shape[0]
N2 = IVS.shape[1]
T = IVS.shape[2]
log_IVS = log_IVS.view(N1*N2, T).t()


lnIVS_train = log_IVS[:756]
lnIVS_val = log_IVS[756:1008]
lnIVS_test = log_IVS[1008:]

T_train = lnIVS_train.shape[0]
T_val = lnIVS_val.shape[0]
T_test = lnIVS_test.shape[0]


'''
'''
lnIVS_train = logIVSreal_train
lnIVS_val = logIVSreal_test
T_train = lnIVS_train.shape[0]
T_val = lnIVS_val.shape[0]


print('Lag '+str(TIME_LAG)+' model')
y_train = lnIVS_train[TIME_LAG:T_train,:]
X_train = torch.zeros([T_train-TIME_LAG,TIME_LAG, INPUT_SIZE])
for i in range(int(T_train - TIME_LAG)):
  for j in range(int(TIME_LAG)):
         X_train[i,j,:] = lnIVS_train[i+TIME_LAG-1-j, :].view(1,1,INPUT_SIZE)
'''
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
'''
y_val = lnIVS_val[TIME_LAG:T_val,:]     
X_val = torch.zeros([T_val-TIME_LAG,TIME_LAG, INPUT_SIZE])
for i in range(int(T_val - TIME_LAG)):
  for j in range(int(TIME_LAG)):
         X_val[i,j,:] = lnIVS_val[i+TIME_LAG-1-j, :].view(1,1,INPUT_SIZE)  
'''            
#logIVSt_x = logIVSt_x.numpy()            
#logIVSt_y = logIVSt_y.numpy()
#TIME_STEPS = logIVSt_x.shape[0]
         


#logIVSt_x = logIVSt_x.numpy()
#logIVSt_y = logIVSt_y.numpy()

#X_train, X_test, y_train, y_test = train_test_split(logIVSt_x, logIVSt_y, test_size = 0.3, shuffle = False)

#X_train = torch.from_numpy(X_train).float()
#y_train = torch.from_numpy(y_train).float()
#X_test = torch.from_numpy(X_test).float()
#y_test = torch.from_numpy(y_test).float()

torch_trainingset = Data.TensorDataset(X_train, y_train)
trainloader = Data.DataLoader(dataset=torch_trainingset, batch_size=BATCH_SIZE, shuffle = False, num_workers = 0)

torch_testset = Data.TensorDataset(X_val, y_val)
testloader = Data.DataLoader(dataset = torch_testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)



regressor = RNN(INPUT_SIZE, OUTPUT_SIZE)



sigma = nn.Parameter(torch.Tensor([0.02]), requires_grad = True)
gamma = nn.Parameter(torch.Tensor([3.33]), requires_grad = True)
optimizer = optim.Adam([{'params': sigma, 'lr': LRsigma}, {'params': regressor.parameters(), 'lr': LRnet}, {'params': gamma, 'lr': LRgamma}], amsgrad = True)

net_opt = optim.Adam(regressor.parameters(), lr = LRnet, amsgrad = True)
kernel_opt = optim.Adam([{'params': sigma, 'lr': LRsigma}, {'params': gamma, 'lr': LRgamma}], amsgrad = True)

reg_loss=Re.Regularization(regressor, weight_decay = 10, p=1) 
#hstate = None


trainlosses = []
testlosses = []
RMSEtrain = []
RMSEtest = []
MAPEtrain = []
MAPEtest = []
#writer = SummaryWriter(log_dir = 'loss',)
start = time.clock()
#if __name__ == "__main__":
scheduler1 = torch.optim.lr_scheduler.StepLR(net_opt,  step_size=50, gamma=0.5)
for epoch in range(EPOCH):
    #print("The", epoch,"st", "epoch")
    #trainloss, rmse, mape = train()
    regressor.train()
    batch_nll = []
    for batch_idx, (batch_x, batch_y) in enumerate(trainloader):
       predict = regressor.forward(batch_x)
       diff = batch_y - predict
       trainloss = nMarginalloglikelihood(sigma, gamma, moneyness, tau, diff) #+ reg_loss.forward(regressor)
       
       
       #trainloss = loss_fn(predict, batch_y) + reg_loss.forward(regressor)
       #optimizer.zero_grad()
       #loss.backward()
       #optimizer.step()
       
       optimizer.zero_grad()
       trainloss.backward()
       net_opt.step()
       rmse = torch.sqrt(torch.mean(diff**2))
       mape = torch.abs(torch.mean(torch.abs(diff)/batch_y))
       batch_nll.append(trainloss.item())
       #rmse = torch.sqrt(loss)
      # mape = torch.abs(torch.mean(torch.abs(diff)/batch_y))
      #batch_nll.append(loss.item())
    trainlosses.append(np.mean(batch_nll))
    RMSEtrain.append(rmse.item())
   # RMSE1train.append((torch.sqrt(torch.sum(diff**2))).item())
    MAPEtrain.append(mape.item())
    if epoch % 10 == 0:
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
           testloss = nMarginalloglikelihood(sigma, gamma, moneyness, tau, diff) #+ reg_loss.forward(regressor)
           #testloss = loss_fn(predict, batch_y) + reg_loss.forward(regressor)
           rmse = torch.sqrt(torch.mean(diff**2))
           #rmse = torch.sqrt(loss)
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

#writer.close()

#store the network
torch.save(regressor, 'lstmReal' + str(TIME_LAG) + '.pkl')
torch.save(regressor.state_dict(), 'param lstmReal'+ str(TIME_LAG) +'.pkl')

#print(sigma, gamma)
   #with open('param.txt', 'wt') as f1:
'''
reghat = torch.load('lstmReal' + str(TIME_LAG) + '.pkl')
scheduler2 = torch.optim.lr_scheduler.StepLR(kernel_opt,  step_size=200, gamma=0.5)
trainloss2 = []
for epoch in range(Kernel_EPOCH):
    g = reghat.forward(X_train)
    d = y_train - g
    loss2 = nMarginalloglikelihood(sigma, gamma, moneyness, tau, d)
    kernel_opt.zero_grad()
    loss2.backward()
    kernel_opt.step()
    trainloss2.append(loss2)
    if epoch % 10 == 0:
        print(epoch, "sigma is:", torch.sqrt(sigma**2).item(), "gamma is:", gamma.item())
    scheduler2.step()
'''
print('LAG '+'TIME_LAG'+' model: ' 'sigma is:',  torch.sqrt(sigma**2).item(), 'gamma is:', gamma.item())#, file = f1) # sigma 0.1810, gamma 1.6471



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



    