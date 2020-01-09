# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 17:04:56 2019

@author: wyzhang
"""

import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

net_to_test = torch.load( 'ConvlstmReal10.pkl')

TIME_LAG = 10
EPOCH = 100
BATCH_SIZE = 128

#TIME_LAG = 3
LRsigma = 1e-2
LRgamma = 1e-2
LRnet = 1e-2

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

logIVSreal_train = difflogval.view(-1, 8, 8)
logIVSreal_test = difflogtest.view(-1, 8, 8)

dataTrain = logIVSreal_train
dataTest = logIVSreal_test

T_train = dataTrain.shape[0]
T_test = dataTest.shape[0]
N1 = dataTrain.shape[1]
N2 = dataTrain.shape[2]

print('Lag '+str(TIME_LAG)+' model')
logIVSt_y1 = dataTrain[TIME_LAG:T_train, :, :]
logIVSt_x1 = torch.zeros([T_train-TIME_LAG,TIME_LAG, N1, N2])
for i in range(int(T_train - TIME_LAG)):
  for j in range(int(TIME_LAG)):
         logIVSt_x1[i,j,:, :] = dataTrain[i+TIME_LAG-1-j, :].view(-1,1, N1, N2)

logIVSt_y = dataTest[TIME_LAG:T_test, :, :]     
logIVSt_x = torch.zeros([T_test-TIME_LAG,TIME_LAG, N1, N2])
for i in range(int(T_test - TIME_LAG)):
  for j in range(int(TIME_LAG)):
         logIVSt_x[i,j,:, :] = dataTest[i+TIME_LAG-1-j, :].view(-1,1, N1, N2)
         
TIME_STEPS = logIVSt_x.shape[0]
         

lvsforPred = torch.cat((dataTrain[-TIME_LAG:,:,:], dataTest), 0)

N_sample = 200
h = 30
def CondExpectErr(N_sample, h):
    with torch.no_grad(): 
        #timeforward = lvsforPred.shape[0]
        c = 0
        MSE = []
        MAPE = []
        AE = []
        for step in range(TIME_LAG, N_sample + TIME_LAG):
            logIVShat = torch.zeros((TIME_LAG + h, N1, N2))
            #print(logIVShat.shape)
            logIVShat[:TIME_LAG, :, :] = lvsforPred[step: step + TIME_LAG, :, :]
            for i in range(h):
                x = logIVShat[i:TIME_LAG + i,:, :].view(1, TIME_LAG, N1, N2)
                net_to_test.eval()
                output,_ = net_to_test.forward(x)
                logIVShat[TIME_LAG + i ,:, :] = output[0].view(-1, 8, 8)
        
            diff = torch.abs(logIVShat[TIME_LAG + h -1,:, :] - lvsforPred[step + TIME_LAG + h - 1, :, :])  
            ae = torch.mean(torch.abs(diff))
            mse = torch.mean(diff**2)
            mape = torch.abs(torch.mean(diff/lvsforPred[step + TIME_LAG + h -1, :, :]))
            MSE.append(mse.item())
            MAPE.append(mape.item())
            AE.append(ae.item())
            c+=1
       # print(c)
        print(np.mean(AE), np.sqrt(np.mean(MSE)), np.mean(MAPE))
    return np.array(AE), np.array(MSE), np.array(MAPE)


ae1, se1, ape1 = CondExpectErr(200, 1) 
ae10, se10, ape10 = CondExpectErr(200, 10)   
ae30, se30, ape30 = CondExpectErr(200, 30)     

np.savetxt('se1 real.txt', se1)    
np.savetxt('ape1 real.txt', ape1)
np.savetxt('se10 real.txt', se10)    
np.savetxt('ape10 real.txt', ape10)
np.savetxt('se30 real.txt', se30)    
np.savetxt('ape30 real.txt', ape30)
np.savetxt('ae1 real.txt', ae1)
np.savetxt('ae10 real.txt', ae10)    
np.savetxt('ae30 real.txt', ae30)








