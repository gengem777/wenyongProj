#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:05:16 2019

@author: apple
"""

import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
#from torch.nn import Parameter
from sklearn.decomposition import PCA
from torch.distributions.normal import Normal
import torch.optim as optim
#from statsmodels.graphics.tsaplots import plot_acf  
#from statsmodels.graphics.tsaplots import plot_pacf 
#from statsmodels.tsa.stattools import pacf
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

torch.manual_seed(1)

def dm_test(actual_lst, pred1_lst, pred2_lst, h = 1, crit="MSE", power = 2):
    # Routine for checking errors
    def error_check():
        rt = 0
        msg = ""
        # Check if h is an integer
        if (not isinstance(h, int)):
            rt = -1
            msg = "The type of the number of steps ahead (h) is not an integer."
            return (rt,msg)
        # Check the range of h
        if (h < 1):
            rt = -1
            msg = "The number of steps ahead (h) is not large enough."
            return (rt,msg)
        len_act = len(actual_lst)
        len_p1  = len(pred1_lst)
        len_p2  = len(pred2_lst)
        # Check if lengths of actual values and predicted values are equal
        if (len_act != len_p1 or len_p1 != len_p2 or len_act != len_p2):
            rt = -1
            msg = "Lengths of actual_lst, pred1_lst and pred2_lst do not match."
            return (rt,msg)
        # Check range of h
        if (h >= len_act):
            rt = -1
            msg = "The number of steps ahead is too large."
            return (rt,msg)
        # Check if criterion supported
        if (crit != "MSE" and crit != "MAPE" and crit != "MAD" and crit != "poly"):
            rt = -1
            msg = "The criterion is not supported."
            return (rt,msg)  
        # Check if every value of the input lists are numerical values
        from re import compile as re_compile
        comp = re_compile("^\d+?\.\d+?$")  
        def compiled_regex(s):
            """ Returns True is string is a number. """
            if comp.match(s) is None:
                return s.isdigit()
            return True
        for actual, pred1, pred2 in zip(actual_lst, pred1_lst, pred2_lst):
            is_actual_ok = compiled_regex(str(abs(actual)))
            is_pred1_ok = compiled_regex(str(abs(pred1)))
            is_pred2_ok = compiled_regex(str(abs(pred2)))
            if (not (is_actual_ok and is_pred1_ok and is_pred2_ok)):  
                msg = "An element in the actual_lst, pred1_lst or pred2_lst is not numeric."
                rt = -1
                return (rt,msg)
        return (rt,msg)
    
    # Error check
    error_code = error_check()
    # Raise error if cannot pass error check
    if (error_code[0] == -1):
        raise SyntaxError(error_code[1])
        return
    # Import libraries
    from scipy.stats import t
    import collections
    import pandas as pd
    import numpy as np
    
    # Initialise lists
    e1_lst = []
    e2_lst = []
    d_lst  = []
    
    # convert every value of the lists into real values
    actual_lst = pd.Series(actual_lst).apply(lambda x: float(x)).tolist()
    pred1_lst = pd.Series(pred1_lst).apply(lambda x: float(x)).tolist()
    pred2_lst = pd.Series(pred2_lst).apply(lambda x: float(x)).tolist()
    
    # Length of lists (as real numbers)
    T = float(len(actual_lst))
    
    # construct d according to crit
    if (crit == "MSE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append((actual - p1)**2)
            e2_lst.append((actual - p2)**2)
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAD"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs(actual - p1))
            e2_lst.append(abs(actual - p2))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAPE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs((actual - p1)/actual))
            e2_lst.append(abs((actual - p2)/actual))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "poly"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(((actual - p1))**(power))
            e2_lst.append(((actual - p2))**(power))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)    
    
    # Mean of d        
    mean_d = pd.Series(d_lst).mean()
    
    # Find autocovariance and construct DM test statistics
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N-k):
              autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
        return (1/(T))*autoCov
    gamma = []
    for lag in range(0,h):
        gamma.append(autocovariance(d_lst,len(d_lst),lag,mean_d)) # 0, 1, 2
    V_d = (gamma[0] + 2*sum(gamma[1:]))/T
    DM_stat=V_d**(-0.5)*mean_d
    harvey_adj=((T+1-2*h+h*(h-1)/T)/T)**(0.5)
    DM_stat = harvey_adj*DM_stat
    # Find p-value
    p_value = 2*t.cdf(-abs(DM_stat), df = T - 1)
    
    # Construct named tuple for return
    dm_return = collections.namedtuple('dm_return', 'DM p_value')
    
    rt = dm_return(DM = DM_stat, p_value = p_value)
    
    return rt

def FuncPCA(XMat, k):
    average = np.mean(XMat, axis = 0) 
    m, n = np.shape(XMat)
    data_adjust = []
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    #delta = 0.00952
    covX = np.cov(data_adjust.T)#*0.00952 #计算协方差矩阵
    featValue, featVec=  np.linalg.eig(covX)  #求解协方差矩阵的特征值和特征向量
    index = np.argsort(-featValue) #按照featValue进行从大到小排序
    finalData = []
    if k > n:
        print ("k must lower than feature number")
        return
    else:
        #注意特征向量时列向量，而numpy的二维矩阵(数组)a[m][n]中，a[1]表示第1行值
        selectVec = np.matrix(featVec.T[index[:k]]) #所以这里需要进行转置
        finalData = data_adjust * selectVec.T 
        reconData = (finalData * selectVec) + average  
    return featValue, featVec, finalData, reconData


def impvolfunc(m, tau, impI, h1 = 1, h2 = 1):  #gaussian interpolation
    Tau = np.array([0.0833, 0.1786, 0.2738, 0.3690, 0.4643, 0.5595, 0.6548, 0.7500])
    moneyness = np.array([0.7000, 0.8000, 0.9000, 1, 1.1000, 1.2000, 1.3000, 1.4000])
    Tau, moneyness = np.meshgrid(Tau, moneyness)
    Tau = Tau.ravel()
    moneyness = moneyness.ravel()
    Nt = len(Tau)
    Nm = len(moneyness)
    def g(x, y, h1, h2):
        g = np.exp(-x**2/2/h1)*np.exp(-y**2/2/h2)/2*np.pi
        return g
    tau = tau*np.ones([Nt])
    m = m*np.ones([Nm])
    impvol = np.exp(impI)
    norminator = np.dot(impvol, g(m - moneyness, tau - Tau, h1, h2))
    denorminator = np.sum(g(m - moneyness, tau - Tau, h1, h2))
    ret = norminator/denorminator
    return ret

def Proj(X, f, X_0):
    N = X.shape[0]
    d = f.shape[0]
    x = np.zeros((N, d))
    delta = 0.00952
    for j in range(N):
       for i in range(d):
          x[j, i] = np.dot(X[j]- X_0.numpy(), f[i])*delta
    return x

def AR1(x, x_bar, l):
    c = (1 - torch.exp(-l))*x_bar
    a = torch.exp(-l)*x
    ret = a + c
    return ret

def AR_sim(x, x_bar, l, phi):
    ar1 = AR1(x, x_bar, l)
    sigma = phi*torch.sqrt((1 - torch.exp(-l))/2*l)
    ret = ar1 + sigma * torch.randn(len(x_bar))
    return ret

def NLL(y, l, phi):
    sigma = phi*torch.sqrt((1 - torch.exp(-l))/2*l)
    mu = torch.Tensor([0])
    logllhood = Normal(mu, sigma).log_prob(y)
    ret = -torch.mean(logllhood)
    return ret

def PredX(x, x_bar, l, phi, X_0):  #predict next time step surface mean, given the global initial value
    xforward = AR1(x, x_bar, l)
    xsimulate = AR_sim(x, x_bar, l, phi)
    pred = X_0
    simul = X_0
    '''
    for i in range(3):
      pred += xforward[i]*eigenface[i]
      simul += xsimulate[i]*eigenface[i]
    '''
    pred = X_0 + xforward[0]*eigenface[0] + xforward[1]*eigenface[1] + xforward[2]*eigenface[2]
    simul = X_0 + xsimulate[0]*eigenface[0] + xsimulate[1]*eigenface[1] + xsimulate[2]*eigenface[2]
    return pred, simul, xforward


def CalPredErr(x, X_test, x_bar, timeahead, l, phi, X_0):
    T = x.shape[0]
    RMSErr = []
    MAPErr = []
    if timeahead < T:
      for t in range(timeahead):
        Xhat,_ ,_= PredX(x[t,:], x_bar, l, phi, X_0)
        Xtarget = X_test[t+1,:]
        rmse = torch.sqrt(torch.mean((Xhat - Xtarget)**2))
        mape = torch.mean(torch.abs((Xhat - Xtarget)/Xtarget))
        RMSErr.append(rmse.item())
        MAPErr.append(mape.item())
    else:
        print('shit')
    
    RMSErr = torch.Tensor(RMSErr)
    MAPErr = torch.Tensor(MAPErr)
    return RMSErr, MAPErr


def MultimeErr(x, X_test, x_bar, timeahead, l, phi):
    T = x.shape[0]
    RMSErr = []
    MAPErr = []
    Xhat = torch.zeros((timeahead+1,64))    
    xhat = torch.zeros((timeahead+1,3)) 
    Xhat[0,:], _, xhat[0,:] = PredX(xtrain1[-1,:], x_bar, l, phi)
    if timeahead <= T:
      for t in range(timeahead):
        Xhat[t+1,:],_ ,xhat[t+1,:]= PredX(xhat[t,:], x_bar, l, phi)
        Xtarget = X_test[t+1,:]
        rmse = torch.sqrt(torch.mean((Xhat[t+1,:] - Xtarget)**2))
        mape = torch.mean(torch.abs((Xhat[t+1,:] - Xtarget)/Xtarget))
        RMSErr.append(rmse.item())
        MAPErr.append(mape.item())
    else:
        print('shit')
    
    RMSErr = torch.Tensor(RMSErr)
    MAPErr = torch.Tensor(MAPErr)
    return RMSErr, MAPErr


def CondExpectErr(x, X_test, x_bar, tau, N, l, phi, X_0):
    T = x.shape[0]
    MSErr = []
    APErr = []
    AErr = []
    if N <= T - tau:
        for i in range(N):
            xhat = x[i, :]
            for j in range(tau):
               Xhat,_,xhat =  PredX(xhat, x_bar, l, phi, X_0)
            mse = torch.mean((Xhat - X_test[i + tau,:])**2)
            ape = torch.mean(torch.abs((Xhat - X_test[i + tau,:])/X_test[i + tau,:]))
            ae = torch.mean(torch.abs(Xhat - X_test[i + tau,:]))
            MSErr.append(mse.item())
            APErr.append(ape.item())
            AErr.append(ae.item())
    else:
        print('shit')
    AErr = torch.Tensor(AErr)
    MSErr = torch.Tensor(MSErr)
    APErr = torch.Tensor(APErr)
    return AErr, MSErr, APErr
    
#==============================================================================
IVS_realmarket = np.loadtxt('IVS')
logIVSreal = np.log(IVS_realmarket)
logIVSreal_train = logIVSreal[:1372]
logIVSreal_test = logIVSreal[1372:]
logIVSreal_train = torch.from_numpy(logIVSreal_train).float()
logIVSreal_test = torch.from_numpy(logIVSreal_test).float()

#===============================================================================


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


lnIVS_out = torch.cat((lnIVS_val, lnIVS_test), 0)
lnIVS_in = torch.cat((lnIVS_train, lnIVS_val), 0)






datafile1 = 'IVS88756.mat'
data1 = scio.loadmat(datafile1)
IVS1 = data1['implyvolsurfaces']
log_IVS1 = np.log(IVS1)
log_IVS1 = torch.from_numpy(log_IVS1).float()
Ntrain1 = log_IVS1.shape[0]
Ntrain2 = log_IVS1.shape[1]
Ttrain = log_IVS1.shape[2]
log_IVS1 = log_IVS1.view(Ntrain1*Ntrain2, Ttrain).t()
#log_IVS1 = log_IVS1.numpy()


datafile2 = 'IVS88252outofsample.mat'
data2 = scio.loadmat(datafile2)
IVS2 = data2['implyvoloutofsample']
log_IVS2 = np.log(IVS2)
log_IVS2 = torch.from_numpy(log_IVS2).float()
Ntest1 = log_IVS2.shape[0]
Ntest2 = log_IVS2.shape[1]
Ttest = log_IVS2.shape[2]
log_IVS2 = log_IVS2.view(Ntest1*Ntest2, Ttest).t()
#log_IVS2 = log_IVS2.numpy()










#------------------------------------------------------------------------------------------------

datatrain = logIVSreal_train
X_0 = logIVSreal_train[-1,:].view(1, 64)
datatest = logIVSreal_test


#dX_train = np.diff(lnIVS_train.numpy(), axis = 0)
dX_val = np.diff(lnIVS_val.numpy(), axis = 0)
dX_test = np.diff(lnIVS_test.numpy(), axis = 0)

dX = np.diff(datatrain.numpy(), axis = 0)
pca = PCA(n_components=10)
pca.fit(dX)
#print(pca.explained_variance_ratio_)
#print('特征值：{}\n特征向量：{}'.format(pca.explained_variance_,pca.components_)) 

plt.figure()
x = np.arange(1,11,1)
plt.grid(axis = 'x', linestyle = '-.')
plt.plot(x, pca.explained_variance_ratio_, 's-', color = 'r')
plt.xlabel('Rank')
#plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio')
plt.show()

eigenface = pca.components_[0:3,:]
'''
eigenface = torch.from_numpy(eigenface).float()
fig = plt.figure()
ax = Axes3D(fig)
Tau = np.array([0.0833, 0.1786, 0.2738, 0.3690, 0.4643, 0.5595, 0.6548, 0.7500])
moneyness = np.array([0.7000, 0.8000, 0.9000, 1, 1.1000, 1.2000, 1.3000, 1.4000])
Tau, moneyness = np.meshgrid(Tau, moneyness)
Xface1 = (eigenface[0].view(8, 8)).numpy()
Xface2 = (eigenface[1].view(8, 8)).numpy()
Xface3 = (eigenface[2].view(8, 8)).numpy()
ax.plot_surface(Tau, moneyness, Xface1, rstride=1, cstride=1, cmap='rainbow')
plt.show()
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(Tau, moneyness, Xface2, rstride=1, cstride=1, cmap='rainbow')
plt.show()
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(Tau, moneyness, Xface3, rstride=1, cstride=1, cmap='rainbow')
plt.show()

delta = 0.00952
X_bar = np.mean(datatrain.numpy(), axis=0)
x_bar = np.dot(X_bar.T, eigenface.T)*delta
xtrain = Proj(log_IVS1.numpy(), eigenface)
xtrain1 = torch.from_numpy(xtrain).float() 
'''         
def train(log_IVS1, x_bar, eigenface):

    xtrain = Proj(log_IVS1, eigenface, X_0)
    #xtest = Proj(log_IVS2, eigenface)
    #X_bar = np.mean(log_IVS1, axis=0)
    #x_bar = np.dot(X_bar.T, eigenface.T)*delta
    
    #prewash data
    xtrain_predictor = xtrain[0:-2,:]
    xtrain_target =xtrain[1:-1,:]
    
    #xtest_predictor = xtest[0:-2,:]
    #xtest_target = xtest[1:-1,:]
    
    x1train = torch.from_numpy(xtrain_predictor).float()
    y1train = torch.from_numpy(xtrain_target).float()
    #print(x1train.shape, y1train.shape)
    
    #train the first component series
    x_bar = torch.from_numpy(x_bar).float()
    lhat = []
    phihat = []
    for i in range(3):
        xtr = x1train[:,i].view(len(x1train[:,i]),1)
        ytr = y1train[:,i].view(len(y1train[:,i]),1)
        #print(xtr.shape, ytr.shape)
        l = nn.Parameter(torch.Tensor([0.1]), requires_grad = True)
        phi = nn.Parameter(torch.Tensor([0.1]), requires_grad = True)
        optimizer = optim.Adam([{'params': l, 'lr': 0.01},{'params': phi, 'lr': 0.01}], amsgrad = True)
        nlltrain = []
        RMSE = []
        MAPE = [] 
        for time in range(3000):
            optimizer.zero_grad()
            pred = AR1(xtr, x_bar[i], l)
            #print(pred)
            nll = NLL(ytr - pred, l, phi)
            nll.backward()
            optimizer.step()
            nlltrain.append(nll.item())
            rmse = torch.sqrt(torch.mean((ytr - pred)**2))
            mape = torch.mean(torch.abs((ytr - pred)/ytr))
            MAPE.append(mape.item())
            RMSE.append(rmse.item())
            #print(mape.item())
        lhat.append(l.item())
        phihat.append(phi.item())
        print(ytr.shape)    
        print(str(i+1)+'st', 'is:', l.item(), phi.item())   
        print('likelihood', 'is:', -np.mean(nlltrain[-100:-1]))
        print('RMSE', 'is:', np.mean(RMSE[-100:-1]))
        print('MAPE', 'is:', np.mean(MAPE[-100:-1]))
        plt.figure()
        plt.plot(nlltrain, alpha = 0.5, color = 'blue')
        plt.show()
        plt.figure()
        plt.plot(RMSE, alpha = 0.5, color = 'green')
        plt.show()
        plt.figure()
        plt.plot(MAPE, alpha = 0.5, color = 'red')
        plt.show()
        
    return torch.Tensor(lhat), torch.Tensor(phihat)


#lhat, phihat = train(datatrain.numpy(), x_bar, eigenface)




if __name__ == '__main__':
    delta = 0.00952
    X_bar = np.mean(datatrain.numpy(), axis=0)
    x_bar = np.dot(X_bar.T, eigenface.T)*delta
    xtrain = Proj(datatrain.numpy(), eigenface, X_0)
    xtrain1 = torch.from_numpy(xtrain).float()
    
    lhat, phihat = train(datatrain.numpy(), x_bar, eigenface)
    
    eigenface = torch.from_numpy(eigenface).float()
    Xtest = datatest
    xtest = Proj(datatest.numpy(), eigenface, X_0)
    xtest1 = torch.from_numpy(xtest).float()
    x_bar1 = torch.from_numpy(x_bar).float()      
    
    
    '''
    #predict just one day step, predict I_{t+1} conditioned on \mathcal{F}_t , for any t. 
    rmserr, maperr = CalPredErr(xtest1, Xtest, x_bar1, xtest1.shape[0]-1)  
    print(rmserr[1].item(), maperr[1].item())
    print(rmserr[7].item(), maperr[7].item())  
    print(rmserr[30].item(), maperr[30].item())
    print(rmserr[-1].item(), maperr[-1].item())
    
    
    print('----------------------------------------------------')
    #predict multi-time period
    rmserr1, maperr1 = MultimeErr(xtest1, Xtest, x_bar1, xtest1.shape[0]-1)  
    print(rmserr1[1].item(), maperr1[1].item())
    print(rmserr1[7].item(), maperr1[7].item())  
    print(rmserr1[30].item(), maperr1[30].item())
    print(rmserr1[-1].item(), maperr1[-1].item())
    
    plt.figure()
    
    plt.plot(maperr, color = 'yellow', label = 'MAPE of single-prediction')
    plt.plot(maperr1, alpha = 0.7, color = 'purple', label = 'MAPE of multi-prediction')
    plt.xlabel('tau')
    plt.ylabel('MAPE')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(maperr1 - maperr)
    plt.show()
    '''
    Aae1, Ase1, Aape1 = CondExpectErr(xtest1, Xtest, x_bar1, 1, 200, lhat, phihat, X_0)
    print(torch.sqrt(torch.mean(Ase1)).item(), torch.mean(Aape1).item())
    Aae10, Ase10, Aape10 = CondExpectErr(xtest1, Xtest, x_bar1, 10, 200, lhat, phihat, X_0)
    print(torch.sqrt(torch.mean(Ase10)).item(), torch.mean(Aape10).item())
    Aae30, Ase30, Aape30 = CondExpectErr(xtest1, Xtest, x_bar1, 30, 200, lhat, phihat, X_0)
    print(torch.sqrt(torch.mean(Ase30)).item(), torch.mean(Aape30).item())
    
    np.savetxt('Ase1.txt', Ase1)    
    np.savetxt('Aape1.txt', Aape1)
    np.savetxt('Ase10.txt', Ase10)    
    np.savetxt('Aape10.txt', Aape10)
    np.savetxt('Ase30.txt', Ase30)    
    np.savetxt('Aape30.txt', Aape30)
    np.savetxt('Aae1.txt', Aae1)
    np.savetxt('Aae10.txt', Aae10)    
    np.savetxt('Aae30.txt', Aae30)
        
    
    