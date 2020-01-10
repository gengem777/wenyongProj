# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:23:28 2019

@author: wyzhang
"""

import numpy as np
import pandas as pd

df = pd.read_csv('IVS.csv')

#获得时间的list
df_date = df['date']
df_date = df_date.drop_duplicates()
date_list = []
for i in df_date.keys():
    date_list.append(df_date[i])
print(date_list)

def g(x, y, h1, h2):
    g = np.exp(-x**2/2/h1)*np.exp(-y**2/2/h2)/2*np.pi
    return g

Tau = np.array([0.0833, 0.1786, 0.2738, 0.3690, 0.4643, 0.5595, 0.6548, 0.7500])
moneyness = np.array([0.7000, 0.8000, 0.9000, 1, 1.1000, 1.2000, 1.3000, 1.4000])

IVS_matrix = []
for date in date_list:
    tau1 = df[df['date']==date]['tau'].values
    m1 = df[df['date']==date]['moneyness'].values
    imp = df[df['date']==date]['impl_volatility'].values
    
    #Tau = np.array([0.0833, 0.1786, 0.2738, 0.3690, 0.4643, 0.5595, 0.6548, 0.7500])
    #moneyness = np.array([0.7000, 0.8000, 0.9000, 1, 1.1000, 1.2000, 1.3000, 1.4000])
    def impvolfunc(m, tau, imp, h1 = 1, h2 = 1):  #gaussian interpolation
        #Tau = np.array([0.0833, 0.1786, 0.2738, 0.3690, 0.4643, 0.5595, 0.6548, 0.7500])
        #moneyness = np.array([0.7000, 0.8000, 0.9000, 1, 1.1000, 1.2000, 1.3000, 1.4000])
        #Tau, moneyness = np.meshgrid(tau1, m1)
        #Tau = Tau.ravel()
       # moneyness = moneyness.ravel()
        Nt = len(tau1)
        Nm = len(m1)
        tau = tau*np.ones([Nt])
        m = m*np.ones([Nm])
        impvol = imp
        norminator = np.dot(impvol, g(m - m1, tau - tau1, h1, h2))
        denorminator = np.sum(g(m - m1, tau - tau1, h1, h2))
        ret = norminator/denorminator
        return ret
    
    #print(impvolfunc(1.1, 1.5, imp, h1 = 1, h2 = 1))
    
    Tau, Mon = np.meshgrid(Tau, moneyness)
    Tau = Tau.ravel()
    Mon = Mon.ravel()
    
    impvol = []
    for i in range(len(Tau)):
        impvol.append(impvolfunc(Tau[i], Mon[i], imp, h1 = 1, h2 = 1))
    
    impvol = np.array(impvol)
    IVS_matrix.append(impvol)

IVS_matrix = np.array(IVS_matrix)
print(IVS_matrix)
np.savetxt('IVS', IVS_matrix)