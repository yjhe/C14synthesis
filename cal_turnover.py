# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 02:58:30 2015

@author: happysk8er
"""
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def cal_recursive_FM(k, atmFM, nyr, idx):
    ''' recursively calculate FM based on previous years' values, use Torn book
        chapter A2.6, difference equation
    arguments: k, decomposition rate (1/tau, yr)
               atmD14C, time series of atmos D14C, end in year the sample was measured
               nyr, number of total years to run
               idx, index used to track nyr
    output: calculated FM in year yr
    '''
    lambdaa = 1.21e-4
    # print 'nyr: ', nyr
    if nyr == 1:
        return k*atmFM[-idx] + atmFM[-idx]*(1.-k-lambdaa)
    return k*atmFM[-idx] + cal_recursive_FM(k, atmFM, nyr-1, idx+1)*(1.-k-lambdaa)

def cal_difference_FM(k, atmFM, nyr, yr):
    ''' non-recursive version of calculate FM based on previous years' values.
        refer to Torn book chapter A2.6. 
        arguments: 
            k, decomposition rate (1/tau, yr)
            atmD14C: complete atm time series
            nyr: total years want to run (start year idx = -(2013-yr-nyr))
            yr: year of end run.
        output: 
            calculated FM in year yr.
    '''
    lambdaa = 1.21e-4
    startyridx = -(2013-yr)-nyr
    endyridx = -(2013-yr)
    fm0 = atmFM[startyridx]
    for i in range(startyridx, endyridx+1):
        fm = k*atmFM[i] + fm0*(1.-k-lambdaa)
        fm0 = fm
    return fm
    
def cal_prebomb_FM(k):
    ''' calculate FM assuming steady-state and no bomb-14C
        F = k/(k + lambdaa)
    '''
    lambdaa = 1.21e-4
    return k/(k + lambdaa)
    
from scipy.optimize import fsolve
Fct = -300.; nyr = 1000; yr = 2012
atm14C = scipy.io.loadmat('atmD14C_50kBP-2012.mat')
atmD14C = atm14C['atmD14C'][:,1]
atmFM = atmD14C/1000. + 1.
k_initial_guess = 1./200; k = k_initial_guess
func = lambda k: (cal_prebomb_FM(k) - Fct)**2.
tau_solution = fsolve(func, k_initial_guess)

#%% plot recursive_FM for year 1950 - 2012, test
out = []
out2 = []
out3 = []
nyr = 20000
k = 1/90000.
startyr = 1900
endyr = 2013
for yr in np.arange(startyr, endyr):
    #dum = recursive_FM(k, atmFM[:-(2013-yr)], nyr, 1)    
    #out.append(1000. *(dum-1.))
    dum = cal_difference_FM(k, atmFM, nyr, yr)
    out2.append(1000. *(dum-1.))
plt.plot(atmD14C[-(endyr-startyr):])
plt.plot(out, label='recursive')
plt.plot(out2, label='differential')
print FMtoD14C(cal_prebomb_FM(k))
#%% 
def cal_tau(d14c, smpyr):
    ''' treat each entry of d14c as a homogeneous one-box soil. 
        search for best tau that agrees most with observation
    '''
    besttau = []
    taufast = np.arange(1,2000,20)
    tauslow = np.arange(2000, 200000, 100)
    atm14C = scipy.io.loadmat('atmD14C_50kBP-2012.mat')
    atmD14C = atm14C['atmD14C'][:,1]
    atmFM = atmD14C/1000. + 1.
    for obsn, (obs, sampleyr) in enumerate(zip(d14c,smpyr)):
        print '\n -----  %d  ------ \n'%obsn
        cost = []        
        if obs > -200.:
            # use difference equation
            if np.isnan(sampleyr):
                print 'missing value for bomb 14C...skip...'                
                continue
            for n, tau in enumerate(taufast):
                #print 'taufast: %d, #%d'%(tau, n)
                dum = obs - FMtoD14C(cal_difference_FM(1./tau, atmFM, sampleyr))
                cost.append(dum**2)    
            besttau.append(taufast[np.argmin(np.asarray(cost))])
        else:
            # use prebomb_FM
            for n, tau in enumerate(tauslow):
                #print 'tauslow: %d, #%d'%(tau, n)
                dum = obs - FMtoD14C(cal_prebomb_FM(1./tau))
                cost.append(dum**2)
            besttau.append(tauslow[np.argmin(np.asarray(cost))])
    return besttau
    
#%% test cal_tau
import D14Cpreprocess as prep
import pandas as pd
import C14tools
filename = 'Non_peat_data_synthesis.csv'
Cave14C = prep.getCweightedD14C2(filename)
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1])  
profid = Cave14C[:,3]
d14C = prep.getvarxls(data,'D14C_BulkLayer', profid, ':')
sampleyr = prep.getvarxls(data, 'SampleYear', profid, ':')

n0 = 40
nend = 60
%timeit tau, cost = C14tools.cal_tau(d14C[n0:nend], sampleyr[n0:nend])

#%%
import numba as nb
#@nb.jit(nb.f8(nb.f8[:]))
#@nb.autojit
def summ(arr):
    summ = 0.
    for i in arr:
        summ = summ + i
    return summ
%timeit out = summ(np.arange(100000,dtype='float'))
#%%  calculate turnover time for d14C of each layer
import C14tools as C14
import D14Cpreprocess as prep
import mystats as mysm

filename = 'Non_peat_data_synthesis.csv'
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1])  
profid = data.index.unique() # index of profile start
d14C = prep.getvarxls(data,'D14C_BulkLayer', profid, ':')
sampleyr = prep.getvarxls(data, 'SampleYear', profid, ':')
layerbot = prep.getvarxls(data, 'Layer_bottom', profid, ':')
tau, cost = C14.cal_tau(d14C, sampleyr, 3, False)
np.savez('./Synthesis_allD14C_tau.npz',tau=tau,cost=cost)

taudata = np.load('./Synthesis_allD14C_tau.npz')
tau = taudata['tau']
cost = taudata['cost']

D14C2000 = np.array(C14.cal_D14Ctosmpyr(tau[:,0], 2000))

is_badcost = cost[:,0]>50
data.D14C_BulkLayer[is_badcost]
a = mysm.cal_RMSE(d14C[~is_badcost], D14C2000[~is_badcost])

D14C2000df = pd.DataFrame(data=D14C2000)
D14C2000df.to_csv('normalizedD14C.csv')
#%% verify the D14C normalization approach
newdata = data.copy()

# index of profiles that have multiple year measurements
def print_normalized(profid, tosmpyr):    
    prof1 = data.loc[profid,['Layer_bottom','D14C_BulkLayer','SampleYear']]
    mod = C14.cal_D14Ctosmpyr(tau[:,0], tosmpyr)
    newdata['D14C_normalized'] = mod
    prof = newdata.loc[profid,['Layer_bottom','D14C_BulkLayer','D14C_normalized','SampleYear']]
    print prof

print_normalized(1, 2013)



