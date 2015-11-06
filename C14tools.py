# -*- coding: utf-8 -*-
"""
A collection of funcitons for 14C analysis
    * calculate turnover time
    * conversion between FM and D14C
    * radiocarbon age

Created on Thu Jun 04 22:28:39 2015

@author: happysk8er
"""
#%%
import numpy as np
import scipy.io
from numba import autojit

@autojit
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

@autojit
def cal_difference_FM(k, atmFM, yr, nyr=30000):
    ''' non-recursive version of calculate FM based on previous years' values.
        refer to Torn book chapter A2.6. 
        arguments: 
            k, decomposition rate (1/tau, yr)
            atmD14C: complete atm time series
            nyr: total years want to run (start year idx = -(2013-yr-nyr))
            yr: year of end run/ sample year
        output: 
            calculated FM in year yr.
    '''
    yr = int(yr)
    lambdaa = 1.21e-4
    startyridx = -(2013-yr)-nyr
    endyridx = -(2013-yr)
    fm0 = atmFM[startyridx]
    for i in range(startyridx, endyridx+1):
        fm = k*atmFM[i] + fm0*(1.-k-lambdaa)
        fm0 = fm
    return fm

@autojit    
def cal_prebomb_FM(k):
    ''' calculate FM assuming steady-state and no bomb-14C
        F = k/(k + lambdaa)
    '''
    lambdaa = 1.21e-4
    return k/(k + lambdaa)

@autojit    
def D14CtoFM(D14C):
    ''' convert to FM
    '''
    return D14C/1000. + 1

@autojit   
def FMtoD14C(FM):
    ''' convert to D14C
    '''
    return 1000. * (FM - 1)

@autojit   
def cal_tau(d14c, smpyr, savecostn, tauonly):
    ''' treat each entry of d14c as a homogeneous one-box soil. 
        search for best tau that agrees most with observation
    parameters: array of d14c 
                its corresonding sample year
                saveconstn is the first n tau/cost you want to output, int.
                tauonly, to return tau or tau + cost, 1 or 0
    output: array of tau
    '''
    besttau = np.zeros((d14c.shape[0],savecostn))
    bestcost = np.zeros((d14c.shape[0],savecostn))
    taufast = np.arange(1,2000,5)
    tauslow = np.arange(2000, 200000, 50)
    atm14C = scipy.io.loadmat('atmD14C_50kBP-2012.mat')
    atmD14C = atm14C['atmD14C'][:,1]
    atmFM = atmD14C/1000. + 1.
    for obsn, (obs, sampleyr) in enumerate(zip(d14c,smpyr)):
        print '\n -----  obs: %d  ------ \n'%obsn
        cost = []        
        if obs > -200.:
            # use difference equation
            if np.isnan(sampleyr):
                print 'missing value for bomb 14C...skip...'                
                continue
            for n, tau in enumerate(taufast):
                #print 'taufast: %d, #%d'%(tau, n)
                dum = obs - FMtoD14C(cal_difference_FM(1./tau, atmFM, sampleyr, 30000))
                cost.append(dum**2)  
            tmp = np.argsort(np.asarray(cost))
            besttau[obsn,:] = taufast[tmp[:savecostn]]
            bestcost[obsn,:] = np.asarray(cost)[tmp[:savecostn]]
        else:
            # use prebomb_FM
            for n, tau in enumerate(tauslow):
                #print 'tauslow: %d, #%d'%(tau, n)
                dum = obs - FMtoD14C(cal_prebomb_FM(1./tau))
                cost.append(dum**2)
            tmp = np.argsort(np.asarray(cost))
            besttau[obsn,:] = tauslow[tmp[:savecostn]]
            bestcost[obsn,:] = np.asarray(cost)[tmp[:savecostn]]
    if tauonly:
        return besttau
    else:
        return besttau, bestcost
        
@autojit 
def cal_D14Ctosmpyr(tau, sampleyr):
    '''
    Given tau, assume homogeneous one-box model, forward calculate
    D14C in a given year (sampleyr)
    parameters:
        tau: {array-like} turnover time of each layer
        sampleyr: {array-like} sample year(to normalize to)
                  corresponding to each tau
    return:
        D14C: {array-like} same shape with tau
    '''
    atm14C = scipy.io.loadmat('atmD14C_50kBP-2012.mat')
    atmD14C = atm14C['atmD14C'][:,1]
    atmFM = atmD14C/1000. + 1.
    outD14C = []
    for obsn, obs in enumerate(tau):
        #print '\n -----  obs: %d  ------ \n'%obsn
        if obs < 500.: # younger than 500 years
            # use difference equation
           outD14C += FMtoD14C(cal_difference_FM(1./obs, atmFM, sampleyr, 30000)),
        else:
            # use prebomb_FM
            outD14C += FMtoD14C(cal_prebomb_FM(1./obs)),
    return outD14C

def C14agetoD14C(age):
    ''' calculate D14C with standard corrected to 1950
    '''
    return (np.exp(-age/8033.)-1.)*1000.
    
    
#%% 
if __name__ == "__main__":
    import D14Cpreprocess as prep
    import pandas as pd
    
    filename = 'Non_peat_data_synthesis.csv'
    Cave14C = prep.getCweightedD14C2(filename)
    data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1])  
    profid = Cave14C[:,3]
    d14C = prep.getvarxls(data,'D14C_BulkLayer', profid, ':')
    sampleyr = prep.getvarxls(data, 'SampleYear', profid, ':')
    
    print 'Run cal_tau to get turnover time for each soil layer...'
    n0 = 40
    nend = 60
    #tau, cost = cal_tau(d14C[n0:nend], sampleyr[n0:nend])
    tau, cost = cal_tau(d14C, sampleyr)
    print 'tau, cost are the calculated turnover and cost'