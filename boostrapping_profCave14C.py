# -*- coding: utf-8 -*-
"""
Prove selected 42 profiles are not biased: boostrap say top 20cm Cave14C of all profiles and 
construct distribution, compare this with the distribution of the 42 profiles, do t- and F-test

Created on Mon Apr 27 11:02:39 2015

@author: Yujie
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import D14Cpreprocess as prep
import scipy.stats as stats
import statsmodels.api as sm
import mystats as mysm
import myplot
import pylab
import random
from collections import Counter
#%% prepare data
# get 14C and SOC of total profiles
filename = 'Non_peat_data_synthesis.csv'
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1])  
layerbot = prep.getvarxls(data, 'Layer_bottom_norm', data.index.unique(), -1)
plt.hist(layerbot, 60)
cutdep = 40.
Cave14C = prep.getCweightedD14C2(filename, cutdep=cutdep)
nprof = Cave14C[(Cave14C[:,1]==0) & (Cave14C[:,2]==cutdep)].shape[0] # n = 138
totprofid = Cave14C[(Cave14C[:,1]==0) & (Cave14C[:,2]==cutdep), 3]
totprof14C = Cave14C[(Cave14C[:,1]==0) & (Cave14C[:,2]==cutdep), 4]
totprofSOC = Cave14C[(Cave14C[:,1]==0) & (Cave14C[:,2]==cutdep), 5]
totprofveg = prep.getvarxls(data,'VegTypeCode_Local',totprofid,0)
dum = list(totprofveg); Counter(dum)

# get 14C and SOC of profiles selected for modeling
pltorisitesonly = 0
filename = 'sitegridid2.txt'
data = np.loadtxt(filename,unpack=True,delimiter=',')[:,:].T
mdlprofid = data[:,2].astype(float)
if pltorisitesonly == 0: # including extra sites
    filename = 'extrasitegridid.txt'
    data = np.loadtxt(filename,unpack=True,delimiter=',')[:,:].T
    mdlprofid = np.r_[mdlprofid, data[:,2].astype(float)]
idx = [i in set(mdlprofid) for i in Cave14C[:,3]]
idx = np.asarray(idx)
np.where(idx)
mdlprof14C = Cave14C[np.nonzero(idx), 4]
mdlprofSOC = Cave14C[np.nonzero(idx), 5]
filename = 'Non_peat_data_synthesis.csv'
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1])  
mdlprofveg = prep.getvarxls(data,'VegTypeCode_Local',mdlprofid,0)
dum = list(mdlprofveg); Counter(dum)

# calculate number of sites in boreal, temperate, tropical biomes
pltorisitesonly = 0
filename = 'tot48prof.txt'
data = np.loadtxt(filename,unpack=True,delimiter=',')[:,:].T
mdlproflon = data[:,1].astype(float)
n = sum(np.logical_or(np.logical_and(mdlproflon> 23,mdlproflon<50),np.logical_and(mdlproflon<-23,mdlproflon>-50)))
print 'temperate sites: ',n
n = sum(mdlproflon> 50)
print 'boreal sites: ',n
n = sum(np.logical_and(mdlproflon<23,mdlproflon>-23))
print 'tropical sites: ',n

#%% test on distribution. bootstrapping 42 profiles, each time compare with 
# population.
X = totprof14C[~np.isnan(totprof14C)]; Y = np.squeeze(mdlprof14C)
print 'N sample of total profile is ',len(X)
out = []
nbootstrapp = 500
for i in range(nbootstrapp):
    print 'i is ',i
    cur = []
    random.shuffle(X)
    Xr = X[:42]
    Y = X
    # check normality
    cur.append(stats.mstats.normaltest(Xr)[1])
    #stats.kstest(Xr,'norm')
    #plt.hist(Xr)
    #stats.probplot(Xr, dist="norm", plot=pylab)
    cur.append(stats.mstats.normaltest(Y)[1])
    #plt.hist(Y)
    #stats.probplot(Y, dist="norm", plot=pylab)
    
    # F-test, strong normality required
    F = np.var(Xr)/np.var(Y)
    df1 = len(Xr) - 1; df2 = len(Y) - 1
    alpha = 0.05 #Or whatever you want your alpha to be.
    p_value = stats.f.sf(F, df1, df2) # p-value = 1-CDF
    cur.append(p_value)    
    if p_value < alpha:
        print "Reject the null hypothesis that Var(X) == Var(Y)"
    else:
        print "equal variance !"
    
    cur.append(stats.bartlett(Xr,Y)[1]) # require normal
    cur.append(stats.levene(Xr,Y,center='median')[1]) # for non-normal samples
    
    # t-test, after equal variance, test for mean
    cur.append(stats.ttest_ind(Xr, Y)[1])
    cur.append(stats.ttest_ind(Xr, Y, equal_var=False)[1])
    cur.append(stats.mannwhitneyu(Xr, Y)[1])
    out.append(cur)

alpha = .5
outary = np.array(out)
col = 0
nrej = sum(outary[:,col]<alpha)
print 'Fraction of reject normaltest on Xr: %.2f' % (nrej*1./nbootstrapp)
col = 1
nrej = sum(outary[:,col]<alpha)
print 'Fraction of reject normaltest on Y: %.2f' % (nrej*1./nbootstrapp)
col = 2
nrej = sum(outary[:,col]<alpha)
print 'Fraction of rejection in F-test: %.2f' % (nrej*1./nbootstrapp)
col = 3
nrej = sum(outary[:,col]<alpha)
print 'Fraction of rejection in Bartlett: %.2f' % (nrej*1./nbootstrapp)
col = 4
nrej = sum(outary[:,col]<alpha)
print 'Fraction of rejection in Levene: %.2f' % (nrej*1./nbootstrapp)
col = 5
nrej = sum(outary[:,col]<alpha)
print 'Fraction of rejection in ttest equal var: %.2f' % (nrej*1./nbootstrapp)
col = 6
nrej = sum(outary[:,col]<alpha)
print 'Fraction of rejection in ttest non-equal var: %.2f' % (nrej*1./nbootstrapp)
col = 7
nrej = sum(outary[:,col]<alpha)
print 'Fraction of rejection in mannwhitney equal mean: %.2f' % (nrej*1./nbootstrapp)

#%% test on equal variance. use total profiles. compare 42 with total popu.
X = totprof14C[~np.isnan(totprof14C)]; Y = np.squeeze(mdlprof14C)
print 'N sample of total profile is ',len(X)
nbootstrapp = 1
out = []
cur = []
Xr = X
# check normality
cur.append(stats.mstats.normaltest(Xr)[1])
#stats.kstest(Xr,'norm')
#plt.hist(Xr)
#stats.probplot(Xr, dist="norm", plot=pylab)
cur.append(stats.mstats.normaltest(Y)[1])
#plt.hist(Y)
#stats.probplot(Y, dist="norm", plot=pylab)

# F-test, strong normality required
F = np.var(Xr)/np.var(Y)
df1 = len(Xr) - 1; df2 = len(Y) - 1
alpha = 0.05 #Or whatever you want your alpha to be.
p_value = stats.f.sf(F, df1, df2) # p-value = 1-CDF
cur.append(p_value)    
if p_value < alpha:
    print "Reject the null hypothesis that Var(X) == Var(Y)"
else:
    print "equal variance !"

cur.append(stats.bartlett(Xr,Y)[1]) # require normal
cur.append(stats.levene(Xr,Y,center='median')[1]) # for non-normal samples

# t-test, after equal variance, test for mean
cur.append(stats.ttest_ind(Xr, Y)[1])
cur.append(stats.ttest_ind(Xr, Y, equal_var=False)[1])
cur.append(stats.mannwhitneyu(Xr, Y)[1]*2.)
out.append(cur)

outary = np.array(out)
col = 0; alpha = .5
nrej = sum(outary[:,col]<alpha)
print 'Fraction of reject normaltest on Xr: %.2f' % (nrej*1./nbootstrapp)
col = 1
nrej = sum(outary[:,col]<alpha)
print 'Fraction of reject normaltest on Y: %.2f' % (nrej*1./nbootstrapp)
col = 2
nrej = sum(outary[:,col]<alpha)
print 'Fraction of rejection in F-test: %.2f' % (nrej*1./nbootstrapp)
col = 3
nrej = sum(outary[:,col]<alpha)
print 'Fraction of rejection in Bartlett: %.2f' % (nrej*1./nbootstrapp)
col = 4
nrej = sum(outary[:,col]<alpha)
print 'Fraction of rejection in Levene: %.2f' % (nrej*1./nbootstrapp)
col = 5
nrej = sum(outary[:,col]<alpha)
print 'Fraction of rejection in ttest equal var: %.2f' % (nrej*1./nbootstrapp)
col = 6
nrej = sum(outary[:,col]<alpha)
print 'Fraction of rejection in ttest non-equal var: %.2f' % (nrej*1./nbootstrapp)
col = 7
nrej = sum(outary[:,col]<alpha)
print 'Fraction of rejection in mannwhitney equal mean: %.2f' % (nrej*1./nbootstrapp)
    
#%% test on equal variance. boostrapping 42 profiles, considering biome
    # construct empirical distribution of moments(mean, var, skewness and kurtosis)
X = totprof14C[~np.isnan(totprof14C)]; Y = np.squeeze(mdlprof14C)
totprofvegnonan = totprofveg[~np.isnan(totprof14C)]
print 'N sample of total profile is ',len(X)
testout = [] # normality, F, ttest etc.
statistics = [] # mean, var, skewness, kurtosis
nbootstrapp = 1000
for i in range(nbootstrapp):
    boot = []    
    for j in set(mdlprofveg): # for each biome, sample same n as in 42 profiles
        nsmp = list(mdlprofveg).count(j)
        a = np.squeeze(np.where(totprofvegnonan==j))
        if a.ndim>=1:
            random.shuffle(a)        
            idx = a[:nsmp]
            boot.append(X[idx])        
        else:
            boot.append([X[a]])
    boot = reduce(lambda x, y: list(x)+list(y), boot)
    print 'i is ',i
    Xr = boot
    Y = X

    # do tests    
    cur = []
    # check normality
    cur.append(stats.mstats.normaltest(Xr)[1])
    cur.append(stats.mstats.normaltest(Y)[1])
    # F-test, strong normality required
    F = np.var(Xr)/np.var(Y)
    df1 = len(Xr) - 1; df2 = len(Y) - 1
    alpha = 0.05 #Or whatever you want your alpha to be.
    p_value = stats.f.sf(F, df1, df2) # p-value = 1-CDF
    cur.append(p_value)    
    if p_value < alpha:
        print "Reject the null hypothesis that Var(X) == Var(Y)"
    else:
        print "equal variance !"    
    cur.append(stats.bartlett(Xr,Y)[1]) # require normal
    cur.append(stats.levene(Xr,Y,center='median')[1]) # for non-normal samples
    # t-test, after equal variance, test for mean
    cur.append(stats.ttest_ind(Xr, Y)[1])
    cur.append(stats.ttest_ind(Xr, Y, equal_var=False)[1])
    cur.append(stats.mannwhitneyu(Xr, Y)[1])
    testout.append(cur)

    # calculate statistics
    cur = []
    cur.append(np.nanmean(Xr))
    cur.append(np.nanvar(Xr))
    cur.append(stats.skew(Xr))
    cur.append(stats.kurtosis(Xr))
    statistics.append(cur)

# print test results
outary = np.array(testout)
col = 0; alpha = .05
nrej = sum(outary[:,col]<alpha)
print 'Fraction of reject normaltest on Xr: %.2f' % (nrej*1./nbootstrapp)
col = 1
nrej = sum(outary[:,col]<alpha)
print 'Fraction of reject normaltest on Y: %.2f' % (nrej*1./nbootstrapp)
col = 2
nrej = sum(outary[:,col]<alpha)
print 'Fraction of rejection in F-test: %.2f' % (nrej*1./nbootstrapp)
col = 3
nrej = sum(outary[:,col]<alpha)
print 'Fraction of rejection in Bartlett: %.2f' % (nrej*1./nbootstrapp)
col = 4
nrej = sum(outary[:,col]<alpha)
print 'Fraction of rejection in Levene: %.2f' % (nrej*1./nbootstrapp)
col = 5
nrej = sum(outary[:,col]<alpha)
print 'Fraction of rejection in ttest equal var: %.2f' % (nrej*1./nbootstrapp)
col = 6
nrej = sum(outary[:,col]<alpha)
print 'Fraction of rejection in ttest non-equal var: %.2f' % (nrej*1./nbootstrapp)
col = 7
nrej = sum(outary[:,col]<alpha)
print 'Fraction of rejection in mannwhitney equal mean: %.2f' % (nrej*1./nbootstrapp)\
# print empirical test on moments
mom = np.array(statistics)
Y = np.squeeze(mdlprof14C)
# mean
plt.hist(mom[:,3])
low = np.percentile(mom[:,0],2.5); up = np.percentile(mom[:,0],97.5)
if np.mean(Y) >= low and np.mean(Y) >= low:
    print 'mean is OK'
low = np.percentile(mom[:,1],2.5); up = np.percentile(mom[:,1],97.5)
if np.var(Y) >= low and np.var(Y) >= low:
    print 'var is OK'# mean
low = np.percentile(mom[:,2],2.5); up = np.percentile(mom[:,2],97.5)
if stats.skew(Y) >= low and stats.skew(Y) >= low:
    print 'skew is OK'# mean
low = np.percentile(mom[:,3],2.5); up = np.percentile(mom[:,3],97.5)
if stats.kurtosis(Y) >= low and stats.kurtosis(Y) >= low:
    print 'kurtosis is OK'