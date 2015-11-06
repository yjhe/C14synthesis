# -*- coding: utf-8 -*-
"""
Plot cumulative SOC and cumulative C-averaged D14C (or D14C) of profiles used in modeling

Created on Sat May 16 02:01:34 2015

@author: happysk8er
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import D14Cpreprocess as prep
import pylab

filename = 'tot48prof.txt'
dum = np.loadtxt(filename, delimiter=',')
profid = dum[:,2]
#%% plot cumuSOC vs cumuC-averaged D14C
cm = plt.get_cmap('gist_rainbow')
numcolr = len(profid) # no repeat in color
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
axes.set_color_cycle([cm(1.*jj/numcolr) for jj in range(numcolr)])
for i in profid:
    data = prep.getprofSOCD14C_interp('Non_peat_data_synthesis.csv',i,cutdep=100.)
    cumuC = np.cumsum(data[:,2])
    cumufracC = cumuC/cumuC[-1]
    cumuD14C = np.cumsum(data[:,1] * data[:,2])/cumuC
    axes.plot(cumufracC, cumuD14C, ':', lw=2)
axes.set_xlabel('Cumulative SOC fraction (%)')
axes.set_ylabel(r"Cumulative $\Delta^{14}C$ ("+ u"\u2030)")

#%% plot cumuSOC vs  D14C
cm = plt.get_cmap('gist_rainbow')
numcolr = len(profid) # no repeat in color
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
axes.set_color_cycle([cm(1.*jj/numcolr) for jj in range(numcolr)])
for i in profid:
    if i==144.:
        continue
    data = prep.getprofSOCD14C_interp('Non_peat_data_synthesis.csv',i,cutdep=100.)
    cumuC = np.cumsum(data[:,2])
    cumufracC = cumuC/cumuC[-1]
    D14C = data[:,1]
    print 'profile ID is ', i
    print D14C
#    raw_input('Enter to continue...')
    axes.plot(cumufracC, D14C, ':', lw=2)
axes.set_xlabel('Cumulative SOC fraction (%)')
axes.set_ylabel(r"$\Delta^{14}C$ ("+ u"\u2030)")
axes.axvline(x=0.8,c='gray')
axes.axhline(y=-800,c='gray')