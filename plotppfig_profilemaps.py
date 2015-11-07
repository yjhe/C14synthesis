# -*- coding: utf-8 -*-
"""
Plot the profile sites locations, color of marker indicates D14C values
Created on Wed Apr 22 15:09:32 2015

@author: Yujie
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import D14Cpreprocess as prep
from mpl_toolkits.basemap import Basemap, cm
import mynetCDF as mync
from netCDF4 import Dataset
import myplot as myplt

pltorisitesonly = 1
filename = 'tot48prof.txt'
data = np.loadtxt(filename,unpack=True,delimiter=',')[:,:].T
lons = data[:,0].astype(float)
lats = data[:,1].astype(float)
D14C = data[:,5].astype(float)
if pltorisitesonly == 0: # including extra sites
    filename = 'extrasitegridid.txt'
    data = np.loadtxt(filename,unpack=True,delimiter=',')[:,:].T
    lons = np.r_[lons, data[:,0].astype(float)]
    lats = np.r_[lats, data[:,1].astype(float)]
    D14C = np.r_[D14C, data[:,5].astype(float)]
#%%
fig = plt.figure(figsize=(8,5))
ax = fig.add_axes([0.07,0.02,0.9,1])
m = Basemap(llcrnrlon=-180,llcrnrlat=-60,urcrnrlon=180,urcrnrlat=80,projection='mill',lon_0=0.,lat_0=0.)
m.drawcoastlines(linewidth=0.25)
m.drawcountries(linewidth=0.25)
m.drawmapboundary(fill_color='White')
m.fillcontinents(color='lightgrey',lake_color='White',zorder=0)
x, y = m(lons,lats)
D14Clim = np.nanmax(D14C) - np.nanmin(D14C)
norm = myplt.MidpointNormalize(midpoint=0)
im = m.scatter(x,y,s=50,marker='^',c=D14C,norm=norm,cmap=plt.cm.seismic)
im.set_clim((-600, 200))
cbar = myplt.hcolorbar('neither', im=im, ticks=np.arange(-600,210,100))
#cbar.set_ticklabels(np.arange(-600,210,100))
#cbar.set_clim((-600, 200))
cbar.set_label(r"C-averaged $\Delta^{14}C$ ("+ u"\u2030)",fontsize=12)
#cbar.set_xlabel
parallels = np.arange(-90.,90.,30.)
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
meridians = np.arange(0.,360.,45.)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#fig.savefig('./figures/ProfileExtracted4modeling2.png')
