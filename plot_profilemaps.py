# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 14:37:03 2015

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
import mystats as mysm

#%% plot all profiles
filename = 'Non_peat_data_synthesis.csv'
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1])  
profid = data.index.unique()
lons = prep.getvarxls(data,'Lon',profid,0)
lats = prep.getvarxls(data,'Lat',profid,0)
fig = plt.figure()
ax = fig.add_axes([0.05,0.05,0.9,0.9])
m = Basemap(llcrnrlon=-180,llcrnrlat=-60,urcrnrlon=180,urcrnrlat=80,projection='mill',lon_0=0,lat_0=0)
lon, lat = np.meshgrid(lons, lats)
x, y = m(lons,lats)
m.drawcoastlines(linewidth=0.25)
m.drawcountries(linewidth=0.25)
m.drawmapboundary(fill_color='#99ffff')
m.fillcontinents(color='grey',lake_color='#99ffff',zorder=0)
m.scatter(lons,lats,15,marker='^',color='r',alpha=0.7,latlon=True)
# draw parallels.
parallels = np.arange(-90.,90.,30.)
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
# draw meridians
meridians = np.arange(0.,360.,45.)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
ax.set_title('Profile Sites')
#fig.savefig('./figures/Allprofiles.png')

#%% plot profiles that are extracted for modeling
filename = 'tot48prof.txt'
data = np.loadtxt(filename,unpack=True,delimiter=',')[0:2,:].T
lons = data[:,0].astype(float)
lats = data[:,1].astype(float)
fig = plt.figure()
ax = fig.add_axes([0.05,0.05,0.9,0.9])
m = Basemap(llcrnrlon=-180,llcrnrlat=-60,urcrnrlon=180,urcrnrlat=80,projection='mill',lon_0=0.,lat_0=0.)
m.drawcoastlines(linewidth=0.25)
#m.drawcountries(linewidth=0.25)
m.drawmapboundary(fill_color='#99ffff')
m.fillcontinents(color='grey',lake_color='#99ffff',zorder=0)
#x, y = m(lons,lats)
#m.scatter(lons,lats,17,marker='^',color='r',alpha=0.7,latlon=True)
x, y = m(lons,lats)
for i in range(x.shape[0]):
    m.scatter(lons[i],lats[i],17,marker='^',color='r',alpha=0.7,latlon=True)
# draw parallels.
parallels = np.arange(-90.,90.,30.)
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
# draw meridians
meridians = np.arange(0.,360.,45.)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
ax.set_title('Profile Sites Extracted for modeling')
fig.savefig('./figures/ProfileExtracted4modeling2.png')




#%% plot all profiles with veg
filename = 'Non_peat_data_synthesis.csv'
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID')  
biome = {1:'Boreal Forest',2:'Temperate Forest',3:'Tropical Forest',4:'Grassland', \
         5:'Cropland',6:'Shrublands',7:'Peatland',8:'Savannas'}
lons = data['Lon'][1:].values.astype(float)
lats = data['Lat'][1:].values.astype(float)
veg = data['VegTypeCode_Local'][1:].values.astype(float)
nveg = len(np.unique(veg[~np.isnan(veg)]))
fig = plt.figure()
ax = fig.add_axes([0.05,0.05,0.9,0.9])
m = Basemap(llcrnrlon=-180,llcrnrlat=-60,urcrnrlon=180,urcrnrlat=80,projection='mill',lon_0=180,lat_0=0)
m.drawcoastlines(linewidth=0.25)
m.drawcountries(linewidth=0.25)
m.drawmapboundary(fill_color='#99ffff')
m.fillcontinents(color='grey',lake_color='#99ffff',zorder=0)
cmm = plt.get_cmap('Set1')
for i in np.unique(veg[~np.isnan(veg)]):
    x, y = m(lons[veg==i],lats[veg==i])
    m.scatter(x,y,55,marker='^',color=cmm(1.*i/nveg*1.),\
              label=biome[i],alpha=1) 
plt.legend(scatterpoints=1,loc=3)
# draw parallels.
parallels = np.arange(-90.,90.,30.)
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
# draw meridians
meridians = np.arange(0.,360.,45.)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
ax.set_title('Profile Sites')
fig.savefig('./figures/Allprofiles.png')

#%% plot HWSD data
sclayfn = '..\\AncillaryData\\HWSD\\Regridded_ORNLDAAC\\S_CLAY.nc4'
tclayfn = '..\\AncillaryData\\HWSD\\Regridded_ORNLDAAC\\T_CLAY.nc4'
scecclayfn= '..\\AncillaryData\\HWSD\\Regridded_ORNLDAAC\\S_CEC_CLAY.nc4'
tcecclayfn= '..\\AncillaryData\\HWSD\\Regridded_ORNLDAAC\\T_CEC_CLAY.nc4'
tbulkdenfn= '..\\AncillaryData\\HWSD\\Regridded_ORNLDAAC\\T_BULK_DEN.nc4'
sbulkdenfn= '..\\AncillaryData\\HWSD\\Regridded_ORNLDAAC\\S_BULK_DEN.nc4'
sawtcfn = '..\\AncillaryData\\HWSD\\Regridded_ORNLDAAC\\AWT_S_SOC.nc4'
tawtcfn = '..\\AncillaryData\\HWSD\\Regridded_ORNLDAAC\\AWT_T_SOC.nc4'
ref_depthfn = '..\\AncillaryData\\HWSD\\Regridded_ORNLDAAC\\REF_DEPTH.nc4'
dictt = {'sclay':sclayfn,'tclay':tclayfn,'scecclay':scecclayfn, \
        'tcecclay': tcecclayfn, 'tbd':tbulkdenfn,'sbd':sbulkdenfn, \
        'sawtc':sawtcfn, 'tawtc':tawtcfn, 'ref_depth':ref_depthfn}
bd = 0  # if the calculation need bulk density
ncfid = Dataset(dictt['ref_depth'], 'r')
nc_attrs, nc_dims, nc_vars = mync.ncdump(ncfid)
lats = ncfid.variables['lat'][:]
lons = ncfid.variables['lon'][:]
var = ncfid.variables[nc_vars[2]][:]
if bd == 1:
    ncfid = Dataset(dictt['tbd'], 'r')
    nc_attrs, nc_dims, nc_vars = mync.ncdump(ncfid)
    tbd = ncfid.variables[nc_vars[2]][:]
    ncfid = Dataset(dictt['sbd'], 'r')
    nc_attrs, nc_dims, nc_vars = mync.ncdump(ncfid)            
    sbd = ncfid.variables[nc_vars[2]][:]
    ncfid = Dataset(dictt['tawtc'], 'r')
    nc_attrs, nc_dims, nc_vars = mync.ncdump(ncfid)            
    tclay = ncfid.variables[nc_vars[2]][:]
    #var = var*sbd*1000*0.7 + tclay*tbd*1000*0.3
    var = tclay/(var + tclay)*100.

var_rebin = mysm.rebin_mean(var, (360, 720))
np.savetxt('soildepthHWSD0.5.txt',var_rebin)
#% plot
#cbtitle = "r'area weighted total soil C $(kgC/m^{2})$'"
#cbtitle = 'Proportion of topsoil C (%)'
cbtitle = "soil depth"
im = myplt.millshow(var_rebin,np.arange(-180,180,0.5),np.arange(-90,90, 0.5),theTitle=None,cmap='GMT_haxby',latmin=-60,latmax=90,
                    cbartitle=cbtitle,cbar='h',extend='neither')
plt.savefig('./AncillaryData/HWSD/figure/'+'topawtc_ratio.png')    

## create 3D soildepth binary array (1 when there is soil, 0 when there is not)       
import numpy.ma as ma 
soildepth = np.floor(var_rebin)
plt.imshow(soildepth); plt.colorbar() # 90S-90N
soildepth_3D = np.ones((100, 360, 720))
valid = soildepth[~soildepth.mask]
idx_valid = np.where(~soildepth.mask)
n = 0
for i,j in zip(idx_valid[0],idx_valid[1]):
    #print i,j
    #if soildepth[i,j] is not ma.masked:
    d = soildepth[i,j]; d = int(d)
    if d == 0:
        soildepth_3D[:,i,j] = 0.
    else:
        soildepth_3D[d:,i,j] = 0.
    n += 1
    print 'n : ',n
np.save('soildepth3D.npy',soildepth_3D)  
#%% calculate HWSD global total csoil
sawtcfn = 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\AWT_S_SOC.nc4'
tawtcfn = 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\AWT_T_SOC.nc4'
ncfid = Dataset(sawtcfn, 'r')
nc_attrs, nc_dims, nc_vars = mync.ncdump(ncfid)
sawtc = ncfid.variables['SUM_s_c_1'][:]
lat = ncfid.variables['lat'][:]
ncfid = Dataset(tawtcfn, 'r')
nc_attrs, nc_dims, nc_vars = mync.ncdump(ncfid)
tawtc = ncfid.variables['SUM_t_c_12'][:]
hwsdsoc = sawtc + tawtc
hwsdsoc[hwsdsoc==0.] = np.nan

# average radius of reference spheroid is 6371km
areaa = np.zeros((1800, 1)) # m2
rad = 6.371e6
height = rad*2.*np.pi/3600.
latt = []
for i in range(1,1801):
    #print 'i is %d, j is %d\n'%(i,j)
    upperedge = rad*np.cos((i-1)/20.*np.pi/180.)*2.*np.pi/7200.
    loweredge = rad*np.cos(i/20.*np.pi/180.)*2.*np.pi/7200.        
    areaa[1800-i-1,0] = (loweredge+upperedge)*height/2.
    latt.append((i-1)/20.*np.pi/180.)
areaa = np.tile(areaa,(1,7200))
areaa = np.r_[areaa, np.flipud(areaa)];
#plt.imshow(areaa)
#plt.imshow(hwsdsoc)
# calculate global total SOC
hwsdglbsoc = np.nansum(hwsdsoc*areaa)/1e12   
print 'HWSD global total SOC is:%.3f PgC' %hwsdglbsoc

areaa = np.zeros((1800, 1))
height = rad*2.*np.pi/7200.
for i in range(1,1801):
    hori = rad*np.cos(lat[i]*np.pi/180.)*2*np.pi/7200.
    areaa[i-1,0] = hori*height
areaa = np.tile(areaa,(1,7200))
areaa = np.r_[areaa, np.flipud(areaa)];
# calculate global total SOC
areaa = mysm.cal_earthgridarea(0.05)
hwsdglbsoc = np.nansum(hwsdsoc*areaa)/1e12   
print 'HWSD global total SOC is:%.3f PgC' %hwsdglbsoc
#%% plot NPP, 2000-2012 average in gC/m2/yr
nppfn = 'AncillaryData\\NPP\\2000_2012meannpp_gCm2yr.nc'
ncfid = Dataset(nppfn, 'r')
nc_attrs, nc_dims, nc_vars = mync.ncdump(ncfid)
lats = ncfid.variables['lat'][:][::-1]
lons = ncfid.variables['lon'][:]
var = ncfid.variables['npp'][:]
var = np.flipud(np.ma.transpose(var))
im= myplt.millshow(var,lons,lats,theTitle=None,cmap='GMT_haxby',latmin=-60,latmax=90,\
                   cbartitle=r'2000-2012 mean annual NPP $(gC\/m^{-2}\/yr^{-1})$',units='aa',cbar='h')
im= myplt.geoshow(var,lons,lats,theTitle=None,cmap='GMT_haxby',\
                   units='aa',cbar='v')

plt.savefig('./AncillaryData/NPP/npp.png')

#%% plot GSDESM soil depth
ref_depthfn = 'C:\\download\\work\\!manuscripts\\dormancy_active\\dormancy_active\\' + \
              'spatialRun\\CESMdataModelout\\GSDESMreadme\\REF_DEPTH1.nc'
ncfid = Dataset(ref_depthfn, 'r')
nc_attrs, nc_dims, nc_vars = mync.ncdump(ncfid)
lats = ncfid.variables['lat'][:]
lons = ncfid.variables['lon'][:]
var = ncfid.variables['REF_DEPTH'][:]

var_rebin = mysm.rebin_mean(var, (280, 720))
# plot
cbtitle = "soil depth (GSDESM)"
im = myplt.millshow(np.flipud(var_rebin),np.arange(-180,180,0.5),np.arange(-56, 84, 0.5),
                    theTitle=None,cmap='GMT_haxby',latmin=-60,latmax=90,
                    cbartitle=cbtitle,cbar='h',extend='neither',clim=[0,100])
plt.savefig('./AncillaryData/HWSD/figure/'+'topawtc_ratio.png')              
#%% plot SoilGrids 50km soil C
SG_fn = 'C:\\download\\work\\!manuscripts\\C14_synthesis\\AncillaryData\\SoilGrids\\SoilGrids50km.nc'
ncfid = Dataset(SG_fn, 'r')
nc_attrs, nc_dims, nc_vars = mync.ncdump(ncfid)
lats = ncfid.variables['LATIXY'][:]
lons = ncfid.variables['LONGXY'][:]
var = ncfid.variables['OCSTHA_M'][:]
scalar = 0.1
im = myplt.millshow(scalar * np.sum(var[[0,1,2],:,:],axis=0),lons, lats,
                    theTitle=None,cmap='GMT_haxby',latmin=-90,latmax=90,
                    cbartitle='C stock (kg/m2)',cbar='h',extend='neither',clim=[0,100])
                    
                    