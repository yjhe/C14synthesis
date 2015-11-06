# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 22:03:13 2015

@author: Yujie
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import D14Cpreprocess as prep
import scipy.stats as stats
from sklearn import linear_model
import statsmodels.api as sm
import mystats as mysm
import myplot
import scipy.io
import C14tools
#%% linear regression of Cave14C. MAT, MAP
filename = 'Non_peat_data_synthesis.csv'
Cave14C = prep.getCweightedD14C2(filename,cutdep=40.)
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1])  
mat = prep.getvarxls(data,'MAT',Cave14C[~np.isnan(Cave14C[:,4]),3],0)
mapp = prep.getvarxls(data,'MAP',Cave14C[~np.isnan(Cave14C[:,4]),3],0)

x = np.c_[mat.astype(float),mapp.astype(float)]
y = Cave14C[~np.isnan(Cave14C[:,4]),4]
notNaNs = ~np.any(np.isnan(x),1) & ~np.isnan(y)
#stats.linregress(x[notNaNs,:],y[notNaNs]) # cannot do multiple-linear reg
#
#clf = linear_model.LinearRegression() # no statistics!!
#clf.fit(x[notNaNs],y[notNaNs])

X = x[notNaNs,:]
y = y[notNaNs]
X= sm.add_constant(X)
ols = sm.OLS(y, X).fit()
print ols.summary()

#%% OLS on Cave14C
filename = 'Non_peat_data_synthesis.csv'
cutdep = 100.
Cave14C = prep.getCweightedD14C2(filename,cutdep=cutdep)
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1])  
mat = prep.getvarxls(data,'MAT',Cave14C[:,3],0)
mapp = prep.getvarxls(data,'MAP',Cave14C[:,3],0)
lon = prep.getvarxls(data,'Lon',Cave14C[:,3],0)
lat = prep.getvarxls(data,'Lat',Cave14C[:,3],0)
vegid = prep.getvarxls(data, 'VegTypeCode_Local', Cave14C[:,3], 0)
vegiduniq = np.unique(vegid[~np.isnan(vegid)])
botD14C = prep.getvarxls(data, 'D14C_BulkLayer',Cave14C[:,3], -1)
clayon = 0
nppon = 0

if clayon == 1:
    sclayfn = 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\S_CLAY.nc4'
    tclayfn = 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\T_CLAY.nc4'
    scecclayfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\S_CEC_CLAY.nc4'
    tcecclayfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\T_CEC_CLAY.nc4'
    tbulkdenfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\T_BULK_DEN.nc4'
    sbulkdenfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\S_BULK_DEN.nc4'
    nppfn = 'AncillaryData\\NPP\\2000_2012meannpp_gCm2yr.nc'
    sclay = prep.getHWSD(sclayfn, lon, lat) # % weight
    tclay = prep.getHWSD(tclayfn, lon, lat)  # % weight
    scecclay = prep.getHWSD(scecclayfn, lon, lat)  # cmol/kg clay
    tcecclay = prep.getHWSD(tcecclayfn, lon, lat)  # cmol/kg clay
    sbulkden = prep.getHWSD(sbulkdenfn, lon, lat)  # g/cm3
    tbulkden = prep.getHWSD(tbulkdenfn, lon, lat)  # g/cm3
    clay = tbulkden*tclay*0.3 + sbulkden*sclay*0.7
    cecclay =  tbulkden*tcecclay*0.3 + sbulkden*scecclay*0.7
if nppon == 1:
    npp = prep.getnpp(nppfn, lon, lat) # gC/m2/yr

dummy = (vegid[:, None] == vegiduniq).astype(float)  # dummy[:,1:]
#x = np.c_[mat.astype(float),mapp.astype(float),dummy[:,1:]]
x = np.c_[mat.astype(float),mapp.astype(float),dummy[:,1:]]
sel = np.logical_and(~np.isnan(Cave14C[:,4]), Cave14C[:,2]==cutdep)

y = Cave14C[:, 4]
#y = botD14C.astype(float)
notNaNs = ~np.any(np.isnan(x),1) & ~np.isnan(y) & sel
X = x[notNaNs,:]
y = y[notNaNs]
#y = y/1000. + 1.
X= sm.add_constant(X)
ols = sm.OLS(y, X).fit()
print ols.summary()
print "inlcuded profiles are:", Cave14C[notNaNs,3]
#% try log transformation
fig, axes = plt.subplots(nrows=1,ncols=1)
#plt.hist(y,30,alpha=0.3, normed=True)
plt.scatter(X[:,1],y)

#%% OLS using slope on profile
filename = 'Non_peat_data_synthesis.csv'
slpFM_lnC = prep.getslopeFM_lnC(filename)
slpD14C_cumC = prep.getslopeD14C_cumC(filename)
y = slpFM_lnC[:,1]
profid = slpFM_lnC[:,0]
# get X for profiles
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1])  
mat = prep.getvarxls(data,'MAT',profid,0)
mapp = prep.getvarxls(data,'MAP',profid,0)
lon = prep.getvarxls(data,'Lon',profid,0)
lat = prep.getvarxls(data,'Lat',profid,0)
vegid = prep.getvarxls(data, 'VegTypeCode_Local', profid, 0)
vegiduniq = np.unique(vegid[~np.isnan(vegid)])
botD14C = prep.getvarxls(data, 'D14C_BulkLayer', profid, -1)
sclayfn = 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\S_CLAY.nc4'
tclayfn = 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\T_CLAY.nc4'
scecclayfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\S_CEC_CLAY.nc4'
tcecclayfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\T_CEC_CLAY.nc4'
tbulkdenfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\T_BULK_DEN.nc4'
sbulkdenfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\S_BULK_DEN.nc4'
nppfn = 'AncillaryData\\NPP\\2000_2012meannpp_gCm2yr.nc'
sclay = prep.getHWSD(sclayfn, lon, lat) # % weight
tclay = prep.getHWSD(tclayfn, lon, lat)  # % weight
scecclay = prep.getHWSD(scecclayfn, lon, lat)  # cmol/kg clay
tcecclay = prep.getHWSD(tcecclayfn, lon, lat)  # cmol/kg clay
sbulkden = prep.getHWSD(sbulkdenfn, lon, lat)  # g/cm3
tbulkden = prep.getHWSD(tbulkdenfn, lon, lat)  # g/cm3
npp = prep.getnpp(nppfn, lon, lat) # gC/m2/yr
clay = tbulkden*tclay*0.3 + sbulkden*sclay*0.7
cecclay =  tbulkden*tcecclay*0.3 + sbulkden*scecclay*0.7
dummy = (vegid[:, None] == vegiduniq).astype(float)  # dummy[:,1:]
x = np.c_[mat.astype(float),mapp.astype(float),dummy[:,1:]]
# regress
notNaNs = ~np.any(np.isnan(x),1) & ~np.isnan(y) & ~np.isinf(y)
X = x[notNaNs,:]
y = y[notNaNs]
X= sm.add_constant(X) 
ols = sm.OLS(y, X).fit()
print ols.summary()
    
fig, axes = plt.subplots(nrows=1,ncols=1)
plt.scatter(y,clay[notNaNs])
axes.set_xlabel('D14C')
axes.set_ylabel('CEC_Clay')

#%% Robust Linear Model on Cave14C
filename = 'Non_peat_data_synthesis.csv'
cutdep = 110.
Cave14C = prep.getCweightedD14C2(filename,cutdep=cutdep)
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1])  
mat = prep.getvarxls(data,'MAT',Cave14C[:,3],0)
mapp = prep.getvarxls(data,'MAP',Cave14C[:,3],0)
lon = prep.getvarxls(data,'Lon',Cave14C[:,3],0)
lat = prep.getvarxls(data,'Lat',Cave14C[:,3],0)
vegid = prep.getvarxls(data, 'VegTypeCode_Local', Cave14C[:,3], 0)
vegiduniq = np.unique(vegid[~np.isnan(vegid)])
botD14C = prep.getvarxls(data, 'D14C_BulkLayer',Cave14C[:,3], -1)
clayon = 0
nppon = 0
plott = 0
plotres = 1

if clayon == 1:
    sclayfn = 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\S_CLAY.nc4'
    tclayfn = 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\T_CLAY.nc4'
    scecclayfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\S_CEC_CLAY.nc4'
    tcecclayfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\T_CEC_CLAY.nc4'
    tbulkdenfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\T_BULK_DEN.nc4'
    sbulkdenfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\S_BULK_DEN.nc4'
    nppfn = 'AncillaryData\\NPP\\2000_2012meannpp_gCm2yr.nc'
    sclay = prep.getHWSD(sclayfn, lon, lat) # % weight
    tclay = prep.getHWSD(tclayfn, lon, lat)  # % weight
    scecclay = prep.getHWSD(scecclayfn, lon, lat)  # cmol/kg clay
    tcecclay = prep.getHWSD(tcecclayfn, lon, lat)  # cmol/kg clay
    sbulkden = prep.getHWSD(sbulkdenfn, lon, lat)  # g/cm3
    tbulkden = prep.getHWSD(tbulkdenfn, lon, lat)  # g/cm3
    clay = tbulkden*tclay*0.3 + sbulkden*sclay*0.7
    cecclay =  tbulkden*tcecclay*0.3 + sbulkden*scecclay*0.7
if nppon == 1:
    npp = prep.getnpp(nppfn, lon, lat) # gC/m2/yr

dummy = (vegid[:, None] == vegiduniq).astype(float)  # dummy[:,1:]
#x = np.c_[mat.astype(float),mapp.astype(float),dummy[:,1:]]
x = np.c_[mat.astype(float),mapp.astype(float),dummy[:,1:]]
sel = np.logical_and(~np.isnan(Cave14C[:,4]), Cave14C[:,2]<=cutdep)
y = Cave14C[:, 4]
#y = botD14C.astype(float)
notNaNs = ~np.any(np.isnan(x),1) & ~np.isnan(y) & sel
X = x[notNaNs,:]
y = y[notNaNs]
#y = y/1000. + 1.
X= sm.add_constant(X)
rlm = sm.RLM(y, X).fit()
print rlm.summary()
yhat = rlm.predict()
print "R2 is: %.3f, R2adj is: %.3f" %(mysm.cal_R2(y,yhat,n=rlm.nobs,p=rlm.df_model))

print "inlcuded profiles are:", Cave14C[notNaNs,3]

if plott == 1:
    fig, axes = plt.subplots(nrows=1,ncols=1)
    #plt.hist(y,30,alpha=0.3, normed=True)
    plt.scatter(X[:,1],y)
if plotres == 1:
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.scatter(rlm.predict(), rlm.resid)
    ax.set_ylabel('residual')
    ax.set_xlabel('yhat')
    
#%% OLS model on profile
# indicate if want to use deltaDelta
ddelta = 1
filename = 'Non_peat_data_synthesis.csv'
cutdep = 40.
Cave14C = prep.getCweightedD14C2(filename)
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1])  
profid = data[data['Start_of_Profile']==1].index # index of profile start
d14C = prep.getvarxls(data,'D14C_BulkLayer', profid, ':')
sampleyr = prep.getvarxls(data, 'SampleYear', profid, ':')
dd14C = prep.getDDelta14C(sampleyr, d14C)
tau, cost = C14tools.cal_tau(d14C, sampleyr)
data['tau'] = pd.Series(tau[:,0], index=data.index)
mat = prep.getvarxls(data,'MAT', profid, ':')
mapp = prep.getvarxls(data,'MAP', profid, ':')
layerbot = prep.getvarxls(data, 'Layer_bottom_norm', profid, ':')
vegid = prep.getvarxls(data, 'VegTypeCode_Local', profid, ':')
vegiduniq = np.unique(vegid[~np.isnan(vegid)])
soilorder = prep.getvarxls(data, 'LEN Order', profid, ':')
soilorder = np.array([str(i) for i in soilorder])
soilorderuniq = np.unique(soilorder[soilorder != 'nan'])
lon = prep.getvarxls(data,'Lon',profid,':')
lat = prep.getvarxls(data,'Lat',profid,':')
clayon = 0
nppon = 0
plott = 0
plotres = 1
plot_y_yhat = 1

if clayon == 1:
    sclayfn = 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\S_CLAY.nc4'
    tclayfn = 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\T_CLAY.nc4'
    scecclayfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\S_CEC_CLAY.nc4'
    tcecclayfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\T_CEC_CLAY.nc4'
    tbulkdenfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\T_BULK_DEN.nc4'
    sbulkdenfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\S_BULK_DEN.nc4'
    nppfn = 'AncillaryData\\NPP\\2000_2012meannpp_gCm2yr.nc'
    sclay = prep.getHWSD(sclayfn, lon, lat) # % weight
    tclay = prep.getHWSD(tclayfn, lon, lat)  # % weight
    scecclay = prep.getHWSD(scecclayfn, lon, lat)  # cmol/kg clay
    tcecclay = prep.getHWSD(tcecclayfn, lon, lat)  # cmol/kg clay
    sbulkden = prep.getHWSD(sbulkdenfn, lon, lat)  # g/cm3
    tbulkden = prep.getHWSD(tbulkdenfn, lon, lat)  # g/cm3
    clay = tbulkden*tclay*0.3 + sbulkden*sclay*0.7
    cecclay =  tbulkden*tclay*tcecclay*0.3 + sbulkden*sclay*scecclay*0.7
if nppon == 1:
    npp = prep.getnpp(nppfn, lon, lat) # gC/m2/yr

dummyveg = (vegid[:, None] == vegiduniq).astype(float)  # dummy[:,1:]
dummysord = (soilorder[:, None] == soilorderuniq).astype(float)
depthind = np.zeros([dummyveg.shape[0], 1])
depthind[layerbot>300.] = 1
depthind = depthind*(np.c_[layerbot**1./2, layerbot**2, layerbot**3.])
intact = np.reshape(layerbot,[len(layerbot),1])*np.c_[mat,mapp]
x = np.c_[layerbot, mat, mapp, intact, dummyveg[:,1:]]
ddelta = 0
vartau = 1
if ddelta == 1:
    y = dd14C
else:
    y = d14C
if vartau == 1:
    y = tau[:,0]
#y = np.log(y/1000. + 10.)
#y, _ = stats.boxcox(y)
#for i in y:
#    if i in ydel:
#        y[i] = np.nan
notNaNs = ~np.any(np.isnan(x),1) & ~np.isnan(y) 
X = x[notNaNs,:]
y = y[notNaNs]
X= sm.add_constant(X)

model = sm.OLS(y, X).fit()
print model.summary()
yhat = model.predict()
print "R2 is: %.3f, R2adj is: %.3f" %(mysm.cal_R2(y,yhat,n=model.nobs,p=model.df_model))
print "rmse is %.3f"%(mysm.cal_RMSE(y, yhat))
if plott == 1:
    fig, axes = plt.subplots(nrows=1,ncols=1)
    #plt.hist(y,30,alpha=0.3, normed=True)
    plt.scatter(X[:,1],y)
if plotres == 1:
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.scatter(model.predict(), model.resid)
    ax.set_ylabel('residual')
    ax.set_xlabel('yhat')
if plot_y_yhat:
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.scatter(y, yhat)
    ax.set_ylabel('yhat')
    ax.set_xlabel('y')
    myplot.refline()
    
boolist = yhat < -1000.
idx = [i for i, elem in enumerate(boolist) if elem]
    
#%% Linear Mixed Effects Model on profile
filename = 'Non_peat_data_synthesis.csv'
cutdep = 110.
Cave14C = prep.getCweightedD14C2(filename)
data = pd.read_csv(filename,encoding='iso-8859-1',skiprows=[1],index_col='ProfileID')  
profid = Cave14C[:,3]
d14C = prep.getvarxls(data,'D14C_BulkLayer', profid, ':')
mat = prep.getvarxls(data,'MAT', profid, ':')
mapp = prep.getvarxls(data,'MAP', profid, ':')
layerbot = prep.getvarxls(data, 'Layer_bottom_norm', profid, ':')
vegid = prep.getvarxls(data, 'VegTypeCode_Local', profid, ':')
vegiduniq = np.unique(vegid[~np.isnan(vegid)])

lon = prep.getvarxls(data,'Lon',Cave14C[:,3],0)
lat = prep.getvarxls(data,'Lat',Cave14C[:,3],0)
clayon = 0
nppon = 0
plott = 0
plotres = 1
plot_y_yhat = 1

if clayon == 1:
    sclayfn = 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\S_CLAY.nc4'
    tclayfn = 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\T_CLAY.nc4'
    scecclayfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\S_CEC_CLAY.nc4'
    tcecclayfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\T_CEC_CLAY.nc4'
    tbulkdenfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\T_BULK_DEN.nc4'
    sbulkdenfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\S_BULK_DEN.nc4'
    nppfn = 'AncillaryData\\NPP\\2000_2012meannpp_gCm2yr.nc'
    sclay = prep.getHWSD(sclayfn, lon, lat) # % weight
    tclay = prep.getHWSD(tclayfn, lon, lat)  # % weight
    scecclay = prep.getHWSD(scecclayfn, lon, lat)  # cmol/kg clay
    tcecclay = prep.getHWSD(tcecclayfn, lon, lat)  # cmol/kg clay
    sbulkden = prep.getHWSD(sbulkdenfn, lon, lat)  # g/cm3
    tbulkden = prep.getHWSD(tbulkdenfn, lon, lat)  # g/cm3
    clay = tbulkden*tclay*0.3 + sbulkden*sclay*0.7
    cecclay =  tbulkden*tcecclay*0.3 + sbulkden*scecclay*0.7
if nppon == 1:
    npp = prep.getnpp(nppfn, lon, lat) # gC/m2/yr

dummy = (vegid[:, None] == vegiduniq).astype(float)  # dummy[:,1:]
#x = np.c_[mat.astype(float),mapp.astype(float),dummy[:,1:]]
intact = np.reshape(layerbot,[len(layerbot),1])*np.c_[mat,mapp]
x = np.c_[layerbot, mat, mapp, intact]
y = d14C
notNaNs = ~np.any(np.isnan(x),1) & ~np.isnan(y) 
X = x[notNaNs,:]
y = y[notNaNs]
#y = np.log(y/1000. + 1.)
X= sm.add_constant(X)

model = sm.regression.mixed_linear_model.MixedLM(y, X, \
                                groups=vegid[notNaNs]).fit()
print model.summary()
yhat = np.dot(np.c_[X,np.ones([X.shape[0],1])], model.params)
print "R2 is: %.3f, R2adj is: %.3f" %(mysm.cal_R2(y,yhat,n=model.nobs,p=model.nobs-model.df_resid))

free = sm.regression.mixed_linear_model.MixedLMParams(2, 2)
free.set_fe_params(np.ones(2))
free.set_cov_re(np.eye(2))
model = sm.MixedLM.from_formula(
             "D14C_BulkLayer ~ Layer_bottom_norm + VegTypeCode_Local",\
             data[notNaNs], \
             re_formula="Layer_bottom_norm", \
             groups=data["VegTypeCode_Local"][notNaNs], missing='drop').fit(free=free)
print model.summary()
yhat = model.predict(model.params,)
print "R2 is: %.3f, R2adj is: %.3f" %(mysm.cal_R2(y,yhat,n=model.nobs,p=model.df_model))

if plott == 1:
    fig, axes = plt.subplots(nrows=1,ncols=1)
    #plt.hist(y,30,alpha=0.3, normed=True)
    plt.scatter(X[:,1],y)
if plotres == 1:
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.scatter(model.predict(), model.resid)
    ax.set_ylabel('residual')
    ax.set_xlabel('yhat')
if plot_y_yhat:
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.scatter(y, yhat)
    ax.set_ylabel('yhat')
    ax.set_xlabel('y')
    myplot.refline()
    
#%% Robust Linear Model on profile
filename = 'Non_peat_data_synthesis.csv'
cutdep = 110.
Cave14C = prep.getCweightedD14C2(filename,cutdep=cutdep)
data = pd.read_csv(filename,encoding='iso-8859-1',skiprows=[1],index_col='ProfileID')  
profid = Cave14C[:,3]
d14C = prep.getvarxls(data,'D14C_BulkLayer', profid, ':')
mat = prep.getvarxls(data,'MAT', profid, ':')
mapp = prep.getvarxls(data,'MAP', profid, ':')
layerbot = prep.getvarxls(data, 'Layer_bottom', profid, ':')
vegid = prep.getvarxls(data, 'VegTypeCode_Local', profid, ':')
vegiduniq = np.unique(vegid[~np.isnan(vegid)])

lon = prep.getvarxls(data,'Lon',Cave14C[:,3],0)
lat = prep.getvarxls(data,'Lat',Cave14C[:,3],0)
clayon = 0
nppon = 0
plott = 0
plotres = 1
plot_y_yhat = 1

if clayon == 1:
    sclayfn = 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\S_CLAY.nc4'
    tclayfn = 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\T_CLAY.nc4'
    scecclayfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\S_CEC_CLAY.nc4'
    tcecclayfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\T_CEC_CLAY.nc4'
    tbulkdenfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\T_BULK_DEN.nc4'
    sbulkdenfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\S_BULK_DEN.nc4'
    nppfn = 'AncillaryData\\NPP\\2000_2012meannpp_gCm2yr.nc'
    sclay = prep.getHWSD(sclayfn, lon, lat) # % weight
    tclay = prep.getHWSD(tclayfn, lon, lat)  # % weight
    scecclay = prep.getHWSD(scecclayfn, lon, lat)  # cmol/kg clay
    tcecclay = prep.getHWSD(tcecclayfn, lon, lat)  # cmol/kg clay
    sbulkden = prep.getHWSD(sbulkdenfn, lon, lat)  # g/cm3
    tbulkden = prep.getHWSD(tbulkdenfn, lon, lat)  # g/cm3
    clay = tbulkden*tclay*0.3 + sbulkden*sclay*0.7
    cecclay =  tbulkden*tcecclay*0.3 + sbulkden*scecclay*0.7
if nppon == 1:
    npp = prep.getnpp(nppfn, lon, lat) # gC/m2/yr

dummy = (vegid[:, None] == vegiduniq).astype(float)  # dummy[:,1:]
#x = np.c_[mat.astype(float),mapp.astype(float),dummy[:,1:]]
intact = np.reshape(layerbot,[len(layerbot),1])*np.c_[mat,mapp]
x = np.c_[layerbot, mat, mapp, intact]
y = d14C
notNaNs = ~np.any(np.isnan(x),1) & ~np.isnan(y) 
X = x[notNaNs,:]
y = y[notNaNs]
#y = np.log(y/1000. + 1.)
X= sm.add_constant(X)

rlm = sm.RLM(y, X).fit()
print rlm.summary()
yhat = rlm.predict()
print "R2 is: %.3f, R2adj is: %.3f" %(mysm.cal_R2(y,yhat,n=rlm.nobs,p=rlm.df_model))

if plott == 1:
    fig, axes = plt.subplots(nrows=1,ncols=1)
    #plt.hist(y,30,alpha=0.3, normed=True)
    plt.scatter(X[:,1],y)
if plotres == 1:
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.scatter(model.predict(), model.resid)
    ax.set_ylabel('residual')
    ax.set_xlabel('yhat')
if plot_y_yhat:
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.scatter(y, yhat)
    ax.set_ylabel('yhat')
    ax.set_xlabel('y')
    myplot.refline()
    
#%% create climate-space based biomes
class Point():
    def __init__(self,x,y):
        self.x = x
        self.y = y
        
def Angle2D(x1,y1,x2,y2):
    ''' Return the angle b/w two vectors on a plane.
    The angle is fro vector1 to vector2, positive anticlockwise
    the result is b/w -pi -> pi
    '''
    theta1 = np.arctan2(y1,x1)
    theta2 = np.arctan2(y2,x2)
    dtheta = theta2 - theta1
    while dtheta > np.pi:
        dtheta -= 2*np.pi
    while dtheta < -np.pi:
        dtheta =+ 2*np.pi
    return dtheta

def insidePoly(polygon, n, p):
    angle = 0
    tol = 2
    for i in range(n):
        p1 = Point(polygon[i,0]-p.x, polygon[i,1]-p.y)
        p2 = Point(polygon[(i+1)%n,0]-p.x, polygon[(i+1)%n,1]-p.y)
        angle += Angle2D(p1.x, p1.y, p2.x, p2.y)
    #print 'angle is: ', angle
    if abs(angle) < 2.*np.pi-tol:
        return False
    else:
        return True

def plot_wittaker(wisk, clim=None):
    if clim:
        plt.scatter(wisk[wisk.clim_code==clim].mat, wisk[wisk.clim_code==clim].map, c='k',s=20)
    else:
        plt.scatter(wisk.mat, wisk.map, c='k',s=20)
    
    
filename = 'C:\\download\\work\\!manuscripts\\C14_synthesis\\figures\\statanalysis\\WISKERDATA.xlsx'  
wisk = pd.read_excel(filename,sheetname='combined')
filename = 'Non_peat_data_synthesis.csv'
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1]) 
pid = data.index.unique()
uniqsite = data.copy().reset_index().drop_duplicates(subset=['MAT','MAP','ProfileID'])
uniqsite.set_index('ProfileID',inplace=True)
for n,(temp,pre) in enumerate(uniqsite[['MAT','MAP']].values):
    #print n, temp, pre
    p = Point(temp, pre)
    for clim in wisk.clim_code.unique():
        matmap = wisk.loc[wisk.clim_code==clim,['mat','map']].values
        if insidePoly(matmap, matmap.shape[0], p):
            uniqsite.loc[n+1,'clim_code'] = clim
            break

# plot
def plot_assigned():
    vegtype = ['tundra','woodland and grassland','savanna','desert',
               'boreal forest','temperate forest','tropical dry forest',
               'temperate wet forest','tropical wet forest']
    cmm = plt.cm.Set1(np.linspace(0,1,9))
    plt.scatter(uniqsite.MAT,uniqsite.MAP,c='grey',
                alpha=0.7,s=10)
    
    for n,bio in enumerate(vegtype):
        idx = uniqsite.clim_code==(n+1.)
        plt.scatter(uniqsite[idx].MAT,uniqsite[idx].MAP,c=cmm[n],
                    alpha=0.8,s=60,label=bio)
    plt.legend(loc=2,fontsize=10)
    plt.gca().set_xlabel(r'MAT ($^{\circ}$C)')
    plt.gca().set_ylabel(r'MAP (mm)')
    plt.gca().set_ylim([-100,6000])
    plot_wittaker(wisk)

# modify other not-correctly assigned points
uniqsite.loc[(uniqsite.clim_code==1)&(uniqsite.MAP>400),'clim_code'] = 2
uniqsite.loc[(uniqsite.clim_code==2)&(uniqsite.MAT>19),'clim_code'] = 3
uniqsite.loc[(uniqsite.clim_code==1),'clim_code'] = 4
uniqsite.loc[(uniqsite.clim_code==2)&uniqsite.MAP>400, 'clim_code'] = 3
uniqsite.loc[(uniqsite.clim_code.isnull())&(uniqsite.MAP>3000), 'clim_code'] = 9
uniqsite.loc[(uniqsite.clim_code.isnull())&(uniqsite.MAP<1000), 'clim_code'] = 2
uniqsite.loc[uniqsite.clim_code.isnull(),'clim_code'] = 6
uniqsite.loc[(uniqsite.clim_code==8)&(uniqsite.MAT>20),'clim_code'] = 9
uniqsite.loc[(uniqsite.clim_code==5)&(uniqsite.MAT>5),'clim_code'] = 2
uniqsite.loc[(uniqsite.clim_code==3)&(uniqsite.MAP<470),'clim_code'] = 4
uniqsite.loc[(uniqsite.clim_code==4)&(uniqsite.MAT<-10),'clim_code'] = 2
uniqsite.loc[(uniqsite.clim_code==2)&(uniqsite.MAP>1300),'clim_code'] = 6
uniqsite.loc[(uniqsite.clim_code==4)&(uniqsite.MAT<10),'clim_code'] = 2

plot_assigned()

uniqsite['clim_code'].to_csv('clim_code.csv')

# merge with the compelete original Non_synthesis dataset
data.reset_index(inplace=True)
uniqsite.reset_index(inplace=True)
dum = uniqsite[['clim_code','ProfileID']].reset_index()
datanew = data.merge(dum,how='left',on='ProfileID')
datanew.to_csv()

for p in data.index.unique():
    data.loc[p:p,'clim_code'] = uniqsite.loc[p,'clim_code']
data.to_csv('clim_code.csv',encoding='iso-8859-1')
#%% test
plt.figure()
prof = 591
print 'MAT, MAP is ', uniqsite.loc[prof, 'MAT'],uniqsite.loc[prof, 'MAP']
p = Point(uniqsite.loc[prof, 'MAT'],uniqsite.loc[prof, 'MAP'])
for clim in wisk.clim_code.unique():
    matmap = wisk.loc[wisk.clim_code==clim,['mat','map']].values
    insidePoly(matmap, matmap.shape[0], p)
plt.scatter(p.x, p.y, s=90)
plot_wittaker(wisk,1)

#%% self-created climate region
temp_range = [-30, 0, 18, 30]
map_range = [0, 1000, 2000, 6000]

filename = 'Non_peat_data_synthesis.csv'
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1]) 
pid = data.index.unique()
uniqsite = data.copy().reset_index().drop_duplicates(subset=['MAT','MAP','ProfileID'])

vegtype = np.arange(0,9,1)
cmm = plt.cm.Set1(np.linspace(0,1,9))
temp_n = 0; map_n = 0;
for n,bio in enumerate(vegtype):
    idx = (uniqsite.MAT>temp_range[temp_n]) & \
          (uniqsite.MAT<=temp_range[temp_n+1]) & \
          (uniqsite.MAP>map_range[map_n]) & \
          (uniqsite.MAT<=map_range[map_n+1]) 
    plt.scatter(uniqsite[idx].MAT,uniqsite[idx].MAP,c=cmm[n],
                alpha=0.8,s=60,label=bio)
    uniqsite.loc[idx,'clim_code'] = bio
    map_n += 1
    if map_n == 3:
        temp_n += 1
        map_n = 0
        
plt.legend(loc=2,fontsize=10)
plt.gca().set_xlabel(r'MAT ($^{\circ}$C)')
plt.gca().set_ylabel(r'MAP (mm)')
plt.gca().set_ylim([0,6000])

# save clim_code to csv
uniqsite.set_index('ProfileID',inplace=True)
for p in data.index.unique():
    data.loc[p:p,'clim_code'] = uniqsite.loc[p,'clim_code']
data.to_csv('clim_code.csv',encoding='iso-8859-1')