# -*- coding: utf-8 -*-
"""
Extrapolation file: use ML models to predict global D14C

Created on Thu Jun 18 02:05:17 2015

@author: happysk8er
"""
#%%
import pandas as pd
import numpy as np
import D14Cpreprocess as prep
import sklearn as sk
from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import mystats as mysm
import C14tools
import scipy.io
import matplotlib.pyplot as plt
import matplotlib as mpl
import myplot as myplt
from netCDF4 import Dataset
import mynetCDF as mync
from mpl_toolkits.basemap import Basemap, cm
from scipy.interpolate import interp1d

def prep_obs(useorilyr=1,cutdep=None,soil='shallow',ddelta=0,vartau=0,clayon=0,
             nppon=0,pred_veg=1,pred_soil=1,pred_climzone=0,y='D14C_BulkLayer'):
    '''
    Prepare observation data for model training
    param: 
        useorilyr : whether use original layer or incremented layer data
        cutdep    : indicate the depth above which to used for training        
        indicate what is y (default: d14C; or ddelta, or vartau)
        indicate what additional input to use (clay or npp)
        y         : 'D14C_BulkLayer', 'ddelta', 'tau', 'pct_C', 'BulkDensity',
                    'O_thickness', if this, then X is one row per profile
    return:
        Xcont, Xdummy : continuous X [n*m], dummy X [n*k], k is total cols after dummy-zation
    '''
    filename = 'Non_peat_data_synthesis.csv'
    data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1])  
    profid = data.index.unique() # index of profile start
    if y != 'O_thickness':    
        if useorilyr == 1:    
            d14C = prep.getvarxls(data,'D14C_BulkLayer', profid, ':')
            pctC = prep.getvarxls(data,'pct_C', profid, ':')
            bd = prep.getvarxls(data,'BulkDensity', profid, ':')
            sampleyr = prep.getvarxls(data, 'SampleYear', profid, ':')
            dd14C = prep.getDDelta14C(sampleyr, d14C)
            #tau, cost = C14tools.cal_tau(d14C, sampleyr, 1, 1)
            #data['tau'] = pd.Series(tau[:,0], index=data.index)
            mat = prep.getvarxls(data,'MAT', profid, ':')
            mapp = prep.getvarxls(data,'MAP', profid, ':')
            layerbot = prep.getvarxls(data, 'Layer_bottom_norm', profid, ':')
            vegid = prep.getvarxls(data, 'VegTypeCode_Local', profid, ':')
            vegiduniq = np.unique(vegid[~np.isnan(vegid)])
            soilorder = prep.getvarxls(data, 'SoilOrder_LEN_USDA', profid, ':')
            soilorder = np.array([str(i) for i in soilorder])
            soilorderuniq = np.unique(soilorder[soilorder != 'nan'])
            climid = prep.getvarxls(data, 'clim_code', profid, ':')
            climuniq = np.unique(climid[~np.isnan(climid)])
            lon = prep.getvarxls(data,'Lon',profid,':')
            lat = prep.getvarxls(data,'Lat',profid,':')
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
                cecclay =  tbulkden*tclay*tcecclay*0.3 + sbulkden*sclay*scecclay*0.7
            if nppon == 1:
                npp = prep.getnpp(nppfn, lon, lat) # gC/m2/yr
            dummyveg = (vegid[:, None] == vegiduniq).astype(float)  # dummy[:,1:]
            dummysord = (soilorder[:, None] == soilorderuniq).astype(float)
            dummyclim = (climid[:, None] == climuniq).astype(float)
            
            # construct X and y
            if clayon == 1:
                x = np.c_[layerbot, mat, mapp, cecclay]
            else:
                x = np.c_[layerbot, mat, mapp]
            if y == 'D14C_BulkLayer':
                y = d14C
            elif y == 'ddelta':
                y = dd14C
            elif y == 'tau':
                y = tau[:,0]
            elif y == 'pct_C':
                y = pctC            
            elif y == 'BulkDensity':
                y = bd           
    
        else:
            colname = ['Layer_top_norm','Layer_bottom_norm','D14C_BulkLayer','BulkDensity','pct_C',
                       'VegTypeCode_Local','MAT','MAP','SoilOrder_LEN_USDA','clim_code','SampleYear']
            newdf = data[colname]
            incre_data = prep.prep_increment(newdf, colname)
            dd14C = prep.getDDelta14C(incre_data['SampleYear'].values.astype(float),
                                      incre_data['D14C_BulkLayer'].values.astype(float))    
            pctC = incre_data['pct_C'].values.astype(float)
            bd = incre_data['BulkDensity'].values.astype(float)
            vegid = incre_data['VegTypeCode_Local'].values.astype(int)
            vegiduniq = np.unique(vegid[~np.isnan(vegid)])
            soilorder = incre_data['SoilOrder_LEN_USDA'].values.astype(str)
            soilorderuniq = np.unique(soilorder[soilorder != 'nan'])
            climid = incre_data['clim_code'].values.astype(int)
            climuniq = np.unique(climid[~np.isnan(climid)])
    
            # construct X and y
            dummyveg = (vegid[:, None] == vegiduniq).astype(float)  # dummy[:,1:]
            dummysord = (soilorder[:, None] == soilorderuniq).astype(float)
            dummyclim = (climid[:, None] == climuniq).astype(float)
            x = incre_data[['Layer_depth_incre','MAT','MAP']].values.astype(float)
            if y == 'D14C_BulkLayer':
                y = incre_data['D14C_BulkLayer'].values.astype(float)
            elif y == 'ddelta':
                y = dd14C
            elif y == 'tau':
                y = tau[:,0]
            elif y == 'pct_C':
                y = pctC
            elif y == 'BulkDensity':
                y = bd
        if pred_veg == 1 and pred_soil == 1:
            x = np.c_[x, dummyveg[:,:], dummysord[:,:]]
        if pred_veg == 1 and pred_soil == 0:
            x = np.c_[x, dummyveg[:,:]]
        if pred_veg == 0 and pred_soil == 1:
            x = np.c_[x, dummysord[:,:]]
        if pred_climzone == 1:
            x = np.c_[x, dummyclim[:,1:]]
        notNaNs = ~np.any(np.isnan(x),1) & ~np.isnan(y) 
        X = x[notNaNs,:]
        y = y[notNaNs]
        # deal with cutdepth
        if cutdep is not None:
            if soil == 'shallow':
                idx = X[:,0] <= cutdep # use onlyl data that have depth < cutdep
            elif soil == 'deep':
                maxdep = np.nanmax(X[:,0])
                if cutdep >= maxdep:
                    print 'cutdep is larger than the maximum depth'
                    return
                idx = X[:,0] >= cutdep
            X = X[idx,:]; y = y[idx]
    elif y == 'O_thickness':
        profid = data.index.unique()
        mat = prep.getvarxls(data, 'MAT', profid, 0)
        mapp = prep.getvarxls(data, 'MAP', profid, 0)
        vegid = prep.getvarxls(data, 'VegTypeCode_Local', profid, 0)
        vegiduniq = np.unique(vegid[~np.isnan(vegid)])
        soilorder = prep.getvarxls(data, 'SoilOrder_LEN_USDA', profid, 0)
        soilorder = np.array([str(i) for i in soilorder])
        soilorderuniq = np.unique(soilorder[soilorder != 'nan'])
        climid = prep.getvarxls(data, 'clim_code', profid, 0)
        climuniq = np.unique(climid[~np.isnan(climid)])
        # construct X and y
        dummyveg = (vegid[:, None] == vegiduniq).astype(float)  # dummy[:,1:]
        dummysord = (soilorder[:, None] == soilorderuniq).astype(float)
        dummyclim = (climid[:, None] == climuniq).astype(float)
        y = np.zeros((profid.shape[0],)); y[:] = np.nan
        for n,p in enumerate(profid):
            tmp = data.loc[p:p,['Layer_top_norm','Layer_bottom_norm']]
            top = tmp.Layer_top_norm.values[0]
            if top < 0:
                y[n] = -top
            elif top == 0:
                y[n] = 0
            else:
                continue
        x = np.c_[mat, mapp]
        if pred_veg == 1 and pred_soil == 1:
            x = np.c_[x, dummyveg[:,:], dummysord[:,:]]
        if pred_veg == 1 and pred_soil == 0:
            x = np.c_[x, dummyveg[:,1:]]
        if pred_veg == 0 and pred_soil == 1:
            x = np.c_[x, dummysord[:,1:]]
        if pred_climzone == 1:
            x = np.c_[x, dummyclim[:,1:]]
        notNaNs = ~np.any(np.isnan(x),1) & ~np.isnan(y) 
        X = x[notNaNs,:]
        y = y[notNaNs]
    return X, y

def get_Xscaled(X, Xextra=None):
    '''
    Construct X_scaled for model training. 
    If Xextra is None, use X to construct, otherwise, use X + Xextra to construct
    '''
    if Xextra is None:
        scaler = sk.preprocessing.StandardScaler().fit(X)
        print 'in get_Xscaled, Xextra is None, constructed scaler.mean is ', scaler.mean_
        X_scaled = scaler.transform(X)
    else:
        scaler = sk.preprocessing.StandardScaler().fit(np.r_[X,Xextra])
        print 'in get_Xscaled, Xextra is given, the constructed scaler.mean for whole dataset is ', scaler.mean_
        X_scaled = scaler.transform(X)
    return scaler, X_scaled
    
def d14C_svr(X_scaled, y, params):
    ''' 
    run svr on X_scaled.
    return:
        svrtrained : trained svr model, can use to predict
    '''
    # train SVM and predict
    svr = sk.svm.SVR(kernel='rbf', **params)
    svrtrained = svr.fit(X_scaled, y)
    yhat = svr.fit(X_scaled, y).predict(X_scaled)
    print "... training results ..."
    print "R2 is: %.3f, R2adj is: %.3f" %(mysm.cal_R2(y,yhat),0)
    print "RMSE is: %.3f" %(mysm.cal_RMSE(y,yhat))
    return svrtrained

def d14C_gbrt(X, y, params):
    '''
    run GBRT on original X. no need to scale.
    return:
        est : trained GBRT model, can use to predict
    '''
    est = GradientBoostingRegressor(**params).fit(X, y)
    yhat = est.predict(X)
    print "... training results ..."
    print "R2 is: %.3f, R2adj is: %.3f" %(mysm.cal_R2(y,yhat),0)
    print "mean absolute error is : %.3f"%sk.metrics.mean_absolute_error(y, yhat)
    return est

def d14C_rf(X, y, params):
    '''
    run random forest on original X. no need to scale
    return:
        rf : trained rf model, can use to predict
    '''
    rf = RandomForestRegressor(**params).fit(X, y)
    yhat = rf.predict(X)
    print "... training results ..."
    print "R2 is: %.3f, R2adj is: %.3f" %(mysm.cal_R2(y,yhat),0)
    print "mean absolute error is : %.3f"%sk.metrics.mean_absolute_error(y, yhat)
    return rf

def d14C_ols(X_scaled, y):
    ols = sk.linear_model.LinearRegression()
    ols.fit(X_scaled, y)
    yhat = ols.predict(X_scaled)
    print "... training results ..."
    print "R2 is: %.3f, R2adj is: %.3f" %(mysm.cal_R2(y,yhat),0)
    print "mean absolute error is : %.3f"%sk.metrics.mean_absolute_error(y, yhat)
    return ols
    
def d14C_rdg(X_scaled, y):
    rdg = sk.linear_model.Ridge(alpha=5, copy_X=True, fit_intercept=True)    
    rdg.fit(X_scaled, y)
    yhat = rdg.predict(X_scaled)
    print "... training results ..."
    print "R2 is: %.3f, R2adj is: %.3f" %(mysm.cal_R2(y,yhat),0)
    print "mean absolute error is : %.3f"%sk.metrics.mean_absolute_error(y, yhat)
    return rdg

def d14C_lso(X_scaled, y):
    lso = sk.linear_model.Lasso(alpha=0.01, copy_X=True, 
                                fit_intercept=True,positive=False,
                                selection='cyclic') 
    lso.fit(X_scaled, y)
    yhat = lso.predict(X_scaled)
    print "... training results ..."
    print "R2 is: %.3f, R2adj is: %.3f" %(mysm.cal_R2(y,yhat),0)
    print "mean absolute error is : %.3f"%sk.metrics.mean_absolute_error(y, yhat)
    return lso
    
def changelccodeto14Csyn(lc):
    '''change the code from original landcover data to the 8 categories
    used in 14C synthesis. lc contains 1-12, 14.
    Remeber to update the file under Ancillary data if you change this LC allocation.
    parameters: 
        lc is the readed in .mat matrix
    return: 
        np.array same with lc dimension, but with the 8 14C categories
    '''
    idx = np.zeros((17,lc.shape[0],lc.shape[1]))
    for i in range(17):
        idx[i,:,:] = lc==i
    lonts, lats = construct_lonlat()
    mapping = {1:[1,2,3,4,5,8],
               2:[1,2,3,4,5], 
               3:[1,2,3,4,5],
               4:[10],
               5:[12,14],
               6:[6,7],
               7:[11],
               8:[9],
               9:[6,7],
              10:[16]}

    for i in mapping.keys():
        tmpidx = np.zeros((lc.shape))
        for j in mapping[i]:
            tmpidx = tmpidx + idx[j,:,:]
        if i == 1: # boreal forest
            tmpidx[lats<50,:] = 0
        if i == 2: # teperate forest
            tmpidx[(lats>=50) | ((lats<23) & (lats>-23)) | (lats<=-50), :] = 0
        if i == 3: # tropical forest
            tmpidx[(lats>=23) | (lats<=-23),:] = 0     
        lc[tmpidx>=1] = i
        if i == 9: # high latitude tundra, assigned from shrubland, now assign back
            lc[(tmpidx>0) & np.tile(lats<50, (720,1)).T] = 6  
    # tropical grassland and shrubland set to savana        
    lc[(lc==4) & np.tile(lats<15,(720,1)).T & np.tile(lats>-15,(720,1)).T] = 8
    lc[(lc==6) & np.tile(lats<15,(720,1)).T & np.tile(lats>-15,(720,1)).T] = 8
    # shrubland > 45N set to boreal forest    
    lc[(lc==6) & np.tile(lats>45,(720,1)).T] = 1
    # grassland > 60N set to tundra
    lc[(lc==4) & np.tile(lats>60,(720,1)).T] = 9
    # grassland from equiter to 20S set to savanna
    lc[(lc==4) & np.tile((lats>-23)&(lats<0),(720,1)).T] = 8
    return lc

def prep_extra_data(plotorilc=0, plotnewlc=0, pred_veg=1, pred_soil=1):
    ''' 
    Prepare input data for extrapolation. Depth is exlcuded. 
    return:
        Xextra : X data matrix of everything except for depth.
        notNaNs: boolean vector indicates notNaN entries.
    '''
    mat = np.load('..\\AncillaryData\\CRUdata\\cruannualtmp.npy')
    mat = np.nanmean(mat[-30:-20,:,:],axis=0)/10.   # top-down 90S-90N
    maprep = np.load('..\\AncillaryData\\CRUdata\\cruannualpre.npy')
    maprep = np.nanmean(maprep[-30:-20,:,:],axis=0)*12./10. # 10 is scaler
    lc = scipy.io.loadmat('..\\AncillaryData\\landcoverdata\\lc2011_05x05.mat') #1-12, 14, 16
    lc = lc['lc05x05']   # top-down 90N-90S
    soilorder = np.load('..\\AncillaryData\\Glb_SoilOrder\\Glb_soilorder.npy')

    # plot original lc
    if plotorilc == 1:
        lons,lats = construct_lonlat()
        vegtype = ['EG needleleaf','EG broadleaf','Deci Needleleaf','Deci broadleaf',
                   'Mixed forest','closed shrublands','Open shrublands','Woody savannas',
                   'savannas','grasslands','permanent wetlands','croplands',
                   'Cropland/natural veg mosaic']
        myplt.geoshow(np.flipud(lc),lons,np.flipud(lats),cbar='h',ticks=range(16),cbarticklab=vegtype,
                      rotation=45)
    lc = changelccodeto14Csyn(lc)
#    np.save('..\\AncillaryData\\landcoverdata\\14CsynthesisLC.npy',lc)  # top-down 90N-90S
    lc = np.flipud(lc) # top-down 90S-90N
#    plt.figure()
#    myplt.geoshow(mat,lons,np.flipud(lats))
#    myplt.geoshow(maprep,lons,np.flipud(lats))
    # plot new lc
    if plotnewlc == 1:
        lons,lats = construct_lonlat()
        vegtype = ['boreal forest','temperate forest','tropical forest','grassland',
                   'cropland','shrublands','peatland','savannas','tundra','desert']
        myplt.geoshow(lc,lons,np.flipud(lats),cbar='h',ticks=np.arange(0,12,1),cbarticklab=vegtype,
                      rotation=45, levels=10, extend='neither')    

    # ravel matrix to column vector. maxtrix is top-down 90S-90N
    mat_1d = np.ravel(mat)
    maprep_1d = np.ravel(maprep)
    lc_1d = np.ravel(lc)
    sord_1d = np.ravel(np.flipud(soilorder))
    
    # create dummy variables for categorical data
    dummyuniq_veg = np.unique(lc_1d[~np.isnan(lc_1d)])
    dummyuniq_sord = [159, 153, 156, 161, 160, 155, 150, 151, 158, 154, 152, 157] # to be consistent
        # in the same order with the training data, which is
        # 'Alf', 'And', 'Ard', 'Ent', 'Ept', 'Ert', 'Gel', 'His', 'Oll', 'Ox',
        # 'Spo', 'Ult'
    dummy_lc = (lc_1d[:,None] == dummyuniq_veg).astype(float)
    dummy_sord = (sord_1d[:,None] == dummyuniq_sord).astype(float)
    if pred_veg == 1 and pred_soil == 1:
        Xextra = np.c_[mat_1d, maprep_1d, dummy_lc[:,:], dummy_sord[:,:]]
    if pred_veg == 1 and pred_soil == 0:
        Xextra = np.c_[mat_1d, maprep_1d, dummy_lc[:,:]]
    if pred_veg == 0 and pred_soil == 1:
        Xextra = np.c_[mat_1d, maprep_1d, dummy_sord[:,:]]    
    notNaNs = ~np.any(np.isnan(Xextra), 1)
    return Xextra, notNaNs, lc_1d

def d14C_pred(mdl, depth, Xextra, scaler=None):
    ''' use trained svr to predict d14C at a particular depth. no NANs are allowed.
    params:
        mdl   : the trained learner
        scaler: scaler to scale Xextra (if you data is scaled, pass scaler in)
        depth : scaler 
        Xextra: missing depth (1st) column
    return:
        yhat: predicted y, depends on what mdl is. (N*1)
    '''    
    N = Xextra.shape[0]
    X = np.c_[np.tile(depth,(N,1)), Xextra]
    if scaler is None:
        return mdl.predict(X)
    X_scaled = scaler.transform(X)
    yhat = mdl.predict(X_scaled)
    return yhat

def extrapolate_paintbynumber():
    
def construct_lonlat():
    latstep = 0.5; lonstep = 0.5;
    lonmax = 180.0 - lonstep/2.0;
    lonmin = -180.0 + lonstep/2.0;
    latmax = 90.0 - latstep/2.0;
    latmin = -90.0 + latstep/2.0;
    lats = np.arange(latmax,latmin-0.1*latstep,-latstep)
    lons = np.arange(lonmin,lonmax+0.1*lonstep,lonstep)
    return lons, lats

def cal_binedtauC(tau, C, bins):
    '''
    calculate the binned tau and total C, used for plotting histogram
    parameters:
        tau  : turnover time, 2-D
        C    : carbon content (kgC/m2)
        bins : number of bins you want to group tau
    return: 
        xtau : 1-D mean tau for each bin [bins,]
        yC   : 1-D total C for each bin [bins,]
    '''
    areaa = mysm.cal_earthgridarea(0.5)
    hist, bin_eg = np.histogram(tau[~np.isnan(tau)], bins)
#    bin_eg = np.arange(0, np.log(30000), np.log(30000)/(bins+1))  
#    bin_eg = np.exp(bin_eg)
    xtau = np.zeros(bins)
    yC   = np.zeros(bins)
    for n in range(bins):
        idx = np.where((tau > bin_eg[n]) & (tau < bin_eg[n+1]))
        xtau[n] = np.nansum(tau[idx] * areaa[idx])/np.nansum(areaa[idx])
        yC[n]   = np.nansum(C[idx] * 1e3 * areaa[idx])
    return xtau, yC, bin_eg

def getbiomed14Cprof(depth, mdl, scaler=None, siteinfo=None, **kwargs):
    '''
    Extrapolated global biome 14C profile (mean, std)
    params:
        depth    ： {list} depth you want to extrpolate at
        mdl      : {sklearn model obj} the learner
        siteinfo : {list} lon, lat, lc at sites. If not None, get data at sites
    return:
        biome_14C_mean : {list} [len(depth),nbiomes]
        biome_14C_std  : {list} [len(depth),nbiomes] 
    '''
    Xextra, notNaNs, lc_1d = prep_extra_data(**kwargs)
    lc = np.reshape(lc_1d,(360,720)); lc = np.flipud(lc) # 90N - 90S
    #print 'total grid notNaNs: ', sum(notNaNs)    
    biomes    = np.unique(lc_1d)[:10]  
    biome_14C_mean = np.zeros((len(depth),len(biomes)))
    biome_14C_std = np.zeros((len(depth),len(biomes)))
    if siteinfo is not None:
        mask = np.zeros((len(biomes), lc.shape[0], lc.shape[1]))
        lons, lats = construct_lonlat()
        for nb, bio in enumerate(biomes):
            dum = np.array(siteinfo)
            cursiteinfo = list(dum[dum[:,2]==bio,:])
            for lon, lat, sitelc in cursiteinfo:  
                if ~np.isnan(lon) and ~np.isnan(lat):
                    ii,jj = np.where(lats<lat)[0][0], np.where(lons>lon)[0][0]
                    if sitelc == lc[ii,jj]:
                        print 'sitelc is consistent with lc map...'
                    else:
                        print 'Not consistent lc ...'
                        print '     site lon, lat is %.2f, %.2f, lc lon,lat is %.2f, %.2f'%(lon, lat, lons[jj], lats[ii])
                        print '     sitelc is %.1f, lc is %.1f, ...'%(sitelc, lc[ii,jj])
                    mask[nb,ii,jj] = 1    
    for nd, d in enumerate(depth):
        print 'depth is ', d
        yhat = np.empty((Xextra.shape[0])); yhat[:] = np.NAN        
        ypred = d14C_pred(mdl, d, Xextra[notNaNs,:], scaler)
        yhat[notNaNs] = ypred
        yhat = np.reshape(yhat,(360,720))
        yhat = np.flipud(yhat) # now 90N-90S
       
        # get biome profile
        if siteinfo is not None:
            for nb, bio in enumerate(biomes):
                tmp = yhat[mask[nb,:,:]==1]
                biome_14C_mean[nd, nb] = np.nanmean(tmp)
                biome_14C_std[nd, nb]  = np.nanstd(tmp)
        else:
            for nb, bio in enumerate(biomes):
                tmp = yhat[lc==bio]
                biome_14C_mean[nd, nb] = np.nanmean(tmp)
                biome_14C_std[nd, nb]  = np.nanstd(tmp)
    return biome_14C_mean, biome_14C_std
 
def getbiomed14Cprof_trained(depth, mdl, scaler, **kwargs):
    '''
    get predicted biome14C profile at training site. This is to show 
    how well the trainning does.
    params:
        mdl      : {sklearn model obj} the learner
        siteinfo : {list} lon, lat, lc at sites. If not None, get data at sites
    return:
        biome_14C_mean : {list} [len(depth),nbiomes]
        biome_14C_std  : {list} [len(depth),nbiomes] 
    '''
    X, y = prep_obs(**kwargs)
    biomes    = np.arange(1,11)
    biome_14C_mean = np.zeros((len(depth),len(biomes)))
    biome_14C_std = np.zeros((len(depth),len(biomes)))

    for nd, d in enumerate(depth):
        print 'depth is ', d
        N = X.shape[0]
        X = np.c_[np.tile(d,(N,1)), X[:,1:]]  # substitute the depth column
        X_scaled = scaler.transform(X)
        yhat = mdl.predict(X_scaled)
            
        # get biome profile
        for nb, bio in enumerate(biomes):
            tmp = yhat[X[:,bio+2]==1]
            biome_14C_mean[nd, nb] = np.nanmean(tmp)
            biome_14C_std[nd, nb]  = np.nanstd(tmp)
    return biome_14C_mean, biome_14C_std
    
def plot_obs_pf(group='veg', axes=None, yvar='D14C_BulkLayer', 
                xlim=[-1000, 500], ylim=[-50, 800]):
    '''
    plot observed biome/soil order profile with shaded area. use incremented data
    params:
        group: {'veg'|'soil'} indicate plot by vegetation or by soil order
        yvar : {Str} y-variable, 'D14C_BulkLayer','pct_C', or 'BulkDensity'
        xlim : {list} plotting range of the yvar
        ylim : {list} depth range 
    '''
    filename = 'Non_peat_data_synthesis.csv'
    data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1]) 
    pid = data.index.unique() # index of profile start
    if group == 'veg':
        colname = ['Layer_top_norm','Layer_bottom_norm',yvar,'VegTypeCode_Local']
    elif group == 'soil':
        colname = ['Layer_top_norm','Layer_bottom_norm',yvar,'SoilOrder_LEN_USDA']
    newdf = data[colname]
    incre_data = prep.prep_increment(newdf, colname)
    
    # plot observed biome/soir order -averaged profile. shaded errorbar
    if group == 'veg':
        vegtype = ['boreal forest','temperate forest','tropical forest','grassland',
                   'cropland','shrublands','peatland','savannas','tundra','desert']
    elif group == 'soil':
        order = {0:'Alf', 1:'And', 2:'Ard', 3:'Ent', 
                 4:'Gel', 5:'His', 6:'Ept', 7:'Oll',
                 8:'Ox', 9:'Spo', 10:'Ult', 11:'Ert'}
    if axes is None:
        if group == 'veg':
            nrows, ncols = 2, 5
            fig, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(15,8),sharey=True)
        elif group == 'soil':
            nrows, ncols = 3, 4
            fig, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(12,10),sharey=True)
    nrows, ncols = axes.shape
    for n,row in enumerate(axes):
        for m,ax in enumerate(row):
            idx = ncols*n+m
            if group == 'veg':
                dum = prep.get_biome_ave(idx+1, incre_data, yvar) # biome index (1-10)
            elif group == 'soil':
                dum = prep.get_soilorder_ave(order[idx], incre_data, yvar) # soil (0-11)
            x = dum['mean']; y = dum['Layer_depth_incre']; err = dum['std']
            ax.plot(x, y, 'k-')    
            ax.locator_params(axis='y',tight=True,nbins=8)
            ax.locator_params(axis='x',tight=True,nbins=6)
            ax.fill_betweenx(y, x-err, x+err, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848',
                            linewidth=0, linestyle='-')
            ax.set_ylabel('depth(cm)')
            ax.set_xlabel(yvar)
            if group == 'veg':
                ax.set_title(vegtype[idx])
            elif group == 'soil':
                ax.set_title(order[idx])
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.grid(True)
    ax.invert_yaxis()
    plt.tight_layout()  
            
def plot_obs_pf_mean(group='veg', yvar='D14C_BulkLayer', xlim=[-1000, 500],
                     ylim=[-50, 800]):
    '''
    plot observed biome profile. use incremented data. plot mean only (no shaded area)
    params:
        group: {'veg'|'soil'} plot all veg-averaged or soil order-averaged profile
        yvar : {Str} y-variable, 'D14C_BulkLayer','pct_C', or 'BulkDensity'
        xlim : {list} plotting range of the yvar
        ylim : {list} depth range 
    '''
    filename = 'Non_peat_data_synthesis.csv'
    data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1]) 
    if group == 'veg':
        colname = ['Layer_top_norm','Layer_bottom_norm',yvar,'VegTypeCode_Local']
        groups = ['boreal forest','temperate forest','tropical forest','grassland',
                   'cropland','shrublands','peatland','savannas','tundra','desert']
    elif group == 'soil':
        colname = ['Layer_top_norm','Layer_bottom_norm',yvar,'SoilOrder_LEN_USDA']
        groups = {0:'Alf', 1:'And', 2:'Ard', 3:'Ent', 
                 4:'Gel', 5:'His', 6:'Ept', 7:'Oll',
                 8:'Ox', 9:'Spo', 10:'Ult', 11:'Ert'}               
    nbiome = len(groups)
    newdf = data[colname]
    incre_data = prep.prep_increment(newdf, colname)
    
    # plot observed biome-averaged profile. shaded errorbar   
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,8),sharey=True)
    cmm = plt.get_cmap('Set1')
    for idx in range(nbiome):  
        if group == 'veg':
            dum = prep.get_biome_ave(idx+1, incre_data, yvar) # biome index (1-10)
        elif group == 'soil':
                dum = prep.get_soilorder_ave(groups[idx], incre_data, yvar) # soil (0-11)
        x = dum['mean']; y = dum['Layer_depth_incre']; 
        ax.plot(x, y, '-',color=cmm(1.*idx/nbiome*1.),label=groups[idx],linewidth=2)    
        ax.locator_params(axis='y',tight=True,nbins=8)
        ax.locator_params(axis='x',tight=True,nbins=6)
        ax.set_ylabel('depth(cm)')
        ax.set_xlabel(yvar)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(loc=4,fontsize=12)
    ax.invert_yaxis()
    ax.grid(True)
    plt.tight_layout()       

def plot_obs_biomepf_pairwise(axes):
    '''
    plot observed biome profile with shaded area. use incremented data. 
    plot paired veg curves
    '''
    filename = 'Non_peat_data_synthesis.csv'
    data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1]) 
    pid = data.index.unique() # index of profile start
    colname = ['Layer_top_norm','Layer_bottom_norm','D14C_BulkLayer','VegTypeCode_Local']
    newdf = data[colname] 
    incre_data = prep.prep_increment(newdf, colname)
    
    # plot observed biome-averaged profile. shaded errorbar
    vegtype = ['boreal forest','temperate forest','tropical forest','grassland',
               'cropland','shrublands','peatland','savannas','tundra','desert']
    cmm = plt.get_cmap('Set1')
    numcolr = 2*len(set(vegtype)) # no repeat in color
    vegcolr = [cmm(1.*jj/numcolr) for jj in range(numcolr)]
    vegn = 1; vegn2 = 2
    for n,row in enumerate(axes):       
        for m,ax in enumerate(row):
            dum = prep.get_biome_ave(vegn, incre_data, 'D14C_BulkLayer') # biome index (1-10)
            x = dum['mean']; y = dum['Layer_depth_incre']; err = dum['std']
            ax.plot(x, y, color=vegcolr[2*(vegn-1)], label=vegtype[vegn-1])    
            ax.fill_betweenx(y, x-err, x+err, alpha=0.5, edgecolor='#CC4F1B', 
                             facecolor=vegcolr[2*(vegn-1)+1],linewidth=0, linestyle='-')
            dum = prep.get_biome_ave(vegn2, incre_data, 'D14C_BulkLayer') # biome index (1-10)
            x = dum['mean']; y = dum['Layer_depth_incre']; err = dum['std']
            ax.plot(x, y, color=vegcolr[2*(vegn2-1)], label=vegtype[vegn2-1])
            ax.fill_betweenx(y, x-err, x+err, alpha=0.5, edgecolor='#CC4F1B', 
                             facecolor=vegcolr[2*(vegn2-1)+1],linewidth=0, linestyle='-')
            ax.locator_params(axis='y',tight=True,nbins=8)
            ax.locator_params(axis='x',tight=True,nbins=6)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=7,loc=4)
            ax.set_xlim([-1000, 500])
            ax.set_ylim([-10, 250])
            ax.grid(True)   
            ax.invert_yaxis()
            if n == 6:
                ax.set_xlabel(r'$\Delta^{14}C$',fontsize=8)
            if m == 0:
                ax.set_ylabel('depth(cm)',fontsize=8)

            if vegn2 < 10:
                vegn2 += 1
            else:
                vegn += 1
                vegn2 = vegn + 1
            if (vegn > 10) or (vegn2 > 10):
                plt.tight_layout() 
                return

def plot_obs_climcodepf(kind='whittaker'):
    '''
    plot observed biome profile with shaded area. use incremented data
    params:
        kind: whittaker or self-defined
    '''
    filename = 'Non_peat_data_synthesis.csv'
    data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1]) 
    pid = data.index.unique() # index of profile start
    if kind == 'whittaker':
        colname = ['Layer_top_norm','Layer_bottom_norm','D14C_BulkLayer','clim_code_whittaker']
        vegtype = ['tundra','woodland and grassland','savanna','desert','boreal forest',
                   'temperate forest','tropical dry forest','temperate wet forest',
                   'tropical wet forest']    
    elif kind == 'self-defined':
        colname = ['Layer_top_norm','Layer_bottom_norm','D14C_BulkLayer','clim_code']   
        vegtype = ['cold dry', 'cold mesic', 'cold wet',
                   'temperate dry', 'temperate mesic', 'temperate wet',
                   'tropical dry', 'tropical mesic', 'tropical wet']
    newdf = data[colname]
    incre_data = prep.prep_increment(newdf, colname)
    
    # plot observed biome-averaged profile. shaded errorbar    
    nrows, ncols = 2, 5
    fig, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(15,8),sharey=True)
    for n,row in enumerate(axes):
        for m,ax in enumerate(row):
            idx = ncols*n+m
    #        if idx == 8:
    #            break     
            try:
                dum = prep.get_climzone_ave(kind, idx+1, incre_data, 'D14C_BulkLayer') # clim_code index (1-9)
            except KeyError:
                continue
            x = dum['mean']; y = dum['Layer_depth_incre']; err = dum['sem']
            ax.plot(x, y, 'k-')    
            ax.locator_params(axis='y',tight=True,nbins=8)
            ax.locator_params(axis='x',tight=True,nbins=6)
            ax.fill_betweenx(y, x-err, x+err, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848',
                            linewidth=0, linestyle='-')
            ax.set_ylabel('depth(cm)')
            ax.set_xlabel(r'$\Delta^{14}C$')
            ax.set_title(vegtype[idx])
            ax.set_xlim([-1000, 500])
            ax.set_ylim([-10, 250])
            ax.grid(True)
    ax.invert_yaxis()
    plt.tight_layout()  

    
def verify_site_lc(siteinfo):
    '''
    Extract MODIS land cover at profile sites, add this info to siteinfo file which has 
    lon, lat, site land cover. 
    '''
    Xextra, notNaNs, lc_1d = prep_extra_data()
    lc = np.reshape(lc_1d,(360,720)); lc = np.flipud(lc) # 90N - 90S
    lons, lats = construct_lonlat() # lon/lat for lc
    lcatsite = np.zeros([len(siteinfo),2]) # 1st col is MODISlc, 2nd col is WHETHER consistent
    lcatsite[:] = np.nan
    for n, (lon, lat, sitelc) in enumerate(siteinfo):  
        if ~np.isnan(lon) and ~np.isnan(lat):
            ii = np.argmin((lon-lons)**2)
            jj = np.argmin((lat-lats)**2)
            lcatsite[n,0] = lc[jj,ii]
            if sitelc == lc[jj,ii]:
                lcatsite[n,1] = 1
                print 'sitelc is consistent with lc map...'
            else:
                lcatsite[n,1] = 0
                print 'Not consistent lc ...'
                print '     site lon, lat is %.2f, %.2f, lc lon,lat is %.2f, %.2f'%(lon, lat, lons[ii], lats[jj])
                print '     sitelc is %.1f, lc is %.1f, ...'%(sitelc, lc[jj,ii])
    out = np.c_[np.array(siteinfo),lcatsite]
    df = pd.DataFrame(out, columns=['lon','lat','sitelc','MODISlc','is_same'])    
    return df
    
def plot_lc_and_sitelc(mapp='veg', plt_veg=None):
    '''
    Plot MODIS land cover/mat/maprep and the site landcover (white edged triangle)
    params:    
        mapp   : {str} plot mat/maprep/veg for the underlying map
        plt_veg: {int} if want to plot a specific vegtype, 1-10
    '''
    Xextra, notNaNs, lc_1d = prep_extra_data()
    lc = np.reshape(lc_1d,(360,720)); # 90S-90N
    mat = np.reshape(Xextra[:,0],(360,720))
    maprep = np.reshape(Xextra[:,1],(360,720))
    lons,lats = construct_lonlat()
    df = pd.read_csv('verify_site_modisls.csv')
    vegtype = ['boreal forest','temperate forest','tropical forest','grassland',
               'cropland','shrublands','peatland','savannas','tundra','desert']
    fig, ax = plt.subplots()
    # plot underlying map
    if mapp == 'veg':
        myplt.geoshow(lc,lons,np.flipud(lats),cbar='h',ticks=np.arange(0,11),
                      cmap='jet',cbarticklab=vegtype,
                      rotation=45, levels=10, extend='neither')        
    elif mapp == 'mat':
        myplt.geoshow(mat,lons,np.flipud(lats),cbar='h',cmap='jet', extend='neither',
                      cbartitle=r'MAT $(^{\circ} C)$')
    elif mapp == 'maprep':
        myplt.geoshow(maprep,lons,np.flipud(lats),cbar='h',cmap='jet',extend='neither',
                      cbartitle='MAP (mm)') 
    # plot site lc
    cmm = plt.get_cmap('jet',10)
    norm = mpl.colors.BoundaryNorm(np.linspace(1,10,10), cmm.N) # make a color map of fixed colors
    if plt_veg is None:
        plt.scatter(df.lon, df.lat,c=df.sitelc,marker='^',edgecolor='w',s=55,cmap=cmm) 
    else:
        idx = df.sitelc==plt_veg       
        a = ax.scatter(df.lon[idx], df.lat[idx],c=df.sitelc[idx],marker='^',
                    edgecolor='w',s=55,cmap=cmm,label=vegtype[plt_veg-1],norm=norm)
        #a.set_label(vegtype[plt_veg-1])
        plt.draw()
        plt.legend([a],[vegtype[plt_veg-1]],loc=4)
        plt.tight_layout()

def plot_mat_and_sitelc(plt_veg=None):
    '''
    Plot CRU mean annual temperature map and the site landcover (white edged triangle)
    params:    
        plt_veg: {int} if want to plot a specific vegtype
    '''
    Xextra, notNaNs, lc_1d = prep_extra_data()
    lc = np.reshape(lc_1d,(360,720)); # 90S-90N
    lons,lats = construct_lonlat()
    df = pd.read_csv('verify_site_modisls.csv')
    vegtype = ['boreal forest','temperate forest','tropical forest','grassland',
               'cropland','shrublands','peatland','savannas','tundra','desert']
    myplt.geoshow(lc,lons,np.flipud(lats),cbar='h',ticks=np.arange(0,11),
                  cmap='jet',cbarticklab=vegtype,
                  rotation=45, levels=10, extend='neither')
    
    cmm = plt.get_cmap('jet',10)
    if plt_veg is None:
        plt.scatter(df.lon, df.lat,c=df.sitelc,marker='^',edgecolor='w',s=55,cmap=cmm) 
        return
    else:
        idx = df.sitelc==plt_veg
        plt.scatter(df.lon[idx], df.lat[idx],c=df.sitelc[idx],marker='^',edgecolor='w',s=55,cmap=cmm) 
        return
        
def getJobaggySOC(vegtype, depth_vec, lon, lat, hwsdsoc):
    '''
    Extrapolate SOC profile (kgC/m2 for each interval) using total SOC from HWSD
    and redistribute using Jobaggy curve. Jobbagy depth is upto 1m, so depth_vec 
    should be upto 1m.
    params: 
        vegtype     : {int 1-d array} 1-10 biome types as used in global 
                      extrapolation, nelms should match grid number.
        depth_vec   : {n-d array} depth vector (cm). start with 0, end with 100
                      shape is hwsd.shape[0] * d-depth
        lon, lat    : {int} lon, lat of the profile
        hwsd        : {1-d array} for differnet grid     
    return:
        extraSOC    : {n-d array} kgC/m2  for each interval. shape is depth_vec.shape
    '''
    jobgydepth = [20, 40, 60, 80, 100]
    # biome code in my xlsx. pctC of the totalC in top 1m from jobaggy
    csvbiome = {1:[50, 25, 13, 7, 5], # boreal forest
                2:[50, 22, 13, 8, 7], # temperate forest
                3:[38, 22, 17, 13, 10], # tropical forest
                4:[41, 23, 15, 12, 9], # grassland
                5:[41, 23, 15, 12, 9], # cropland
                6:[39, 22, 16, 13, 10], # shrubland
                7:[46, 46, 46, 46, 46], # peatland
                8:[36, 23, 18, 13, 10], # Savanas
                9:[40, 29, 19, 7, 5], # tundra
                10:[33, 22, 18, 15, 13]} # desert
    out = np.zeros_like(hwsd)
    for i in range(hwsdsoc.shape[1]):
        jobgypctC = np.array(csvbiome[vegtype[i]])/100.
        f_i = interp1d(np.r_[0,jobgydepth], np.r_[jobgypctC[0],jobgypctC])
        f_x = prep.extrap1d(f_i)
        extpctC = f_x(depth_vec[:,i])/sum(f_x(depth_vec[:,i]))*1.
        extrasoc = hwsdsoc[i] * np.reshape(extpctC,-1,1)
        out[:,i] = extrasoc
    return out

#%% learned parameters for all models
svr_params = {'D14C_BulkLayer':{'C':1e4, 'gamma':0.05},
              'pct_C':{'C':100, 'gamma':10},
              'BulkDensity':{'C':10, 'gamma':1.8}}

gbrt_params = {'D14C_BulkLayer':{'n_estimators': 1400, 'max_depth': 4, 'min_samples_split': 2,
                       'learning_rate': 0.01, 'loss': 'ls'},
               'pct_C':{'n_estimators': 1200, 'max_depth': 4, 'min_samples_split': 2,
                        'learning_rate': 0.01, 'loss': 'ls'},
               'BulkDensity':{'n_estimators': 1500, 'max_depth': 7, 'min_samples_split': 2,
                      'learning_rate': 0.01, 'loss': 'ls'}}

rf_params = {'D14C_BulkLayer':{'n_estimators': 200, 'max_depth': 16},
             'pct_C':{'n_estimators': 180, 'max_depth': 19},
             'BulkDensity':{'n_estimators': 150, 'max_depth': 20},
             'tau':{'n_estimators': 300, 'max_depth': 3},
             'O_thickness':{'n_estimators': 150, 'max_depth': 30}}
            
#%% train svr and predict (extrapolate) and map. calculate tau, and save
scalarfromobs = 1
yvar = 'pct_C'
kwargs = {'pred_veg': 1, 'pred_soil': 1}
# use only observations to contruct scaler 
depth = 50.
X, y = prep_obs(useorilyr=1,y=yvar,**kwargs)
Xextra, notNaNs, lc_1d = prep_extra_data(**kwargs)
scaler, X_scaled = get_Xscaled(X)

#svr = d14C_svr(X_scaled, y, svr_params[yvar])
#ypred = d14C_pred(svr, depth, Xextra[notNaNs,:], scaler=scaler)

gbrt = d14C_gbrt(X, y, gbrt_params[yvar])
ypred = d14C_pred(gbrt, depth, Xextra[notNaNs,:])

#rf = d14C_gbrt(X, y, rf_params[yvar])
#ypred = d14C_pred(rf, depth, Xextra[notNaNs,:])

#####  rf on O_thickness
X, y = prep_obs(y='O_thickness',**kwargs)
Xextra, notNaNs, lc_1d = prep_extra_data(**kwargs)
rf = d14C_gbrt(X, y, rf_params['O_thickness'])
ypred = rf.predict(Xextra[notNaNs,:])

yhat = np.empty((Xextra.shape[0])); yhat[:] = np.NAN
yhat[notNaNs] = ypred
yhat = np.reshape(yhat,(360,720))

if scalarfromobs == 0:
    # use the whole dataset to construct scaler.
    depth = 100.
    X, y = prep_obs(useorilyr=1)
    Xextra, notNaNs, lc_1d = prep_extra_data()
    Xextra_wdepth = np.c_[np.tile(depth,(Xextra.shape[0],1)), Xextra]
    scaler, X_scaled = get_Xscaled(X, Xextra_wdepth[~np.any(np.isnan(Xextra_wdepth),1),:])
    svr = d14C_svr(X_scaled, y)
    ypred = d14C_pred(svr, scaler, depth, Xextra[notNaNs,:])
    yhat = np.empty((Xextra.shape[0])); yhat[:] = np.NAN
    yhat[notNaNs] = ypred
    yhat = np.reshape(yhat,(360,720))


# plot
plt.figure()
lons,lats = construct_lonlat()
im = myplt.geoshow(yhat,lons,np.flipud(lats),extend='neither')
# clim=[0 100]
plt.title('O_thickness (cm)')
#plt.title('depth ' +  str(depth) + 'cm')



#%% read in HWSD and resample to 0.5 X 0.5, save data to npz. 
from scipy.interpolate import griddata
from scipy.misc import imresize
pathh = 'C:\\download\work\\!manuscripts\\C14_synthesis\\AncillaryData\\HWSD\Regridded_ORNLDAAC\\'
orifn = 'AWT_S_SOC.nc4'
orifn = 'AWT_T_SOC.nc4'
ncfid = Dataset(pathh + orifn, 'r')
nc_attrs, nc_dims, nc_vars = mync.ncdump(ncfid)
lats = ncfid.variables['lat'][:]
lons = ncfid.variables['lon'][:]
var = ncfid.variables[nc_vars[-1]][:]

awt_s_0505 = imresize(var, (360, 720))
lons,lats = construct_lonlat()
awt_s_0505[awt_s_0505 == 0] = np.nan
plt.figure()
myplt.geoshow(awt_s_0505,lons,np.flipud(lats))
outlats = np.arange()

awt_t_0505 = mysm.rebin(var,[360, 720])
myplt.geoshow(awt_t_0505,lons,np.flipud(lats))
awt_t_0505.fill_value = np.nan
awt_s_0505.fill_value = np.nan
t = awt_t_0505.filled()
s = awt_s_0505.filled()
np.savez('awt_soc_0505',awt_s_0505=s,awt_t_0505=t) # kgC/m2
#%%  plot histogram of turnover time and its associated C stock
# get turnover time
import C14tools
# yhat is 360*720, ypred is 1-D vec with no NaN
smpyr = 2000
tmp = C14tools.cal_tau(ypred,np.repeat(smpyr,ypred.shape[0]),1,1)
tau = np.empty(yhat.shape[0]*yhat.shape[1]); tau[:] = np.NAN
tau[notNaNs] = tmp
tau = np.reshape(tau, (360, 720))
im = myplt.geoshow(tau,lons,np.flipud(lats))
tau20cm = tau
np.save('extra_tau_20cm',tau20cm)

# read in HWSD
pathh = '..\\AncillaryData\\HWSD\\Regridded_ORNLDAAC\\'
hwsd = np.load(pathh + 'awt_soc_0505.npz')

tau = np.load('extra_tau_70cm.npy')
C     = hwsd['awt_t_0505']
bins  = 30
x, y, bin_eg  = cal_binedtauC(tau, C, bins)
#plt.hist(tau[~np.isnan(tau)],bins=bins,normed=True)

# plot barplot
fig, ax = plt.subplots(figsize=(8,6))
ax.bar(bin_eg[1:], y/1e15, width=np.diff(bin_eg)[0])
#ax.set_yscale('log')
ax.set_ylabel('SOC in 0-30cm (PgC)')
ax.set_xlabel('tau (yr)')
#%% get biome mean profile from global extrapolation
yvar = 'BulkDensity'
kwargs = {'pred_veg': 1, 'pred_soil': 1}
getglbextra_atsites = 0
# get biome from global extrapolation
depth = [-50, 0, 30, 70, 100]
X, y = prep_obs(useorilyr=1, y=yvar, **kwargs)
scaler, X_scaled = get_Xscaled(X)

#svr = d14C_svr(X_scaled, y, svr_params[yvar])
#meann, std = getbiomed14Cprof(depth, svr, scaler=scaler, **kwargs)

#gbrt = d14C_gbrt(X, y, gbrt_params[yvar])
#meann, std = getbiomed14Cprof(depth, gbrt, **kwargs)

rf = d14C_rf(X, y, rf_params[yvar])
meann, std = getbiomed14Cprof(depth, rf, **kwargs)

#ols = d14C_ols(X_scaled, y)
#meann, std = getbiomed14Cprof(depth, svr, **kwargs, scaler=scaler)
vegtype = ['boreal forest','temperate forest','tropical forest','grassland',
           'cropland','shrublands','peatland','savannas','tundra','desert']

if getglbextra_atsites == 1:
    # get biome from global extrapolation at obs sites
    filename = 'Non_peat_data_synthesis.csv'
    data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1]) 
    pid = data.index.unique() # index of profile start
    d14C = prep.getvarxls(data,'D14C_BulkLayer', pid, ':')
    sitelon = prep.getvarxls(data,'Lon', pid, 0)
    sitelat = prep.getvarxls(data,'Lat', pid, 0)
    sitelc = prep.getvarxls(data,'VegTypeCode_Local', pid, 0)
    siteinfo = zip(sitelon, sitelat, sitelc)
    meann, std = getbiomed14Cprof(depth, ols, scaler, siteinfo=siteinfo)
    vegtype = ['boreal forest','temperate forest','tropical forest','grassland',
               'cropland','shrublands','peatland','savannas','tundra','desert']

# plot
nrows, ncols = 2, 5
fig, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(15,8),sharey=True)
xlim = [0, 2]
plot_obs_pf(group='veg',axes=axes,yvar=yvar,xlim=xlim)
for n,row in enumerate(axes):
    for m,col in enumerate(row):
        idx = ncols*n+m
#        if idx == 8:
#            break
        col.errorbar(meann[:,idx], depth, xerr=std[:,idx], fmt='-o')    
        col.locator_params(nbins=4)
#        col.set_ylabel('depth(cm)')
#        col.set_xlabel(r'$\Delta^{14}C$')
        col.set_title(vegtype[idx])
        col.set_xlim(xlim)
        col.set_ylim([-50, 150])
col.invert_yaxis()
plt.tight_layout()

#%% get biome mean profile from trainning process
# get biome from training
yvar = 'BulkDensity'
kwargs = {'pred_veg': 1, 'pred_soil': 1, 'useorilyr':1}

# get biome from global extrapolation
depth = [-50, -20, -5, 0, 5, 10, 20, 30, 70, 100]
X, y = prep_obs(y=yvar, **kwargs)
scaler, X_scaled = get_Xscaled(X)

#svr = d14C_svr(X_scaled, y, svr_params[yvar])
#meann, std = getbiomed14Cprof_trained(depth, svr, scaler=scaler, **kwargs)

gbrt = d14C_gbrt(X_scaled, y, gbrt_params[yvar])
meann, std = getbiomed14Cprof_trained(depth, gbrt, scaler, **kwargs)

#rf = d14C_rf(X_scaled, y, rf_params[yvar])
#meann, std = getbiomed14Cprof_trained(depth, rf, scaler, **kwargs)

#ols = d14C_ols(X_scaled, y)
#meann, std = getbiomed14Cprof_trained(depth, svr, **kwargs, scaler=scaler)
vegtype = ['boreal forest','temperate forest','tropical forest','grassland',
           'cropland','shrublands','peatland','savannas','tundra','desert']

# plot
nrows, ncols = 2, 5
fig, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(15,8),sharey=True)
xlim = [0, 2]
plot_obs_pf(group='veg',axes=axes,yvar=yvar,xlim=xlim)
for n,row in enumerate(axes):
    for m,col in enumerate(row):
        idx = ncols*n+m
#        if idx == 8: 
#            break
        col.errorbar(meann[:,idx], depth, xerr=std[:,idx], fmt='-o')    
        col.locator_params(nbins=4)
#        col.set_ylabel('depth(cm)')
#        col.set_xlabel(r'$\Delta^{14}C$')
        col.set_title(vegtype[idx])
        col.set_xlim(xlim)
        col.set_ylim([-50, 150])
col.invert_yaxis()
plt.tight_layout()
#%% plot site lc vs. MODIS lc. profile map
whittaker_map = {1:'tundra',2:'woodland and grassland',3:'savana',4:'desert',
                 5:'boreal forest',6:'temperate forest',7:'tropical dry forest',
                 8:'temperate wet forest',9:'tropical wet forest'}
clim_code_map = {1:'cold dry',2:'cold mesic',3:'cold wet',
                 4:'temperate dry',5:'temperate mesic',6:'temperate wet',
                 7:'tropical dry',8:'tropical mesic',9:'tropical wet'}
lc_map = {1:'boreal forest',2:'temperate forest',3:'tropical forest',
          4:'grassland',5:'cropland',6:'shrublands',7:'peatland',
          8:'savannas',9:'tundra',10:'desert'}
filename = 'Non_peat_data_synthesis.csv'
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1]) 
pid = data.index.unique() # index of profile start
d14C = prep.getvarxls(data,'D14C_BulkLayer', pid, ':')
sitelon = prep.getvarxls(data,'Lon', pid, 0)
sitelat = prep.getvarxls(data,'Lat', pid, 0)
sitelc = prep.getvarxls(data,'VegTypeCode_Local', pid, 0)
siteinfo = zip(sitelon, sitelat, sitelc)
verify_lc_df = verify_site_lc(siteinfo)
verify_lc_df['Site'] = prep.getvarxls(data, 'Site', pid, 0)
verify_lc_df['Country'] = prep.getvarxls(data, 'Country', pid, 0)
verify_lc_df['MAT'] = prep.getvarxls(data, 'MAT', pid, 0)
verify_lc_df['MAP'] = prep.getvarxls(data, 'MAP', pid, 0)
verify_lc_df['Elevation'] = prep.getvarxls(data, 'Elevation', pid, 0)
verify_lc_df['SoilOrder'] = prep.getvarxls(data, 'SoilOrder_LEN_USDA', pid, 0)
verify_lc_df['VegLocal'] = prep.getvarxls(data, 'VegLocal', pid, 0)
verify_lc_df['VegType_Species'] = prep.getvarxls(data, 'VegType_Species', pid, 0)
verify_lc_df['VegTypeCodeStr_Local'] = prep.getvarxls(data, 'VegTypeCodeStr_Local', pid, 0)
verify_lc_df['VegTypeCode_MODIS'] = verify_lc_df.MODISlc.apply(lambda x: lc_map[int(x)] if ~np.isnan(x) else x)
verify_lc_df['clim_code_whittaker'] = map(lambda x: whittaker_map[x], 
                                          prep.getvarxls(data, 'clim_code_whittaker', pid, 0))
verify_lc_df['clim_code'] = map(lambda x: clim_code_map[x], 
                                prep.getvarxls(data, 'clim_code', pid, 0))

verify_lc_df.to_csv('verify_site_modisls2.csv', encoding='iso-8859-1')     
plot_lc_and_sitelc()
#%% plot individual biome site lc vs. MODIS lc. profile map
vegtype = ['boreal forest','temperate forest','tropical forest','grassland',
           'cropland','shrublands','peatland','savannas','tundra','desert']
mapp = 'veg'
for i in range(1,len(vegtype)+1):
    plot_lc_and_sitelc(mapp=mapp, plt_veg=i)
    plt.savefig('..\\figures\\global_extrapolation\\withGCBdata\\site_locations\\'+ \
                mapp+'_'+vegtype[i-1]+'.png')
    plt.close()
#%% plot pairwise obs veg D14C with shade
nrows, ncols = 7,7
fig, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(10,15),sharey=True)
plot_obs_biomepf_pairwise(axes)
plt.tight_layout()
plt.savefig('..\\figures\\biome_profiles\\withGCBdata\\biome_averaged_profiles_pairwise_modifybiome.pdf',
            dpi=300)

#%% pctC, BD, global extrapolation and global total

scalarfromobs = 1
kwargs = {'pred_veg': 1, 'pred_soil': 1}
# use only observations to contruct scaler 
depth = range(0, 110, 10)
X, y = prep_obs(useorilyr=1,y='pctC',**kwargs)
Xextra, notNaNs, lc_1d = prep_extra_data(**kwargs)
scaler, X_scaled = get_Xscaled(X)

svr = d14C_svr(X_scaled, y, **svr_pctC_params)
ypred = d14C_pred(svr, depth, Xextra[notNaNs,:], scaler=scaler)

#gbrt = d14C_gbrt(X, y, **gbrt_pctC_params)
#ypred = d14C_pred(gbrt, depth, Xextra[notNaNs,:])

#rf = d14C_gbrt(X, y, **rf_pctC_params)
#ypred = d14C_pred(rf, depth, Xextra[notNaNs,:])

yhat = np.empty((Xextra.shape[0])); yhat[:] = np.NAN
yhat[notNaNs] = ypred
yhat = np.reshape(yhat,(360,720))
  
if scalarfromobs == 0:
    # need to use the whole dataset to construct scaler.
    depth = 100.
    X, y = prep_obs(useorilyr=1)
    Xextra, notNaNs, lc_1d = prep_extra_data()
    Xextra_wdepth = np.c_[np.tile(depth,(Xextra.shape[0],1)), Xextra]
    scaler, X_scaled = get_Xscaled(X, Xextra_wdepth[~np.any(np.isnan(Xextra_wdepth),1),:])
    svr = d14C_svr(X_scaled, y)
    ypred = d14C_pred(svr, scaler, depth, Xextra[notNaNs,:])
    yhat = np.empty((Xextra.shape[0])); yhat[:] = np.NAN
    yhat[notNaNs] = ypred
    yhat = np.reshape(yhat,(360,720))


# plot
plt.figure()
lons,lats = construct_lonlat()
im = myplt.geoshow(yhat,lons,np.flipud(lats),extend='neither',clim=[300, -400])
plt.title('depth ' +  str(depth) + 'cm')

# global total
area = mysm.cal_earthgridarea(0.5)
globC = np.nansum(area*yhat)/1e6
