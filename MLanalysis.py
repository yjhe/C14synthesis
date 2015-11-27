# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:24:34 2015

@author: Yujie
"""
#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import D14Cpreprocess as prep
import sklearn as sk
from sklearn import svm
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from mpl_toolkits.mplot3d import Axes3D
import mystats as mysm
import myplot
import C14tools
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# prepare data
def prep_data(useorilyr=1, cutdep=None, soil='shallow', ddelta=0, vartau=0, 
              clayon=0, nppon=0, pred_veg=1, pred_soil=1, pred_climzone=0, y='D14C'):
    '''
    params:
        y:  'D14C', 'ddelta', 'tau', 'pctC', 'BD', 'O_thickness'. 
            If 'O_thickness', then X is one row per profile
    '''
    filename = 'Non_peat_data_synthesis.csv'
    #Cave14C = prep.getCweightedD14C2(filename)
    data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1])  
    if y != 'O_thickness':
        if useorilyr == 1:
            profid = data.index.unique() # index of profile start
            d14C = prep.getvarxls(data,'D14C_BulkLayer', profid, ':')
            pctC = prep.getvarxls(data,'pct_C', profid, ':')
            bd = prep.getvarxls(data, 'BulkDensity', profid, ':')
            sampleyr = prep.getvarxls(data, 'SampleYear', profid, ':')
            dd14C = prep.getDDelta14C(sampleyr, d14C)
            #tau, cost = C14tools.cal_tau(d14C, sampleyr)
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
            
            if clayon == 1:
                sclayfn = '..\\AncillaryData\\HWSD\\Regridded_ORNLDAAC\\S_CLAY.nc4'
                tclayfn = '..\\AncillaryData\\HWSD\\Regridded_ORNLDAAC\\T_CLAY.nc4'
                scecclayfn= '..\\AncillaryData\\HWSD\\Regridded_ORNLDAAC\\S_CEC_CLAY.nc4'
                tcecclayfn= '..\\AncillaryData\\HWSD\\Regridded_ORNLDAAC\\T_CEC_CLAY.nc4'
                tbulkdenfn= '..\\AncillaryData\\HWSD\\Regridded_ORNLDAAC\\T_BULK_DEN.nc4'
                sbulkdenfn= '..\\AncillaryData\\HWSD\\Regridded_ORNLDAAC\\S_BULK_DEN.nc4'
                sclay = prep.getHWSD(sclayfn, lon, lat) # % weight
                tclay = prep.getHWSD(tclayfn, lon, lat)  # % weight
                scecclay = prep.getHWSD(scecclayfn, lon, lat)  # cmol/kg clay
                tcecclay = prep.getHWSD(tcecclayfn, lon, lat)  # cmol/kg clay
                sbulkden = prep.getHWSD(sbulkdenfn, lon, lat)  # g/cm3
                tbulkden = prep.getHWSD(tbulkdenfn, lon, lat)  # g/cm3
                clay = tbulkden*tclay*0.3 + sbulkden*sclay*0.7
                cecclay =  tbulkden*tcecclay*0.3 + sbulkden*scecclay*0.7
            if nppon == 1:
                nppfn = '..\\AncillaryData\\NPP\\2000_2012meannpp_gCm2yr.nc'
                npp = prep.getnpp(nppfn, lon, lat) # gC/m2/yr
            
            # construct X and y
            dummyveg = (vegid[:, None] == vegiduniq).astype(float)  # dummy[:,1:]
            dummysord = (soilorder[:, None] == soilorderuniq).astype(float)
            dummyclim = (climid[:, None] == climuniq).astype(float)
            if clayon == 1:
                x = np.c_[layerbot, mat, mapp, cecclay]
            else:
                x = np.c_[layerbot, mat, mapp]            
            if y == 'D14C':
                y = d14C
            elif y == 'ddelta':
                y = dd14C
            elif y == 'tau':
                y = tau[:,0]
            elif y == 'pctC':
                y = pctC
            elif y == 'BD':
                y = bd
            
        else: # use incremented data
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
            if y == 'D14C':
                y = incre_data['D14C_BulkLayer'].values.astype(float)
            elif y == 'ddelta':
                y = dd14C
            elif y == 'tau':
                y = tau[:,0]
            elif y == 'pctC':
                y = pctC
            elif y == 'BD':
                y = bd
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

    # form final X
    #normalizer = sk.preprocessing.Normalizer().fit(X) # very bad, because this is sample normalization
    #X_scaled = normalizer.transform(X)
    scalar = sk.preprocessing.StandardScaler().fit(X)
    X_scaled = scalar.transform(X)
    return X, X_scaled, y

def prep_Othickness_df():
    filename = 'Non_peat_data_synthesis.csv'
    #Cave14C = prep.getCweightedD14C2(filename)
    data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1])  

    profid = data.index.unique()
    mat = prep.getvarxls(data, 'MAT', profid, 0)
    mapp = prep.getvarxls(data, 'MAP', profid, 0)
    vegidstr = prep.getvarxls(data, 'VegTypeCodeStr_Local', profid, 0)
    soilorder = prep.getvarxls(data, 'SoilOrder_LEN_USDA', profid, 0)
    soilorder = np.array([str(i) for i in soilorder])
    climid = prep.getvarxls(data, 'clim_code', profid, 0)
    # construct X and y
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
    x = np.c_[profid, y, mat, mapp, climid]
    df = pd.DataFrame(data=x[:,1:],columns=['O_thickness','MAT','MAP','clim_code'],
                      index=x[:,0])
    df['SoilOrder_LEN_USDA'] = soilorder
    df['VegTypeCodeStr_Local'] = vegidstr
    return df 
                      
                      
def prep_gridsearchplot(j_train, j_v, x, y, xlab, ylab):   
    '''
    the outmost loop is y. i.e., j_train[:,0] or j_v[:,0]. row number
    the inner loop is x. i.e., j_train[:,1] or j_v[:,1]. col number
    '''
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,8))
    train_rmse = j_train[:,2].reshape((y.shape[0],x.shape[0]))
    train_r2 = j_train[:,3].reshape((y.shape[0],x.shape[0]))
    val_rmse = j_v[:,2].reshape((y.shape[0],x.shape[0]))
    val_r2 = j_v[:,3].reshape((y.shape[0],x.shape[0]))
    rmse_ylim = [np.min(np.c_[j_train[:,2],j_v[:,2]]), 
                 np.max(np.c_[j_train[:,2],j_v[:,2]])]
    r2_ylim = [0.6, 1.0]
    
    curax = ax[0][0]
    cax = curax.pcolor(train_rmse, cmap=plt.cm.jet)
    curax.set_xticks(np.arange(train_rmse.shape[1])+0.5, minor=False)
    curax.set_yticks(np.arange(train_rmse.shape[0])+0.5, minor=False)
    curax.set_ylim((0, y.shape[0]))
    curax.set_xticklabels(x)
    curax.set_yticklabels(y)
    curax.set_xlabel(xlab)
    curax.set_ylabel(ylab)
    cbar = fig.colorbar(cax,ax=curax)
    cbar.ax.set_xlabel('RMSE')
    curax.set_title('Train')
    cax.set_clim(rmse_ylim)
    
    curax = ax[0][1]
    cax = curax.pcolor(val_rmse, cmap=plt.cm.jet)
    curax.set_xticks(np.arange(val_rmse.shape[1])+0.5, minor=False)
    curax.set_yticks(np.arange(val_rmse.shape[0])+0.5, minor=False)
    curax.set_ylim((0, y.shape[0]))
    curax.set_xticklabels(x)
    curax.set_yticklabels(y)
    curax.set_xlabel(xlab)
    curax.set_ylabel(ylab)
    cbar = fig.colorbar(cax,ax=curax)
    cbar.ax.set_xlabel('RMSE')
    curax.set_title('Validation')
    cax.set_clim(rmse_ylim)
    
    curax = ax[1][0]
    cax = curax.pcolor(train_r2, cmap=plt.cm.coolwarm)
    curax.set_xticks(np.arange(train_r2.shape[1])+0.5, minor=False)
    curax.set_yticks(np.arange(train_r2.shape[0])+0.5, minor=False)
    curax.set_ylim((0, y.shape[0]))
    curax.set_xticklabels(x)
    curax.set_yticklabels(y)
    curax.set_xlabel(xlab)
    curax.set_ylabel(ylab)
    cbar = fig.colorbar(cax,ax=curax)
    cbar.ax.set_xlabel(r'$R^{2}$')
    cax.set_clim(r2_ylim)
    
    curax = ax[1][1]
    cax = curax.pcolor(val_r2, cmap=plt.cm.coolwarm)
    curax.set_xticks(np.arange(val_r2.shape[1])+0.5, minor=False)
    curax.set_yticks(np.arange(val_r2.shape[0])+0.5, minor=False)
    curax.set_ylim((0, y.shape[0]))
    curax.set_xticklabels(x)
    curax.set_yticklabels(y)
    curax.set_xlabel(xlab)
    curax.set_ylabel(ylab)
    cbar = fig.colorbar(cax,ax=curax)
    cbar.ax.set_xlabel(r'$R^{2}$')
    cax.set_clim(r2_ylim)
    
def plottraindev_and_meanfeatimt(mdl, params, X_test, y_test, X_names, cluster_cat=None):
    '''
    plot feature importance and standard deviation
    params:
        cluster_cat  : list of tuple which contains the column index of features that belong
                       to one categorical variable. e.g. [(1,2,3),(6,7,8,9)] 
    '''
    # plot train deviance vs test deviance
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
    for i, y_pred_test in enumerate(mdl.staged_decision_function(X_test)):
        test_score[i] = mdl.loss_(y_test, y_pred_test)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    ax1 = fig.axes[0]
    ax1.set_title('Deviance')
    ax1.plot(np.arange(params['n_estimators']) + 1, mdl.train_score_, 'b-',
             label='Training Set Deviance')
    ax1.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
             label='Test Set Deviance')
    ax1.legend(loc='upper right')
    ax1.set_xlabel('Boosting Iterations')
    ax1.set_ylabel('Deviance')
    
    # plot feature importance
    if cluster_cat is None: # DO NOT sum up importance of a categorical feature        
        featimt = mdl.feature_importances_
        # calculate relative importance to the max feature
        #featimt = 100. * (featimt / featimt.max())
        sorted_idx = np.argsort(featimt)
        std = np.std([tree.feature_importances_ for tree in np.squeeze(mdl.estimators_)], axis=0)
        pos = np.arange(sorted_idx.shape[0]) + .5
        ax2 = fig.axes[1]
        ax2.barh(pos, featimt[sorted_idx], xerr=std[sorted_idx], align='center')
        ax2.set_yticks(pos)
        ax2.set_yticklabels(X_names[sorted_idx])
        ax2.set_xlabel('Relative Importance')
        ax2.set_title('Variable Importance')
    else: # sum-up categorical features
        featimt = mdl.feature_importances_
        cluster_featimt = featimt[:cluster_cat[0][0]]
        for n in cluster_cat:    
            cluster_featimt = np.r_[cluster_featimt, np.sum(featimt[n])]
        sorted_idx = np.argsort(cluster_featimt)
        std_tot = np.array([tree.feature_importances_ for tree in np.squeeze(mdl.estimators_)])
        tmp = std_tot[:,:cluster_cat[0][0]]        
        for n in cluster_cat:    
            tmp = np.c_[tmp, np.sum(std_tot[:,n],axis=1)]
        std = np.std(tmp, axis=0)
        pos = np.arange(sorted_idx.shape[0]) + .5
        ax2 = fig.axes[1]
        ax2.barh(pos, cluster_featimt[sorted_idx], xerr=std[sorted_idx], align='center')
        ax2.set_yticks(pos)
        ax2.set_yticklabels(X_names[sorted_idx])
        ax2.set_xlabel('Relative Importance')
        ax2.set_title('Variable Importance')
        

def plottraindev_and_featimt(mdl, params, X_test, y_test, X_names):
    # plot train deviance vs test deviance
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
    for i, y_pred_test in enumerate(mdl.staged_decision_function(X_test)):
        test_score[i] = mdl.loss_(y_test, y_pred_test)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    ax1 = fig.axes[0]
    ax1.set_title('Deviance')
    ax1.plot(np.arange(params['n_estimators']) + 1, mdl.train_score_, 'b-',
             label='Training Set Deviance')
    ax1.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
             label='Test Set Deviance')
    ax1.legend(loc='upper right')
    ax1.set_xlabel('Boosting Iterations')
    ax1.set_ylabel('Deviance')

    # plot feature importance
    featimt = mdl.feature_importances_
    # calculate relative importance to the max feature
    featimt = 100. * (featimt / featimt.max())
    sorted_idx = np.argsort(featimt)
    pos = np.arange(sorted_idx.shape[0]) + .5
    ax2 = fig.axes[1]
    ax2.barh(pos, featimt[sorted_idx], align='center')
    ax2.set_yticks(pos)
    ax2.set_yticklabels(X_names[sorted_idx])
    ax2.set_xlabel('Relative Importance')
    ax2.set_title('Variable Importance')

def plot_featimt(mdl, params, X_names):
    # plot feature importance
    featimt = mdl.feature_importances_
    # calculate relative importance to the max feature
    featimt = 100. * (featimt / featimt.max())
    sorted_idx = np.argsort(featimt)
    pos = np.arange(sorted_idx.shape[0]) + .5
    fig, ax = plt.subplots()
    ax.barh(pos, featimt[sorted_idx], align='center')
    ax.set_yticks(pos)
    ax.set_yticklabels(X_names[sorted_idx])
    ax.set_xlabel('Relative Importance')
    ax.set_title('Variable Importance')
    
def plot_meanfeatimt(mdl, params, X_names, cluster_cat=None):
    '''
    plot feature importance and standard deviation
    params:
        cluster_cat  : list of tuple which contains the column index of features that belong
                       to one categorical variable. e.g. [(1,2,3),(6,7,8,9)] 
    '''
    fig, ax2 = plt.subplots(figsize=(12,6))
    # plot feature importance
    if cluster_cat is None: # DO NOT sum up importance of a categorical feature        
        featimt = mdl.feature_importances_
        # calculate relative importance to the max feature
        #featimt = 100. * (featimt / featimt.max())
        sorted_idx = np.argsort(featimt)
        std = np.std([tree.feature_importances_ for tree in np.squeeze(mdl.estimators_)], axis=0)
        pos = np.arange(sorted_idx.shape[0]) + .5
        ax2.barh(pos, featimt[sorted_idx], xerr=std[sorted_idx], align='center')
        ax2.set_yticks(pos)
        ax2.set_yticklabels(X_names[sorted_idx])
        ax2.set_xlabel('Relative Importance')
        ax2.set_title('Variable Importance')
    else: # sum-up categorical features
        featimt = mdl.feature_importances_
        cluster_featimt = featimt[:cluster_cat[0][0]]
        for n in cluster_cat:    
            cluster_featimt = np.r_[cluster_featimt, np.sum(featimt[n])]
        sorted_idx = np.argsort(cluster_featimt)
        std_tot = np.array([tree.feature_importances_ for tree in np.squeeze(mdl.estimators_)])
        tmp = std_tot[:,:cluster_cat[0][0]]        
        for n in cluster_cat:    
            tmp = np.c_[tmp, np.sum(std_tot[:,n],axis=1)]
        std = np.std(tmp, axis=0)
        pos = np.arange(sorted_idx.shape[0]) + .5
        ax2.barh(pos, cluster_featimt[sorted_idx], xerr=std[sorted_idx], align='center')
        ax2.set_yticks(pos)
        ax2.set_yticklabels(X_names[sorted_idx])
        ax2.set_xlabel('Relative Importance')
        ax2.set_title('Variable Importance')
#%% --- SVM regression
X, X_scaled, y = prep_data(useorilyr=1, cutdep=None, soil='shallow', y='D14C',pred_climzone=0)
plott = 0
plot_y_yhat = 1

# train SVM and predict
#svr = sk.svm.SVR(C=50, kernel='poly', degree=3, coef0=2)
C = 1e4;
svr = sk.svm.SVR(C=C, kernel='rbf', gamma=0.15)
svr.fit(X_scaled, y)
yhat = svr.fit(X_scaled, y).predict(X_scaled)
print "R2 is: %.3f, R2adj is: %.3f" %(mysm.cal_R2(y,yhat),0)
print "RMSE is: %.3f" %(mysm.cal_RMSE(y,yhat))
if plott == 1:
    fig, axes = plt.subplots(nrows=1,ncols=1)
    #plt.hist(y,30,alpha=0.3, normed=True)
    plt.scatter(X_scaled[:,1], y, label='data', c='k')
    plt.plot(X_scaled[:,1], yhat, label='poly', c='g')
if plot_y_yhat:
    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
    ax.scatter(y[X[:,0]<=100], yhat[X[:,0]<=100],c='g',label='<100cm')
    ax.scatter(y[X[:,0]>100], yhat[X[:,0]>100],c='k',label='>100cm')
    ax.legend(loc=4,scatterpoints=1)
    ax.set_ylabel('yhat')
    ax.set_xlabel('y')
    # add support vectors
    #supvec = np.logical_and(svr.dual_coef_ != C, svr.dual_coef_ != -1.*C)
    #ax.scatter(y[:-1][np.ravel(supvec)], yhat[:-1][np.ravel(supvec)], c='r')
    myplot.refline()

#%% SVM learned parameters
# for pctC: gamma = 10, C = 100
# for BD: gamma = 1.8, C = 10
#%% SVM cross validation, using polynomial kernel
n_folds = 5
n_degrees = 10
j_train = []
j_v = []
kf = sk.cross_validation.KFold(y.shape[0], n_folds=n_folds, shuffle=True)
for degree in range(1,n_degrees):
    print("..... degree is %d ....."%degree)
    dumj_train = []
    dumj_v = []
    for n, [train, test] in enumerate(kf):
        print("round %d ..."%n)
        # print("%s %s" % (train, test))
        svr = sk.svm.SVR(C=1e3, kernel='poly', degree=degree)
        svr.fit(X_scaled[train], y[train])
        yhat = svr.predict(X_scaled[train])
        dumj_train.append(mysm.cal_RMSE(y[train], yhat))
        dumj_v.append(mysm.cal_RMSE(y[test], svr.predict(X_scaled[test])))
    j_train.append(np.nanmean(dumj_train))
    j_v.append(np.nanmean(dumj_v))
        
# plot cross validation curve
fig = plt.figure()
ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
ax.plot(range(1,n_degrees), j_train, label='train')
ax.plot(range(1,n_degrees), j_v, label='validation')
plt.legend()
ax.set_ylabel('RMSE')
ax.set_xlabel('degrees')
ax.set_ylim([0, 500])

#%% SVM cross validation, using rbf kernel, vary gamma and C, grid search
n_folds = 5
n_gamma_range = np.array([0.01, 0.1, 1, 10, 100])
n_C_range = np.array([0.1, 1, 10, 100, 1000])
j_train = []
j_v = []
kf = sk.cross_validation.KFold(y.shape[0], n_folds=n_folds, shuffle=True)

for gamma in n_gamma_range:
    print("..... gamma is %d ....."%gamma)
    for C in n_C_range:
        print("..... C is %d ....."%C)
        dumj_train = []
        dumj_v = []
        dumj_train_r2 = []
        dumj_v_r2 = []
        for n, [train, test] in enumerate(kf):
            print("round %d ..."%n)
            # print("%s %s" % (train, test))
            svr = sk.svm.SVR(C=C, kernel='rbf', gamma=gamma)
            svr.fit(X_scaled[train], y[train])
            yhat = svr.predict(X_scaled[train])
            dumj_train.append(mysm.cal_RMSE(y[train], yhat))
            dumj_v.append(mysm.cal_RMSE(y[test], svr.predict(X_scaled[test])))
            dumj_train_r2.append(mysm.cal_R2(y[train], yhat))
            dumj_v_r2.append(mysm.cal_R2(y[test], svr.predict(X_scaled[test])))
            if n == 2:
                ploty = y[test]
                plotyhat = svr.predict(X_scaled[test])
        j_train.append([gamma, C, np.nanmean(dumj_train), np.nanmean(dumj_train_r2)])
        j_v.append([gamma, C, np.nanmean(dumj_v), np.nanmean(dumj_v_r2)])

j_train = np.asarray(j_train)
j_v = np.asarray(j_v)

# plot grid search plot  
prep_gridsearchplot(j_train, j_v, n_C_range, n_gamma_range, 'C', 'n_gamma')


#%% SVM cross validation, using rbf kernel, vary gamma
n_folds = 5
n_gamma = 5
x = [1.5, 1.8, 2.1, 2.4]
#x = np.arange(.001, .1, .02)
#x = [10, 20, 40, 60, 80]
j_train = []
j_v = []
kf = sk.cross_validation.KFold(y.shape[0], n_folds=n_folds, shuffle=True)
for gamma in x:
    print("..... gamma is %.3f ....."%gamma)
    dumj_train = []
    dumj_v = []
    dumj_train_r2 = []
    dumj_v_r2 = []
    for n, [train, test] in enumerate(kf):
        print("round %d ..."%n)
        # print("%s %s" % (train, test))
        svr = sk.svm.SVR(C=10, kernel='rbf', gamma=gamma)
        svr.fit(X_scaled[train], y[train])
        yhat = svr.predict(X_scaled[train])
        dumj_train.append(mysm.cal_RMSE(y[train], yhat))
        dumj_v.append(mysm.cal_RMSE(y[test], svr.predict(X_scaled[test])))
        dumj_train_r2.append(mysm.cal_R2(y[train], yhat))
        dumj_v_r2.append(mysm.cal_R2(y[test], svr.predict(X_scaled[test])))
    j_train.append([np.nanmean(dumj_train),np.nanstd(dumj_train),
                    np.nanmean(dumj_train_r2),np.nanstd(dumj_train_r2)])
    j_v.append([np.nanmean(dumj_v), np.nanstd(dumj_v),
                np.nanmean(dumj_v_r2), np.nanstd(dumj_v_r2)])
j_train = np.asarray(j_train)
j_v = np.asarray(j_v)

# plot cross validation curve
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
ax[0].errorbar(x, j_train[:,0], yerr=j_train[:,1], label='train')
ax[0].errorbar(x, j_v[:,0], yerr=j_v[:,1]/n_gamma, label='validation')
ax[0].set_ylabel('RMSE')
ax[0].set_xlabel('gamma')   
#ax.set_ylim([0, 500])

ax[1].errorbar(x, j_train[:,2], yerr=j_train[:,3], label='train')
ax[1].errorbar(x, j_v[:,2], yerr=j_v[:,3], label='validation')
plt.legend(loc=3)
ax[1].set_ylabel('R2')
ax[1].set_xlabel('gamma')
ax[1].set_ylim([0, 1])
plt.tight_layout()
#%%  SVM, using BEST rbf kernel, vary train/test size
trainsize = np.arange(0.1, 1, 0.1)
j_train = []
j_v = []
for curtrainsize in trainsize:
    print "..... curtrainsize is %.2f ....."%curtrainsize
    dumj_train = []
    dumj_v = []
    dumj_train_r2 = []
    dumj_v_r2 = []
    for rs in range(10):
        print 'rs is ',rs
        X_train, X_test, y_train, y_test = sk.cross_validation.train_test_split(
                                X_scaled, y, test_size=1-curtrainsize, random_state=rs)   
        svr = sk.svm.SVR(C=1e4, kernel='rbf', gamma=0.02)
        svr.fit(X_train, y_train)
        yhat = svr.predict(X_train)
        dumj_train.append(mysm.cal_RMSE(y_train, yhat))
        dumj_v.append(mysm.cal_RMSE(y_test, svr.predict(X_test)))
        dumj_train_r2.append(mysm.cal_R2(y_train, yhat))
        dumj_v_r2.append(mysm.cal_R2(y_test, svr.predict(X_test)))
    j_train.append([np.nanmean(dumj_train), np.nanmean(dumj_train_r2)])
    j_v.append([np.nanmean(dumj_v), np.nanmean(dumj_v_r2)])
j_train = np.asarray(j_train)
j_v = np.asarray(j_v)

# plot cross validation curve
x = trainsize
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].plot(x, j_train[:,0], label='train')
ax[0].plot(x, j_v[:,0], label='validation')
plt.legend()
ax[0].set_ylabel('RMSE')
ax[0].set_xlabel('Fraction of data used for training')
#ax.set_xlim([0, 200])

ax[1].plot(x, j_train[:,1], label='train')
ax[1].plot(x, j_v[:,1], label='validation')
plt.legend()
ax[1].set_ylabel('R2')
ax[1].set_xlabel('Fraction of data used for training')
ax[1].set_ylim([0, 1])
plt.tight_layout()
#%% --- Gradient Tree Boosting
plott = 0; plot_y_yhat = 1
X, X_scaled, y = prep_data(useorilyr=0,cutdep=None, soil='shallow', 
                           y='D14C',pred_soil=1,pred_veg=1)
#X_names = np.array(['layerbot', 'MAT', 'MAP', 'BorealFor', 'TempFor', 'TropFor', \
#           'Grassland', 'Cropland', 'Shrubland', 'Peatland', 'Savannas','Tundra','Desert'])
X_names = np.array(['layer depth', 'MAT', 'MAP', 'BorFor','TempFor', 'TropFor', \
           'Grassland', 'Cropland', 'Shrubland', 'Peatland', 'Savannas','Tundra','Desert', \
           'Alf','And','Ard','Ent','Ept','Ert','Gel','His','Oll','Ox','Spo','Ult'])
X_names_cluster = np.array(['layerbot','MAT','MAP','Veg','SoilOrder'])
#cluster_cat = [range(3,12+1),range(13,24+1),range(25,30)]  # with clim_code
cluster_cat = [range(3,15),range(15,20)]
# train GBRT and predict. X_scaled
params = {'n_estimators': 1200, 'max_depth': 4,
          'learning_rate': 0.01, 'loss': 'ls'}
          # 300, 3, for tau. 200, 10 for d14C
est = GradientBoostingRegressor(**params).fit(X_scaled, y)
yhat = est.predict(X_scaled)
#plottraindev_and_featimt(est, params, X_scaled, y, X_names)
#plottraindev_and_meanfeatimt(est, params, X_scaled, y, X_names_cluster, cluster_cat)
print "R2 is: %.3f, R2adj is: %.3f" %(mysm.cal_R2(y,yhat),0)
print "mean squared error is : %.3f"%sk.metrics.mean_absolute_error(y, yhat)

# train GBRT, original X
params = {'n_estimators': 1200, 'max_depth': 4,
          'learning_rate': 0.01, 'loss': 'ls'}
          # 300, 3, for tau. 200, 10 for d14C
est = GradientBoostingRegressor(**params).fit(X, y)
yhat = est.predict(X)
#plottraindev_and_featimt(est, params, X_scaled, y, X_names)
#plottraindev_and_meanfeatimt(est, params, X, y, X_names_cluster, cluster_cat)
print "R2 is: %.3f, R2adj is: %.3f" %(mysm.cal_R2(y,yhat),0)
print "mean squared error is : %.3f"%sk.metrics.mean_absolute_error(y, yhat)

# plot partial dependence plots
features = [0,1,2,(0,1),(0,2)]
fig, ax = plot_partial_dependence(est, X, features, feature_names=X_names, n_cols=2,
                                  figsize=(10,8))
plt.tight_layout()

# plot 3-D partial dependence plot
fig = plt.figure()
target_feature = (0,1)
pdp, (x_axis, y_axis) = partial_dependence(est, target_feature,
                                           X=X_scaled, grid_resolution=50)
XX, YY = np.meshgrid(x_axis, y_axis)
Z = pdp.T.reshape(XX.shape).T
ax = Axes3D(fig)
surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.BuPu)
ax.set_xlabel(X_names[target_feature[0]])
ax.set_ylabel(X_names[target_feature[1]])
ax.set_zlabel('Partial dependence')
#  pretty init view
ax.view_init(elev=22, azim=122)
cax = fig.add_axes([0.9,0.1,0.05,0.8])
cbar = fig.colorbar(surf,cax=cax)
plt.subplots_adjust(top=0.9)

# plot y~yhat
if plott == 1:
    fig, axes = plt.subplots(nrows=1,ncols=1)
    #plt.hist(y,30,alpha=0.3, normed=True)
    plt.scatter(X_scaled[:,1], y, label='data', c='k')
    plt.scatter(X_scaled[:,1], yhat, label='poly', c='g')
if plot_y_yhat:
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([0.12, 0.1, 0.85, 0.85])
    ax.scatter(y, yhat)
    ax.set_ylabel('yhat')
    ax.set_xlabel('y') 
    myplot.refline()
#%% GBRT for O_thickness
plott = 0; plot_y_yhat = 1
X, X_scaled, y = prep_data(y='O_thickness',pred_soil=1,pred_veg=1)
#X_names = np.array(['layerbot', 'MAT', 'MAP', 'BorealFor', 'TempFor', 'TropFor', \
#           'Grassland', 'Cropland', 'Shrubland', 'Peatland', 'Savannas','Tundra','Desert'])
X_names = np.array(['MAT', 'MAP', 'BorFor','TempFor', 'TropFor', \
           'Grassland', 'Cropland', 'Shrubland', 'Peatland', 'Savannas','Tundra','Desert', \
           'Alf','And','Ard','Ent','Ept','Ert','Gel','His','Oll','Ox','Spo','Ult'])
X_names_cluster = np.array(['MAT','MAP','Veg','SoilOrder'])
#cluster_cat = [range(3,12+1),range(13,24+1),range(25,30)]  # with clim_code
cluster_cat = [range(2,11),range(11,23)]

# train GBRT, original X
params = {'n_estimators': 1200, 'max_depth': 4,
          'learning_rate': 0.01, 'loss': 'ls'}
          # 300, 3, for tau. 200, 10 for d14C
est = GradientBoostingRegressor(**params).fit(X, y)
yhat = est.predict(X)
#plottraindev_and_featimt(est, params, X_scaled, y, X_names)
#plottraindev_and_meanfeatimt(est, params, X, y, X_names_cluster, cluster_cat)
print "R2 is: %.3f, R2adj is: %.3f" %(mysm.cal_R2(y,yhat),0)
print "mean squared error is : %.3f"%sk.metrics.mean_absolute_error(y, yhat)

# plot y~yhat
if plott == 1:
    fig, axes = plt.subplots(nrows=1,ncols=1)
    #plt.hist(y,30,alpha=0.3, normed=True)
    plt.scatter(X_scaled[:,1], y, label='data', c='k')
    plt.scatter(X_scaled[:,1], yhat, label='poly', c='g')
if plot_y_yhat:
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([0.12, 0.1, 0.85, 0.85])
    ax.scatter(y, yhat)
    ax.set_ylabel('yhat')
    ax.set_xlabel('y') 
    myplot.refline()
#%% GBRT learned parameters from gridsearch for different training dataset
# D14C
# cutdep = 50, soil = 'deep'
params = {'n_estimators':1300, 'max_depth':5}
# cutdep = 50, soil = 'shallow'
params = {'n_estimators':1200, 'max_depth':4}
# cutdep = None. Whole profile
params = {'n_estimators':1500, 'max_depth':5}

# pctC
# cutdep = 50, soil = 'deep'
params = {'n_estimators':1200, 'max_depth':3}
# cutdep = 50, soil = 'shallow'
params = {'n_estimators':1200, 'max_depth':4}
# cutdep = None. Whole profile
params = {'n_estimators':1200, 'max_depth':4}

#%% GBRT cross validation, vary n_estimator and max_depth, grid search
n_folds = 5
n_estimator_range = np.array([10, 50, 100, 500, 1000])
n_depth_range = np.array([3, 7, 12])
j_train = []
j_v = []
kf = sk.cross_validation.KFold(y.shape[0], n_folds=n_folds, shuffle=True)

for maxdep in n_depth_range:
    print("..... max depth is %d ....."%maxdep)
    for n_estimators in n_estimator_range:
        print("..... num estimator is %d ....."%n_estimators)
        dumj_train = []
        dumj_v = []
        dumj_train_r2 = []
        dumj_v_r2 = []
        for n, [train, test] in enumerate(kf):
            print("round %d ..."%n)
            # print("%s %s" % (train, test))
            params = {'n_estimators': n_estimators, 'max_depth': maxdep, 'min_samples_split': 2,
                      'learning_rate': 0.01, 'loss': 'ls'}
            est = GradientBoostingRegressor(**params).fit(X_scaled[train], y[train])
            yhat = est.predict(X_scaled[train])
            dumj_train.append(mysm.cal_RMSE(y[train], yhat))
            dumj_v.append(mysm.cal_RMSE(y[test], est.predict(X_scaled[test])))
            dumj_train_r2.append(mysm.cal_R2(y[train], yhat))
            dumj_v_r2.append(mysm.cal_R2(y[test], est.predict(X_scaled[test])))
            if n == 2:
                ploty = y[test]
                plotyhat = est.predict(X_scaled[test])
        j_train.append([maxdep, n_estimators, np.nanmean(dumj_train), np.nanmean(dumj_train_r2)])
        j_v.append([maxdep, n_estimators, np.nanmean(dumj_v), np.nanmean(dumj_v_r2)])
        #plottraindev_and_featimt(est, params, X_scaled[test], y[test], X_names)
        # raw_input('pause, press any key to continue')
        #plt.close()
j_train = np.asarray(j_train)
j_v = np.asarray(j_v)

# plot grid search plot  
prep_gridsearchplot(j_train, j_v, n_estimator_range, n_depth_range, 'n_estimator','max depth')

if plot_y_yhat:
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.scatter(ploty, plotyhat)
    ax.set_ylabel('yhat')
    ax.set_xlabel('y')
    myplot.refline()
    
#%% GBRT cross validation, vary n_estimator 
n_folds = 5
#n_estimator_range = np.arange(900, 3500, 600)
n_estimator_range = np.array([500, 700, 900, 1100, 1500, 2000])
j_train = []
j_v = []
kf = sk.cross_validation.KFold(y.shape[0], n_folds=n_folds, shuffle=True)

for n_estimators in n_estimator_range:
    print("..... num estimator is %d ....."%n_estimators)
    dumj_train = []
    dumj_v = []
    dumj_train_r2 = []
    dumj_v_r2 = []
    for n, [train, test] in enumerate(kf):
        print("round %d ..."%n)
        # print("%s %s" % (train, test))
        params = {'n_estimators': n_estimators, 'max_depth': 7,
                  'learning_rate': 0.01, 'loss': 'ls'}
        est = GradientBoostingRegressor(**params).fit(X_scaled[train], y[train])
        yhat = est.predict(X_scaled[train])
        dumj_train.append(mysm.cal_RMSE(y[train], yhat))
        dumj_v.append(mysm.cal_RMSE(y[test], est.predict(X_scaled[test])))
        dumj_train_r2.append(mysm.cal_R2(y[train], yhat))
        dumj_v_r2.append(mysm.cal_R2(y[test], est.predict(X_scaled[test])))
        if n == 2:
            ploty = y[test]
            plotyhat = est.predict(X_scaled[test])
    j_train.append([np.nanmean(dumj_train),np.nanstd(dumj_train),
                    np.nanmean(dumj_train_r2),np.nanstd(dumj_train_r2)])
    j_v.append([np.nanmean(dumj_v), np.nanstd(dumj_v),
                np.nanmean(dumj_v_r2), np.nanstd(dumj_v_r2)])
    #plottraindev_and_featimt(est, params, X_scaled[test], y[test], X_names)
    # raw_input('pause, press any key to continue')
    #plt.close()
j_train = np.asarray(j_train)
j_v = np.asarray(j_v)

# plot cross validation curve
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
ax[0].errorbar(n_estimator_range, j_train[:,0], yerr=j_train[:,1], label='train')
ax[0].errorbar(n_estimator_range, j_v[:,0], yerr=j_v[:,1], label='validation')
ax[0].set_ylabel('RMSE')
ax[0].set_xlabel('n_estimator')   
#ax.set_ylim([0, 500])

ax[1].errorbar(n_estimator_range, j_train[:,2], yerr=j_train[:,3], label='train')
ax[1].errorbar(n_estimator_range, j_v[:,2], yerr=j_v[:,3], label='validation')
plt.legend(loc=3)
ax[1].set_ylabel('R2')
ax[1].set_xlabel('n_estimator')
ax[1].set_ylim([0, 1])

if plot_y_yhat:
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.scatter(ploty, plotyhat)
    ax.set_ylabel('yhat')
    ax.set_xlabel('y')
    myplot.refline()
#%% GBRT cross validation, vary depth
n_folds = 5
n_depth = np.arange(2, 7, 1)
j_train = []
j_v = []
kf = sk.cross_validation.KFold(y.shape[0], n_folds=n_folds, shuffle=True)
for max_depth in n_depth:
    print("..... n_depth is %d ....."%max_depth)
    dumj_train = []
    dumj_v = []
    dumj_train_r2 = []
    dumj_v_r2 = []
    for n, [train, test] in enumerate(kf):
        print("round %d ..."%n)
        # print("%s %s" % (train, test))
        params = {'n_estimators': 1200, 'max_depth': max_depth, 'min_samples_split': 2,
                  'learning_rate': 0.01, 'loss': 'ls'}
        est = GradientBoostingRegressor(**params).fit(X_scaled[train], y[train])
        yhat = est.predict(X_scaled[train])
        dumj_train.append(mysm.cal_RMSE(y[train], yhat))
        dumj_v.append(mysm.cal_RMSE(y[test], est.predict(X_scaled[test])))
        dumj_train_r2.append(mysm.cal_R2(y[train], yhat))
        dumj_v_r2.append(mysm.cal_R2(y[test], est.predict(X_scaled[test])))
    j_train.append([np.nanmean(dumj_train), np.nanmean(dumj_train_r2)])
    j_v.append([np.nanmean(dumj_v), np.nanmean(dumj_v_r2)])
    plottraindev_and_meanfeatimt(est, params, X_scaled[test], y[test], X_names_cluster, 
                                 cluster_cat=cluster_cat)
    #raw_input('pause, press any key to continue')
    #plt.close()
j_train = np.asarray(j_train)
j_v = np.asarray(j_v)

# plot cross validation curve
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
ax[0].plot(n_depth, j_train[:,0], label='train')
ax[0].plot(n_depth, j_v[:,0], label='validation')
plt.legend()
ax[0].set_ylabel('RMSE')
ax[0].set_xlabel('n_depth')   
#ax.set_ylim([0, 500])

ax[1].plot(n_depth, j_train[:,1], label='train')
ax[1].plot(n_depth, j_v[:,1], label='validation')
plt.legend()
ax[1].set_ylabel('R2')
ax[1].set_xlabel('n_depth')
ax[1].set_ylim([0, 1])

#%% GBRT, using BEST tree, vary train/test size
trainsize = np.arange(0.1, 1, 0.1)
j_train = []
j_v = []
for curtrainsize in trainsize:
    print("..... curtrainsize is %.1f ....."%curtrainsize)
    dumj_train = []
    dumj_v = []
    dumj_train_r2 = []
    dumj_v_r2 = []
    for rs in range(20):
        print "rs is ",rs
        X_train, X_test, y_train, y_test = sk.cross_validation.train_test_split(
                        X_scaled, y, test_size=1-curtrainsize, random_state=rs)  
        params = {'n_estimators': 300, 'max_depth': 4, 'min_samples_split': 1,
                  'learning_rate': 0.01, 'loss': 'ls'}
        est = GradientBoostingRegressor(**params).fit(X_train, y_train)
        yhat = est.predict(X_train)
        dumj_train.append(mysm.cal_RMSE(y_train, yhat))
        dumj_v.append(mysm.cal_RMSE(y_test, est.predict(X_test)))
        dumj_train_r2.append(mysm.cal_R2(y_train, yhat))
        dumj_v_r2.append(mysm.cal_R2(y_test, est.predict(X_test)))
    j_train.append([np.nanmean(dumj_train), np.nanmean(dumj_train_r2)])
    j_v.append([np.nanmean(dumj_v), np.nanmean(dumj_v_r2)])
    #plottraindev_and_featimt(est, params, X_scaled[test], y[test], X_names)
j_train = np.asarray(j_train)
j_v = np.asarray(j_v)

# plot cross validation curve
x = trainsize
fig = plt.figure()
ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
ax.plot(x, j_train[:,0], label='train')
ax.plot(x, j_v[:,0], label='validation')
plt.legend()
ax.set_ylabel('RMSE')
ax.set_xlabel('Fraction of data used for training')
#ax.set_ylim([0, 500])

fig = plt.figure()
ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
ax.plot(x, j_train[:,1], label='train')
ax.plot(x, j_v[:,1], label='validation')
plt.legend()
ax.set_ylabel('R2')
ax.set_xlabel('Fraction of data used for training')
ax.set_ylim([0, 1])

#%% --- RF, Random Forest
plott = 0; plot_y_yhat = 1
X, X_scaled, y = prep_data(useorilyr=1,cutdep=None, soil='deep',y='BD')
X_names = np.array(['layer depth', 'MAT', 'MAP', 'BorFor','TempFor', 'TropFor', \
           'Grassland', 'Cropland', 'Shrubland', 'Peatland', 'Savannas','Tundra','Desert', \
           'Alf','And','Ard','Ent','Ept','Ert','Gel','His','Oll','Ox','Spo','Ult'])
X_names_cluster = np.array(['layerbot','MAT','MAP','Veg','SoilOrder'])
cluster_cat = [range(3,12+1),range(13,24+1)] 

# train and predict
params = {'n_estimators': 100, 'max_depth': 14}
rft = RandomForestRegressor(**params)
rft.fit(X, y)
yhat = rft.fit(X, y).predict(X)
print "R2 is: %.3f, R2adj is: %.3f" %(mysm.cal_R2(y,yhat),0)
print "RMSE is: %.3f" %(mysm.cal_RMSE(y,yhat))

# plot feature importance
#plot_featimt(rft, params, X_names)
plot_meanfeatimt(rft, params, X_names_cluster, cluster_cat)

if plott == 1:
    fig, axes = plt.subplots(nrows=1,ncols=1)
    #plt.hist(y,30,alpha=0.3, normed=True)
    plt.scatter(X_scaled[:,1], y, label='data', c='k')
    plt.plot(X_scaled[:,1], yhat, label='poly', c='g')
if plot_y_yhat:
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.scatter(y[X[:,0]<50], yhat[X[:,0]<50],c='g')
    ax.scatter(y[X[:,0]>50], yhat[X[:,0]>50],c='k')
    
    ax.set_ylabel('yhat')
    ax.set_xlabel('y')
    myplot.refline()    
#%% RF for O_thickness
plott = 0; plot_y_yhat = 1
X, X_scaled, y = prep_data(y='O_thickness',pred_soil=1,pred_veg=1)
#X_names = np.array(['layerbot', 'MAT', 'MAP', 'BorealFor', 'TempFor', 'TropFor', \
#           'Grassland', 'Cropland', 'Shrubland', 'Peatland', 'Savannas','Tundra','Desert'])
X_names = np.array(['MAT', 'MAP', 'BorFor','TempFor', 'TropFor', \
           'Grassland', 'Cropland', 'Shrubland', 'Peatland', 'Savannas','Tundra','Desert', \
           'Alf','And','Ard','Ent','Ept','Ert','Gel','His','Oll','Ox','Spo','Ult'])
X_names_cluster = np.array(['MAT','MAP','Veg','SoilOrder'])
#cluster_cat = [range(3,12+1),range(13,24+1),range(25,30)]  # with clim_code
cluster_cat = [range(2,11),range(11,23)]

# train RF, original X
params = {'n_estimators': 100, 'max_depth': 14}
rft = RandomForestRegressor(**params)
rft.fit(X, y)
yhat = rft.fit(X, y).predict(X)
print "R2 is: %.3f, R2adj is: %.3f" %(mysm.cal_R2(y,yhat),0)
print "RMSE is: %.3f" %(mysm.cal_RMSE(y,yhat))
plot_meanfeatimt(rft, params, X_names_cluster, cluster_cat)

# plot y~yhat
if plott == 1:
    fig, axes = plt.subplots(nrows=1,ncols=1)
    #plt.hist(y,30,alpha=0.3, normed=True)
    plt.scatter(X_scaled[:,1], y, label='data', c='k')
    plt.scatter(X_scaled[:,1], yhat, label='poly', c='g')
if plot_y_yhat:
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([0.12, 0.1, 0.85, 0.85])
    ax.scatter(y, yhat)
    ax.set_ylabel('yhat')
    ax.set_xlabel('y') 
    myplot.refline()

#%% RF, Random Forest cross validation, grid search for n_est and max_depth
n_folds = 5
n_estimator_range = np.arange(60, 200, 10)
n_depth_range = np.arange(3, 80, 10)

j_train = []
j_v = []
kf = sk.cross_validation.KFold(y.shape[0], n_folds=n_folds, shuffle=True)
for maxdep in n_depth_range:
    print("..... max depth is %d ....."%maxdep)    
    for n_estimators in n_estimator_range:
        print("..... num estimator is %d ....."%n_estimators)
        dumj_train = []
        dumj_v = []
        dumj_train_r2 = []
        dumj_v_r2 = []
        for n, [train, test] in enumerate(kf):
            print("round %d ..."%n)
            # print("%s %s" % (train, test))
            params = {'n_estimators': n_estimators, 'max_depth': maxdep}
            est = RandomForestRegressor(**params).fit(X[train], y[train])
            yhat = est.predict(X[train])
            dumj_train.append(mysm.cal_RMSE(y[train], yhat))
            dumj_v.append(mysm.cal_RMSE(y[test], est.predict(X[test])))
            dumj_train_r2.append(mysm.cal_R2(y[train], yhat))
            dumj_v_r2.append(mysm.cal_R2(y[test], est.predict(X[test])))
            if n == 2:
                ploty = y[test]
                plotyhat = est.predict(X[test])
        j_train.append([maxdep, n_estimators, np.nanmean(dumj_train), np.nanmean(dumj_train_r2)])
        j_v.append([maxdep, n_estimators, np.nanmean(dumj_v), np.nanmean(dumj_v_r2)])
        #plottraindev_and_featimt(est, params, X_scaled[test], y[test], X_names)
        # raw_input('pause, press any key to continue')
        #plt.close()
j_train = np.asarray(j_train)
j_v = np.asarray(j_v)

# plot grid search plot  
prep_gridsearchplot(j_train, j_v, n_estimator_range, n_depth_range, 'n_trees','max depth')

#%% RF, Random Forest cross validation, vary n_estimator
# n_estimator does not have much influence on result. see gridsearch plot
n_folds = 10
n_estimator_range = np.arange(50, 200, 20)
j_train = []
j_v = []
kf = sk.cross_validation.KFold(y.shape[0], n_folds=n_folds, shuffle=True)
for n_estimators in n_estimator_range:
    print("..... num estimator is %d ....."%n_estimators)
    dumj_train = []
    dumj_v = []
    dumj_train_r2 = []
    dumj_v_r2 = []
    for n, [train, test] in enumerate(kf):
        print("round %d ..."%n)
        # print("%s %s" % (train, test))
        params = {'n_estimators': n_estimators, 'max_depth': 10}
        est = RandomForestRegressor(**params).fit(X[train], y[train])
        yhat = est.predict(X[train])
        dumj_train.append(mysm.cal_RMSE(y[train], yhat))
        dumj_v.append(mysm.cal_RMSE(y[test], est.predict(X[test])))
        dumj_train_r2.append(mysm.cal_R2(y[train], yhat))
        dumj_v_r2.append(mysm.cal_R2(y[test], est.predict(X[test])))
        if n == 2:
            ploty = y[test]
            plotyhat = est.predict(X[test])
    j_train.append([np.nanmean(dumj_train), np.nanmean(dumj_train_r2)])
    j_v.append([np.nanmean(dumj_v), np.nanmean(dumj_v_r2)])
    #plottraindev_and_featimt(est, params, X_scaled[test], y[test], X_names)
    # raw_input('pause, press any key to continue')
    #plt.close()
j_train = np.asarray(j_train)
j_v = np.asarray(j_v)

# plot cross validation curve 
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
ax[0].plot(n_estimator_range, j_train[:,0], label='train')
ax[0].plot(n_estimator_range, j_v[:,0], label='validation')
plt.legend()
ax[0].set_ylabel('RMSE')
ax[0].set_xlabel('n_estimator')
#ax.set_ylim([0, 500])

ax[1].plot(n_estimator_range, j_train[:,1], label='train')
ax[1].plot(n_estimator_range, j_v[:,1], label='validation')
plt.legend()
ax[1].set_ylabel('R2')
ax[1].set_xlabel('n_estimator')
ax[1].set_ylim([0, 1])

if plot_y_yhat:
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.scatter(ploty, plotyhat)
    ax.set_ylabel('yhat')
    ax.set_xlabel('y')
    myplot.refline()
#%% RF, Random forest cross validation, vary depth
n_folds = 5
n_depth = np.arange(10, 50, 10)
j_train = []
j_v = []
kf = sk.cross_validation.KFold(y.shape[0], n_folds=n_folds, shuffle=True)
for max_depth in n_depth:
    print("..... n_depth is %d ....."%max_depth)
    dumj_train = []
    dumj_v = []
    dumj_train_r2 = []
    dumj_v_r2 = []
    for n, [train, test] in enumerate(kf):
        print("round %d ..."%n)
        # print("%s %s" % (train, test))
        params = {'n_estimators': 200, 'max_depth': max_depth}
        est = RandomForestRegressor(**params).fit(X[train], y[train])
        yhat = est.predict(X[train])
        dumj_train.append(mysm.cal_RMSE(y[train], yhat))
        dumj_v.append(mysm.cal_RMSE(y[test], est.predict(X[test])))
        dumj_train_r2.append(mysm.cal_R2(y[train], yhat))
        dumj_v_r2.append(mysm.cal_R2(y[test], est.predict(X[test])))
    j_train.append([np.nanmean(dumj_train),np.nanstd(dumj_train),
                    np.nanmean(dumj_train_r2),np.nanstd(dumj_train_r2)])
    j_v.append([np.nanmean(dumj_v), np.nanstd(dumj_v),
                np.nanmean(dumj_v_r2), np.nanstd(dumj_v_r2)])
    #plottraindev_and_featimt(est, params, X_scaled[test], y[test], X_names)
    #raw_input('pause, press any key to continue')
    #plt.close()
j_train = np.asarray(j_train)
j_v = np.asarray(j_v)

# plot cross validation curve
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
ax[0].errorbar(n_depth, j_train[:,0], yerr=j_train[:,1], label='train')
ax[0].errorbar(n_depth, j_v[:,0], yerr=j_v[:,1], label='validation')
ax[0].set_ylabel('RMSE')
ax[0].set_xlabel('n_depth')   
#ax.set_ylim([0, 500])

ax[1].errorbar(n_depth, j_train[:,2], yerr=j_train[:,3], label='train')
ax[1].errorbar(n_depth, j_v[:,2], yerr=j_v[:,3], label='validation')
plt.legend(loc=3)
ax[1].set_ylabel('R2')
ax[1].set_xlabel('n_depth')
ax[1].set_ylim([0, 1])
#%% --- AdaBoost 
plott = 0; plot_y_yhat = 1
X, X_scaled, y = prep_data(cutdep=None, soil='shallow', y='pctC')
#X_names = np.array(['layerbot', 'MAT', 'MAP', 'BorealFor', 'TempFor', 'TropFor', \
#           'Grassland', 'Cropland', 'Shrubland', 'Peatland', 'Savannas','Tundra','Desert'])
X_names = np.array(['layer depth', 'MAT', 'MAP', 'BorFor','TempFor', 'TropFor', \
           'Grassland', 'Cropland', 'Shrubland', 'Peatland', 'Savannas','Tundra','Desert', \
           'Alf','And','Ard','Ent','Ept','Ert','Gel','His','Oll','Ox','Spo','Ult'])
X_names_cluster = np.array(['layerbot','MAT','MAP','Veg','SoilOrder'])
cluster_cat = [range(3,12+1),range(13,24+1)] 

# train GBRT and predict. original X
params = {'n_estimators': 2000, 'loss':'linear'}
base_est = DecisionTreeRegressor(max_depth=6)
est = AdaBoostRegressor(base_estimator=base_est,**params).fit(X, y)
yhat = est.predict(X)
#plottraindev_and_featimt(est, params, X_scaled, y, X_names)
#plottraindev_and_meanfeatimt(est, params, X, y, X_names_cluster, cluster_cat)
print "R2 is: %.3f, R2adj is: %.3f" %(mysm.cal_R2(y,yhat),0)
print "mean squared error is : %.3f"%sk.metrics.mean_absolute_error(y, yhat)

#%% AdaBoost cross validation, vary n_estimator and max_depth, grid search
n_folds = 5
n_estimator_range = np.array([100, 500, 1000, 3000])
n_depth_range = np.array([3, 7, 12, 17, 25, 35, 50])
j_train = []
j_v = []
kf = sk.cross_validation.KFold(y.shape[0], n_folds=n_folds, shuffle=True)

for maxdep in n_depth_range:
    print("..... max depth is %d ....."%maxdep)
    for n_estimators in n_estimator_range:
        print("..... num estimator is %d ....."%n_estimators)
        dumj_train = []
        dumj_v = []
        dumj_train_r2 = []
        dumj_v_r2 = []
        for n, [train, test] in enumerate(kf):
            print("round %d ..."%n)
            params = {'n_estimators': n_estimators, 'loss':'linear'}
            base_est = DecisionTreeRegressor(max_depth=maxdep)
            est = AdaBoostRegressor(base_estimator=base_est, **params).fit(X[train], y[train])
            yhat = est.predict(X[train])
            dumj_train.append(mysm.cal_RMSE(y[train], yhat))
            dumj_v.append(mysm.cal_RMSE(y[test], est.predict(X_scaled[test])))
            dumj_train_r2.append(mysm.cal_R2(y[train], yhat))
            dumj_v_r2.append(mysm.cal_R2(y[test], est.predict(X_scaled[test])))
            if n == 2:
                ploty = y[test]
                plotyhat = est.predict(X_scaled[test])
        j_train.append([maxdep, n_estimators, np.nanmean(dumj_train), np.nanmean(dumj_train_r2)])
        j_v.append([maxdep, n_estimators, np.nanmean(dumj_v), np.nanmean(dumj_v_r2)])
        #plottraindev_and_featimt(est, params, X_scaled[test], y[test], X_names)
        # raw_input('pause, press any key to continue')
        #plt.close()
j_train = np.asarray(j_train)
j_v = np.asarray(j_v)

# plot grid search plot  
prep_gridsearchplot(j_train, j_v, n_estimator_range, n_depth_range, 'n_estimator','max depth')

if plot_y_yhat:
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.scatter(ploty, plotyhat)
    ax.set_ylabel('yhat')
    ax.set_xlabel('y')
    myplot.refline()
    
#%% --- Extra-trees
X, X_scaled, y = prep_data(useorilyr=0)
plott = 0
plot_y_yhat = 1

# train and predict
C = 1e4;
svr = sk.svm.SVR(C=C, kernel='rbf', gamma=0.4)
svr.fit(X_scaled, y)
yhat = svr.fit(X_scaled, y).predict(X_scaled)
print "R2 is: %.3f, R2adj is: %.3f" %(mysm.cal_R2(y,yhat),0)
print "RMSE is: %.3f" %(mysm.cal_RMSE(y,yhat))
if plott == 1:
    fig, axes = plt.subplots(nrows=1,ncols=1)
    #plt.hist(y,30,alpha=0.3, normed=True)
    plt.scatter(X_scaled[:,1], y, label='data', c='k')
    plt.plot(X_scaled[:,1], yhat, label='poly', c='g')
if plot_y_yhat:
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.scatter(y[X[:,0]<50], yhat[X[:,0]<50],c='g')
    ax.scatter(y[X[:,0]>50], yhat[X[:,0]>50],c='k')
    
    ax.set_ylabel('yhat')
    ax.set_xlabel('y')
    # add support vectors
    supvec = np.logical_and(svr.dual_coef_ != C, svr.dual_coef_ != -1.*C)
    ax.scatter(y[:-1][np.ravel(supvec)], yhat[:-1][np.ravel(supvec)], c='r')
    myplot.refline()
#%% Extra-trees cross validation, vary n_estimator
n_folds = 5
n_gamma = 5
x = [20, 40, 60, 80]
j_train = []
j_v = []
kf = sk.cross_validation.KFold(y.shape[0], n_folds=n_folds, shuffle=True)
for gamma in x:
    print("..... gamma is %.2f ....."%gamma)
    dumj_train = []
    dumj_v = []
    dumj_train_r2 = []
    dumj_v_r2 = []
    for n, [train, test] in enumerate(kf):
        print("round %d ..."%n)
        # print("%s %s" % (train, test))
        svr = sk.svm.SVR(C=1e4, kernel='rbf', gamma=gamma)
        svr.fit(X_scaled[train], y[train])
        yhat = svr.predict(X_scaled[train])
        dumj_train.append(mysm.cal_RMSE(y[train], yhat))
        dumj_v.append(mysm.cal_RMSE(y[test], svr.predict(X_scaled[test])))
        dumj_train_r2.append(mysm.cal_R2(y[train], yhat))
        dumj_v_r2.append(mysm.cal_R2(y[test], svr.predict(X_scaled[test])))
    j_train.append([np.nanmean(dumj_train),np.nanmean(dumj_train_r2)])
    j_v.append([np.nanmean(dumj_v),np.nanmean(dumj_v_r2)])
j_train = np.asarray(j_train)
j_v = np.asarray(j_v)

# plot cross validation curve
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
ax[0].plot(x, j_train[:,0], label='train')
ax[0].plot(x, j_v[:,0], label='validation')
plt.legend()
ax[0].set_ylabel('RMSE')
ax[0].set_xlabel('gamma')
#ax.set_ylim([0, 200])

ax[1].plot(x, j_train[:,1], label='train')
ax[1].plot(x, j_v[:,1], label='validation')
plt.legend()
ax[1].set_ylabel('R2')
ax[1].set_xlabel('gamma')
ax[1].set_ylim([0, 1])
plt.tight_layout()
#%% Extra-trees cross validation, vary depth
n_folds = 5
n_depth = np.arange(4, 10, 1)
j_train = []
j_v = []
kf = sk.cross_validation.KFold(y.shape[0], n_folds=n_folds, shuffle=True)
for max_depth in n_depth:
    print("..... n_depth is %d ....."%max_depth)
    dumj_train = []
    dumj_v = []
    dumj_train_r2 = []
    dumj_v_r2 = []
    for n, [train, test] in enumerate(kf):
        print("round %d ..."%n)
        # print("%s %s" % (train, test))
        params = {'n_estimators': 500, 'max_depth': max_depth, 'min_samples_split': 1,
                  'learning_rate': 0.01, 'loss': 'ls'}
        est = GradientBoostingRegressor(**params).fit(X_scaled[train], y[train])
        yhat = est.predict(X_scaled[train])
        dumj_train.append(mysm.cal_RMSE(y[train], yhat))
        dumj_v.append(mysm.cal_RMSE(y[test], est.predict(X_scaled[test])))
        dumj_train_r2.append(mysm.cal_R2(y[train], yhat))
        dumj_v_r2.append(mysm.cal_R2(y[test], est.predict(X_scaled[test])))
    j_train.append([np.nanmean(dumj_train), np.nanmean(dumj_train_r2)])
    j_v.append([np.nanmean(dumj_v), np.nanmean(dumj_v_r2)])
    plottraindev_and_featimt(est, params, X_scaled[test], y[test], X_names)
    #raw_input('pause, press any key to continue')
    #plt.close()
j_train = np.asarray(j_train)
j_v = np.asarray(j_v)

# plot cross validation curve
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
ax[0].plot(n_depth, j_train[:,0], label='train')
ax[0].plot(n_depth, j_v[:,0], label='validation')
plt.legend()
ax[0].set_ylabel('RMSE')
ax[0].set_xlabel('n_depth')   
#ax.set_ylim([0, 500])

ax[1].plot(n_depth, j_train[:,1], label='train')
ax[1].plot(n_depth, j_v[:,1], label='validation')
plt.legend()
ax[1].set_ylabel('R2')
ax[1].set_xlabel('n_depth')
ax[1].set_ylim([0, 1])

#%% --- OLS
X, X_scaled, y = prep_data(useorilyr=1)
plott = 0
plot_y_yhat = 1
dum = X_scaled

# train OLS and predict
ols = sk.linear_model.LinearRegression()
ols.fit(dum, y)
yhat = ols.fit(dum, y).predict(dum)
print "R2 is: %.3f, R2adj is: %.3f" %(mysm.cal_R2(y,yhat),0)
print "RMSE is: %.3f" %(mysm.cal_RMSE(y,yhat))
if plott == 1:
    fig = plt.figure()
    ax = fig.add_axes([0.05,0.05,0.9,0.9])
    #plt.hist(y,30,alpha=0.3, normed=True)
    plt.scatter(dum[:,1], y, label='data', c='k')
    plt.plot(dum[:,1], yhat, label='poly', c='g')
if plot_y_yhat:
    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
    ax.scatter(y[X[:,0]<50], yhat[X[:,0]<50],c='b')
    ax.scatter(y[X[:,0]>50], yhat[X[:,0]>50],c='k') 
    ax.set_ylabel('yhat')
    ax.set_xlabel('y')
ols.coef_
ols.intercept_

#%% OLS cross validation
n_folds = 10
j_train = []
j_v = []
kf = sk.cross_validation.KFold(y.shape[0], n_folds=n_folds, shuffle=True)

print("..... gamma is %.2f ....."%gamma)
dumj_train = []
dumj_v = []
dumj_train_r2 = []
dumj_v_r2 = []
for n, [train, test] in enumerate(kf):
    print("round %d ..."%n)
    # print("%s %s" % (train, test))
    ols = sk.linear_model.LinearRegression()
    ols.fit(X_scaled[train], y[train])
    yhat = ols.predict(X_scaled[train])
    dumj_train.append(mysm.cal_RMSE(y[train], yhat))
    dumj_v.append(mysm.cal_RMSE(y[test], ols.predict(X_scaled[test])))
    dumj_train_r2.append(mysm.cal_R2(y[train], yhat))
    dumj_v_r2.append(mysm.cal_R2(y[test], ols.predict(X_scaled[test])))
j_train.append([np.nanmean(dumj_train),np.nanmean(dumj_train_r2)])
j_v.append([np.nanmean(dumj_v),np.nanmean(dumj_v_r2)])
print 'mean training RMSE & R2 is ',j_train
print 'mean validation RMSE & R2 is ',j_v
print 'mean std of training RMSE is ',np.std(dumj_train)
print 'mean std of validation RMSE is ',np.std(dumj_v)

# plot cross validation curve
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
ax[0].plot(np.arange(1,n_folds+1), dumj_train, label='train')
ax[0].plot(np.arange(1,n_folds+1), dumj_v, label='validation')
plt.legend()
ax[0].set_ylabel('RMSE')
ax[0].set_xlabel('gamma')
#ax.set_ylim([0, 200])

ax[1].plot(np.arange(1,n_folds+1), dumj_train_r2, label='train')
ax[1].plot(np.arange(1,n_folds+1), dumj_v_r2, label='validation')
plt.legend()
ax[1].set_ylabel('R2')
ax[1].set_xlabel('gamma')
ax[1].set_ylim([0, 1])
plt.tight_layout()

#%% --- Ridge
X, X_scaled, y = prep_data(useorilyr=1)
plott = 0
plot_y_yhat = 1
dum = X_scaled

# train Ridge and predict
rdg = sk.linear_model.Ridge(alpha=5, copy_X=True, fit_intercept=True)
rdg.fit(dum, y)
yhat = rdg.fit(dum, y).predict(dum)
print "R2 is: %.3f, R2adj is: %.3f" %(mysm.cal_R2(y,yhat),0)
print "RMSE is: %.3f" %(mysm.cal_RMSE(y,yhat))
if plott == 1:
    fig = plt.figure()
    ax = fig.add_axes([0.05,0.05,0.9,0.9])
    #plt.hist(y,30,alpha=0.3, normed=True)
    plt.scatter(dum[:,1], y, label='data', c='k')
    plt.plot(dum[:,1], yhat, label='poly', c='g')
if plot_y_yhat:
    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
    ax.scatter(y[X[:,0]<50], yhat[X[:,0]<50],c='b')
    ax.scatter(y[X[:,0]>50], yhat[X[:,0]>50],c='k') 
    ax.set_ylabel('yhat')
    ax.set_xlabel('y')
rdg.coef_
rdg.intercept_
#%% Ridge cross validation, vary alpha
n_folds = 10
n_alpha = 10
alphaa = [0, 0.1, 1, 5, 10, 15, 20]
j_train = []
j_v = []
kf = sk.cross_validation.KFold(y.shape[0], n_folds=n_folds, shuffle=True)
for a in alphaa:
    print("..... alpha is %.2f ....."%a)
    dumj_train = []
    dumj_v = []
    dumj_train_r2 = []
    dumj_v_r2 = []
    for n, [train, test] in enumerate(kf):
        print("round %d ..."%n)
        # print("%s %s" % (train, test))
        mdl = sk.linear_model.Ridge(alpha=a, copy_X=True, fit_intercept=True)
        mdl.fit(X_scaled[train], y[train])
        yhat = mdl.predict(X_scaled[train])
        dumj_train.append(mysm.cal_RMSE(y[train], yhat))
        dumj_v.append(mysm.cal_RMSE(y[test], mdl.predict(X_scaled[test])))
        dumj_train_r2.append(mysm.cal_R2(y[train], yhat))
        dumj_v_r2.append(mysm.cal_R2(y[test], mdl.predict(X_scaled[test])))
    j_train.append([np.nanmean(dumj_train),np.nanmean(dumj_train_r2)])
    j_v.append([np.nanmean(dumj_v),np.nanmean(dumj_v_r2)])
j_train = np.asarray(j_train)
j_v = np.asarray(j_v)

# plot cross validation curve
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
ax[0].plot(alphaa, j_train[:,0], label='train')
ax[0].plot(alphaa, j_v[:,0], label='validation')
plt.legend()
ax[0].set_ylabel('RMSE')
ax[0].set_xlabel('alpha')
#ax.set_ylim([0, 200])

ax[1].plot(alphaa, j_train[:,1], label='train')
ax[1].plot(alphaa, j_v[:,1], label='validation')
plt.legend()
ax[1].set_ylabel('R2')
ax[1].set_xlabel('alpha')
ax[1].set_ylim([0, 1])
plt.tight_layout()

#%% --- Lasso
X, X_scaled, y = prep_data(useorilyr=1)
plott = 0
plot_y_yhat = 1
dum = X_scaled

# train Ridge and predict
lso = sk.linear_model.Lasso(alpha=0.01, copy_X=True, fit_intercept=True,positive=False,
                            selection='cyclic')
lso.fit(dum, y)
yhat = lso.fit(dum, y).predict(dum)
print "R2 is: %.3f, R2adj is: %.3f" %(mysm.cal_R2(y,yhat),0)
print "RMSE is: %.3f" %(mysm.cal_RMSE(y,yhat))
if plott == 1:
    fig = plt.figure()
    ax = fig.add_axes([0.05,0.05,0.9,0.9])
    #plt.hist(y,30,alpha=0.3, normed=True)
    plt.scatter(dum[:,1], y, label='data', c='k')
    plt.plot(dum[:,1], yhat, label='poly', c='g')
if plot_y_yhat:
    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
    ax.scatter(y[X[:,0]<50], yhat[X[:,0]<50],c='b')
    ax.scatter(y[X[:,0]>50], yhat[X[:,0]>50],c='k') 
    ax.set_ylabel('yhat')
    ax.set_xlabel('y')
lso.coef_
lso.intercept_
#%% Lasso cross validation, vary alpha
n_folds = 10
n_alpha = 10
alphaa = [0.0001, 0.001, 0.005, 0.01,0.05, 0.1,0.5,1]
j_train = []
j_v = []
kf = sk.cross_validation.KFold(y.shape[0], n_folds=n_folds, shuffle=True)
for a in alphaa:
    print("..... alpha is %.2f ....."%a)
    dumj_train = []
    dumj_v = []
    dumj_train_r2 = []
    dumj_v_r2 = []
    for n, [train, test] in enumerate(kf):
        print("round %d ..."%n)
        # print("%s %s" % (train, test))
        mdl = sk.linear_model.Lasso(alpha=a, copy_X=True, fit_intercept=True,positive=False,
                                    selection='cyclic',max_iter=5000)
        mdl.fit(X_scaled[train], y[train])
        yhat = mdl.predict(X_scaled[train])
        dumj_train.append(mysm.cal_RMSE(y[train], yhat))
        dumj_v.append(mysm.cal_RMSE(y[test], mdl.predict(X_scaled[test])))
        dumj_train_r2.append(mysm.cal_R2(y[train], yhat))
        dumj_v_r2.append(mysm.cal_R2(y[test], mdl.predict(X_scaled[test])))
    j_train.append([np.nanmean(dumj_train),np.nanmean(dumj_train_r2)])
    j_v.append([np.nanmean(dumj_v),np.nanmean(dumj_v_r2)])
j_train = np.asarray(j_train)
j_v = np.asarray(j_v)

# plot cross validation curve
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
ax[0].plot(alphaa, j_train[:,0], label='train')
ax[0].plot(alphaa, j_v[:,0], label='validation')
plt.legend()
ax[0].set_ylabel('RMSE')
ax[0].set_xlabel('alpha')
#ax.set_ylim([0, 200])

ax[1].plot(alphaa, j_train[:,1], label='train')
ax[1].plot(alphaa, j_v[:,1], label='validation')
plt.legend()
ax[1].set_ylabel('R2')
ax[1].set_xlabel('alpha')
ax[1].set_ylim([0, 1])
plt.tight_layout()

#%% --- OLS statsmodels
import statsmodels.api as sm
import scipy 
import mystats as mysm
from pandas.tools.plotting import scatter_matrix

X, X_scaled, y = prep_data(useorilyr=1, cutdep=100., soil='shallow')
plott = 0
plot_y_yhat = 1
Xaddone = sm.add_constant(X) 

# train and predict
# for deep soil. delete all-zero cols
# X: const (if any), depth, MAT, MAP
# veg: 4:10 (discard last col 10); soil: 11:22 (discard last col 22)
Xaddone = np.delete(Xaddone,[9,11,12,20],axis=1)
Xaddone = sm.add_constant(Xaddone) 
model = sm.OLS(y, Xaddone[:,[0,3]]).fit() # for deep soil. col -5 is all zeros
model = sm.OLS(y, Xaddone[:,[0,1,2]+range(4,10)+range(11,22)]).fit()

# for shallow soil or whole profile, 
# X: const (if any), depth, MAT, MAP
# veg: 4:13 (discard last col 13); soil: 14:25 (discard last col 25)
model = sm.OLS(y, Xaddone[:,[0,1,2]]).fit()
model = sm.OLS(y, np.delete(Xaddone,[1,3,13,25],axis=1)).fit()
model = sm.OLS(y, Xaddone[:,[0,1,2]+range(4,25)]).fit()

print model.summary()
yhat = model.predict()

# person corr b/w const and MAP
scipy.stats.pearsonr(Xaddone[:,0],Xaddone[:,3])
np.corrcoef(Xaddone[:,0],Xaddone[:,3])

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

#%% --- exploratory plots    
filename = 'Non_peat_data_synthesis.csv'
#Cave14C = prep.getCweightedD14C2(filename)
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1]) 

scatter_matrix(data[['D14C_BulkLayer','MAT','MAP','Layer_top','Elevation','pct_C']])
scatter_matrix(data[['D14C_BulkLayer','Elevation','pct_C']])

# sweep error entries in Elevation
for n,i in enumerate(data.Elevation.values):
    try:
        a = float(i)
    except ValueError:
        print 'valueError: n', n
        print 'valueError: i', i

# plot D14C among different veg      
y = ['D14C_BulkLayer','Layer_bottom']
gp_byveg = data.groupby('VegTypeCodeStr_Local')
means = gp_byveg.mean()[y]
errors = gp_byveg.std()[y]
fig, ax = plt.subplots(figsize=(15,6))
means.plot(yerr=errors, ax=ax, kind='bar')   
plt.xticks(rotation=45)   
ax.grid(True, which='both')
plt.tight_layout()

y = ['D14C_BulkLayer','Layer_bottom','VegTypeCodeStr_Local']
newdata = prep.prep_increment(data, y)
gp_byveg = newdata.groupby('VegTypeCodeStr_Local')
y = ['D14C_BulkLayer','Layer_bottom']
means = gp_byveg.mean()[y]
errors = gp_byveg.std()[y]
fig, ax = plt.subplots(figsize=(15,6))
means.plot(yerr=errors, ax=ax, kind='bar')      
ax.grid(True, which='both')
plt.xticks(rotation=45) 
plt.tight_layout()
  
# plot D14C among different soils    
y = ['D14C_BulkLayer','pct_C','MAT','Layer_top']
y = ['pct_C','MAT','Layer_bottom']
gp_byveg = data.groupby('SoilOrder_LEN_USDA')
means = gp_byveg.mean()[y]
errors = gp_byveg.std()[y]
fig, ax = plt.subplots(figsize=(15,6))
means.plot(yerr=errors, ax=ax, kind='bar')        
plt.tight_layout()

y = ['D14C_BulkLayer','Layer_bottom']
gp_byveg = data.groupby('SoilOrder_LEN_USDA')
means = gp_byveg.mean()[y]
errors = gp_byveg.std()[y]
fig, ax = plt.subplots(figsize=(15,6))
means.plot(yerr=errors, ax=ax, kind='bar')      
ax.grid(True, which='both')
plt.tight_layout()

y = ['D14C_BulkLayer','Layer_bottom','SoilOrder_LEN_USDA','pct_C']
newdata = prep.prep_increment(data, y)
gp_byveg = newdata.groupby('SoilOrder_LEN_USDA')
y = ['D14C_BulkLayer','Layer_bottom']
y = ['pct_C','Layer_bottom']
means = gp_byveg.mean()[y]
errors = gp_byveg.std()[y]
fig, ax = plt.subplots(figsize=(15,6))
means.plot(yerr=errors, ax=ax, kind='bar')      
ax.grid(True, which='both')

# plot number of profiles in each veg/soil type
fig = plt.figure(figsize=(7,6))
site = data[['SoilOrder_LEN_USDA','Site']].reset_index()
siteuniq = site.drop_duplicates(subset=['ProfileID'])
gp_bysoil = siteuniq.groupby('SoilOrder_LEN_USDA')
gp_bysoil.count().Site.plot(kind='pie',autopct=(lambda x: '%d'%(x*6)),
                colormap=plt.get_cmap('Set1'),fontsize=10)
plt.tight_layout()

fig = plt.figure(figsize=(7,6))
site = data[['VegTypeCodeStr_Local','Site']].reset_index()
siteuniq = site.drop_duplicates(subset=['ProfileID'])
gp_bysoil = siteuniq.groupby('VegTypeCodeStr_Local')
Nsite= gp_bysoil.count().Site
gp_bysoil.count().Site.plot(kind='pie',autopct=(lambda x: '%d'%(x*6)),
                colormap=plt.get_cmap('Set1'),fontsize=10)
fig.tight_layout()

# check number of profiles has 14C and of certain depth. 
# examine correlation b/w MAT and D14C teasing out depth
filename = 'Non_peat_data_synthesis.csv'
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1]) 
Cave14C = prep.getCweightedD14C2(data,cutdep=20.)
Cave14C_top0 = Cave14C[Cave14C.Layer_top==0]
shallowdf = prep.cal_pfNumber(0, 20)
deepdf = prep.cal_pfNumber(80,100)
def extract_D14C(df, loc):
    outdf = pd.DataFrame()
    for prof in df.index.unique():
        outdf = pd.concat([outdf, df.loc[prof:prof].iloc[loc]],axis=1)
    return outdf.transpose()
shallowdf = extract_D14C(shallowdf,0)
deepdf = extract_D14C(deepdf,-1)
shallowdf = shallowdf[shallowdf.Layer_bottom<=30]
deepdf = deepdf[(deepdf.Layer_bottom<=110)&(deepdf.Layer_bottom>=70)]
Cave14C_top0 = Cave14C_top0[(Cave14C_top0.Layer_bottom<=110)&(Cave14C.Layer_bottom>=70)]

shallowdf.plot(kind='scatter',x='MAT',y='D14C_BulkLayer')
shallowdf.plot(kind='scatter',x='MAP',y='D14C_BulkLayer')
deepdf.plot(kind='scatter',x='MAT',y='D14C_BulkLayer')
deepdf.plot(kind='scatter',x='MAP',y='D14C_BulkLayer')
Cave14C_top0.plot(kind='scatter',x='MAT',y='SOCave_14C')
myplt.add_regress(Cave14C_top0.MAT.astype(float), Cave14C_top0.SOCave_14C.astype(float))
Cave14C_top0.plot(kind='scatter',x='totSOC',y='SOCave_14C')
Cave14C_top0[Cave14C_top0.SOCave_14C<-600]

#%% interpolate available D14C, pctC, BD sample size
filename = 'Non_peat_data_synthesis.csv'
#Cave14C = prep.getCweightedD14C2(filename)
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1]) 

# linear interpolate to increments
filename = 'Non_peat_data_synthesis.csv'
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1]) 
data_lin, statprop_lin = prep.prep_lnrinterp(data)
shallowdf = data_lin[data_lin.Layer_depth_incre==10]
deepdf = data_lin[data_lin.Layer_depth_incre==100]
shallowdf.plot(kind='scatter',x='MAT',y='D14C_BulkLayer')
shallowdf.plot(kind='scatter',x='MAP',y='D14C_BulkLayer')
deepdf.plot(kind='scatter',x='MAT',y='D14C_BulkLayer')
deepdf.plot(kind='scatter',x='MAP',y='D14C_BulkLayer')

# exp interpolate to increments
filename = 'Non_peat_data_synthesis.csv'
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1]) 
data_exp, statprop_exp = prep.prep_expinterp(data)
data_exp_goodfit = data_exp.loc[statprop_exp.D14C_R2>0.7]
shallowdf = data_exp_goodfit[data_exp_goodfit.Layer_depth_incre==10]
deepdf = data_exp_goodfit[data_exp_goodfit.Layer_depth_incre==100]
shallowdf.plot(kind='scatter',x='MAT',y='D14C_BulkLayer')
shallowdf.plot(kind='scatter',x='MAP',y='D14C_BulkLayer')
deepdf.plot(kind='scatter',x='MAT',y='D14C_BulkLayer')
deepdf.plot(kind='scatter',x='MAP',y='D14C_BulkLayer')

from numpy.linalg import lstsq
from scipy.optimize import curve_fit
from scipy import stats

def f(x, A, B): # this is your 'straight line' y=f(x)
    return A*x + B

shallowdf.plot(kind='scatter',x='MAT',y='D14C_BulkLayer')
x = shallowdf.MAT.astype(float); y = shallowdf.D14C_BulkLayer.astype(float)
notNANs = np.all(~np.isnan(np.c_[x,y]),axis=1)
A = np.vstack([x, np.ones(x.shape[0])]).T
#m, c = np.linalg.lstsq(A, y)[0]  # not working
#m, c = curve_fit(f, x, y, maxfev=10000, p0=(100,-100))[0] # not working well
m, c = stats.linregress(x[notNANs],y[notNANs])[:2]
myplt.add_regress(x[notNANs],y[notNANs])


deepdf.plot(kind='scatter',x='MAT',y='D14C_BulkLayer')
x = deepdf.MAT.astype(float); y = deepdf.D14C_BulkLayer.astype(float)
notNANs = np.all(~np.isnan(np.c_[x,y]),axis=1)
A = np.vstack([x, np.ones(x.shape[0])]).T
#m, c = np.linalg.lstsq(A, y)[0]  # not working
#m, c = curve_fit(f, x, y, maxfev=10000, p0=(100,-100))[0] # not working well
m, c = stats.linregress(x[notNANs],y[notNANs])[:2]
plt.plot(x, y, 'o', label='Original data', markersize=10)
plt.plot(x, m*x + c, 'r', label='Fitted line')
plt.gca().set_ylim([-800,400])

# repeat incre to increments
filename = 'Non_peat_data_synthesis.csv'
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1]) 
data_incre =  prep.prep_increment(data,['D14C_BulkLayer','MAT','MAP'])
shallowdf = data_incre[data_incre.Layer_depth_incre==10]
deepdf = data_incre[data_incre.Layer_depth_incre==100]
shallowdf.plot(kind='scatter',x='MAT',y='D14C_BulkLayer')
myplt.add_regress(shallowdf.MAT.astype(float), shallowdf.D14C_BulkLayer.astype(float))

shallowdf.plot(kind='scatter',x='MAP',y='D14C_BulkLayer')
deepdf.plot(kind='scatter',x='MAT',y='D14C_BulkLayer')
myplt.add_regress(deepdf.MAT.astype(float), deepdf.D14C_BulkLayer.astype(float))

deepdf.plot(kind='scatter',x='MAP',y='D14C_BulkLayer')

#%% examine linear interpolation
# linear interpolate to increments
filename = 'Non_peat_data_synthesis.csv'
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1]) 
data_linear, statprop_linear= prep.prep_lnrinterp(data, min_layer=4)
data_linear_goodfit = data_linear.loc[statprop_linear.D14C_R2>0.7]
out = []
# calculate mean rmse at each layer
for prof in data_linear_goodfit.index.unique():
    m = 0; nlayer = 0
    for n,(t,b) in enumerate(np.round(data.loc[prof:prof,['Layer_top_norm','Layer_bottom_norm']].values)):
        #print n,t,b
        dd = data_linear_goodfit.loc[prof:prof,'Layer_depth_incre'].values
        idx = (dd <= b) & (dd >= t)
        m += (np.nanmean(data_linear_goodfit.loc[prof:prof,'D14C_BulkLayer'].values[idx]) - \
                        data.loc[prof:prof,'D14C_BulkLayer'].values[n])**2.
        nlayer = n+1.
    out += np.sqrt(m/nlayer),
out = np.array(out)    


plt.hist(out[~np.isnan(out)],bins=50)    
plt.gca().set_xlim([0, 2000])

# plot goodfit
pid_goodfit = data_linear_goodfit[data_linear_goodfit.VegTypeCode_Local==2].index.unique()
for p in pid_goodfit[1:10]:
    fig = plt.figure()
    ax = fig.add_axes([0.05,0.05,0.9,0.9])
    ax.scatter(data.loc[p:p,'D14C_BulkLayer'], 
               data.loc[p:p,['Layer_top_norm','Layer_bottom_norm']].mean(axis=1), marker='x')
    ax.plot(data_linear_goodfit.loc[p:p,'D14C_BulkLayer'], 
            data_linear_goodfit.loc[p:p,'Layer_depth_incre'])
    plt.gca().invert_yaxis() 
    plt.draw()
    #raw_input('press Enter to continue...') 
#%% examine exp interpolation
# exp interpolate to increments
filename = 'Non_peat_data_synthesis.csv'
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1]) 
data_exp, statprop_exp = prep.prep_expinterp(data)
data_exp_goodfit = data_exp.loc[statprop_exp.D14C_R2>0.7]
out = []
# calculate mean rmse at each layer
for prof in data_exp_goodfit.index.unique():
    m = 0; nlayer = 0
    for n,(t,b) in enumerate(np.round(data.loc[prof:prof,['Layer_top_norm','Layer_bottom_norm']].values)):
        #print n,t,b
        dd = data_exp_goodfit.loc[prof:prof,'Layer_depth_incre'].values
        idx = (dd <= b) & (dd >= t)
        m += (np.nanmean(data_exp_goodfit.loc[prof:prof,'D14C_BulkLayer'].values[idx]) - \
                        data.loc[prof:prof,'D14C_BulkLayer'].values[n])**2.
        nlayer = n+1.
    out += np.sqrt(m/nlayer),
out = np.array(out)    


plt.hist(out[~np.isnan(out)],bins=50)    
plt.gca().set_xlim([0, 2000])

# plot goodfit
pid_goodfit = data_exp_goodfit.index.unique()
for p in pid_goodfit[1:10]:
    fig = plt.figure()
    ax = fig.add_axes([0.05,0.05,0.9,0.9])
    ax.scatter(data.loc[p:p,'D14C_BulkLayer'], 
               data.loc[p:p,['Layer_top_norm','Layer_bottom_norm']].mean(axis=1), marker='x')
    ax.plot(data_exp_goodfit.loc[p:p,'D14C_BulkLayer'], 
            data_exp_goodfit.loc[p:p,'Layer_depth_incre'])
    plt.gca().invert_yaxis()
    plt.draw()
    #raw_input('press Enter to continue...') 

#%% examine 2ndorder_poly interpolation
# poly interpolate to increments
filename = 'Non_peat_data_synthesis.csv'
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1]) 
data_poly, statprop_poly = prep.prep_polyinterp(data, 3, min_layer=4)
data_poly_goodfit = data_poly.loc[statprop_poly.D14C_R2>0.7]
out = []
# calculate mean rmse at each layer
for prof in data_poly_goodfit.index.unique():
    m = 0; nlayer = 0
    notNANs = ~data.loc[prof:prof,'D14C_BulkLayer'].isnull()
    for n,(t,b) in enumerate(np.round(data.loc[prof:prof,['Layer_top','Layer_bottom']][notNANs].values)):
        #print n,t,b
        dd = data_poly_goodfit.loc[prof:prof,'Layer_depth_incre'].values
        idx = (dd <= b) & (dd >= t)
        m += (np.nanmean(data_poly_goodfit.loc[prof:prof,'D14C_BulkLayer'].values[idx]) - \
                        data.loc[prof:prof,'D14C_BulkLayer'][notNANs].values[n])**2.
        nlayer = n+1.
        #print n,m
    out += np.sqrt(m/nlayer),
out = np.array(out)    
np.mean(out)


#plt.hist(out[~np.isnan(out)],bins=50) 
plt.ylabel('# profiles')
plt.hist(out[~np.isnan(out)],bins=50,cumulative=True,normed=1) 
plt.xlabel('profile mean RMSE of each layer (permill)')
plt.gca().set_xlim([0, 2000])

# plot goodfit
pid_goodfit = data_poly_goodfit.index.unique()
for p in pid_goodfit[1:10]:
    fig = plt.figure()
    ax = fig.add_axes([0.05,0.05,0.9,0.9])
    ax.scatter(data.loc[p:p,'D14C_BulkLayer'], 
               data.loc[p:p,['Layer_t op','Layer_bottom']].mean(axis=1), marker='x')
    ax.plot(data_poly_goodfit.loc[p:p,'D14C_BulkLayer'], 
            data_poly_goodfit.loc[p:p,'Layer_depth_incre'])
    plt.gca().invert_yaxis()
    plt.draw()
    #raw_input('press Enter to continue...') 
#%% plot climate region (whittakers)
    
# plot whittakers
filename = 'Non_peat_data_synthesis.csv'
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1]) 
pid = data.index.unique()
matt = prep.getvarxls(data, 'MAT', pid, 0)
mapp = prep.getvarxls(data, 'MAP', pid, 0)
dataresetidx = data.copy().reset_index()
uniqsite = data.copy().reset_index().drop_duplicates(subset=['MAT','MAP','ProfileID'])
# find duplicated profiles
uniqsite[uniqsite.duplicated(subset='ProfileID')].ProfileID
uniqsitematmap = uniqsite.drop_duplicates(subset=['MAT','MAP'])
plt.scatter(uniqsite.MAT,uniqsite.MAP,alpha=0.1)
plt.gca().set_xlabel('MAT')
plt.gca().set_ylabel('MAP')

vegtype = ['boreal forest','temperate forest','tropical forest','grassland',
           'cropland','shrublands','peatland','savannas','tundra','desert']
cmm = plt.cm.Paired(np.linspace(0,1,10))
for n,bio in enumerate(vegtype):
    idx = uniqsite.VegTypeCode_Local==(n+1.)
    plt.scatter(uniqsite[idx].MAT,uniqsite[idx].MAP,c=cmm[n],
                alpha=0.8,s=60,label=bio)
plt.legend(loc=2,fontsize=10)
plt.gca().set_xlabel(r'MAT ($^{\circ}$C)')
plt.gca().set_ylabel(r'MAP (mm)')
plt.gca().set_ylim([0,6000])

uniqsite.ProfileID[(uniqsite.VegTypeCode_Local==9) & (uniqsite.MAT>0.)].unique()
prep.sweepdata(filename)

#%% O_thickness exploratory analysis
df = prep_Othickness_df()
df = df[~df.O_thickness.isnull()]
gr = df.groupby('VegTypeCodeStr_Local')
y = 'O_thickness'
means = gr.mean()[y]
errors = gr.std()[y]
fig = plt.figure()
ax = fig.add_axes()
means.plot(yerr=errors, ax=ax, kind='bar') 
plt.gca().set_ylabel('Thickness of O layer (cm)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.gca().set_ylim([0,50])
 