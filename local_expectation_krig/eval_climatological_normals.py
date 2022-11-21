#!/usr/bin/env python

#-----------------------------------------------------------------------
# PROGRAM: eval_climatological_normals.py
#-----------------------------------------------------------------------
# Version 0.3
# 18 November, 2022
# Dr Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#-----------------------------------------------------------------------


#-----------------------------------------------------------------------
# OUTPUTS:
#
# 1) Boxplots of yearly bias as a function of binned temperature
# 2) Boxplots of yearly bias as a function of binned latitude
# 3) Boxplots of yearly bias as a function of binned count
# 4) Scatterplot plot of yearly bias as a function of temperature (unbinned)
# 5) Scatterplot plot of yearly bias as a function of latitude (unbinned)
# 6) Scatterplot plot of yearly bias as a function of counts (unbinned)
# 7) Facetgrid of boxplots of bias per calendar month as a function of binned temperature
# 8) Facetgrid of boxplots of bias per calendar month as a function of binned latitude
# 9) Facetgrid of boxplots of bias per calendar month as a function of binned counts
# 10) Facetgrid of normal correlation per calendar month
# 11) Lineplot of monthly median per calendar month
#-----------------------------------------------------------------------

# Numerics and dataframe libraries:
import numpy as np
import pandas as pd
import xarray as xr
import pickle
from netCDF4 import Dataset

# Datetime libraries:
from datetime import datetime
import nc_time_axis
import cftime
from cftime import num2date, DatetimeNoLeap

# Stats libraries:
import scipy
import scipy.stats as stats    
import random
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

# Plotting libraries:
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt; plt.close('all')
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib import rcParams
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
matplotlib.rcParams['text.usetex'] = False
import seaborn as sns; sns.set()

#-----------------------------------------------------------------------------
# PACKAGE VERSIONS
#-----------------------------------------------------------------------------

import platform
print("python       : ", platform.python_version())
print("pandas       : ", pd.__version__)
print("matplotlib   : ", matplotlib.__version__)

#-----------------------------------------------------------------------------
# SETTINGS
#-----------------------------------------------------------------------------

fontsize = 16
fontsize_small = 12

file_normals = '../DATA/df_normals_qc.pkl'

glosat_version = 'GloSAT.p04c.EBC'

file_lek = 'df_temp_expect.pkl'
lekstr = 'LEK (HCA,iterative,unadjusted)'

ymin, ymax = -1,1
binwidth = 5    # (default = 5)
binwidthstr = '_bw_' + str(binwidth)

sourcecode = 0  # (default = 0) 0 --> all sourcecodes > 0
if sourcecode == 0:
    sourcecodestr = ''
else:
    sourcecodestr = '_sc_' + str(sourcecode)    

#------------------------------------------------------------------------------
# METHODS: 
#------------------------------------------------------------------------------
   
def linear_regression_ols(x,y,model):
    
    if model == 0:
        regr = linear_model.LinearRegression()
    elif model == 1:
        regr = TheilSenRegressor(random_state=42)
    
    X = x.reshape(len(x),1)
    xpred = np.linspace(X.min(),X.max(),len(X)) # dummy var spanning [xmin,xmax]        
    regr.fit(X, y)
    ypred = regr.predict(xpred.reshape(-1, 1))
    slope = regr.coef_[0]
    intercept = regr.intercept_     
        
    return xpred, ypred, slope, intercept

def bin_statistics( x, y, xmin, xmax, binwidth ):
    
    bins = np.arange( xmin, xmax + binwidth, binwidth)     
    Z = np.vstack([X,Y]).T
    df_Z = pd.DataFrame( Z, columns=['X','Y']) 
    df_Z['X_bins'] = pd.cut( x=df_Z['X'], bins=bins )
    df_Z_binned = df_Z.groupby('X_bins').mean()
    X_binned = df_Z_binned.index
    df_Z_binned['p025'] = df_Z.groupby('X_bins').quantile(q=0.025)['Y'].values
    df_Z_binned['p25'] = df_Z.groupby('X_bins').quantile(q=0.25)['Y'].values
    df_Z_binned['p50'] = df_Z.groupby('X_bins').quantile(q=0.50)['Y'].values
    df_Z_binned['p75'] = df_Z.groupby('X_bins').quantile(q=0.75)['Y'].values
    df_Z_binned['p975'] = df_Z.groupby('X_bins').quantile(q=0.975)['Y'].values
    df_Z_binned['mean'] = df_Z.groupby('X_bins').mean()['Y'].values
    df_Z['Y_squared'] = np.power( df_Z['Y'], 2 ).values
    df_Z_binned['rms'] = np.sqrt( df_Z.groupby('X_bins').mean()['Y_squared'].values )

    return df_Z_binned

#------------------------------------------------------------------------------
# LOAD: GloSAT normals pkl
#------------------------------------------------------------------------------

df_normals = pd.read_pickle( file_normals, compression='bz2' )
df_normals = df_normals.reset_index(drop=True)

#------------------------------------------------------------------------------
# LOAD: LEK pkl
#------------------------------------------------------------------------------

df_lek = pd.read_pickle( file_lek, compression='bz2' )

#Index(['year', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
#       'stationcode', 'stationlat', 'stationlon', 'stationelevation',
#       'stationname', 'stationcountry', 'stationfirstyear', 'stationlastyear',
#       'stationsource', 'stationfirstreliable', 'n1', 'n2', 'n3', 'n4', 'n5',
#       'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'e1', 'e2', 'e3', 'e4',
#       'e5', 'e6', 'e7', 'e8', 'e9', 'e10', 'e11', 'e12', 's1', 's2', 's3',
#       's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12'],

# SORT: by stationcode and year

df_lek = df_lek.sort_values(['stationcode','year'], ascending=[True, True]).reset_index(drop=True)

# DROP: stationcrustr (to not impact column indexing)

df_lek = df_lek.drop(columns='stationcrustr')

#------------------------------------------------------------------------------
# COMPUTE: climatological normals and counts per station
#------------------------------------------------------------------------------

df_counts_list = df_lek[ (df_lek.year>=1961) & (df_lek.year<=1990) ].groupby('stationcode')[[ 'n'+str(i) for i in range(1,13)]].count()
df_normals_list = df_lek[ (df_lek.year>=1961) & (df_lek.year<=1990) ].groupby('stationcode')[[ 'n'+str(i) for i in range(1,13)]].mean() 
df_normals_list[ df_counts_list <= 15 ] = np.nan
df_lats_list = df_lek[ (df_lek.year>=1961) & (df_lek.year<=1990) ].groupby('stationcode')['stationlat'].mean()

#------------------------------------------------------------------------------
# ALIGN: stationcode list ( if comparing different versions)
#------------------------------------------------------------------------------

df_normals_list_reindexed = df_normals_list.reset_index()
df_counts_list_reindexed = df_counts_list.reset_index()
df_lats_list_reindexed = df_lats_list.reset_index()

mask1 = np.setdiff1d( np.array(df_normals.stationcode), np.array(df_normals_list.index) )  # stations in CRUTEM not in LEK
mask2 = np.setdiff1d( np.array(df_normals_list.index), np.array(df_normals.stationcode) )  # stations in LEK not in CRUTEM

def filter_rows_by_values(df, col, values):
    return df[~df[col].isin(values)]

df_normals = filter_rows_by_values(df_normals, 'stationcode', mask1 ).reset_index(drop=True)
df_normals_list = filter_rows_by_values(df_normals_list_reindexed, 'stationcode', mask2 ).groupby('stationcode').mean().reset_index(drop=True)
df_counts_list = filter_rows_by_values(df_counts_list_reindexed, 'stationcode', mask2 ).groupby('stationcode').mean().reset_index(drop=True)
df_lats_list = filter_rows_by_values(df_lats_list_reindexed, 'stationcode', mask2 ).groupby('stationcode').mean().reset_index(drop=True)

#------------------------------------------------------------------------------
# EXTRACT: CRUTEM normals
#------------------------------------------------------------------------------

df_normals_crutem = df_normals.iloc[:,6:18]

# SOURCODE MASK: apply to arrays

if sourcecode == 0:
    mask = df_normals.sourcecode > 0
else:    
    mask = df_normals.sourcecode == sourcecode
    
counts_list = np.array( df_counts_list )[mask,:]
normals_list = np.array( df_normals_list )[mask,:]
normals_crutem = np.array( df_normals_crutem )[mask,:]
lats_list = np.array( df_lats_list )[mask]

stationcodelist = df_normals.stationcode.unique()[mask]

#==============================================================================
# PLOTS
#==============================================================================

#------------------------------------------------------------------------------
# PLOT 1: boxplots of yearly bias as a function of binned temperature
#------------------------------------------------------------------------------

print('plotting yearly bias with binned temperature ... ')   

binwidth_yearly = 5
binwidthstr_yearly = '_bw_' + str(binwidth_yearly)

mask = np.isfinite( np.mean( normals_crutem, axis=1) ) & np.isfinite( np.mean( normals_list, axis=1) )
X = np.array( np.mean( normals_crutem[mask,:], axis=1) ).ravel()
Y = np.subtract( np.array( np.mean( normals_crutem[mask,:], axis=1) ), np.array( np.mean( normals_list[mask,:], axis=1) ) )

nstations = mask.sum()
unbinned_mean = np.nanmean(Y)
unbinned_rms = np.sqrt( np.nanmean( Y**2.0 ) )

# COMPUTE: binned statistics

xmin, xmax = -60, 40
df_Z_binned = bin_statistics( X, Y, xmin, xmax, binwidth_yearly )

# COMPUTE: OLS fit to binned mean

idx = pd.IntervalIndex(df_Z_binned.index)          # bin intervals (x1,x2] 
X_binned = idx.left.values + binwidth_yearly/2            # bin centers
Y_binned = df_Z_binned['mean']
mask = np.isfinite(X_binned) & np.isfinite(Y_binned)
if len(X_binned[mask]) < 2:
    corrcoef = np.nan
else:
    corrcoef = scipy.stats.pearsonr(X_binned[mask], Y_binned[mask])[0]
OLS_X, OLS_Y, OLS_slope, OLS_intercept = linear_regression_ols(X_binned[mask], Y_binned[mask], 1)
OLS_start = [ xmin, xmax ]
OLS_end = [ OLS_slope*xmin+OLS_intercept, OLS_slope*xmax+OLS_intercept ]

# PLOT: bin stats
    
figstr = 'normals_yearly_diff_temperature_binned' + binwidthstr_yearly + sourcecodestr + '.png'
titlestr = 'Climatological 1961-1990 normals (MIN 15/30): yearly-averaged (12/12 months) with binned temperature' + ': n(stations)=' + str(nstations)
xstr = glosat_version + r', $^{\circ}C$'
ystr = 'Bias: ' + glosat_version + r' - LEK, $^{\circ}C$'

fig,ax = plt.subplots(figsize=(15,10))        
plt.plot(OLS_start, OLS_end, color='navy', ls='-', lw=2, zorder=2, label='Binned statistics (bin width=' + str(binwidth_yearly) + ') OLS fit (Thiel-Sen): ' + r'$\alpha$=' + str(np.round(OLS_slope,3)) + ' ' + r'$\beta$=' + str(np.round(OLS_intercept,3)) + ' ' + r'$\rho$='+str(np.round(corrcoef,3)).zfill(3))
plt.scatter( X_binned, df_Z_binned['p50'], marker='*', color='red', alpha=1, label='Bin Median (unbinned mean bias=' + str(np.round(unbinned_mean,2)) + r'$^{\circ}$C ' + 'RMS=' + str(np.round(unbinned_rms,2)) + r'$^{\circ}$C' + ')')
plt.scatter( X_binned, df_Z_binned['p25'], marker='.', color='blue', alpha=1, label='Bin IQR' )
plt.scatter( X_binned, df_Z_binned['p75'], marker='.', color='blue', alpha=1 )
plt.scatter( X_binned, df_Z_binned['p025'], marker='.', color='black', alpha=1, label='Bin CI [2.5-97.5%]' )
plt.scatter( X_binned, df_Z_binned['p975'], marker='.', color='black', alpha=1)
plt.fill_between( X_binned, 0, df_Z_binned['rms'], color='grey', alpha=0.2, label='Bin RMS')
plt.axhline( y=0, color='k', ls='--', lw=1, zorder=4 )
plt.xlim( xmin, xmax )
plt.ylim( ymin, ymax )
plt.tick_params(labelsize=fontsize)    
plt.xlabel( xstr, fontsize=fontsize)
plt.ylabel( ystr, fontsize=fontsize)
plt.title( titlestr, fontsize=fontsize)
plt.legend(loc='lower left', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)        
fig.tight_layout()
plt.savefig( figstr, dpi=300, bbox_inches='tight')
plt.close('all')

#------------------------------------------------------------------------------
# PLOT 2: boxplots of yearly bias as a function of binned latitude
#------------------------------------------------------------------------------

print('plotting yearly bias with binned latitude ... ')   

binwidth_yearly = 5
binwidthstr_yearly = '_bw_' + str(binwidth_yearly)

mask = np.isfinite( np.mean( normals_crutem, axis=1) ) & np.isfinite( np.mean( normals_list, axis=1) )
X = np.array( lats_list[mask] ).ravel()
Y = np.subtract( np.array( np.mean( normals_crutem[mask,:], axis=1) ), np.array( np.mean( normals_list[mask,:], axis=1) ) )

unbinned_mean = np.nanmean(Y)
unbinned_rms = np.sqrt( np.nanmean( Y**2.0 ) )

# COMPUTE: binned statistics

xmin, xmax = -90, 90
df_Z_binned = bin_statistics( X, Y, xmin, xmax, binwidth_yearly )

# COMPUTE: OLS fit to binned mean

idx = pd.IntervalIndex(df_Z_binned.index)          # bin intervals (x1,x2] 
X_binned = idx.left.values + binwidth_yearly/2            # bin centers
Y_binned = df_Z_binned['mean']
mask = np.isfinite(X_binned) & np.isfinite(Y_binned)
if len(X_binned[mask]) < 2:
    corrcoef = np.nan
else:
    corrcoef = scipy.stats.pearsonr(X_binned[mask], Y_binned[mask])[0]
OLS_X, OLS_Y, OLS_slope, OLS_intercept = linear_regression_ols(X_binned[mask], Y_binned[mask], 1)
OLS_start = [ xmin, xmax ]
OLS_end = [ OLS_slope*xmin+OLS_intercept, OLS_slope*xmax+OLS_intercept ]

# PLOT: bin stats
    
figstr = 'normals_yearly_diff_latitude_binned' + binwidthstr_yearly + sourcecodestr + '.png'
titlestr = 'Climatological 1961-1990 normals (MIN 15/30): yearly-averaged (12/12 months) with binned latitude' + ': n(stations)=' + str(nstations)
xstr =  r'Latitude, $^{\circ}N$'
ystr = 'Bias: ' + glosat_version + r' - LEK, $^{\circ}C$'

fig,ax = plt.subplots(figsize=(15,10))        
plt.plot(OLS_start, OLS_end, color='navy', ls='-', lw=2, zorder=2, label='Binned statistics (bin width=' + str(binwidth_yearly) + ') OLS fit (Thiel-Sen): ' + r'$\alpha$=' + str(np.round(OLS_slope,3)) + ' ' + r'$\beta$=' + str(np.round(OLS_intercept,3)) + ' ' + r'$\rho$='+str(np.round(corrcoef,3)).zfill(3)) 
plt.scatter( X_binned, df_Z_binned['p50'], marker='*', color='red', alpha=1, label='Bin Median (unbinned mean bias=' + str(np.round(unbinned_mean,2)) + r'$^{\circ}$C ' + 'RMS=' + str(np.round(unbinned_rms,2)) + r'$^{\circ}$C' + ')') 
plt.scatter( X_binned, df_Z_binned['p25'], marker='.', color='blue', alpha=1, label='Bin IQR' )
plt.scatter( X_binned, df_Z_binned['p75'], marker='.', color='blue', alpha=1 )
plt.scatter( X_binned, df_Z_binned['p025'], marker='.', color='black', alpha=1, label='Bin CI [2.5-97.5%]' )
plt.scatter( X_binned, df_Z_binned['p975'], marker='.', color='black', alpha=1)
plt.fill_between( X_binned, 0, df_Z_binned['rms'], color='grey', alpha=0.2, label='Bin RMS')
plt.axhline( y=0, color='k', ls='--', lw=1, zorder=4 )
plt.xlim( xmin, xmax )
plt.ylim( ymin, ymax )
plt.tick_params(labelsize=fontsize)    
plt.xlabel( xstr, fontsize=fontsize)
plt.ylabel( ystr, fontsize=fontsize)
plt.title( titlestr, fontsize=fontsize)
plt.legend(loc='lower left', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)        
fig.tight_layout()
plt.savefig( figstr, dpi=300, bbox_inches='tight')
plt.close('all')

#------------------------------------------------------------------------------
# PLOT 7: facetgrid of boxplots of bias per calendar month as a function of binned temperature
#------------------------------------------------------------------------------

print('plotting bias per calendar month with binned temperature ... ')   
      
figstr = 'normals_monthly_diff_temperature_binned' + binwidthstr + sourcecodestr + '.png'
titlestr = 'Climatological 1961-1990 normals (MIN 15/30): with binned temperature' + ': n(stations)=' + str(nstations)
xstr = glosat_version + r', $^{\circ}C$'
ystr = 'GloSAT-LEK, $^{\circ}C$'
xmin, xmax = -60, 40
ymin, ymax = -1, 1

fig,ax = plt.subplots(3,4, sharex=True, sharey=True, figsize=(15,10))    

for i in range(12):

    plt.subplot(3, 4, i+1)
    
    mask = np.isfinite( normals_crutem[:,i] ) & np.isfinite( normals_list[:,i] )
    X = np.array( normals_crutem[mask,i] ).ravel()
    Y = np.subtract( np.array( normals_crutem[mask,i] ),np.array( normals_list[mask,i] ) )
    
    # COMPUTE: binned statistics
    
    df_Z_binned = bin_statistics( X, Y, xmin, xmax, binwidth )
    
    # COMPUTE: OLS fit to binned mean
    
    idx = pd.IntervalIndex(df_Z_binned.index)          # bin intervals (x1,x2] 
    X_binned = idx.left.values + binwidth/2            # bin centers
    Y_binned = df_Z_binned['mean']
    mask = np.isfinite(X_binned) & np.isfinite(Y_binned)
    if len(X_binned[mask]) < 2:
        corrcoef = np.nan
    else:
        corrcoef = scipy.stats.pearsonr(X_binned[mask], Y_binned[mask])[0]
    OLS_X, OLS_Y, OLS_slope, OLS_intercept = linear_regression_ols(X_binned[mask], Y_binned[mask], 1)
    OLS_start = [ xmin, xmax ]
    OLS_end = [ OLS_slope*xmin+OLS_intercept, OLS_slope*xmax+OLS_intercept ]
        
    bias_mean = np.nanmean(Y_binned)
    bias_rms = np.sqrt( np.nanmean( Y_binned**2.0 ) )
   
    plt.plot(OLS_start, OLS_end, color='navy', ls='-', lw=2, zorder=2, label=r'$\alpha$=' + str(np.round(OLS_slope,3)) + ' ' + r'$\beta$=' + str(np.round(OLS_intercept,3)) ) 
    plt.scatter( X_binned, df_Z_binned['p50'], marker='*', color='red', alpha=1)
    plt.scatter( X_binned, df_Z_binned['p25'], marker='.', color='blue', alpha=1)
    plt.scatter( X_binned, df_Z_binned['p75'], marker='.', color='blue', alpha=1)
    plt.scatter( X_binned, df_Z_binned['p025'], marker='.', color='black', alpha=1)
    plt.scatter( X_binned, df_Z_binned['p975'], marker='.', color='black', alpha=1)
    plt.fill_between( X_binned, 0, df_Z_binned['rms'], color='grey', alpha=0.2)
    plt.axhline( y=0, color='k', ls='--', lw=1, zorder=4 )
    plt.xlim( xmin, xmax )
    plt.ylim( ymin, ymax )
    plt.tick_params(labelsize=fontsize)    
    if i > 7: plt.xlabel( xstr, fontsize=fontsize)
    if i % 4 == 0: plt.ylabel( ystr, fontsize=fontsize)
    plt.title( 'month: ' + str(i+1).zfill(2), fontsize=fontsize)
    plt.legend(loc='lower left', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    
plt.suptitle( titlestr, fontsize=fontsize)
fig.tight_layout()
plt.savefig( figstr, dpi=300, bbox_inches='tight')
plt.close('all')

#------------------------------------------------------------------------------
# PLOT 8: facetgrid of boxplots of bias per calendar month as a function of binned latitude
#------------------------------------------------------------------------------

print('plotting bias per calendar month with binned latitude ... ')   
      
figstr = 'normals_monthly_diff_latitude_binned' + binwidthstr + sourcecodestr + '.png'
titlestr = 'Climatological 1961-1990 normals (MIN 15/30): with binned latitude' + ': n(stations)=' + str(nstations)
xstr = r'Latitude, $^{\circ}N$'
ystr = 'GloSAT-LEK, $^{\circ}C$'
xmin, xmax = -90, 90
ymin, ymax = -1, 1

fig,ax = plt.subplots(3,4, sharex=True, sharey=True, figsize=(15,10))    

for i in range(12):

    plt.subplot(3, 4, i+1)
    
    mask = np.isfinite( normals_crutem[:,i] ) & np.isfinite( normals_list[:,i] )
    X = np.array( lats_list[mask] ).ravel()
    Y = np.subtract( np.array( normals_crutem[mask,i] ),np.array( normals_list[mask,i] ) )
    
    # COMPUTE: binned statistics
    
    df_Z_binned = bin_statistics( X, Y, xmin, xmax, binwidth )
    
    # COMPUTE: OLS fit to binned mean
    
    idx = pd.IntervalIndex(df_Z_binned.index)          # bin intervals (x1,x2] 
    X_binned = idx.left.values + binwidth/2            # bin centers
    Y_binned = df_Z_binned['mean']
    mask = np.isfinite(X_binned) & np.isfinite(Y_binned)
    if len(X_binned[mask]) < 2:
        corrcoef = np.nan
    else:
        corrcoef = scipy.stats.pearsonr(X_binned[mask], Y_binned[mask])[0]    
    OLS_X, OLS_Y, OLS_slope, OLS_intercept = linear_regression_ols(X_binned[mask], Y_binned[mask], 1)
    OLS_start = [ xmin, xmax ]
    OLS_end = [ OLS_slope*xmin+OLS_intercept, OLS_slope*xmax+OLS_intercept ]
        
    bias_mean = np.nanmean(Y_binned)
    bias_rms = np.sqrt( np.nanmean( Y_binned**2.0 ) )
   
    plt.plot(OLS_start, OLS_end, color='navy', ls='-', lw=2, zorder=2, label=r'$\alpha$=' + str(np.round(OLS_slope,3)) + ' ' + r'$\beta$=' + str(np.round(OLS_intercept,3)) )
    plt.scatter( X_binned, df_Z_binned['p50'], marker='*', color='red', alpha=1)
    plt.scatter( X_binned, df_Z_binned['p25'], marker='.', color='blue', alpha=1)
    plt.scatter( X_binned, df_Z_binned['p75'], marker='.', color='blue', alpha=1)
    plt.scatter( X_binned, df_Z_binned['p025'], marker='.', color='black', alpha=1)
    plt.scatter( X_binned, df_Z_binned['p975'], marker='.', color='black', alpha=1)
    plt.fill_between( X_binned, 0, df_Z_binned['rms'], color='grey', alpha=0.2)
    plt.axhline( y=0, color='k', ls='--', lw=1, zorder=4 )
    plt.xlim( xmin, xmax)
    plt.ylim( ymin, ymax)        
    plt.tick_params(labelsize=fontsize)    
    if i > 7: plt.xlabel( xstr, fontsize=fontsize)
    if i % 4 == 0: plt.ylabel( ystr, fontsize=fontsize)
    plt.title( 'month: ' + str(i+1).zfill(2), fontsize=fontsize)
    plt.legend(loc='lower left', ncol=2, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    
plt.suptitle( titlestr, fontsize=fontsize)
fig.tight_layout()
plt.savefig( figstr, dpi=300, bbox_inches='tight')
plt.close('all')

#------------------------------------------------------------------------------
# PLOT 10: facetgrid of normal correlation per calendar month
#------------------------------------------------------------------------------

print('plotting normals correlation per calendar month ... ')   
      
figstr = 'normals_monthly_correlation' + sourcecodestr + '.png'
titlestr = 'Climatological 1961-1990 normals (MIN 15/30): correlation per calendar month' + ': n(stations)=' + str(nstations)
xstr = glosat_version + r', $^{\circ}C$'
ystr = r'LEK, $^{\circ}C$'
xmin, xmax = -70, 50

fig,ax = plt.subplots(sharex=True, sharey=True, figsize=(15,10))    
for i in range(12):

    plt.subplot(3, 4, i+1)

    X = normals_crutem[:,i].ravel()
    Y = normals_list[:,i]
    mask = np.isfinite(X) & np.isfinite(Y)
    corrcoef = scipy.stats.pearsonr(X[mask], Y[mask])[0]    
    OLS_X, OLS_Y, OLS_slope, OLS_intercept = linear_regression_ols(X[mask], Y[mask], 1)
    OLS_start = [ xmin, xmax ]
    OLS_end = [ OLS_slope*xmin+OLS_intercept, OLS_slope*xmax+OLS_intercept ]

    plt.plot(OLS_start, OLS_end, color='navy', ls='-', lw=2, zorder=1, label=r'$\alpha$=' + str(np.round(OLS_slope,3)) + ' ' + r'$\beta$=' + str(np.round(OLS_intercept,3)) )
    plt.scatter( X, Y, marker='s', color='teal', alpha=0.05, zorder=2 )
    plt.xlim( xmin, xmax )
    plt.ylim( xmin, xmax )
    ax.set_aspect('equal') 
    ax.xaxis.grid(True, which='major', color='lightgrey', alpha=0.5, zorder=0 )  
    ax.yaxis.grid(True, which='major', color='lightgrey', alpha=0.5, zorder=0 )  
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    if i > 7: plt.xlabel( xstr, fontsize=fontsize)
    if i % 4 == 0: plt.ylabel( ystr, fontsize=fontsize)
    plt.title( 'month: ' + str(i+1).zfill(2), fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)    
    plt.legend(loc='upper left', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    

plt.suptitle( titlestr, fontsize=fontsize)
fig.tight_layout()
plt.savefig( figstr, dpi=300, bbox_inches='tight')
plt.close('all')

#------------------------------------------------------------------------------
# PLOT 11: lineplot of normals median per calendar month
#------------------------------------------------------------------------------

print('plotting normals median per calendar month ... ')   

mask = np.isfinite( normals_list ) & np.isfinite( normals_crutem ) 
# nstations = mask.sum(axis=0).max()

medians_lek = []
medians_crutem = []

for i in range(12):
#    mask = np.isfinite( normals_list[:,i] ) & np.isfinite( normals_crutem[:,i] )        
    medians_lek.append( np.nanmedian( normals_list[:,i][mask[:,i]] ) )
    medians_crutem.append ( np.nanmedian( normals_crutem[:,i][mask[:,i]] ) )
                                                              
figstr = 'normals_monthly_median' + sourcecodestr + '.png'
titlestr = 'Climatological 1961-1990 normals (MIN 15/30): median per calendar month' + ': n(stations)=' + str(nstations)
xstr = 'Calendar month'
ystr = r'Median normal, $^{\circ}C$'

fig, ax = plt.subplots( nrows=2, ncols=1, sharex=True, sharey=False, gridspec_kw={'height_ratios': [4,1]}, figsize=(15,10) )             
ax[0].plot( np.arange(1,13), medians_crutem, marker='o', color='red', alpha=0.5, ls='-', label=glosat_version )
ax[0].plot( np.arange(1,13), medians_lek, marker='o', color='blue', alpha=0.5, ls='-', label=lekstr ) 
ylimits = ax[0].get_ylim()
xlimits = ax[0].get_xlim()
ax[0].yaxis.grid(True, which='major', color='lightgrey', alpha=0.5)  
ax[0].yaxis.set_major_locator(ticker.MultipleLocator(1))
ax[0].set_ylabel( ystr, fontsize=fontsize)
ax[0].set_title( titlestr, fontsize=fontsize)
ax[0].legend( loc='upper left', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize) 
ax[0].set_xticks( np.arange(1,13), [ str(i) for i in range(1,13) ] )
ax[0].tick_params(labelsize=fontsize)   
ax[1].plot( np.arange(1,13), np.array(medians_crutem)-np.array(medians_lek), marker='o', color='teal', ls='-', label='GloSAT-LEK')
ax[1].axhline(y=0, ls='dotted', lw=2, color='black')
ylimits = ax[1].get_ylim()
ymax = np.max(ylimits)
ax[1].legend(loc='upper left', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)        
ax[1].set_xlabel('Year', fontsize=fontsize)
ax[1].set_ylabel('Difference, $^{\circ}C$', fontsize=fontsize)
ax[1].set_xlabel( xstr, fontsize=fontsize)
ax[1].tick_params(labelsize=fontsize)   
fig.tight_layout()
plt.savefig( figstr, dpi=300, bbox_inches='tight')
plt.close('all')

#------------------------------------------------------------------------------
# SAVE: LEK normals and monthly counts + diffs
#------------------------------------------------------------------------------          
                                                  
#------------------------------------------------------------------------------
print('** END')


