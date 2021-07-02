#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: best-fit-means-auto.py
#------------------------------------------------------------------------------
# Verion 0.2
# 2 July, 2021
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#------------------------------------------------------------------------------
# Dataframe libraries:
import numpy as np
import numpy.ma as ma
import itertools
import pandas as pd
import xarray as xr
import pickle
from datetime import datetime
import nc_time_axis
import cftime

# Plotting libraries:
import matplotlib
import matplotlib.pyplot as plt; plt.close('all')
import matplotlib.cm as cm
from matplotlib import colors as mcol
from matplotlib.cm import ScalarMappable
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
from matplotlib.collections import PolyCollection
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import cmocean
import seaborn as sns; sns.set()

# OS libraries:
import os
import os.path
from pathlib import Path
import sys
import subprocess
from subprocess import Popen
import time

# Stats libraries:
import random
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

# Datetime libraries
import cftime
import calendar 
# print(calendar.calendar(2020))
# print(calendar.month(2020,2))
# calendar.isleap(2020)
from datetime import date, time, datetime, timedelta
#today = datetime.now()
#tomorrow = today + pd.to_timedelta(1,unit='D')
#tomorrow = today + timedelta(days=1)
#birthday = datetime(1970,11,1,0,0,0).strftime('%Y-%m-%d %H:%M')
#print('Week:',today.isocalendar()[1])

# Silence library version notifications
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------

fontsize = 16
nsmooth = 60

load_glosat = True
plot_monthly = True
plot_fit_model_1 = True
plot_fit_model_2 = True

use_correlated = True # --> Model B based on differences
use_fahrenheit = True

if use_fahrenheit: 
    temperature_unit = 'F'
else:
    temperature_unit = 'C'

segment_start = pd.to_datetime('1850-01-01')
segment_end = pd.to_datetime('1899-12-01')
normal_start = pd.to_datetime('1961-01-01')
normal_end = pd.to_datetime('1990-12-01')

test_station = 'blue_hill'

#------------------------------------------------------------------------------
# METHODS: 
#------------------------------------------------------------------------------

def fahrenheit_to_centigrade(x):
    y = (5.0/9.0) * (x - 32.0)
    return y

def centigrade_to_fahrenheit(x):
    y = (x * (9.0/5.0)) + 32.0
    return y
    
def calculate_normals_and_SEs(a1,a2,r2):
    
    # CALCULATE: monthly normals
    
    x1r_A = []
    x1r_B = []
    SE1r_A = []
    SE1r_B = []

    for i in range(12):
    
        x1a = np.nanmean( a1[a1.index.month==(i+1)] )
        x2a = np.nanmean( a2[a2.index.month==(i+1)] )
        x2r = np.nanmean( r2[r2.index.month==(i+1)] )    
        SE1a = np.nanstd( a1[a1.index.month==(i+1)] ) / np.sqrt( np.isfinite(a1[a1.index.month==(i+1)]).sum() )
        SE2a = np.nanstd( a2[a2.index.month==(i+1)] ) / np.sqrt( np.isfinite(a2[a2.index.month==(i+1)]).sum() )
        SE2r = np.nanstd( r2[r2.index.month==(i+1)] ) / np.sqrt( np.isfinite(r2[r2.index.month==(i+1)]).sum() )
   
        # CASE A: uncorrelated errors
        
        x1r_A_month = x1a + (x2r - x2a)
        x1r_A.append(x1r_A_month)
        SE1r_A_month = np.sqrt( SE1a**2. + SE2r**2. + SE2a**2. )
        SE1r_A.append(SE1r_A_month)

        # CASE B: 'modeling out the correlation' 

        a12 = a1 - a2

        x12a = np.nanmean( a12[a12.index.month==(i+1)] )
        x1r_B_month = x2r + x12a 
        x1r_B.append(x1r_B_month)

        SE12a = np.nanstd( a12[a12.index.month==(i+1)] ) / np.sqrt( np.isfinite(a12[a12.index.month==(i+1)]).sum() )
        SE1r_B_month = np.sqrt( SE2r**2. + SE12a**2. )    
        SE1r_B.append(SE1r_B_month)
                        
    return x1r_A, x1r_B, SE1r_A, SE1r_B

#==============================================================================
# LOAD: absolute temperatures --> df
#==============================================================================

if load_glosat == True:
    
    #------------------------------------------------------------------------------    
    # LOAD: GloSAT absolute temperature and anomaly archives: CRUTEM5.0.1.0
    #------------------------------------------------------------------------------
        
    print('loading temperatures ...')
                
    df_temp = pd.read_pickle('DATA/df_temp.pkl', compression='bz2') # dataframe of GloSAT absolute temperatures in degrees C
    
    stationcode_list = ['720218', '720219', '744920', '725092', '753011', '720222', '720223', '725045', '756213', '725070', '725091', '725090', '720225', '744900', '744902']
    stationname_list = ['amherst', 'bedford', 'blue_hill', 'boston_city_wso', 'kingston', 'lawrence', 'new_bedford', 'new_haven', 'plymouth_kingston', 'providence_wso', 'provincetown', 'reading', 'taunton', 'walpole_2', 'west_medway']
    stationcode_dict = dict(zip(range(len(stationcode_list)), stationcode_list))
    stationname_dict = dict(zip(range(len(stationcode_list)), stationname_list))

    dates = pd.date_range(start='1700-01-01', end='2021-12-01', freq='MS')
    df = pd.DataFrame(index=dates)
    for i in range(len(stationcode_list)):        
        
        dt = df_temp[df_temp['stationcode']==stationcode_list[i]]
        ts = np.array(dt.groupby('year').mean().iloc[:,0:12]).ravel()        
        
        if use_fahrenheit == True: 
            
            ts = centigrade_to_fahrenheit(ts)        
            
        t = pd.date_range(start=str(dt.year.iloc[0]), periods=len(ts), freq='MS')
        df[stationname_list[i]] = pd.DataFrame({stationname_list[i]:ts}, index=t) 

#------------------------------------------------------------------------------
# CALCULATE: test station and neighbouring station mean timeseries in the segment and normal
#------------------------------------------------------------------------------

df_segment = df[ (df.index>=segment_start) & (df.index<=segment_end) ]
df_normal = df[ (df.index>=normal_start) & (df.index<=normal_end) ]
df_test_station = df[test_station]
df_test_station_segment = df_segment[test_station]
df_test_station_normal = df_normal[test_station]
df_neighbours = df.drop([test_station], axis=1)
df_neighbours_segment = df_segment.drop([test_station], axis=1)
df_neighbours_normal = df_normal.drop([test_station], axis=1)
df_neighbours_mean = df_neighbours.mean(axis=1)
df_neighbours_mean_segment = df_neighbours_segment.mean(axis=1)
df_neighbours_mean_normal = df_neighbours_normal.mean(axis=1)

#------------------------------------------------------------------------------
# LOOP: over ref_stations and estimate normal
#------------------------------------------------------------------------------

df_errors = pd.DataFrame(columns=[
        'test_station', 
        'ref_station', 
        'error_x1r_CASE_1A',
        'error_x1r_CASE_1B',
        'error_x1r_CASE_2A',
        'error_SE1r_CASE_1A',
        'error_SE1r_CASE_1B',
        'error_SE1r_CASE_2A'])
df_errors['test_station'] = len(stationcode_list)*[test_station]    

for station in range(len(stationcode_list)):
    
    ref_station = stationname_list[station]
    df_ref_station = df[ref_station]
    df_ref_station_segment = df_segment[ref_station]
    df_ref_station_normal = df_normal[ref_station]
    
    # CALCULATE: expected 'true' monthly means from test station
    
    r1 = df_test_station_normal
    x1r_truth = []
    SE1r_truth = []
    for i in range(12):
        
        x1r = np.nanmean( r1[r1.index.month==(i+1)] )    
        x1r_truth.append(x1r)
        SE1r = np.nanstd( r1[r1.index.month==(i+1)] ) / np.sqrt( np.isfinite(r1[r1.index.month==(i+1)]).sum() )   
        SE1r_truth.append(SE1r)
        
    #------------------------------------------------------------------------------
    # MODEL 1: CASE 1A & CASE 1B: single reference: x1=test_station, x2=BHO (referemce_station)
    #------------------------------------------------------------------------------
    
    a1 = df_test_station_segment
    a2 = df_ref_station_segment
    r2 = df_ref_station_normal
    
    x1r_CASE_1A, x1r_CASE_1B, SE1r_CASE_1A, SE1r_CASE_1B = calculate_normals_and_SEs(a1,a2,r2)
    x1r_CASE_1A_normal = pd.Series(np.tile(x1r_CASE_1A, reps=30), index=r2.index)
    x1r_CASE_1B_normal = pd.Series(np.tile(x1r_CASE_1B, reps=30), index=r2.index)
    
    #------------------------------------------------------------------------------
    # MODEL 2A: multple co-located neighbours x1=mean of neighbours, x2=BHO
    #------------------------------------------------------------------------------
    
    a1 = df_test_station_segment
    a2 = df_neighbours_mean_segment
    r2 = df_neighbours_mean_normal
    
    x1r_CASE_2A, x1r_CASE_2B, SE1r_CASE_2A, SE1r_CASE_2B = calculate_normals_and_SEs(a1,a2,r2)
    x1r_CASE_2A_normal = pd.Series(np.tile(x1r_CASE_2A, reps=30), index=r2.index)
    x1r_CASE_2B_normal = pd.Series(np.tile(x1r_CASE_2B, reps=30), index=r2.index)
    
    if use_correlated == True: # (default)
        x1r_CASE_1_normal = x1r_CASE_1B_normal
        x1r_CASE_2_normal = x1r_CASE_2B_normal
    else:
        x1r_CASE_1_normal = x1r_CASE_1A_normal
        x1r_CASE_2_normal = x1r_CASE_2A_normal    
    
    #------------------------------------------------------------------------------
    # STATISTICS: model versus truth monthly mean errors
    #------------------------------------------------------------------------------
    
    error_x1r_CASE_1A = np.nanmean( np.array(x1r_CASE_1A) - np.array(x1r_truth) )
    error_x1r_CASE_1B = np.nanmean( np.array(x1r_CASE_1B) - np.array(x1r_truth) )
    error_x1r_CASE_2A = np.nanmean( np.array(x1r_CASE_2A) - np.array(x1r_truth) )
    error_SE1r_CASE_1A = np.nanmean( np.array(SE1r_CASE_1A) - np.array(SE1r_truth) )
    error_SE1r_CASE_1B = np.nanmean( np.array(SE1r_CASE_1B) - np.array(SE1r_truth) )
    error_SE1r_CASE_2A = np.nanmean( np.array(SE1r_CASE_2A) - np.array(SE1r_truth) )
    
    X_1 = df_test_station.rolling(nsmooth,center=True).mean()                                           # BHO
    X_1a = X_1[ (X_1.index>=segment_start) & (X_1.index<=segment_end) ]                                 # BHO (segment)
    X_1r_truth = X_1[ (X_1.index>=normal_start) & (X_1.index<=normal_end) ]                             # BHO (normal) truth
    X_1r_estimate_CASE_1 = X_1r_truth + ( np.nanmean( x1r_CASE_1_normal ) - np.nanmean( X_1r_truth ) )  # BHO (normal) estimate (CASE_1)
    X_2_CASE_1 = df_ref_station.rolling(nsmooth,center=True).mean()                                     # single neighbour
    X_2a_CASE_1 = X_2_CASE_1[ (X_2_CASE_1.index>=segment_start) & (X_2_CASE_1.index<=segment_end) ]     # single neighbour (segment)
    X_2r_CASE_1 = X_2_CASE_1[ (X_2_CASE_1.index>=normal_start) & (X_2_CASE_1.index<=normal_end) ]       # single neighbour (normal)            
    error_CASE_1 = X_1r_estimate_CASE_1[0] -  X_1r_truth[0]                                             # error relative to expected true normal
    
    X_1 = df_test_station.rolling(nsmooth,center=True).mean()                                           # BHO
    X_1a = X_1[ (X_1.index>=segment_start) & (X_1.index<=segment_end) ]                                 # BHO (segment)
    X_1r_truth = X_1[ (X_1.index>=normal_start) & (X_1.index<=normal_end) ]                             # BHO (normal) truth
    X_1r_estimate_CASE_2 = X_1r_truth + ( np.nanmean( x1r_CASE_2_normal ) - np.nanmean( X_1r_truth ) )  # BHO (normal) estimate (CASE_2)
    X_2_CASE_2 = df_neighbours_mean.rolling(nsmooth,center=True).mean()                                 # All neighbours mean
    X_2a_CASE_2 = X_2_CASE_2[ (X_2_CASE_2.index>=segment_start) & (X_2_CASE_2.index<=segment_end) ]     # All neighbours mean (segment)
    X_2r_CASE_2 = X_2_CASE_2[ (X_2_CASE_2.index>=normal_start) & (X_2_CASE_2.index<=normal_end) ]       # All neighbours mean (normal)
    error_CASE_2 = X_1r_estimate_CASE_2[0] -  X_1r_truth[0]                                             # error relative to expected true normal
    
    df_errors['ref_station'].iloc[station] = ref_station
    df_errors['test_station'].iloc[station] = test_station
    df_errors['error_x1r_CASE_1A'].iloc[station] = error_x1r_CASE_1A
    df_errors['error_x1r_CASE_1B'].iloc[station] = error_x1r_CASE_1B
    df_errors['error_x1r_CASE_2A'].iloc[station] = error_x1r_CASE_2A
    df_errors['error_SE1r_CASE_1A'].iloc[station] = error_SE1r_CASE_1A
    df_errors['error_SE1r_CASE_1B'].iloc[station] = error_SE1r_CASE_1B
    df_errors['error_SE1r_CASE_2A'].iloc[station] = error_SE1r_CASE_2A
        
    #==============================================================================
    
    if plot_monthly == True: # one plot per ref_station
        
        # PLOT: monthly normals and standard errors
        
        print('plotting x1r and SE1r (CASE 1A vs CASE 1B vs CASE 2A) ...')
            
        figstr = 'models-monthly-normals-sterr-CASE-1A-1B-2A' + '-' + test_station + '-' + ref_station + '.png'
        titlestr = r'Monthly $x_{1,r}$ and $SE_{1,r}$ (ref = ' + ref_station.title() + ') for test station = ' + test_station.title()
        
        fig, axs = plt.subplots(2,1, figsize=(15,10))
        axs[0].plot(x1r_truth, marker='s', markersize=20, fillstyle='none', linestyle='none', label=r'Truth: $\bar{X}_{1,r}$')
        axs[0].plot(x1r_CASE_1A, marker='^', markersize=10, alpha=0.5, linestyle='none', label=r'Model 1A: $\bar{X}_{1,r}$ $\mu_{\epsilon}$=' + str(np.round(error_x1r_CASE_1A,2)) + '$^{\circ}$' + temperature_unit) 
        axs[0].plot(x1r_CASE_1B, marker='v', markersize=10, alpha=0.5, linestyle='none', label=r'Model 1B: $\bar{X}_{1,r}$ $\mu_{\epsilon}$=' + str(np.round(error_x1r_CASE_1B,2)) + '$^{\circ}$' + temperature_unit) 
        axs[0].plot(x1r_CASE_2A, marker='o', markersize=10, alpha=0.5, linestyle='none', label=r'Model 2A: $\bar{X}_{1,r}$ $\mu_{\epsilon}$=' + str(np.round(error_x1r_CASE_2A,2)) + '$^{\circ}$' + temperature_unit) 
        axs[0].set_xticks(np.arange(0,12))
        axs[0].set_xticklabels(np.arange(1,13))
        axs[0].tick_params(labelsize=fontsize)    
        axs[0].set_xlabel('Month', fontsize=fontsize)
        axs[0].set_ylabel(r'BFM: mean $X_{1,r}$, $^{\circ}$' + temperature_unit, fontsize=fontsize)
        axs[0].legend(loc='upper left', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=12)            
        axs[1].plot(SE1r_truth, marker='s', markersize=20, fillstyle='none', linestyle='none', label=r'Truth: $SE_{1,r}$')
        axs[1].plot(SE1r_CASE_1A, marker='^', markersize=10, alpha=0.5, label=r'Model 1A: $SE_{1,r}$ $\mu_{\epsilon}$=' + str(np.round(error_SE1r_CASE_1A,2)) + '$^{\circ}$' + temperature_unit) 
        axs[1].plot(SE1r_CASE_1B, marker='v', markersize=10, alpha=0.5, label=r'Model 1B: $SE_{1,r}$ $\mu_{\epsilon}$=' + str(np.round(error_SE1r_CASE_1B,2)) + '$^{\circ}$' + temperature_unit)
        axs[1].plot(SE1r_CASE_2A, marker='o', markersize=10, alpha=0.5, label=r'Model 2A: $SE_{1,r}$ $\mu_{\epsilon}$=' + str(np.round(error_SE1r_CASE_2A,2)) + '$^{\circ}$' + temperature_unit)
        axs[1].sharex(axs[0]) 
        axs[1].set_xticks(np.arange(0,12))
        axs[1].set_xticklabels(np.arange(1,13))
        axs[1].tick_params(labelsize=fontsize)    
        axs[1].set_xlabel('Month', fontsize=fontsize)
        axs[1].set_ylabel(r'BFM: standard error $SE_{1,r}$ , $^{\circ}$' + temperature_unit, fontsize=fontsize)
        axs[1].legend(loc='lower left', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=12)            
        if use_fahrenheit == True:
            axs[0].set_ylim(20,80)
            axs[1].set_ylim(0,2)
        else:
            ax.set_xlim(-20,40)
            ax.set_ylim(0,0.05)    
        plt.suptitle(titlestr)        
        fig.tight_layout()
        plt.savefig(figstr, dpi=300)
        plt.close('all')
    
    if plot_fit_model_1 == True: # one plot per ref_station
        
        print('plotting MODEL 1 (single neighbour) Xr vs Xa ...')
            
        figstr = 'model-1-fit' + '-' + test_station + '-' + ref_station + '.png'
        titlestr = 'Model fit: $X_{r}$ vs $X_{a}$ (ref = ' + ref_station.title() + ') for test station = ' + test_station.title()
            
        fig, axs = plt.subplots(figsize=(15,10))
        axs.plot(X_1.index, X_1, marker='.', color='lightgrey', alpha=1, label='$T_{g}$ ' + test_station)
        axs.plot(X_2a_CASE_1.index, X_2a_CASE_1, marker='.', color='lightblue', alpha=1, label='$X_{2,a}$ ' + ref_station + ' (segment)')
        axs.plot(X_2r_CASE_1.index, X_2r_CASE_1, marker='.', color='pink', alpha=1, label='$X_{2,r}$ ' + ref_station + ' (1961-1990)')
        axs.plot(X_1a.index, X_1a, marker='.', color='blue', alpha=1, label='$X_{1,a}$ ' + test_station + ' (segment)')
        axs.plot(X_1r_truth.index, X_1r_truth, marker='.', color='red', alpha=1, label='$X_{1,r}$ ' + test_station + ' (1961-1990) truth')
        axs.plot(X_1r_truth.index, X_1r_estimate_CASE_1, marker='.', color='k', alpha=1, label='$X_{1,r}$ ' + test_station + ' (1961-1990) estimate: $\epsilon$=' + str(np.round(error_CASE_1,2)) + '$^{\circ}$' + temperature_unit)   
        axs.plot(X_2a_CASE_1.index, len(X_2a_CASE_1)*[ np.nanmean( X_2a_CASE_1 ) ], ls='--', lw=2, color='lightblue', alpha=1)            
        axs.plot(X_2r_CASE_1.index, len(X_2r_CASE_1)*[ np.nanmean( X_2r_CASE_1 ) ], ls='--', lw=2, color='pink', alpha=1)            
        axs.plot(X_1a.index, len(X_1a)*[ np.nanmean( X_1a ) ], ls='--', lw=2, color='blue', alpha=1)            
        axs.plot(X_1r_truth.index, len(X_1r_truth)*[ np.nanmean( X_1r_truth ) ], ls='--', lw=2, color='red', alpha=1)            
        axs.plot(X_1r_truth.index, len(X_1r_estimate_CASE_1)*[ np.nanmean( x1r_CASE_1_normal ) ], ls='--', lw=2, color='k', alpha=1)           
        axs.legend(loc='upper left', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=12)        
        axs.set_xlabel('Year', fontsize=fontsize)
        axs.set_ylabel(r'Absolute temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
        axs.set_title(titlestr, fontsize=fontsize)
        axs.tick_params(labelsize=fontsize)   
        if use_fahrenheit == True:
            axs.set_ylim(42,54)
        else:
            ax.set_ylim(-20,40)
        fig.tight_layout()
        plt.savefig(figstr, dpi=300)
        plt.close('all')

# SAVE: model errors array
        
df_errors.to_csv('df_errors.csv')

#==============================================================================
                 
if plot_fit_model_2 == True: # one all neighbours plot
    
    print('plotting MODEL 2 (all neighbours) Xr vs Xa ...')
        
    X = df.rolling(nsmooth,center=True).mean()                                                          # All neighbours
    X_segment = X[ (X.index>=segment_start) & (X.index<=segment_end) ]                                  # All neighbours (segment)
    X_normal = X[ (X.index>=normal_start) & (X.index<=normal_end) ]                                     # All neighbours (normal)
        
    figstr = 'model-2-fit' + '-' + test_station + '-' + 'all-neighbours' + '.png'
    titlestr = 'Model fit: $X_{r}$ vs $X_{a}$ (ref = all neighbours) for test station = ' + test_station.title()
        
    fig, axs = plt.subplots(figsize=(15,10))
    axs.plot(X_segment.index, X_segment, marker='.', color='lightblue', alpha=1)
    axs.plot(X_normal.index, X_normal, marker='.', color='pink', alpha=1)
    axs.plot(X_1.index, X_1, marker='.', color='lightgrey', alpha=1, label='$T_{g}$ ' + test_station)
    axs.plot(X_2a_CASE_2.index, X_2a_CASE_2, marker='.', color='teal', alpha=1, label='$X_{2,a}$ all neighbours mean (segment)')
    axs.plot(X_2r_CASE_2.index, X_2r_CASE_2, marker='.', color='purple', alpha=1, label='$X_{2,r}$ all neighbours mean (1961-1990)')
    axs.plot(X_1a.index, X_1a, marker='.', color='blue', alpha=1, label='$X_{1,a}$ ' + test_station + ' (segment)')
    axs.plot(X_1r_truth.index, X_1r_truth, marker='.', color='red', alpha=1, label='$X_{1,r}$ ' + test_station + ' (1961-1990) truth')    
    axs.plot(X_1r_truth.index, X_1r_estimate_CASE_2, marker='.', color='k', alpha=1, label='$X_{1,r}$ ' + test_station + ' (1961-1990) estimate: $\epsilon$=' + str(np.round(error_CASE_2,2)) + '$^{\circ}$' + temperature_unit)    
    axs.plot(X_2a_CASE_2.index, len(X_2a_CASE_2)*[ np.nanmean( X_2a_CASE_2 ) ], ls='--', lw=2, color='teal', alpha=1)            
    axs.plot(X_2r_CASE_2.index, len(X_2r_CASE_2)*[ np.nanmean( X_2r_CASE_2 ) ], ls='--', lw=2, color='purple', alpha=1)            
    axs.plot(X_1a.index, len(X_1a)*[ np.nanmean( X_1a ) ], ls='--', lw=2, color='blue', alpha=1)            
    axs.plot(X_1r_truth.index, len(X_1r_truth)*[ np.nanmean( X_1r_truth ) ], ls='--', lw=2, color='red', alpha=1)            
    axs.plot(X_1r_truth.index, len(X_1r_estimate_CASE_2)*[ np.nanmean( x1r_CASE_2_normal ) ], ls='--', lw=2, color='k', alpha=1)           
    axs.legend(loc='upper left', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=12)        
    axs.set_xlabel('Year', fontsize=fontsize)
    axs.set_ylabel(r'Absolute temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs.set_title(titlestr, fontsize=fontsize)
    axs.tick_params(labelsize=fontsize)   
    if use_fahrenheit == True:
        axs.set_ylim(42,54)
    else:
        ax.set_ylim(-20,40)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')    
    
#------------------------------------------------------------------------------
print('** END')

# axs.plot(df.index, df.rolling(nsmooth,center=True).mean().ewm(span=nsmooth, adjust=True).mean(), marker='.', color='lightgrey', alpha=1.0, label=label)

