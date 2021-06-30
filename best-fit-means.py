#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: best-fit-means.py
#------------------------------------------------------------------------------
# Version 0.2
# 30 June, 2021
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
color_palette = 'viridis_r'
use_fahrenheit = True

load_glosat = True

if use_fahrenheit: 
    temperature_unit = 'F'
else:
    temperature_unit = 'C'

plot_monthly = True
plot_fit = True

#------------------------------------------------------------------------------
# METHODS: 
#------------------------------------------------------------------------------

def fahrenheit_to_centigrade(x):
    y = (5.0/9.0) * (x - 32.0)
    return y

def centigrade_to_fahrenheit(x):
    y = (x * (9.0/5.0)) + 32.0
    return y
    
#==============================================================================
# LOAD: Datasets
#==============================================================================

if load_glosat == True:
    
    #------------------------------------------------------------------------------    
    # LOAD: GloSAT absolute temperature and anomaly archives: CRUTEM5.0.1.0
    #------------------------------------------------------------------------------
        
    print('loading temperatures ...')
        
    # df = pd.read_csv(csv, index_col='date', parse_dates=True)
        
    df_temp = pd.read_pickle('DATA/df_temp.pkl', compression='bz2')    
    df_anom = pd.read_pickle('DATA/df_anom.pkl', compression='bz2')    
    
    stationcode_amherst = '720218'
    stationcode_bedford = '720219'
    stationcode_blue_hill = '744920'
    stationcode_boston_city_wso = '725092'
    stationcode_kingston = '753011'
    stationcode_lawrence = '720222'
    stationcode_new_bedford = '720223'
    stationcode_new_haven = '725045'    
    stationcode_plymouth_kingston = '756213'
    stationcode_providence_wso = '725070'
    stationcode_provincetown = '725091'
    stationcode_reading = '725090'
    stationcode_taunton = '720225'
    stationcode_walpole_2 = '744900'
    stationcode_west_medway = '744902'
    
    # GloSAT: absolutes

    da_amherst = df_temp[df_temp['stationcode']==stationcode_amherst]                       # USC00190120	AMHERST, MA US 1893-2021
    da_bedford = df_temp[df_temp['stationcode']==stationcode_bedford]                       # USC00190538	BEDFORD, MA US 1893-1923
    da_blue_hill = df_temp[df_temp['stationcode']==stationcode_blue_hill]                   # USC00190736	BLUE HILL COOP, MA US 1893-2021
    da_boston_city_wso = df_temp[df_temp['stationcode']==stationcode_boston_city_wso]       # USW00094701	BOSTON CITY WEATHER SERVICE OFFICE, MA US 1893-1935
    da_kingston = df_temp[df_temp['stationcode']==stationcode_kingston]                      
    da_lawrence = df_temp[df_temp['stationcode']==stationcode_lawrence]                     # USC00194105	LAWRENCE, MA US 1893-2021
    da_new_bedford = df_temp[df_temp['stationcode']==stationcode_new_bedford]            
    da_new_haven = df_temp[df_temp['stationcode']==stationcode_new_haven]                    
    da_plymouth_kingston = df_temp[df_temp['stationcode']==stationcode_plymouth_kingston]    
    da_providence_wso = df_temp[df_temp['stationcode']==stationcode_providence_wso]         # USC00376712	PROVIDENCE 2, RI US 1893-1913
    da_provincetown = df_temp[df_temp['stationcode']==stationcode_provincetown]             
    da_reading = df_temp[df_temp['stationcode']==stationcode_reading]                       
    da_taunton = df_temp[df_temp['stationcode']==stationcode_taunton]                       
    da_walpole_2 = df_temp[df_temp['stationcode']==stationcode_walpole_2]                   
    da_west_medway = df_temp[df_temp['stationcode']==stationcode_west_medway]        
             
    ts = np.array(da_amherst.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_amherst.year.iloc[0]), periods=len(ts), freq='MS')
    df_amherst_absolute = pd.DataFrame({'amherst':ts}, index=t) 
    ts = np.array(da_bedford.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_bedford.year.iloc[0]), periods=len(ts), freq='MS')
    df_bedford_absolute = pd.DataFrame({'bedford':ts}, index=t) 
    ts = np.array(da_blue_hill.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_blue_hill.year.iloc[0]), periods=len(ts), freq='MS')
    df_blue_hill_absolute = pd.DataFrame({'blue_hill':ts}, index=t)    
    ts = np.array(da_boston_city_wso.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_boston_city_wso.year.iloc[0]), periods=len(ts), freq='MS')
    df_boston_city_wso_absolute = pd.DataFrame({'boston_city_wso':ts}, index=t)        
    ts = np.array(da_kingston.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_kingston.year.iloc[0]), periods=len(ts), freq='MS')
    df_kingston_absolute = pd.DataFrame({'kingston':ts}, index=t) 
    ts = np.array(da_lawrence.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_lawrence.year.iloc[0]), periods=len(ts), freq='MS')
    df_lawrence_absolute = pd.DataFrame({'lawrence':ts}, index=t) 
    ts = np.array(da_new_bedford.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_new_bedford.year.iloc[0]), periods=len(ts), freq='MS')
    df_new_bedford_absolute = pd.DataFrame({'new_bedford':ts}, index=t) 
    ts = np.array(da_new_haven.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_new_haven.year.iloc[0]), periods=len(ts), freq='MS')
    df_new_haven_absolute = pd.DataFrame({'new_haven':ts}, index=t)     
    ts = np.array(da_plymouth_kingston.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_plymouth_kingston.year.iloc[0]), periods=len(ts), freq='MS')
    df_plymouth_kingston_absolute = pd.DataFrame({'plymouth_kingston':ts}, index=t)     
    ts = np.array(da_providence_wso.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_providence_wso.year.iloc[0]), periods=len(ts), freq='MS')
    df_providence_wso_absolute = pd.DataFrame({'providence_wso':ts}, index=t) 
    ts = np.array(da_provincetown.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_provincetown.year.iloc[0]), periods=len(ts), freq='MS')
    df_provincetown_absolute = pd.DataFrame({'provincetown':ts}, index=t) 
    ts = np.array(da_reading.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_reading.year.iloc[0]), periods=len(ts), freq='MS')
    df_reading_absolute = pd.DataFrame({'reading':ts}, index=t) 
    ts = np.array(da_taunton.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_taunton.year.iloc[0]), periods=len(ts), freq='MS')
    df_taunton_absolute = pd.DataFrame({'taunton':ts}, index=t) 
    ts = np.array(da_walpole_2.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_walpole_2.year.iloc[0]), periods=len(ts), freq='MS')
    df_walpole_2_absolute = pd.DataFrame({'walpole_2':ts}, index=t) 
    ts = np.array(da_west_medway.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_west_medway.year.iloc[0]), periods=len(ts), freq='MS')
    df_west_medway_absolute = pd.DataFrame({'west_medway':ts}, index=t) 

    # GloSAT: anomalies

    da_amherst = df_anom[df_anom['stationcode']==stationcode_amherst]                       # USC00190120	AMHERST, MA US 1893-2021
    da_bedford = df_anom[df_anom['stationcode']==stationcode_bedford]                       # USC00190538	BEDFORD, MA US 1893-1923
    da_blue_hill = df_anom[df_anom['stationcode']==stationcode_blue_hill]                   # USC00190736	BLUE HILL COOP, MA US 1893-2021
    da_boston_city_wso = df_anom[df_anom['stationcode']==stationcode_boston_city_wso]       # USW00094701	BOSTON CITY WEATHER SERVICE OFFICE, MA US 1893-1935
    da_kingston = df_anom[df_anom['stationcode']==stationcode_kingston]                      
    da_lawrence = df_anom[df_anom['stationcode']==stationcode_lawrence]                     # USC00194105	LAWRENCE, MA US 1893-2021
    da_new_bedford = df_anom[df_anom['stationcode']==stationcode_new_bedford]            
    da_new_haven = df_anom[df_anom['stationcode']==stationcode_new_haven]                    
    da_plymouth_kingston = df_anom[df_anom['stationcode']==stationcode_plymouth_kingston]    
    da_providence_wso = df_anom[df_anom['stationcode']==stationcode_providence_wso]         # USC00376712	PROVIDENCE 2, RI US 1893-1913
    da_provincetown = df_anom[df_anom['stationcode']==stationcode_provincetown]             
    da_reading = df_anom[df_anom['stationcode']==stationcode_reading]                       
    da_taunton = df_anom[df_anom['stationcode']==stationcode_taunton]                       
    da_walpole_2 = df_anom[df_anom['stationcode']==stationcode_walpole_2]                   
    da_west_medway = df_anom[df_anom['stationcode']==stationcode_west_medway]        

    ts = np.array(da_amherst.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_amherst.year.iloc[0]), periods=len(ts), freq='MS')
    df_amherst_anomaly = pd.DataFrame({'amherst':ts}, index=t) 
    ts = np.array(da_bedford.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_bedford.year.iloc[0]), periods=len(ts), freq='MS')
    df_bedford_anomaly = pd.DataFrame({'bedford':ts}, index=t) 
    ts = np.array(da_blue_hill.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_blue_hill.year.iloc[0]), periods=len(ts), freq='MS')
    df_blue_hill_anomaly = pd.DataFrame({'blue_hill':ts}, index=t)    
    ts = np.array(da_boston_city_wso.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_boston_city_wso.year.iloc[0]), periods=len(ts), freq='MS')
    df_boston_city_wso_anomaly = pd.DataFrame({'boston_city_wso':ts}, index=t)        
    ts = np.array(da_kingston.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_kingston.year.iloc[0]), periods=len(ts), freq='MS')
    df_kingston_anomaly = pd.DataFrame({'kingston':ts}, index=t) 
    ts = np.array(da_lawrence.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_lawrence.year.iloc[0]), periods=len(ts), freq='MS')
    df_lawrence_anomaly = pd.DataFrame({'lawrence':ts}, index=t) 
    ts = np.array(da_new_bedford.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_new_bedford.year.iloc[0]), periods=len(ts), freq='MS')
    df_new_bedford_anomaly = pd.DataFrame({'new_bedford':ts}, index=t) 
    ts = np.array(da_new_haven.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_new_haven.year.iloc[0]), periods=len(ts), freq='MS')
    df_new_haven_anomaly = pd.DataFrame({'new_haven':ts}, index=t)     
    ts = np.array(da_plymouth_kingston.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_plymouth_kingston.year.iloc[0]), periods=len(ts), freq='MS')
    df_plymouth_kingston_anomaly = pd.DataFrame({'plymouth_kingston':ts}, index=t)     
    ts = np.array(da_providence_wso.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_providence_wso.year.iloc[0]), periods=len(ts), freq='MS')
    df_providence_wso_anomaly = pd.DataFrame({'providence_wso':ts}, index=t) 
    ts = np.array(da_provincetown.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_provincetown.year.iloc[0]), periods=len(ts), freq='MS')
    df_provincetown_anomaly = pd.DataFrame({'provincetown':ts}, index=t) 
    ts = np.array(da_reading.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_reading.year.iloc[0]), periods=len(ts), freq='MS')
    df_reading_anomaly = pd.DataFrame({'reading':ts}, index=t) 
    ts = np.array(da_taunton.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_taunton.year.iloc[0]), periods=len(ts), freq='MS')
    df_taunton_anomaly = pd.DataFrame({'taunton':ts}, index=t) 
    ts = np.array(da_walpole_2.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_walpole_2.year.iloc[0]), periods=len(ts), freq='MS')
    df_walpole_2_anomaly = pd.DataFrame({'walpole_2':ts}, index=t) 
    ts = np.array(da_west_medway.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_west_medway.year.iloc[0]), periods=len(ts), freq='MS')
    df_west_medway_anomaly = pd.DataFrame({'west_medway':ts}, index=t) 
    
#------------------------------------------------------------------------------
# CONVERT: to Fahrenheit is selected
#------------------------------------------------------------------------------

if use_fahrenheit == True:
    
    df_amherst_absolute = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_amherst_absolute['amherst'] )})
    df_bedford_absolute = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_bedford_absolute['bedford'] )})
    df_blue_hill_absolute = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_blue_hill_absolute['blue_hill'] )})      
    df_boston_city_wso_absolute = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_boston_city_wso_absolute['boston_city_wso'] )})       
    df_kingston_absolute = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_kingston_absolute['kingston'] )})
    df_lawrence_absolute = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_lawrence_absolute['lawrence'] )})
    df_new_bedford_absolute = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_new_bedford_absolute['new_bedford'] )})
    df_new_haven_absolute = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_new_haven_absolute['new_haven'] )})
    df_plymouth_kingston_absolute = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_plymouth_kingston_absolute['plymouth_kingston'] )})   
    df_providence_wso_absolute = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_providence_wso_absolute['providence_wso'] )})
    df_provincetown_absolute = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_provincetown_absolute['provincetown'] )})
    df_reading_absolute = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_reading_absolute['reading'] )})
    df_taunton_absolute = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_taunton_absolute['taunton'] )})
    df_walpole_2_absolute = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_walpole_2_absolute['walpole_2'] )})
    df_west_medway_absolute = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_west_medway_absolute['west_medway'] )})

    df_amherst_anomaly = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_amherst_anomaly['amherst'] )-32 })
    df_bedford_anomaly = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_bedford_anomaly['bedford'] )-32 })
    df_blue_hill_anomaly = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_blue_hill_anomaly['blue_hill'] )-32 })      
    df_boston_city_wso_anomaly = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_boston_city_wso_anomaly['boston_city_wso'] )-32 })       
    df_kingston_anomaly = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_kingston_anomaly['kingston'] )-32 })
    df_lawrence_anomaly = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_lawrence_anomaly['lawrence'] )-32 })
    df_new_bedford_anomaly = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_new_bedford_anomaly['new_bedford'] )-32 })
    df_new_haven_anomaly = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_new_haven_anomaly['new_haven'] )-32 })
    df_plymouth_kingston_anomaly = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_plymouth_kingston_anomaly['plymouth_kingston'] )-32 })   
    df_providence_wso_anomaly = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_providence_wso_anomaly['providence_wso'] )-32 })
    df_provincetown_anomaly = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_provincetown_anomaly['provincetown'] )-32 })
    df_reading_anomaly = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_reading_anomaly['reading'] )-32 })
    df_taunton_anomaly = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_taunton_anomaly['taunton'] )-32 })
    df_walpole_2_anomaly = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_walpole_2_anomaly['walpole_2'] )-32 })
    df_west_medway_anomaly = pd.DataFrame({'Tg':centigrade_to_fahrenheit( df_west_medway_anomaly['west_medway'] )-32 })
        
else:

    df_amherst_absolute = pd.DataFrame({'Tg':( df_amherst_absolute['amherst'] )})
    df_bedford_absolute = pd.DataFrame({'Tg':( df_bedford_absolute['bedford'] )})
    df_blue_hill_absolute = pd.DataFrame({'Tg':( df_blue_hill_absolute['blue_hill'] )})      
    df_boston_city_wso_absolute = pd.DataFrame({'Tg':( df_boston_city_wso_absolute['boston_city_wso'] )})       
    df_kingston_absolute = pd.DataFrame({'Tg':( df_kingston_absolute['kingston'] )})
    df_lawrence_absolute = pd.DataFrame({'Tg':( df_lawrence_absolute['lawrence'] )})
    df_new_bedford_absolute = pd.DataFrame({'Tg':( df_new_bedford_absolute['new_bedford'] )})
    df_new_haven_absolute = pd.DataFrame({'Tg':( df_new_haven_absolute['new_haven'] )})
    df_plymouth_kingston_absolute = pd.DataFrame({'Tg':( df_plymouth_kingston_absolute['plymouth_kingston'] )})   
    df_providence_wso_absolute = pd.DataFrame({'Tg':( df_providence_wso_absolute['providence_wso'] )})
    df_provincetown_absolute = pd.DataFrame({'Tg':( df_provincetown_absolute['provincetown'] )})
    df_reading_absolute = pd.DataFrame({'Tg':( df_reading_absolute['reading'] )})
    df_taunton_absolute = pd.DataFrame({'Tg':( df_taunton_absolute['taunton'] )})
    df_walpole_2_absolute = pd.DataFrame({'Tg':( df_walpole_2_absolute['walpole_2'] )})
    df_west_medway_absolute = pd.DataFrame({'Tg':( df_west_medway_absolute['west_medway'] )})

    df_amherst_anomaly = pd.DataFrame({'Tg':( df_amherst_anomaly['amherst'] )-32 })
    df_bedford_anomaly = pd.DataFrame({'Tg':( df_bedford_anomaly['bedford'] )-32 })
    df_blue_hill_anomaly = pd.DataFrame({'Tg':( df_blue_hill_anomaly['blue_hill'] )-32 })      
    df_boston_city_wso_anomaly = pd.DataFrame({'Tg':( df_boston_city_wso_anomaly['boston_city_wso'] )-32 })       
    df_kingston_anomaly = pd.DataFrame({'Tg':( df_kingston_anomaly['kingston'] )-32 })
    df_lawrence_anomaly = pd.DataFrame({'Tg':( df_lawrence_anomaly['lawrence'] )-32 })
    df_new_bedford_anomaly = pd.DataFrame({'Tg':( df_new_bedford_anomaly['new_bedford'] )-32 })
    df_new_haven_anomaly = pd.DataFrame({'Tg':( df_new_haven_anomaly['new_haven'] )-32 })
    df_plymouth_kingston_anomaly = pd.DataFrame({'Tg':( df_plymouth_kingston_anomaly['plymouth_kingston'] )-32 })   
    df_providence_wso_anomaly = pd.DataFrame({'Tg':( df_providence_wso_anomaly['providence_wso'] )-32 })
    df_provincetown_anomaly = pd.DataFrame({'Tg':( df_provincetown_anomaly['provincetown'] )-32 })
    df_reading_anomaly = pd.DataFrame({'Tg':( df_reading_anomaly['reading'] )-32 })
    df_taunton_anomaly = pd.DataFrame({'Tg':( df_taunton_anomaly['taunton'] )-32 })
    df_walpole_2_anomaly = pd.DataFrame({'Tg':( df_walpole_2_anomaly['walpole_2'] )-32 })
    df_west_medway_anomaly = pd.DataFrame({'Tg':( df_west_medway_anomaly['west_medway'] )-32 })
        
#------------------------------------------------------------------------------
# MARGE: stations into dataframe
#------------------------------------------------------------------------------

dates = pd.date_range(start='1700-01-01', end='2021-12-01', freq='MS')
df = pd.DataFrame(index=dates)
df['amherst'] = df_amherst_absolute
df['bedford'] = df_bedford_absolute
df['blue_hill'] = df_blue_hill_absolute
df['boston_city_wso'] = df_boston_city_wso_absolute
df['kingston'] = df_kingston_absolute
df['lawrence'] = df_lawrence_absolute
df['new_bedford'] = df_new_bedford_absolute
df['new_haven'] = df_new_haven_absolute
df['plymouth_kingston'] = df_plymouth_kingston_absolute
df['providence_wso'] = df_providence_wso_absolute
df['provincetown'] = df_provincetown_absolute
df['reading'] = df_reading_absolute
df['taunton'] = df_taunton_absolute
df['walpole_2'] = df_walpole_2_absolute
df['west_medway'] = df_west_medway_absolute

df_anomaly = pd.DataFrame(index=dates)
df_anomaly['amherst'] = df_amherst_anomaly
df_anomaly['bedford'] = df_bedford_anomaly
df_anomaly['blue_hill'] = df_blue_hill_anomaly
df_anomaly['boston_city_wso'] = df_boston_city_wso_anomaly
df_anomaly['kingston'] = df_kingston_anomaly
df_anomaly['lawrence'] = df_lawrence_anomaly
df_anomaly['new_bedford'] = df_new_bedford_anomaly
df_anomaly['new_haven'] = df_new_haven_anomaly
df_anomaly['plymouth_kingston'] = df_plymouth_kingston_anomaly
df_anomaly['providence_wso'] = df_providence_wso_anomaly
df_anomaly['provincetown'] = df_provincetown_anomaly
df_anomaly['reading'] = df_reading_anomaly
df_anomaly['taunton'] = df_taunton_anomaly
df_anomaly['walpole_2'] = df_walpole_2_anomaly
df_anomaly['west_medway'] = df_west_medway_anomaly

#------------------------------------------------------------------------------
# SLICE: to segment
#------------------------------------------------------------------------------

segment_start = pd.to_datetime('1850-01-01')
segment_end = pd.to_datetime('1899-12-01')
normal_start = pd.to_datetime('1961-01-01')
normal_end = pd.to_datetime('1990-12-01')
df_segment = df[ (df.index>=segment_start) & (df.index<=segment_end) ]
df_normal = df[ (df.index>=normal_start) & (df.index<=normal_end) ]
df_anomaly_normal = df_anomaly[ (df_anomaly.index>=normal_start) & (df_anomaly.index<=normal_end) ]

#------------------------------------------------------------------------------
# CALCULATE: segment mean for test station (excluded from mean)
#------------------------------------------------------------------------------

ref_station = 'blue_hill'
test_station = 'new_haven'

df_ref_station_segment = df_segment[ref_station]
df_test_station_segment = df_segment[test_station]
df_neighbours_segment = df_segment.drop([ref_station], axis=1)
df_neighbours_segment_mean = df_neighbours_segment.mean(axis=1)

df_ref_station_normal = df_normal[ref_station]
df_test_station_normal = df_normal[test_station]
df_test_station_anomaly = df_anomaly_normal[test_station]

#------------------------------------------------------------------------------
# MODEL 1: single reference: x1=New Haven, x2=BHO
#------------------------------------------------------------------------------

a1 = df_test_station_segment
a2 = df_ref_station_segment
a12 = df_test_station_segment - df_ref_station_segment
r2 = df_ref_station_normal

x1rA = []
x1rB = []
SE1rA = []
SE1rB = []
for i in range(12):

    x1a = np.nanmean(a1[a1.index.month==(i+1)])
    x2a = np.nanmean(a2[a2.index.month==(i+1)])
    x12a = np.nanmean(a12[a12.index.month==(i+1)])
    x2r = np.nanmean(r2[r2.index.month==(i+1)])

    SE1a = np.nanstd( a1[a1.index.month==(i+1)] ) / np.sqrt( np.isfinite(a1[a1.index.month==(i+1)]).sum() )
    SE2a = np.nanstd( a2[a2.index.month==(i+1)] ) / np.sqrt( np.isfinite(a2[a2.index.month==(i+1)]).sum() )
    SE12a = np.nanstd( a12[a12.index.month==(i+1)] ) / np.sqrt( np.isfinite(a12[a12.index.month==(i+1)]).sum() )
    SE2r = np.nanstd( r2[r2.index.month==(i+1)] ) / np.sqrt( np.isfinite(r2[r2.index.month==(i+1)]).sum() )

    x1rA_month = x1a + (x2r-x2a)
    x1rB_month = x2r + x12a
    SE1rA_month = np.sqrt( SE1a**2. + SE2r**2. + SE2a**2. )
    SE1rB_month = np.sqrt( SE2r**2. + SE12a**2. )

    x1rA.append(x1rA_month)
    x1rB.append(x1rB_month)
    SE1rA.append(SE1rA_month)
    SE1rB.append(SE1rB_month)
        
x1a_full = np.nanmean( a1 )
x2a_full = np.nanmean( a2 )
x12a_full = np.nanmean( a12 )
x2r_full = np.nanmean( r2 )
x1rA_full = x1a_full + (x2r_full-x2a_full)
x1rB_full = x2r_full + x12a_full

SE1a_full = np.nanstd( a1 ) / np.sqrt(np.isfinite( a1 ).sum())
SE2a_full = np.nanstd( a2 ) / np.sqrt(np.isfinite( a2 ).sum())
SE12a_full = np.nanstd( a12 ) / np.sqrt(np.isfinite( a12 ).sum())
SE2r_full = np.nanstd( r2 ) / np.sqrt(np.isfinite( r2 ).sum())
#SE1r_full = np.sqrt( SE1a**2. + SE2r**2. + SE2a**2. )
SE1r_full = np.sqrt( SE2r**2. + SE12a**2. )

#------------------------------------------------------------------------------
# MODEL 2A: multple co-located neighbours x1=mean, x2=BHO
#------------------------------------------------------------------------------

a1 = df_neighbours_segment_mean
a2 = df_ref_station_segment
a12 = df_neighbours_segment_mean - df_ref_station_segment
r2 = df_ref_station_normal

x1r = []
SE1r = []
for i in range(12):

    x1a = np.nanmean(a1[a1.index.month==(i+1)])
    x2a = np.nanmean(a2[a2.index.month==(i+1)])
    x12a = np.nanmean(a12[a12.index.month==(i+1)])
    x2r = np.nanmean(r2[r2.index.month==(i+1)])

    SE1a = np.nanstd( a1[a1.index.month==(i+1)] ) / np.sqrt( np.isfinite(a1[a1.index.month==(i+1)]).sum() )
    SE2a = np.nanstd( a2[a2.index.month==(i+1)] ) / np.sqrt( np.isfinite(a2[a2.index.month==(i+1)]).sum() )
    SE12a = np.nanstd( a12[a12.index.month==(i+1)] ) / np.sqrt( np.isfinite(a12[a12.index.month==(i+1)]).sum() )
    SE2r = np.nanstd( r2[r2.index.month==(i+1)] ) / np.sqrt( np.isfinite(r2[r2.index.month==(i+1)]).sum() )

#   x1r_month = x1a + (x2r-x2a)
    x1r_month = x2r + x12a
#   SE1r_month = np.sqrt( SE1a**2. + SE2r**2. + SE2a**2. )
    SE1r_month = np.sqrt( SE2r**2. + SE12a**2. )

    x1r.append(x1r_month)
    SE1r.append(SE1r_month)
        
x1a_full = np.nanmean( a1 )
x2a_full = np.nanmean( a2 )
x12a_full = np.nanmean( a12 )
x2r_full = np.nanmean( r2 )
# x1r_full = x1a_full + (x2r_full-x2a_full)
x1r_full = x2r_full + x12a_full

SE1a_full = np.nanstd( a1 ) / np.sqrt(np.isfinite( a1 ).sum())
SE2a_full = np.nanstd( a2 ) / np.sqrt(np.isfinite( a2 ).sum())
SE12a_full = np.nanstd( a12 ) / np.sqrt(np.isfinite( a12 ).sum())
SE2r_full = np.nanstd( r2 ) / np.sqrt(np.isfinite( r2 ).sum())
#SE1r_full = np.sqrt( SE1a**2. + SE2r**2. + SE2a**2. )
SE1r_full = np.sqrt( SE2r**2. + SE12a**2. )

#==============================================================================

if plot_monthly == True:
    
    # PLOT: monthly normals and standard errors
    
    print('plotting x1r and SE1r (A vs B) ...')
        
    figstr = 'monthly_normals_sterr.png'
#    titlestr = 'CASE 1: single reference: x1r and SE1r (A vs B)'
    titlestr = 'Monthly x1r and SE1r'
    
    fig, axs = plt.subplots(2,1, figsize=(15,10))
    axs[0].plot(x1rA, marker='o', markersize=10, alpha=0.5, label=r'Model 1A: $\bar{X}_{1,r}$')
    axs[0].plot(x1rB, marker='o', markersize=10, ls='None', alpha=0.5, label=r'Model 1B: $\bar{X}_{1,r}$')
    axs[0].plot(x1r, marker='o', markersize=10, alpha=0.5, label=r'Model 2A: $\bar{X}_{1,r}$')
    axs[0].set_xticks(np.arange(0,12))
    axs[0].set_xticklabels(np.arange(1,13))
    axs[0].tick_params(labelsize=fontsize)    
    axs[0].set_xlabel('Month', fontsize=fontsize)
    axs[0].set_ylabel(r'BFM: mean $X_{1,r}$, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs[0].legend(loc='upper left', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=12)            
    axs[1].plot(SE1rA, marker='o', markersize=10, alpha=0.5, label=r'Model 1A: $SE_{1,r}$')
    axs[1].plot(SE1rB, marker='o', markersize=10, alpha=0.5, label=r'Model 1B: $SE_{1,r}$')
    axs[1].plot(SE1r, marker='o', markersize=10, alpha=0.5, label=r'Model 2A: $SE_{1,r}$')
    axs[1].sharex(axs[0]) 
    axs[1].set_xticks(np.arange(0,12))
    axs[1].set_xticklabels(np.arange(1,13))
    axs[1].tick_params(labelsize=fontsize)    
    axs[1].set_xlabel('Month', fontsize=fontsize)
    axs[1].set_ylabel(r'BFM: standard error $SE_{1,r}$ , $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs[1].legend(loc='lower left', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=12)            
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')
   
if plot_fit == True:

    print('plotting Xr vs Xa ...')
        
    figstr = 'model_fit.png'
#    titlestr = 'CASE 1: single reference: $X_{r}$ vs $X_{a}$'
    titlestr = 'Model fit: $X_{r}$ vs $X_{a}$'
    
    fig, axs = plt.subplots(figsize=(15,10))
    axs.plot(df[ref_station].index, df[ref_station].rolling(12).mean(), marker='.', color='lightgrey', alpha=1.0, label='$T_{g}$ ' + ref_station)
    axs.plot(df_ref_station_segment.index, df_ref_station_segment.rolling(12).mean(), marker='.', color='pink', alpha=1, label='$X_{2,a}$ ' + ref_station + ' (segment)')
    axs.plot(df_ref_station_normal.index, df_ref_station_normal.rolling(12).mean(), marker='.', color='red', alpha=0.5, label='$X_{2,r}$ ' + ref_station + ' (1961-1990)')
    axs.plot(df_test_station_segment.index, df_test_station_segment.rolling(12).mean(), marker='.', color='cyan', alpha=1, label='$X_{1,a}$ ' + test_station + ' (segment)')
    axs.plot(df_test_station_normal.index, df_test_station_normal.rolling(12).mean(), marker='.', color='blue', alpha=0.5, label='$X_{1,r}$ ' + test_station + ' (1961-1990)')
    axs.plot(df_test_station_anomaly.index, df_test_station_anomaly.rolling(12).mean() + x1rA_full, marker='.', color='k', alpha=0.5, label='$X_{1,r}$ ' + test_station + ' (1961-1990) estimate')

    axs.plot(a1.index, len(a1)*[ np.nanmean( a1 ) ], ls='--', lw=2, color='cyan', alpha=1)            
    axs.plot(a2.index, len(a2)*[ np.nanmean( a2 ) ], ls='--', lw=2, color='pink', alpha=1)            
    axs.plot(r2.index, len(r2)*[ np.nanmean( r2 ) ], ls='--', lw=2, color='red', alpha=1)            
    axs.plot(r2.index, len(r2)*[ np.nanmean( df_test_station_normal ) ], ls='--', lw=2, color='blue', alpha=1)            
    axs.plot(r2.index, len(r2)*[ x1rA_full ], ls='--', lw=2, color='k', alpha=1)            

    axs.legend(loc='upper left', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=12)        
    axs.set_xlabel('Year', fontsize=fontsize)
    axs.set_ylabel(r'Absolute temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs.set_title(titlestr, fontsize=fontsize)
    axs.tick_params(labelsize=fontsize)   

    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')
                                
#------------------------------------------------------------------------------
print('** END')


