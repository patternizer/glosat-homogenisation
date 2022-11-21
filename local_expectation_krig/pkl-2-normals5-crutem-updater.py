#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: pkl-2-normals5-crtuem-updater.py
#
# updates CRUTEM normals5 station line when any months are missing: LEK flag=6 or 7 (see Processing Table)
#------------------------------------------------------------------------------
# Version 0.4
# 19 November, 2022
# Michael Taylor
# https://patternizer.github.io
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#------------------------------------------------------------------------------
# Dataframe libraries:
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------

fontsize = 16

lekfile = 'df_temp_expect.pkl'

normals5file = '../DATA/normals5.GloSAT.prelim04c.EBC_FRYuse_ocPLAUS1_iqr3.600reg0.3_19411990_MIN15_OCany_19611990_MIN15_PERDEC00_NManySDreq.txt' 	# INPUT
normals5file_lek = 'normals5_crutem_updated.txt'																									# OUTPUT

#------------------------------------------------------------------------------
# LOAD: LEK pickled pandas dataframe
#------------------------------------------------------------------------------

df_lek = pd.read_pickle( lekfile, compression='bz2' ) 

# DROP: stationcrustr (to not impact column indexing)

df_lek = df_lek.drop(columns='stationcrustr')

#------------------------------------------------------------------------------
# LOAD: normals5 file and extract fields
#------------------------------------------------------------------------------

stationcodelist = []
firstyearlist = []
lastyearlist = []
firstyear1961list = []
lastyear1990list = []
narray = []
flaglist = []
pcarray = []
crutemlist = []
   
with open (normals5file, 'r', encoding="ISO-8859-1") as f:                      
    for line in f:   

        if len(line)>1: # ignore empty lines         
            stationcode = line.split()[0]
            firstyear = int(line.split()[1])
            lastyear = int(line.split()[2])
            firstyear1961 = int(line.split()[3])
            lastyear1990 = int(line.split()[4])
            flag = int(line.split()[17])
            nvec = [ float(line.split()[5+i]) for i in range(12) ]
            pcvec = [ float(line.split()[18+i]) for i in range(12) ]
					                          
            stationcodelist.append( stationcode )
            firstyearlist.append( firstyear )                
            lastyearlist.append( lastyear )                
            firstyear1961list.append( firstyear1961 )                
            lastyear1990list.append( lastyear1990 )                
            narray.append( nvec )                
            flaglist.append( flag )                
            pcarray.append( pcvec )                            
            crutemlist.append( line[0:-1] )
        else:                
            continue
f.close

narray = np.array(narray)       # normals [n,12]
pcarray = np.array(pcarray)     # percentage [n,12]

#------------------------------------------------------------------------------
# WRITE: station header + yearly rows of monthly values in CRUTEM format
#------------------------------------------------------------------------------

with open(normals5file_lek,'w') as f:
    
    for k in range(len(stationcodelist)):
                        
        if len( df_lek[ df_lek['stationcode'] == stationcodelist[k] ]['year'] ) > 0:
                    
            # EXTRACT: CRUTEM station data
    
            stationcode = stationcodelist[k]
            firstyear = firstyearlist[k]
            lastyear = lastyearlist[k]
            firstyear1961 = firstyear1961list[k]
            lastyear1990 = lastyear1990list[k]            
            
            flag = flaglist[k]            
            nvec = narray[k,:]
            pcvec = pcarray[k,:]
                                                
            # EXTRACT: LEK station data array
    
            station_data_lek = df_lek[ (df_lek['stationcode'] == stationcodelist[k]) & ((df_lek['year']>= 1961) & ((df_lek['year']<= 1990))) ].reset_index(drop=True)
            nvec_lek = np.nanmean( station_data_lek.iloc[:,range(23,35)], axis=0 )

            # INITIALISE: normals vector and flag vector

            nvec_new = []
            flag_new = []
    
            for i in range(12):
                                    
                n_raw = np.count_nonzero( ~np.isnan( station_data_lek[str(i+1)] ) )
                n_lek = np.count_nonzero( ~np.isnan( station_data_lek['n'+str(i+1)] ) )
    
                if n_lek >= 15: # lek normal exists
    
                    if n_raw == 30:
                        nvec_new.append( nvec[i] )                    
                        flag_new.append( 4 )
                    elif n_raw == 0:
                        nvec_new.append( nvec_lek[i] )
                        flag_new.append( 6 )
                    elif (n_raw > 0) & (n_raw<30):
                        
                        # COMPUTE: LEK in-filled normal in actuals space
                        
                        ts_station_actuals = np.array(station_data_lek[str(i+1)]) 
                        if nvec[i] == -999:
                            normal_old = np.nan
                            ts_lek_actuals = np.array(station_data_lek['e'+str(i+1)]) + nvec_lek[i]                   
                        else:                        
                            normal_old = nvec[i]
                            ts_lek_actuals = np.array(station_data_lek['e'+str(i+1)]) + normal_old                   
                        missing = np.arange(30)[np.isnan(ts_station_actuals)]
                        ts_station_actuals_updated = np.array(station_data_lek[str(i+1)])
                        ts_station_actuals_updated[missing] = ts_lek_actuals[missing]
                        normal_new = np.mean(ts_station_actuals_updated)
                        nvec_new.append( normal_new ) 
                        flag_new.append( 7 )
                        
                        '''
                        # PLOT: station normal vs LEK-infilled normal
    
                        fig,ax = plt.subplots(figsize=(15,10))
                        plt.plot(np.arange(1961,1991), ts_station_actuals_updated, marker='.', markersize=10, color='r', ls='-', lw=0.5, label='Station data')
                        plt.plot(np.arange(1961,1991), ts_lek_actuals, marker='.', markersize=10, color='b', ls='-', lw=0.5, label='LEK estimates')
                        plt.axhline(y=normal_old, ls='dashed', color='r', label='Station normal='+str( np.round(normal_old,3) ) + r'$^{\circ}$C' )
                        plt.axhline(y=normal_new, ls='dashed', color='b', label='LEK-infilled normal='+str( np.round(normal_new,3) ) + r'$^{\circ}$C' )
                        plt.plot(np.arange(1961,1991)[missing], ts_lek_actuals[missing], marker='o', markersize=10, color='r', markerfacecolor="none", ls='none', label='LEK-infilled value(s): n=' + str(len(missing)) )
                        plt.legend( loc='upper left', fontsize=fontsize )
                        plt.xlabel( 'Year', fontsize=fontsize)
                        plt.ylabel( '2m Temperature, ' + r'$^{\circ}$C', fontsize=fontsize)
                        plt.title( 'LEK-infilled normal: ' + stationcodelist[k] + ': month=' + str(i+1).zfill(2), fontsize=fontsize)
                        plt.tick_params(labelsize=fontsize)    
                        plt.savefig( stationcodelist[k] + '-' + str(i+1).zfill(2) + '-' + 'lek-infilled-normal.png' ,dpi=300)
                        plt.close('all')
                        '''
                                            
                else:
    
                    if n_raw >= 15:
                        nvec_new.append( nvec[i] )                    
                        flag_new.append( 4 )
                    elif n_raw < 15:
                        if flag == 1:
                            nvec_new.append( np.nan )
                            flag_new.append( 1 )
                        else: 
                            nvec_new.append( nvec[i] )
                            flag_new.append( flag )
    
            nvec_new = np.array(nvec_new)    
            flag_new = np.array(flag_new)    
    
            # EXTRACT: model of source code flags (HadCRUT5-style)
    
            vals, counts = np.unique(flag_new, return_counts=True)         
            flag = vals[ np.argmax( counts ) ]           

            # EXTRACT: normals and percent strings

            nvec_str = ""      
            pcvec_str = ""

            for i in range(12):
                
                if nvec[i] < -500.0:
                    					
                    if np.isnan(nvec_new[i]): 
                        nvec_str += '{:.3f}'.format(np.round(-999.0,3)).rjust(9)
                    else:
                        nvec_str += '{:.3f}'.format(np.round(nvec_new[i],3)).rjust(9)

                    pcvec_str += '{:3}'.format(int(pcvec[i])).rjust(4)        

                else:					
                    nvec_str += '{:.3f}'.format(np.round(nvec_new[i],3)).rjust(9)					
                    pcvec_str += '{:3}'.format(int(pcvec[i])).rjust(4)        
                                   
            rowstr = stationcode + ' ' + str(firstyear) + ' ' + str(lastyear) + ' ' + str(firstyear1961) + ' '  + str(lastyear1990) + nvec_str + '   ' + str(flag) + pcvec_str
            
        else:
            rowstr = crutemlist[k]

        if k % 100 == 0: print(k, rowstr)

        f.write(rowstr+'\n')            

f.close
    
#------------------------------------------------------------------------------
print('** END')

