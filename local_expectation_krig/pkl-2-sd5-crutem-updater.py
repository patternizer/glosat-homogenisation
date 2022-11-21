#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: pkl-2-sd5-crtuem-updater.py
#
# updates CRUTEM sd5 station line when any months are missing: LEK flag=6 or 7 (see Processing Table)
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

sd5file = '../DATA/sd5.GloSAT.prelim04c.EBC_FRYuse_ocPLAUS1_iqr3.600reg0.3_19411990_MIN15_OCany_19611990_MIN15_PERDEC00_NManySDreq.txt' 	# INPUT
sd5file_lek = 'sd5_crutem_updated.txt'																									# OUTPUT

#------------------------------------------------------------------------------
# LOAD: LEK pickled pandas dataframe
#------------------------------------------------------------------------------

df_lek = pd.read_pickle( lekfile, compression='bz2' ) 

# DROP: stationcrustr (to not impact column indexing)

df_lek = df_lek.drop(columns='stationcrustr')

#------------------------------------------------------------------------------
# LOAD: sd5 file and extract fields
#------------------------------------------------------------------------------

stationcodelist = []
firstyearlist = []
lastyearlist = []
firstyear1941list = []
lastyear1990list = []
sdarray = []
flaglist = []
pcarray = []
crutemlist = []
   
with open (sd5file, 'r', encoding="ISO-8859-1") as f:                      
    for line in f:   

        if len(line)>1: # ignore empty lines         
            stationcode = line.split()[0]
            firstyear = int(line.split()[1])
            lastyear = int(line.split()[2])
            firstyear1941 = int(line.split()[3])
            lastyear1990 = int(line.split()[4])
            flag = int(line.split()[17])
            sdvec = [ float(line.split()[5+i]) for i in range(12) ]
            pcvec = [ float(line.split()[18+i]) for i in range(12) ]
					                          
            stationcodelist.append( stationcode )
            firstyearlist.append( firstyear )                
            lastyearlist.append( lastyear )                
            firstyear1941list.append( firstyear1941 )                
            lastyear1990list.append( lastyear1990 )                
            sdarray.append( sdvec )                
            flaglist.append( flag )                
            pcarray.append( pcvec )                            
            crutemlist.append( line[0:-1] )
        else:                
            continue
f.close

sdarray = np.array(sdarray)     # SDs [n,12]
pcarray = np.array(pcarray)     # percentage [n,12]

#------------------------------------------------------------------------------
# WRITE: station header + yearly rows of monthly values in CRUTEM format
#------------------------------------------------------------------------------

with open(sd5file_lek,'w') as f:
    
    for k in range(len(stationcodelist)):
                        
        if len( df_lek[ df_lek['stationcode'] == stationcodelist[k] ]['year'] ) > 0:
                    
            # EXTRACT: CRUTEM station data
    
            stationcode = stationcodelist[k]
            firstyear = firstyearlist[k]
            lastyear = lastyearlist[k]
            firstyear1941 = firstyear1941list[k]
            lastyear1990 = lastyear1990list[k]            
            
            flag = flaglist[k]            
            sdvec = sdarray[k,:]
            pcvec = pcarray[k,:]
                                                
            # EXTRACT: LEK station data array
    
            station_data_lek_full = df_lek[ (df_lek['stationcode'] == stationcodelist[k]) & ((df_lek['year']>= 1781) & ((df_lek['year']<= 2022))) ].reset_index(drop=True)
            station_data_lek = df_lek[ (df_lek['stationcode'] == stationcodelist[k]) & ((df_lek['year']>= 1941) & ((df_lek['year']<= 1990))) ].reset_index(drop=True)

            # COMPUTE: LEK SDs from 1941-1990 local expectations

            sdvec_full = np.nanstd( station_data_lek_full.iloc[:,range(1,13)], axis=0 ) # 1781-2022 data SDs
            sdvec_lek = np.nanstd( station_data_lek.iloc[:,range(47,59)], axis=0 )
            pcvec_lek = np.array( np.isfinite( station_data_lek.iloc[:,range(47,59)]).sum()/50*100 )                      
            
            # INITIALISE: SDs vector and flag vector

            sdvec_new = []
            flag_new = []
    
            for i in range(12):
                                    
                n_raw_full = np.count_nonzero( ~np.isnan( station_data_lek_full[str(i+1)] ) )
                n_raw = np.count_nonzero( ~np.isnan( station_data_lek[str(i+1)] ) )
                n_lek = np.count_nonzero( ~np.isnan( station_data_lek['n'+str(i+1)] ) )
    
                if n_raw < 15: # actual values
    
                    if (flag >= 2) & (flag <= 5): # use existing SD

                        sdvec_new.append( sdvec[i] )
                        flag_new.append( flag )

                    elif flag == 1: 

                        if n_raw_full >= 10: # use 1781-2022 data to compute SD 

                            sdvec_new.append( sdvec_full[i] )
                            flag_new.append( 4 )

                        elif n_raw_full < 10:

                            if np.isfinite( sdvec_lek[i] ): # use LEK SD

                                sdvec_new.append( sdvec_lek[i] )
                                flag_new.append( 6 )

                            else: # missing

                                sdvec_new.append( np.nan )                    
                                flag_new.append( 1 )
                                           
                else:
    
                    if n_raw >= 15:
                        
                        sdvec_new.append( sdvec[i] )                    
                        flag_new.append( 4 )
                            
            sdvec_new = np.array(sdvec_new)    
            flag_new = np.array(flag_new)    
    
            # EXTRACT: model of source code flags (HadCRUT5-style)
    
            vals, counts = np.unique(flag_new, return_counts=True)         
            flag = vals[ np.argmax( counts ) ]           

            # EXTRACT: SDs and percent strings

            sdvec_str = ""      
            pcvec_str = ""

            for i in range(12):
                
                if sdvec[i] < -500.0:
                    					
                    if np.isnan(sdvec_new[i]): 
                        sdvec_str += '{:.3f}'.format(np.round(-999.0,3)).rjust(9)
                    else:
                        sdvec_str += '{:.3f}'.format(np.round(sdvec_new[i],3)).rjust(9)

                    pcvec_str += '{:3}'.format(int(pcvec[i])).rjust(4)        

                else:					
                    sdvec_str += '{:.3f}'.format(np.round(sdvec_new[i],3)).rjust(9)					
                    pcvec_str += '{:3}'.format(int(pcvec[i])).rjust(4)        
                                   
            rowstr = stationcode + ' ' + str(firstyear) + ' ' + str(lastyear) + ' ' + str(firstyear1941) + ' '  + str(lastyear1990) + sdvec_str + '   ' + str(flag) + pcvec_str
            
        else:
            rowstr = crutemlist[k]

        if k % 100 == 0: print(k, rowstr)

        f.write(rowstr+'\n')            

f.close
    
#------------------------------------------------------------------------------
print('** END')

