import os
import glob
import pandas as pd

pkllist = glob.glob("*.pkl")
df_temp_expect = pd.concat([pd.read_pickle(pkllist[i], compression='bz2') for i in range(len(pkllist))])
df_temp_expect.to_pickle('df_temp_expect.pkl', compression='bz2')


