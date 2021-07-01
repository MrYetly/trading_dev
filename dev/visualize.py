#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:36:52 2021

@author: ianich
"""

import pandas as pd
import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.regression.linear_model import OLS
from statsmodels.iolib.summary2 import summary_col
from matplotlib import pyplot as plt
import scipy
from datetime import datetime, timedelta

#Params (remember, with morning check)
exit_day = 1
exit_time = 'open'
#don't set checking=True if exit_day=1 and exit_time='open'
checking = False

#create new variables
master = pd.read_csv('../data/gap d 15 01112021 check agg -253 5.csv')
master = master.append(pd.read_csv('../data/gap d 15 01112021 agg -253 5.csv'))
print('Building master DF')
master = master.rename(columns = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume',
            },
)

master['constant'] = 1
master['date'] = pd.to_datetime(master['date'])
master['init date'] = pd.to_datetime(master['init date'])
earliest = datetime.strptime('2020-04-01', '%Y-%m-%d')
latest = datetime.strptime('2021-01-08', '%Y-%m-%d')
master = master.loc[master['init date']>=earliest]
master = master.loc[master['init date']<latest]

#throw out innaccuracies
master = master.loc[master['ticker'] != 'EFOI']
master = master.loc[master['ticker'] != 'UCO']
master = master.loc[master['ticker'] != 'XSPA']
master = master.loc[master['ticker'] != 'BFIIW']
master = master.loc[master['ticker'] != 'CHKR']
master = master.loc[master['ticker'] != 'SPHS']
master = master.loc[master['ticker'] != 'GLBS']
master = master.loc[master['ticker'] != 'NTEC']
master = master.loc[master['ticker'] != 'HJLI']
master = master.loc[master['ticker'] != 'NBRV']
master = master.loc[master['ticker'] != 'HTBX']




print('building swing DF')

#build swing DF
swing_df = pd.DataFrame()



day_0 = master.loc[
            (master.delta == 0)  
            & (master.open <= 5)  
            & (master.open >= 0.01)
            #remove fed announcement day
            & (master['init date'] != np.datetime64('2020-06-11T00:00:00.000000000'))
]
for i, row in day_0.iterrows():
    swing = pd.Series(row)
    
     #get historical TS
    hist_ts = master.loc[
            (master.ticker == row.ticker) 
            & (master['init date'] == row['init date'])
            & (master.delta < 0)
    ].copy()
    swing['hist_ts'] = hist_ts
    
    
    #create SMA 20 day
    swing.hist_ts['sma20'] = swing.hist_ts.close.rolling(20).mean()
    
    
    entry = row.open
    for delta in range(exit_day+1):
        if delta == 0:
            _exit = master.loc[
                (master['delta'] == delta) &
                (master['init date'] == row['init date']) &
                (master.ticker == row['ticker'])
                ][exit_time].iloc[0]
            swing[f'return {delta}'] = (_exit - entry)/entry
        elif delta < exit_day:
            check = master.loc[
                (master['delta'] == delta) &
                (master['init date'] == row['init date']) &
                (master.ticker == row['ticker'])
                ]['open'].iloc[0]
            swing[f'check {delta}'] = (check - entry)/check
        else:
            _exit = master.loc[
                (master['delta'] == delta) &
                (master['init date'] == row['init date']) &
                (master.ticker == row['ticker'])
                ][exit_time].iloc[0]
            
            check = master.loc[
                (master['delta'] == delta) &
                (master['init date'] == row['init date']) &
                (master.ticker == row['ticker'])
                ]['open'].iloc[0]
            swing[f'check {delta}'] = (check - entry)/entry
            swing[f'return {delta}'] = (_exit - entry)/entry
    
    swing_df = swing_df.append(swing, ignore_index = True)
        



### visualize

#format and output plots  
fig, ax = plt.subplots(figsize = (8,8))

test_df = swing_df.loc[swing_df['return 1']>.40]

for i, row in test_df.iterrows():
    ax.plot(row.hist_ts['delta'], row.hist_ts['sma20'])
    




