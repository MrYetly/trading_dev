#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 11:16:25 2021

@author: ivananich
"""


import pandas as pd
from datetime import datetime


def check_returns(order_by = 0, plot = True):
    
    df = sample
    #find all return columns
    return_cols = [i for i in list(df.columns) if 'return' in i and 'o' not in i]
    sub_df = df.sort_values(return_cols[order_by-1])[['ticker', 'init_date']+return_cols]
    
    if plot == True:
        #create histogram
        return sub_df, sub_df.hist(
            column = return_cols,
            figsize = (8,5),
            bins = 50,
            )
        
    
    return sub_df

def check_qualifications(order_by = 0):
    
    df = sample
    
    return_col = [
        'return_o0m',
        'return_o1m',
        'return_o2m',
        'return_o3m',
    ]
    
    sub_df = df[['ticker', 'init_date',] + return_col]
    
 
    return sub_df.hist(
        column = return_col,
        figsize = (8,5),
        bins = 50,
        ), sub_df.sort_values(f'return_o{order_by}m')
        
    

#Params (remember, with morning check)
final_day = 20
exit_day = 20
entry_day = 1
exit_time = 'open'
entry_time = 'open'
initial_val = 25000
#don't set checking=True if exit_day=1 and exit_time='open'
checking = False
check_range = range(1, exit_day)


#create new variables
master = pd.read_csv('../../data/4x_up_10_otm_2021-06-30 agg -253 20.csv')
master = master.append(
    pd.read_csv('../../data/4x_up_10_otm_2021-06-30 agg -253 20 part 2.csv'),
    ignore_index = True,
)
master = master.append(
    pd.read_csv('../../data/4x_up_10_otm_2021-06-30 agg -253 20 part 3.csv'),
    ignore_index = True,
)
print('Building master DF')
master = master.rename(columns = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume',
                'init date': 'init_date',
            },
)

master['date'] = pd.to_datetime(master['date'])
master['init_date'] = pd.to_datetime(master['init_date'])
earliest = datetime.strptime('2021-02-08', '%Y-%m-%d')
latest = datetime.strptime('2021-06-30', '%Y-%m-%d')
master = master.loc[master['init_date']>=earliest]
master = master.loc[master['init_date']<=latest]


#build sample DF
print('building sample DF')

#start sample with day 0 trade signals
sample = master.loc[master.delta == 0].copy()

#dictionary of months in the past and lags
m_dic = {
    '1':-20,
    '2':-40,
    '3':-60,
    '4':-80,
}

#find close on lag of month in past
for m,lag in m_dic.items():
    close = master.loc[master.delta == lag][['ticker', 'init_date', 'close', 'date']]
    close = close.rename(columns = {'close': f'close_{m}m','date': f'date_{m}m'})
    sample = sample.merge(close, how='inner', on=['init_date','ticker'], validate='1:1')

months = [0,1,2,3]
#calculate return from previous mongth to current month
for m in months:
    
    cur = m
    prev = m+1
    
    cur = str(cur)
    prev = str(prev)
    
    #find close at lag, if current is zero substitute 'close' of sample row
    if cur == '0':
        cur = sample.open
    else:
        cur = sample[f'close_{cur}m']
    
    prev = sample[f'close_{prev}m']
    
    sample[f'return_o{m}m'] = (cur - prev)/prev



#create entry price column
entries = master.loc[master.delta == entry_day][['ticker', 'init_date',entry_time]]
entries = entries.rename(columns = {entry_time:'entry'})
sample = sample.merge(entries, how='inner', on=['init_date','ticker'], validate='1:1')

#create returns
if exit_time == 'open':
    adjust = 1
if exit_time == 'close':
    adjust = 0 

for delta in range(0+adjust, final_day+1):
    if delta % 5 == 0:
        day = master.loc[master.delta == delta][['init_date','ticker', exit_time]]
        day = day.rename(columns={exit_time:f'exit_{delta}'})
        sample = sample.merge(day, how='inner', on=['init_date','ticker'], validate='1:1')
        sample[f'return_{delta}'] = (sample[f'exit_{delta}'] - sample.entry)/sample.entry


#isolate data that doesn't meet qualifications
dq_df = check_qualifications()[1]
dq_df = dq_df.loc[
            (dq_df.return_o0m < .1)
            | (dq_df.return_o1m <0.1)
            | (dq_df.return_o2m <0.1)
            | (dq_df.return_o3m <0.1)
]

#isolate date with strange returns
s_ret_df = check_returns(plot=False)
h_ret = 2
l_ret = -0.6
s_ret_df = s_ret_df.loc[
            (s_ret_df.return_5 > h_ret)
            | (s_ret_df.return_5 < l_ret)
            | (s_ret_df.return_10 > h_ret)
            | (s_ret_df.return_10 < l_ret)
            | (s_ret_df.return_15 > h_ret)
            | (s_ret_df.return_15 < l_ret)
            | (s_ret_df.return_20 > h_ret)
            | (s_ret_df.return_20 < l_ret)
]

iso_index= dq_df.index.union(s_ret_df.index)

iso_df = sample.loc[iso_index]

sample = sample.drop(iso_index)






