#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 12:30:23 2021

@author: ivananich
"""
import pandas as pd
from datetime import datetime

def import_master(file_path, start_date = None, end_date = None):
    '''
    imports file_path to dataframe, renames columns, and sets date and init_date to 
    datetime objects
    '''
    master = pd.read_csv(file_path)
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
    
    if start_date !=None:
        
        earliest = datetime.strptime('2021-03-01', '%Y-%m-%d')
        master = master.loc[master['init_date']>=earliest]
    
    if end_date !=None:
        latest = datetime.strptime('2021-05-12', '%Y-%m-%d')
        master = master.loc[master['init_date']<=latest]
    
    return master

def create_swing_returns(
        sample,
        master,
        entry_time = 'open', 
        entry_day = 0, 
        exit_time = 'open', 
        start = 0, 
        end = 1):
    '''
    creates entry price column according to entry_time and entry_date, calculates return fro each
    delta between start and end at exit_time

    Parameters
    ----------
    sample : TYPE
        DESCRIPTION.
    master : TYPE
        DESCRIPTION.
    entry_time : TYPE, optional
        DESCRIPTION. The default is 'open'.
    entry_day : TYPE, optional
        DESCRIPTION. The default is 0.
    exit_time : TYPE, optional
        DESCRIPTION. The default is 'open'.
    start : TYPE, optional
        DESCRIPTION. The default is 0.
    end : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    sample : dataframe.

    '''
    #create entry price column
    entries = master.loc[master.delta == entry_day][['ticker', 'init_date',entry_time]]
    entries = entries.rename(columns = {entry_time:'entry'})
    sample = sample.merge(entries, how='inner', on=['init_date','ticker'], validate='1:1')
    
    #create returns
    if exit_time == 'open':
        adjust = 1
    if exit_time == 'close':
        adjust = 0 
    
    for delta in range(start+adjust, end+1):
        day = master.loc[master.delta == delta][['init_date','ticker', exit_time]]
        day = day.rename(columns={exit_time:f'exit_{delta}'})
        sample = sample.merge(day, how='inner', on=['init_date','ticker'], validate='1:1')
        sample[f'return_{delta}'] = (sample[f'exit_{delta}'] - sample.entry)/sample.entry
        
    return sample

def create_momentum_indicators(sample, master, on = 'entry'):
    '''
    Creates indicators for momentum on the week, month, quarter, half-year, year

    Parameters
    ----------
    sample : TYPE
        DESCRIPTION.
    master : TYPE
        DESCRIPTION.

    Returns
    -------
    sample : TYPE
        DESCRIPTION.

    '''
    #create momentum indicators
    momentum_dic = {
        'w': -5,
        'm': -20,
        'q': -63,
        'hy': -126,
        'y': -253,
    }
    
    for label, lag in momentum_dic.items():
        day = master.loc[master.delta == lag][['init_date','ticker', 'close']]
        day = day.rename(columns={'close':f'close_{label}'})
        sample = sample.merge(day, how='inner', on=['init_date','ticker'], validate='1:1')
        sample[f'm_{label}'] = (sample[on] - sample[f'close_{label}'])/sample[f'close_{label}']
    
    return sample
    
    
    
def check_returns(sample, order_by = 0, plot = True):
    '''
    provides dataframe of retunrs for each return calculated, and histograms if plot==True.
    '''
    
    df = sample
    
    #find all return columns
    return_cols = [i for i in list(df.columns) if 'return' in i]
    sub_df = df.sort_values(return_cols[order_by])[['ticker', 'init_date']+return_cols]
    
    if plot == True:
        #create histogram
        return sub_df, sub_df.hist(
            column = return_cols,
            figsize = (8,5),
            bins = 50,
            )
        
    
    return sub_df
    
    
    
    
    