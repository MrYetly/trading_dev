#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 12:48:23 2020

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

#create new variables
master = pd.read_csv('../data/gap u 25 12302020 agg -253 5.csv')
print('uncleaned shape:', master.shape)

master['constant'] = 1
master['date'] = pd.to_datetime(master['date'])
master['init date'] = pd.to_datetime(master['init date'])
earliest = datetime.strptime('2020-08-07', '%Y-%m-%d')
latest = datetime.strptime('2020-12-23', '%Y-%m-%d')
master = master.loc[master['init date']>=earliest]
master = master.loc[master['init date']<latest]
master = master.loc[master.ticker != 'JE']
master = master.loc[master.ticker != 'SMLP']
master = master.loc[master.ticker != 'SCON']

#find short returns (close at open for delta >0)
master['return'] = np.nan
for delta in range(6):
    entry = master.loc[master['delta']== 0].copy()
    _exit = master.loc[master['delta'] == delta]
    entry.index = _exit.index
    if delta == 0:
        _return = (entry['1. open'] - _exit['4. close'])/entry['1. open']
    else:
        _return = (entry['1. open'] - _exit['1. open'])/entry['1. open']
    master['return'] = master['return'].combine_first(_return)

#limit volume
vol_drop = master.loc[(master['5. volume'] > 80000000) & (master['delta'] ==0)]
for i, row in vol_drop.iterrows():
    master = master.drop(index = master.loc[(master.ticker == row['ticker']) & (master['init date'] == row['init date'])].index)

master = master.rename(columns = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume',
            },
)


print('cleaned shape:', master.shape)

###Analysis

ret_df = pd.DataFrame(
            {
                'expected return': [],
                'avg backtest return':[],
                'delta': [],
            },
 )

swing_df = pd.DataFrame()

#build swing DF
day_0 = master.loc[(master.delta == 0) & (master.open >= 5)]
for i, row in day_0.iterrows():
    entry = pd.Series(row)
    entry = entry.rename({'return': 'return 0'})
    for delta in range(1,6):
        r_d = master.loc[
                (master['delta'] == delta) &
                (master['init date'] == row['init date']) &
                (master.ticker == row['ticker'])
        ]['return'].iloc[0]
        entry[f'return {delta}'] = r_d
    swing_df = swing_df.append(entry, ignore_index = True)
 
swing_df.index = day_0.index 
swing_df['final ret'] = np.nan
swing_df['position'] = np.nan


#backtest strategy
    
exit_df = pd.DataFrame()

all_dates = list(swing_df['date'].unique())
all_dates.sort()
    
inv_val = [4700,]
net_returns = [0,]
at_risk = 0.5
cost = 0.02    

print('initial value:', inv_val[-1])
print('at risk:', at_risk)

blank = pd.DataFrame(columns = swing_df.columns)

exit_day = 1
inplay = [blank for i in range(exit_day +1)]
    
for d in all_dates:
    
    position = inv_val[-1]*at_risk*(1/(exit_day+2))
    new_trades = swing_df.loc[swing_df['init date'] == d].copy()
    new_trades['position'] = position/new_trades.shape[0]
    inplay = [new_trades,]+inplay
    
    exiting = pd.DataFrame(columns = swing_df.columns)
    lastday = inplay.pop()
    
    if lastday.shape[0] != 0:
        
        lastday['final ret'] = lastday[f'return {len(inplay)-1}']
        exiting = exiting.append(lastday)
    
    for i in range(1,exit_day):
        
        swinging = inplay[i]
        
        if swinging.shape[0] != 0:
            
            swingloss = swinging.loc[swinging[f'return {i}'] <= 0].copy()
            swingloss['final ret'] = swingloss[f'return {i}']
            exiting = exiting.append(swingloss)
            
            swingkeep = swinging.loc[swinging[f'return {i}'] > 0]
            if swingkeep.shape[0] == 0:
                swingkeep = blank
            inplay[i] = swingkeep
    
    exiting['net ret'] = exiting['position']*(exiting['final ret'] - cost)

    if exiting.shape[0] == 0:
        sum_net_ret = 0
    else:
        sum_net_ret = exiting['net ret'].sum()
        exit_df = exit_df.append(exiting, ignore_index = True)
        
    net_returns.append(sum_net_ret)
    inv_val.append(inv_val[-1] + sum_net_ret)
    
    if inv_val[-1] <=0:
        break
    
    
        
ad = ['init date',] + all_dates[:len(inv_val)-1]
test_res = pd.DataFrame(
        {
                'value': inv_val,
                'date': ad,
                'net_ret': net_returns
        }
)
    
#calculate % return
r = test_res['net_ret']
r = r.iloc[1:]
v = test_res['value']
v = v*at_risk*(1/(exit_day+2))
v = v.iloc[:-1].copy()
v.index = r.index
test_res['% return'] = r/v
    
print('final value:', inv_val[-1])
print('avg strategy return:', test_res['% return'].mean())
print('trading days:', len(ad))
#format and output plots  
fig, ax = plt.subplots(2,1, figsize = (8,11))
test_res['value'].plot(
        ax=ax[0],
        title = f'Value of portfolio',
)
test_res['% return'].plot(
        ax=ax[1],
        title = f'Return to Strategy',
)
    
ax[0].set(xlabel = 'Trading Day', ylabel = 'Value of portfolio ($)')
ax[1].set(xlabel = 'Trading Day', ylabel = 'Percent return')
    
#fig.savefig(f'backtest gap 25 short swing.png')

#analyze winners and losers


exit_df_win = exit_df.loc[exit_df['final ret'] > 0]
exit_df_loss = exit_df.loc[exit_df['final ret'] < 0]
    
    
print('total trades:', exit_df.shape[0])
print('unique tickers:', exit_df['ticker'].unique().shape[0])
    
#analysis of strategy
win_rate = exit_df_win.shape[0]/exit_df.shape[0]
loss_rate = exit_df_loss.shape[0]/exit_df.shape[0]
print('# winners:', exit_df_win.shape[0])
print('# losers:', exit_df_loss.shape[0])
print('win rate:', win_rate)
print('loss rate:', loss_rate)
    
#mu, sigma = scipy.stats.norm.fit(exit_df['return'])
#print('mu:', mu, 'sigma:', sigma)
    
    
#winners
win_qr = QuantReg(exit_df_win['final ret'], exit_df_win['constant'])
res_wl = win_qr.fit(q=.05)
res_wu = win_qr.fit(q=.95)
res_wmed = win_qr.fit(q=.5)
    
win_ols = OLS(exit_df_win['final ret'], exit_df_win['constant'])
res_wols = win_ols.fit()
    
#losers
#CHANGING SIGN OF LOSS
loss_qr = QuantReg(-exit_df_loss['final ret'], exit_df_loss['constant'])
res_ll = loss_qr.fit(q=.05)
res_lu = loss_qr.fit(q=.95)
res_lmed = loss_qr.fit(q=.5)
    
loss_ols = OLS(-exit_df_loss['final ret'], exit_df_loss['constant'])
res_lols = loss_ols.fit()

#calculate expected final ret
w_avg = res_wols.params['constant']
l_avg = res_lols.params['constant']
exp_ret = win_rate*w_avg - loss_rate*l_avg
print('expected final ret:', exp_ret)
kelly = win_rate - (loss_rate)/(w_avg/l_avg)
print('kelly percentage:', kelly)
tab_win = summary_col(
        [res_wl, res_wu, res_wmed, res_wols],
        model_names = ['5th', '95th', 'median', 'average'],
        info_dict={
        'N':lambda x: "{0:d}".format(int(x.nobs)),
        'Win rate':lambda x: "{:.2f}".format(win_rate),
}
)
tab_loss = summary_col(
        [res_ll, res_lu, res_lmed, res_lols],
        model_names = ['5th', '95th', 'median', 'average'],
        info_dict={
        'N':lambda x: "{0:d}".format(int(x.nobs)),
        'Loss rate':lambda x: "{:.2f}".format(loss_rate),
}
)
tab_win.title = f'Analysis of Gap 25% Winners'
tab_loss.title = f'Analysis of Gap 25% Losers'

print(tab_win)
print(tab_loss)

#    with open(f'tab_win_{delta} vol limit.tex', 'w') as f:
#        f.write(tab_win.as_latex())
#        
#    with open(f'tab_loss_{delta} vol limit.tex', 'w') as f:
#        f.write(tab_loss.as_latex())'''
