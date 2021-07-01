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

#Params (remember, with moring check)
exit_day = 1
checking = False
exit_time = 'close'

#create new variables
master = pd.read_csv('../data/gap d 25 12262020 agg -253 5.csv')
print('uncleaned shape:', master.shape)

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
earliest = datetime.strptime('2020-08-04', '%Y-%m-%d')
latest = datetime.strptime('2020-12-18', '%Y-%m-%d')
master = master.loc[master['init date']>=earliest]
master = master.loc[master['init date']<latest]
master = master.loc[master['ticker'] != 'GLBS']




#find lowest
master['52w low'] = np.nan
for ticker in master.ticker.unique():
    inits = master.loc[master.ticker == ticker]['init date'].unique()
    for init in inits:
        ts = master.loc[
                (master.ticker == ticker) 
                & (master['init date'] == init)
        ]
        dex = ts.index
        ts = ts.loc[ts.delta < 0]
        lowest = ts.low.min()
        master.loc[dex, '52w low'] = lowest

#find 

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

'''
promising sub-ponds

31% kelly, 15% exp ret,  45% win rate
(master.delta == 0)  
& (master.open <= 5)  
& (master.open >= 0.01) 
& (master.open >= 3*master['52w low'])

27% kelly, 8.7% exp ret, 47% win rate
(master.delta == 0)  
& (master.open <= 5)  
& (master.open >= 0.01) 
& (master.open >= 2*master['52w low'])

'''
day_0 = master.loc[
            (master.delta == 0)  
            & (master.open <= 5)  
            & (master.open >= 0.01) 
            & (master.open >= 2*master['52w low'])
]
for i, row in day_0.iterrows():
    swing = pd.Series(row)
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
        

 
swing_df.index = day_0.index 
swing_df['final ret'] = np.nan
swing_df['position'] = np.nan


#backtest strategy
    
exit_df = pd.DataFrame()

all_dates = list(swing_df['date'].unique())
all_dates.sort()
    
inv_val = [4700,]
net_returns = [np.nan,]
at_risk = 0.5
cost = 0.00    

print('initial value:', inv_val[-1])
print('at risk:', at_risk)

blank = pd.DataFrame(columns = swing_df.columns)

inplay = [blank for i in range(exit_day +1)]
    
for d in all_dates:
    
    
    
    position = inv_val[-1]*at_risk*(1/(exit_day+2))
    new_trades = swing_df.loc[swing_df['init date'] == d].copy()
    new_trades['position'] = position/new_trades.shape[0]
    inplay = [new_trades,]+inplay
    
    exiting = pd.DataFrame(columns = swing_df.columns)
    lastday = inplay.pop().copy()
    
    if lastday.shape[0] != 0:
        
        lastday['final ret'] = lastday[f'return {len(inplay)-1}']
        exiting = exiting.append(lastday)
    
    if checking == True:
        for i in range(1,exit_day+1):
            
            swinging = inplay[i]
            
            if swinging.shape[0] != 0:
                
                swingloss = swinging.loc[swinging[f'check {i}'] <= 0].copy()
                swingloss['final ret'] = swingloss[f'check {i}']
                exiting = exiting.append(swingloss)
                
                swingkeep = swinging.loc[swinging[f'check {i}'] > 0]
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
    
l = 0.25
h = 0.75
#winners
win_qr = QuantReg(exit_df_win['final ret'], exit_df_win['constant'])
res_wl = win_qr.fit(q=l)
res_wu = win_qr.fit(q=h)
res_wmed = win_qr.fit(q=.5)
    
win_ols = OLS(exit_df_win['final ret'], exit_df_win['constant'])
res_wols = win_ols.fit()
    
#losers
#CHANGING SIGN OF LOSS
loss_qr = QuantReg(-exit_df_loss['final ret'], exit_df_loss['constant'])
res_ll = loss_qr.fit(q=l)
res_lu = loss_qr.fit(q=h)
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
t_dbl = (np.log(2)/np.log(1+(exp_ret*at_risk/(exit_day+1)))*((latest-earliest)/len(ad)))
print('time to double:', t_dbl.days)
tab_win = summary_col(
        [res_wl, res_wu, res_wmed, res_wols],
        model_names = [f'{l}', f'{h}', 'median', 'average'],
        info_dict={
        'N':lambda x: "{0:d}".format(int(x.nobs)),
        'Win rate':lambda x: "{:.2f}".format(win_rate),
}
)
tab_loss = summary_col(
        [res_ll, res_lu, res_lmed, res_lols],
        model_names = [f'{l}', f'{h}', 'median', 'average'],
        info_dict={
        'N':lambda x: "{0:d}".format(int(x.nobs)),
        'Loss rate':lambda x: "{:.2f}".format(loss_rate),
}
)
tab_win.title = f'Analysis of Winners'
tab_loss.title = f'Analysis of Gap Losers'

print(tab_win)
print(tab_loss)

#    with open(f'tab_win_{delta} vol limit.tex', 'w') as f:
#        f.write(tab_win.as_latex())
#        
#    with open(f'tab_loss_{delta} vol limit.tex', 'w') as f:
#        f.write(tab_loss.as_latex())'''
