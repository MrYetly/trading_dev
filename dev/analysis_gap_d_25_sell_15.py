#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 12:48:23 2020

@author: ianich
"""

import pandas as pd
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.regression.linear_model import OLS
from statsmodels.iolib.summary2 import summary_col
from matplotlib import pyplot as plt

#create new variables
master = pd.read_csv('../data/gap d 25 12262020 agg.csv')
print('uncleaned shape:', master.shape)
master = master.loc[master['ticker'] != 'GLBS']
master = master.loc[master['1. open'] > 0.1]

master['constant'] = 1

entry = master.loc[master['delta']== 0]
_exit = master.loc[master['delta'] == 15].copy()
_exit.index = entry.index
master['return'] = (_exit['4. close'] - entry['1. open'])/entry['1. open']

print('cleaned shape:', master.shape)

#isolate winners and losers
day_0 = master.loc[master['delta']== 0]
day_0_win = day_0.loc[day_0['return'] > 0]
day_0_loss = day_0.loc[day_0['return'] < 0]

#analysis of strategy
win_rate = day_0_win.shape[0]/day_0.shape[0]
loss_rate = day_0_loss.shape[0]/day_0.shape[0]
print('win rate:', win_rate)
print('loss rate:', loss_rate)

#winners
win_qr = QuantReg(day_0_win['return'], day_0_win['constant'])
res_wl = win_qr.fit(q=.25)
res_wu = win_qr.fit(q=.75)
res_wmed = win_qr.fit(q=.5)

win_ols = OLS(day_0_win['return'], day_0_win['constant'])
res_wols = win_ols.fit()

#losers
#CHANGING SIGN OF LOSS
loss_qr = QuantReg(-day_0_loss['return'], day_0_loss['constant'])
res_ll = loss_qr.fit(q=.25)
res_lu = loss_qr.fit(q=.75)
res_lmed = loss_qr.fit(q=.5)

loss_ols = OLS(-day_0_loss['return'], day_0_loss['constant'])
res_lols = loss_ols.fit()

tab_win = summary_col(
        [res_wl, res_wu, res_wmed, res_wols],
        model_names = ['25th', '75th', 'median', 'average'],
)
tab_loss = summary_col(
        [res_ll, res_lu, res_lmed, res_lols],
        model_names = ['25th', '75th', 'median', 'average'],
)
tab_win.title = 'Analysis of Gap -25% 7 day Winners'
tab_loss.title = 'Analysis of Gap -25% 7 day Losers'

print(tab_win)
print(tab_loss)

#calculate expected return
w_avg = res_wols.params['constant']
l_avg = res_lols.params['constant']
exp_ret = win_rate*w_avg - loss_rate*l_avg
print('expected return:', exp_ret)

#with open('tab_win.tex', 'w') as f:
 #   f.write(tab_win.as_latex())
    
#with open('tab_loss.tex', 'w') as f:
 #   f.write(tab_loss.as_latex())

#backtest strategy

all_dates = list(day_0['date'].unique())

inv_val = [100000,]
net_returns = [0,]

print('initial value:', inv_val[-1])

for d in all_dates:
    trades = day_0.loc[day_0['date'] == d]
    positions = inv_val[-1]/(trades.shape[0])
    day_returns = positions*trades['return']
    net = day_returns.sum()
    net_returns.append(net)
    inv = inv_val[-1] + net
    inv_val.append(inv)
    
ad = ['2020-08-03',] + all_dates
test_res = pd.DataFrame(
        {
                'value': inv_val,
                'date': ad,
                'net_ret': net_returns
        }
)
test_res['% return'] = test_res['net_ret']/test_res['value']

print('final value:', inv_val[-1])
print('avg strategy return:', test_res['% return'].mean())

#format and output plots  
fig, ax = plt.subplots(2,1, figsize = (8,11))
test_res['value'].plot(
        ax=ax[0],
        title = 'Value of portfolio',
)
test_res['% return'].plot(
        ax=ax[1],
        title = 'Return to Strategy',
)

ax[0].set(xlabel = 'Trading Day', ylabel = 'Value of portfolio ($)')
ax[1].set(xlabel = 'Trading Day', ylabel = 'Percent return')

#fig.savefig('results.png')

