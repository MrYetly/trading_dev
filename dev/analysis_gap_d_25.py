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
master['return'] = 100*(master['4. close'] - master['1. open'])/master['1. open']

master = master.rename(columns = {
        '1. open': 'open',
        '2. high': 'high',
        '3. low': 'low',
        '4. close': 'close',
        '5. volume': 'volume',
        }
)

master.volume = master.volume/1000000

print('cleaned shape:', master.shape)

#isolate winners and losers
day_0 = master.loc[(master['delta']== 0) & (master.open < 5)]
day_0_win = day_0.loc[day_0['return'] > 0]
day_0_loss = day_0.loc[day_0['return'] < 0]

#analysis of strategy
win_rate = day_0_win.shape[0]/day_0.shape[0]
loss_rate = day_0_loss.shape[0]/day_0.shape[0]
print('win rate:', win_rate)
print('loss rate:', loss_rate)

regressors = ['constant',]
#winners
win_qr = QuantReg(day_0_win['return'], day_0_win[regressors])
res_wl = win_qr.fit(q=.25)
res_wu = win_qr.fit(q=.75)
res_wmed = win_qr.fit(q=.5)

win_ols = OLS(day_0_win['return'], day_0_win[regressors])
res_wols = win_ols.fit()

#losers
#CHANGING SIGN OF LOSS
loss_qr = QuantReg(-day_0_loss['return'], day_0_loss[regressors])
res_ll = loss_qr.fit(q=.25)
res_lu = loss_qr.fit(q=.75)
res_lmed = loss_qr.fit(q=.5)

loss_ols = OLS(-day_0_loss['return'], day_0_loss[regressors])
res_lols = loss_ols.fit()

tab_win = summary_col(
        [res_wl, res_wu, res_wmed, res_wols],
        model_names = ['25th', '75th', 'median', 'average'],
        info_dict={
            'N':lambda x: "{0:d}".format(int(x.nobs)),
            'Win rate':lambda x: "{:.2f}".format(win_rate),
        }
)
tab_loss = summary_col(
        [res_ll, res_lu, res_lmed, res_lols],
        model_names = ['25th', '75th', 'median', 'average'],
        info_dict={
            'N':lambda x: "{0:d}".format(int(x.nobs)),
            'loss rate':lambda x: "{:.2f}".format(loss_rate),
        },
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

inv_val = [4700,]
net_returns = [0,]
at_risk = 0.25

print('initial value:', inv_val[-1])

for d in all_dates:
    trades = day_0.loc[day_0['date'] == d]
    positions = (inv_val[-1]*at_risk)/(trades.shape[0])
    day_returns = positions*trades['return']/100
    net = day_returns.sum()
    net_returns.append(net)
    inv = inv_val[-1] + net
    inv_val.append(inv)
    if inv <=0:
        break
    
ad = ['2020-08-03',] + all_dates
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
v = v*at_risk
v = v.iloc[:-1].copy()
v.index = r.index
test_res['% return'] = r/v

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

