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


### handle historical price data

#create new variables
hist = pd.read_csv('../data/gap u 25 12302020 agg -253 5.csv')
print('hist uncleaned shape:', hist.shape)

rename = {
        '1. open': 'open',
        '2. high': 'high',
        '3. low': 'low',
        '4. close': 'close',
        '5. volume': 'volume',
}

hist = hist.rename(columns = rename)
hist['constant'] = 1
hist['date'] = pd.to_datetime(hist['date'])
hist['init date'] = pd.to_datetime(hist['init date'])
earliest = datetime.strptime('2020-08-07', '%Y-%m-%d')
latest = datetime.strptime('2020-12-23', '%Y-%m-%d')
hist = hist.loc[hist['init date']>=earliest]
hist = hist.loc[hist['init date']<latest]
hist = hist.loc[hist.ticker != 'JE']
hist = hist.loc[hist.ticker != 'SMLP']
hist = hist.loc[hist.ticker != 'SCON']



pennies = hist.loc[hist.open < 0.01]['ticker'].unique()
for i in pennies:
    hist = hist.loc[hist.ticker != i]


#find short returns
hist['return'] = np.nan
for delta in range(6):
    entry = hist.loc[hist['delta']== 0].copy()
    _exit = hist.loc[hist['delta'] == delta]
    entry.index = _exit.index
    _return = (entry['open'] - _exit['close'])/entry['open']
    hist['return'] = hist['return'].combine_first(_return)

#limit volume
vol_drop = hist.loc[(hist['volume'] > 80000000) & (hist['delta'] ==0)]
for i, row in vol_drop.iterrows():
    hist = hist.drop(index = hist.loc[(hist.ticker == row['ticker']) & (hist['init date'] == row['init date'])].index)


print('hist cleaned shape:', hist.shape)

#handle intraday data

intra_1 = pd.read_csv('../data/gap u 25 12302020 agg intra 5 min.csv')
intra_2 = pd.read_csv('../data/gap u 25 12302020 agg intra 5 min part 2.csv')
intra = pd.concat([intra_1, intra_2], sort = True, ignore_index = True)
intra['time'] = pd.to_datetime(intra['time'])
intra['init date'] = pd.to_datetime(intra['init date'])

print('intra uncleaned shape:', intra.shape)

intra_ts = pd.DataFrame()

ts_key = hist[['ticker', 'init date']].drop_duplicates()

for i, row in ts_key.iterrows():
    ts = intra.loc[(intra['ticker'] == row['ticker']) & (intra['init date'] == row['init date'])]
    entry = {
            'ticker': row['ticker'],
            'init date': row['init date'],
            'ts': ts[[
                    'close',
                    'high',
                    'low',
                    'open',
                    'time',
                    'volume'
                   ]],
    }
    intra_ts = intra_ts.append(entry, ignore_index = True)

intra_ts.index = ts_key.index


###find pushes


pushes = pd.DataFrame()

for i, row in intra_ts.iterrows():
    push = np.nan
    p_open = np.nan
    d = row['init date']
    o = d + np.timedelta64(9, 'h') + np.timedelta64(35, 'm')
    if o in set(row['ts']['time']):
        candle = row['ts'].loc[row['ts']['time'] == o]
        move = (candle['high'] - candle['open'])/candle['open']
        if move.iloc[0] >= 0.07:
            push = 1.0
        else:
            push = 0.0
        p_open = candle['close'].iloc[0]
    entry = {
            'ticker': row['ticker'],
            'init date': row['init date'],
            'push': push,
            'p open': p_open
    }
    pushes = pushes.append(entry, ignore_index = True)
    
pushes.index = intra_ts.index

###find stop losses

#stop_loss = -.8
#stop_losses = pd.DataFrame()
#
#for i, row in intra_ts.iterrows():
#    stop = np.nan
#    d = row['init date']
#    t = row['ticker']
#    o = d + np.timedelta64(9, 'h') + np.timedelta64(35, 'm')
#    o_price = hist.loc[(hist['init date'] == d) & (hist.delta == 0) & (hist.ticker == t)]['open'].iloc[0]
#    candles = row['ts']
#    candles = candles.sort_values('time')
#    candles = candles.loc[candles.time >= o]
#    for j, candle in candles.head().iterrows():
#        r = (o_price - candle['high'])/o_price
#        if r <= stop_loss:
#            stop = 1
#        else:
#            stop= 0
#    entry = {
#            'ticker': row['ticker'],
#            'init date': row['init date'],
#            'stop': stop,
#            'o_price': o_price,
#            'c high': candle['high'],
#    }  
#    stop_losses = stop_losses.append(entry, ignore_index = True)
#
#stop_losses.index = intra_ts.index    
    

    


#isolate only delta=0, push=1
data = hist.loc[hist.delta == 0].copy()

data['qret'] = (data['open'] - data['high'])/data['open']

#data['push'] = pushes['push']
#data['open'] = pushes['p open']
#data['return'] = (data['open'] - data['close'])/data['open']
#data = data.loc[data['push'] == 0]

#data['stop'] = stop_losses['stop']


#chekc accuracy of open and close between daily and 5min

#tk = 'POAI'
#test = hist.loc[(hist.ticker == tk) & (hist.delta == 0)]
#d = test['init date'].iloc[0]
#o = d + np.timedelta64(9, 'h') + np.timedelta64(35, 'm')
#c = d + np.timedelta64(16, 'h')
#compare = intra_ts.loc[intra_ts.ticker == tk]
#compare = compare.loc[compare['init date'] == test['init date'].iloc[0]]
#compare = compare.iloc[0].ts
#_open = compare.loc[compare['time'] == o]
#_close = compare.loc[compare['time'] == c]
#print(test[['open', 'close', 'init date', 'ticker']])
#print(_open)
#print(_close)



###Analysis

data_win = data.loc[data['return'] > 0]
data_loss = data.loc[data['return'] < 0]
    
print('total trades:', data.shape[0])
print('unique tickers:', data['ticker'].unique().shape[0])

#analysis of strategy
win_rate = data_win.shape[0]/data.shape[0]
loss_rate = data_loss.shape[0]/data.shape[0]
print('# winners:', data_win.shape[0])
print('# losers:', data_loss.shape[0])
print('win rate:', win_rate)
print('loss rate:', loss_rate)

#mu, sigma = scipy.stats.norm.fit(data['return'])
#print('mu:', mu, 'sigma:', sigma)

h = 0.75
l = 0.25

#winners
win_qr = QuantReg(data_win['return'], data_win['constant'])
res_wl = win_qr.fit(q=l)
res_wu = win_qr.fit(q=h)
res_wmed = win_qr.fit(q=.5)

win_ols = OLS(data_win['return'], data_win['constant'])
res_wols = win_ols.fit()

#losers
#CHANGING SIGN OF LOSS
loss_qr = QuantReg(-data_loss['return'], data_loss['constant'])
res_ll = loss_qr.fit(q=l)
res_lu = loss_qr.fit(q=h)
res_lmed = loss_qr.fit(q=.5)

loss_ols = OLS(-data_loss['return'], data_loss['constant'])
res_lols = loss_ols.fit()

#calculate expected return
w_avg = res_wols.params['constant']
l_avg = res_lols.params['constant']
exp_ret = win_rate*w_avg - loss_rate*l_avg
print('expected return:', exp_ret)

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
tab_win.title = f'Analysis of Gap 25% Winners, less than 7% push, vol<80mil'
tab_loss.title = f'Analysis of Gap 25% Losers, less than 7% push, vol<80mil'

print(tab_win)
print(tab_loss)

#with open(f'tab_win_>7push5.tex', 'w') as f:
#    f.write(tab_win.as_latex())
#
#with open(f'tab_loss_>7push5.tex', 'w') as f:
#    f.write(tab_loss.as_latex())

#backtest strategy

all_dates = list(data['date'].unique())
    
inv_val = [26000,]
net_returns = [0,]
ppd = 0.03
    
print('initial value:', inv_val[-1])
    
for d in all_dates:
    trades = data.loc[data['date'] == d]
    positions = (inv_val[-1]*ppd)/(trades.shape[0])
    day_returns = positions*trades['return']
    net = day_returns.sum()
    net_returns.append(net)
    inv = inv_val[-1] + net
    inv_val.append(inv)
    if inv <= 0:
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
v = v*ppd
v = v.iloc[:-1].copy()
v.index = r.index
test_res['% return'] = r/v

print('final value:', inv_val[-1])
print('avg strategy return:', test_res['% return'].mean())


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

#fig.savefig(f'backtest gap 25 short >7push5.png')

