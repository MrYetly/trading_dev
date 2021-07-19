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

#Params (remember, with morning check)
exit_day = 0
exit_time = 'close'
#don't set checking=True if exit_day=1 and exit_time='open'
checking = False

#create new variables
print('building master')

master = pd.read_csv('../../data/gap u 25 12302020 agg -253 5.csv')
master = master.append(pd.read_csv('../../data/gap u 25 12302020 check agg -253 5.csv'))

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
latest = datetime.strptime('2020-12-23', '%Y-%m-%d')
master = master.loc[master['init date']>=earliest]
master = master.loc[master['init date']<latest]

#remove innaccuracies
master = master.loc[master.ticker != 'JE']
master = master.loc[master.ticker != 'SMLP']
master = master.loc[master.ticker != 'SCON']



#build sample DF

print('building sample')

sample = pd.DataFrame()


day_0 = master.loc[(master.delta == 0)]
for i, row in day_0.iterrows():
    trade = pd.Series(row)
    
    #get TS
    ts = master.loc[
            (master.ticker == row.ticker) 
            & (master['init date'] == row['init date'])
            & (master.delta < 0)
    ]
    trade['ts'] = ts
    
    #calculate returns and checks
    entry = row.open
    
    for delta in range(exit_day+1):
        if delta == 0:
            _exit = master.loc[
                (master['delta'] == delta) &
                (master['init date'] == row['init date']) &
                (master.ticker == row['ticker'])
                ]['close'].iloc[0]
            trade[f'return {delta}'] = -(_exit - entry)/entry
        elif delta < exit_day:
            check = master.loc[
                (master['delta'] == delta) &
                (master['init date'] == row['init date']) &
                (master.ticker == row['ticker'])
                ]['open'].iloc[0]
            trade[f'check {delta}'] = -(check - entry)/check
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
            trade[f'check {delta}'] = -(check - entry)/entry
            trade[f'return {delta}'] = -(_exit - entry)/entry
    
    sample = sample.append(trade, ignore_index = True)
        

 
sample.index = day_0.index 
sample['win'] = 0
dex = sample.loc[sample[f'return {exit_day}']>0].index
sample.loc[dex, 'win'] = 1
sample.index = day_0.index 
sample['final ret'] = np.nan
sample['position'] = np.nan

###limit sample
sample = sample.loc[
            (sample.open > 0.01)
            & (sample.volume <= 80000000)
]


### backtest

    
exit_df = pd.DataFrame()

all_dates = list(sample['date'].unique())
all_dates.sort()
    
inv_val = [26000,]
net_returns = [np.nan,]
n_trades = [np.nan,]
at_risk = 0.25
cost = 0.02    

print('initial value:', inv_val[-1])
print('at risk:', at_risk)

blank = pd.DataFrame(columns = sample.columns)

inplay = [blank for i in range(exit_day +1)]
    
for d in all_dates:
    
    
    ###
    position = inv_val[-1]*at_risk*(1/(exit_day+1))
    new_trades = sample.loc[sample['init date'] == d].copy()
    new_trades['position'] = position/new_trades.shape[0]
    n_trades.append(new_trades.shape[0])
    inplay = [new_trades,]+inplay
    
    exiting = pd.DataFrame(columns = sample.columns)
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
    
    #if d in list(q):
     #   print(exiting[['ticker', 'init date', 'open', 'check 1', 'return 1', 'final ret']])
    
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
                'net_ret': net_returns,
                'n_trades': n_trades,
        }
)
    
#calculate % return
r = test_res['net_ret']
r = r.iloc[1:]
v = test_res['value']
v = v*at_risk*(1/(exit_day+1))
v = v.iloc[:-1].copy()
v.index = r.index
test_res['% return'] = r/v
    
print('final value:', inv_val[-1])
print('avg strategy return:', test_res['% return'].mean())
print('trading days:', len(ad))
print('avg trades per open exec:', test_res.n_trades.mean())
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
regressors = ['constant',]
#winners
win_qr = QuantReg(exit_df_win['final ret'], exit_df_win[regressors])
res_wl = win_qr.fit(q=l)
res_wu = win_qr.fit(q=h)
res_wmed = win_qr.fit(q=.5)
    
win_ols = OLS(exit_df_win['final ret'], exit_df_win[regressors])
res_wols = win_ols.fit()
    
#losers
#CHANGING SIGN OF LOSS
loss_qr = QuantReg(-exit_df_loss['final ret'], exit_df_loss[regressors])
res_ll = loss_qr.fit(q=l)
res_lu = loss_qr.fit(q=h)
res_lmed = loss_qr.fit(q=.5)
    
loss_ols = OLS(-exit_df_loss['final ret'], exit_df_loss[regressors])
res_lols = loss_ols.fit()

#calculate expected final ret
w_avg = res_wols.params['constant']
l_avg = res_lols.params['constant']
exp_ret = win_rate*w_avg - loss_rate*l_avg
print('expected final ret:', exp_ret)
kelly = win_rate - (loss_rate)/(w_avg/l_avg)
print('kelly percentage:', kelly)
t_dbl = (np.log(2)/np.log(1+((exp_ret-cost)*at_risk/(exit_day+1)))*((latest-earliest)/len(ad)))
print('time to double (calendar days):', t_dbl.days)
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

###linear prediction model

l = 0.25
h = 0.75
regressors = ['constant', 'open']


qr = QuantReg(sample['win'], sample[regressors])
res_l = qr.fit(q=l)
res_u = qr.fit(q=h)
res_med = qr.fit(q=.5)
    
ols = OLS(sample['win'], sample[regressors])
res_ols = ols.fit()
    
tab = summary_col(
        [res_l, res_u, res_med, res_ols],
        model_names = [f'{l}', f'{h}', 'median', 'average'],
        info_dict={
        'N':lambda x: "{0:d}".format(int(x.nobs)),
}
)

tab.title = f'Linear Prediction Model'

print(tab)
