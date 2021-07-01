#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 09:30:39 2021

@author: ianich
"""


import pandas as pd
import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.regression.linear_model import OLS
from statsmodels.iolib.summary2 import summary_col
from matplotlib import pyplot as plt
from datetime import datetime

#Params 
final_day = 5
exit_time = 'close'
entry_day = 0
#entry level = change w.r.t. open
entry_level = 0.2
exit_day = 0
initial_val = 25000
#don't set checking=True if exit_day=1 and exit_time='open'
#checking is broken for now
checking = False
check_range = range(4, exit_day)

#create new variables
master = pd.read_csv('../data/high up 20 1 lt prev close lt 2 06072021 agg -253 5.csv')
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

master.date = pd.to_datetime(master.date)
master.init_date = pd.to_datetime(master.init_date)
earliest = datetime.strptime('2021-02-26', '%Y-%m-%d')
latest = datetime.strptime('2021-05-28', '%Y-%m-%d')
master = master.loc[master.init_date >= earliest]
master = master.loc[master.init_date <= latest]

#throw out innaccuracies
inaccuracies = [
        master.loc[master['ticker'] == 'SPHS'],
        master.loc[master['ticker'] == 'SNNAQ'],
        #master.loc[master['ticker'] == 'EMMS'],
        #master.loc[master['ticker'] == 'HYMCW'],
]

for i in inaccuracies:
    master = master.drop(index=i.index)


#build sample DF
print('building sample DF')

#start sample with day 0 trade signals
sample = master.loc[master.delta == 0].copy()

#create rise column
sample['rise'] = (sample.high - sample.open)/sample.open
print('Describe rise')
print(sample.rise.describe())

#create entry column (leave out stocks that don't rise high enough)
sample = sample.loc[sample.rise >= entry_level]
sample['entry'] = sample.open * (1+entry_level)

#create returns
if exit_time == 'open':
    adjust = 1
if exit_time == 'close':
    adjust = 0 

for delta in range(0+adjust, final_day+1):
    day = master.loc[master.delta == delta][['init_date','ticker', exit_time]]
    day = day.rename(columns={exit_time:f'exit_{delta}'})
    sample = sample.merge(day, how='inner', on=['init_date','ticker'], validate='1:1')
    sample[f'return_{delta}'] = (sample.entry - sample[f'exit_{delta}'])/sample.entry

#calculate lagged opens and closes
    
for delta in range(1,5):
    day = master.loc[master.delta == -delta][['init_date', 'ticker', 'open', 'close', 'volume']]
    day = day.rename(
            columns={
                    'open': f'open_lag{delta}',
                    'close': f'close_lag{delta}',
                    'volume': f'volume_lag{delta}',
                    }
    )
    sample = sample.merge(day, how='inner', on=['init_date', 'ticker'], validate='1:1')

#drop any rows with blanks
print('sample with NA:', sample.shape)
sample = sample.dropna()
print('sampel without NA:', sample.shape)

#create constant for regression analysis
sample['constant'] = 1.0


#backtest strategy

print('backtesting')

#limit sample for analysis

#variation 1 (entry at 0.2)
sample = sample.loc[ 
            (sample.open <= sample.close_lag1)
]




    
sample['final_ret'] = np.nan
sample['position'] = np.nan

exit_df = pd.DataFrame()

all_dates = list(sample.date.unique())
all_dates.sort()
    
inv_val = [initial_val,]
net_returns = [np.nan,]
pc_returns = [np.nan,]
n_trades = [np.nan,]
positions = [np.nan,]

at_risk = 0.1
cost = 0.00    

print('initial value:', inv_val[-1])
print('at risk:', at_risk)

#filler DF for inplay
blank = pd.DataFrame(columns = sample.columns)

inplay = [blank for i in range(exit_day-entry_day+1)]
    
#add extra days
extra_dates = list(master.date.unique())
extra_dates.sort()
ref = extra_dates.index(all_dates[-1])
extra_dates = extra_dates[ref+1:ref+1+exit_day-entry_day+1]

for d in all_dates+extra_dates:
    
        
    #add new trades signalled today (day 0) into inplay list
    new_trades = sample.loc[sample.init_date == d].copy()
    inplay = [new_trades,]+inplay
    
    #get trades played today
    play = inplay[entry_day]
    n_trades.append(play.shape[0])
    
    #calculate position, constrained by unexposed capital available to invest
    if play.shape[0] == 0:
        position = 0
        play['position'] = 0
    else:
        inv_unexposed = inv_val[-1] - sum(positions[-(exit_day-entry_day+1):-1])
        position = inv_val[-1]*at_risk*(1/(exit_day-entry_day+1))
        if position > inv_unexposed:
            position = inv_unexposed
        #distribute position among trades played that day
        inplay[entry_day]['position'] = position/play.shape[0]
    
    positions.append(position)
    
    #get trades exited that day
    exiting = pd.DataFrame(columns = sample.columns)
    lastday = inplay.pop().copy()
    
    if lastday.shape[0] != 0:
        
        lastday['final_ret'] = lastday[f'return_{exit_day}']
        exiting = exiting.append(lastday)
    
    #if checking, add losers to exiting
    if checking == True:
        
        for i in check_range:
            
            swinging = inplay[i]
            
            if swinging.shape[0] != 0:
                
                swingloss = swinging.loc[swinging[f'return_{i}'] <= 0].copy()
                swingloss['final_ret'] = swingloss[f'return_{i}']
                exiting = exiting.append(swingloss)
                
                swingkeep = swinging.loc[swinging[f'return_{i}'] > 0]
                if swingkeep.shape[0] == 0:
                    swingkeep = blank
                inplay[i] = swingkeep
                    
    exiting['net_ret'] = exiting['position']*(exiting['final_ret'] - cost)
    
    if exiting.shape[0] == 0:
        sum_net_ret = 0
        pc_day = 0
    else:
        sum_net_ret = exiting['net_ret'].sum()
        pc_day = exiting['final_ret'].mean()
        exit_df = exit_df.append(exiting, ignore_index = True)
            
    net_returns.append(sum_net_ret)
    pc_returns.append(pc_day)
    inv_val.append(inv_val[-1] + sum_net_ret)
    
    
    if inv_val[-1] <=0:
        break
    
    
        
ad = ['init_date',] + all_dates+extra_dates
test_res = pd.DataFrame(
        {
                'value': inv_val,
                'date': ad,
                'net_ret': net_returns,
                'n_trades': n_trades,
                'position': positions,
                'pc_return': pc_returns,
        }
)
    
    
print('final value:', inv_val[-1])
print('avg strategy return:', test_res['pc_return'].mean())
print('trading days:', len(ad))
print('avg trades per open exec:', test_res.n_trades.mean())
#format and output plots  
fig, ax = plt.subplots(3,1, figsize = (8,17))
test_res.value.plot(
        ax=ax[0],
        title = f'Value of portfolio',
        grid = True,
)
test_res.pc_return.plot(
        ax=ax[1],
        title = f'Return to Strategy',
        grid = True,
)
test_res.n_trades.plot(
        ax=ax[2],
        title = f'Number of Trades',
        grid = True,
)

fig.savefig('pump_dump_sim_oltc.png')

ax[0].set(xlabel = 'Trading Day', ylabel = 'Value of portfolio ($)')
ax[1].set(xlabel = 'Trading Day', ylabel = 'Percent return')
ax[2].set(xlabel = 'Trading Day', ylabel = 'Count')

    
#fig.savefig(f'backtest gap 25 short swing.png')

#analyze winners and losers


exit_df_win = exit_df.loc[exit_df['final_ret'] > 0]
exit_df_loss = exit_df.loc[exit_df['final_ret'] < 0]
    
    
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
win_qr = QuantReg(exit_df_win['final_ret'], exit_df_win[regressors])
res_wl = win_qr.fit(q=l)
res_wu = win_qr.fit(q=h)
res_wmed = win_qr.fit(q=.5)
    
win_ols = OLS(exit_df_win['final_ret'], exit_df_win[regressors])
res_wols = win_ols.fit()
    
#losers
#CHANGING SIGN OF LOSS
loss_qr = QuantReg(-exit_df_loss['final_ret'], exit_df_loss[regressors])
res_ll = loss_qr.fit(q=l)
res_lu = loss_qr.fit(q=h)
res_lmed = loss_qr.fit(q=.5)
    
loss_ols = OLS(-exit_df_loss['final_ret'], exit_df_loss[regressors])
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

with open(f'pump_dump_var_2_after_win.tex', 'w') as f:
    f.write(tab_win.as_latex()[59:-25])
with open(f'pump_dump_var_2_after_loss.tex', 'w') as f:
    f.write(tab_loss.as_latex()[62:-25])
#extended hold analysis

avg_ret = [sample[f'return_{i}'].mean() for i in range(0+adjust,final_day+1)]

ret_input = {
                'day': list(range(0+adjust,final_day+1)),
                'avg_ret': avg_ret,
}

quantiles = [0.25, 0.5, 0.75]

for q in quantiles:
    ret_input[f'{round(q*100)}_pc'] = [sample[f'return_{i}'].quantile(q) for i in range(0+adjust,final_day+1)]




ext_ret = pd.DataFrame(ret_input)

fig2, ax2 = plt.subplots(figsize = (8,5))
ext_ret.plot(
        x = 'day',
        y = 'avg_ret',
        ax=ax2,
        title = 'Extended Hold Returns',
        ylim = (-.2,.2),
        grid = True,
)

for q in quantiles:
    ext_ret.plot(
        x = 'day',
        y = f'{round(q*100)}_pc',
        ax=ax2,
        title = 'Extended Hold Returns',
        ylim = (-.2,.35),
        grid = True,
        linestyle = 'dashed',
)

ax2.set(xlabel = 'Holding Day', ylabel = 'Percent return')


fig2.savefig('pump_dump_ext_hold_oltc.png')
#Give back analysis


