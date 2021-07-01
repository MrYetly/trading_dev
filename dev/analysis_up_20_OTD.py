#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 07:49:21 2021

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
entry_day = 1
exit_time = 'open'
entry_time = 'open'
exit_day = 3
initial_val = 1000
#don't set checking=True if exit_day=1 and exit_time='open'
#checking is broken for now
checking = False
check_range = range(4, exit_day)

#create new variables
master = pd.read_csv('../data/up 20 OTD 05192021 agg -253 5.csv')
df = pd.read_csv('../data/up 20 OTD 05192021 check agg -253 5.csv')
df2 = pd.read_csv('../data/up 20 OTD 05192021 check pt2 agg -253 5.csv')
master = master.append(df)
master = master.append(df2)
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
earliest = datetime.strptime('2021-01-02', '%Y-%m-%d')
latest = datetime.strptime('2021-05-12', '%Y-%m-%d')
master = master.loc[master.init_date >= earliest]
master = master.loc[master.init_date <= latest]

#throw out innaccuracies
inaccuracies = [
        master.loc[master['ticker'] == 'SPHS'],
        master.loc[master['ticker'] == 'SNNAQ'],
        master.loc[(master['ticker'] == 'RELI') & (master.init_date == '2021-02-04')],
        master.loc[(master['ticker'] == 'NVOS') & (master.init_date == '2021-01-28')],
        master.loc[(master['ticker'] == 'NVOS') & (master.init_date == '2021-01-29')],
        master.loc[(master['ticker'] == 'REPX') & (master.init_date == '2021-02-24')],
]

for i in inaccuracies:
    master = master.drop(index=i.index)


#build sample DF
print('building sample DF')

#start sample with day 0 trade signals
sample = master.loc[master.delta == 0].copy()

#create entry price column
entries = master.loc[master.delta == entry_day][['ticker', 'init_date',entry_time]]
entries = entries.rename(columns = {entry_time:'entry'})
sample = sample.merge(entries, how='inner', on=['init_date','ticker'], validate='1:1')

#calculate returns
for delta in range(entry_day+1, final_day+1):
    day = master.loc[master.delta == delta][['init_date','ticker', exit_time]]
    day = day.rename(columns={exit_time:f'exit_{delta}'})
    sample = sample.merge(day, how='inner', on=['init_date','ticker'], validate='1:1')
    sample[f'return_{delta}'] = (sample[f'exit_{delta}'] - sample.entry)/sample.entry

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


##calculate  momentum (from open of signal day)
##specify 'to-date' lags
#td_dict = {
#        'ytd': -253,
#        'hytd': -126,
#        'qtd': -63,
#        'mtd': -20,
#        'wtd': -5,
#}
#
#for period, lag in td_dict.items():
#    td_prices = master.loc[master.delta == lag][['init_date', 'ticker', 'close']]
#    td_prices = td_prices.rename(columns={'close':f'close_{period}'})
#    #momentum w.r.t open
#    sample = sample.merge(td_prices, how='left', on=['init_date','ticker'])
#    sample[period] = (sample.open - sample[f'close_{period}'])/sample[f'close_{period}']
    
#drop any rows with blanks
print('sample with NA:', sample.shape)
sample = sample.dropna()
print('sampel without NA:', sample.shape)

#create constant for regression analysis
sample['constant'] = 1.0

##mometum indicator is 1 if all to-date perfromance metrics are positive/negative
#sample['umomentum'] = sample[['wtd','mtd','qtd','hytd','ytd']].gt(0).all(axis=1)
#sample['dmomentum'] = sample[['wtd','mtd','qtd','hytd','ytd']].lt(0).all(axis=1)

#calculate signal day change
sample['day_change'] = (sample.close-sample.open)/sample.open

#backtest strategy

print('backtesting')

#limit sample for analysis


##variation 1
sample = sample.loc[ 
            (sample.close >= 0.01)
            & (sample.day_change >= .4)
            #& (sample.close <= 21)
            & (sample.volume <= 20000000)
            #& ((sample.high - sample.close)/sample.close <= 0.2 )
]

#variation 2
#sample = sample.loc[
#        (sample.close >= 0.01)
#        & ((sample.high - sample.close)/sample.close >= 0.4 )
#        #& (sample.open < sample.close_lag1)
#        #& (sample.close_lag1 < sample.open_lag1)
#        #& (sample.open_lag1 > sample.close_lag2)
#        #& (sample.close_lag2 < sample.open_lag2)
#        #& (sample.open_lag2 > sample.close_lag3)
#        #& (sample.close_lag3 < sample.open_lag3)
#        #& (sample.open_lag3 > sample.close_lag4)
#        #& (sample.close_lag4 < sample.open_lag4)
#]
    
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

at_risk = 1.0
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
)
test_res.pc_return.plot(
        ax=ax[1],
        title = f'Return to Strategy',
)
test_res.n_trades.plot(
        ax=ax[2],
        title = f'Number of Trades',
)
    
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

#extended hold analysis

hold_ret = [sample[f'return_{i}'].mean() for i in range(entry_day+1,exit_day+1)]

ext_ret = pd.DataFrame(
        {
                'day': list(range(entry_day+1,exit_day+1)),
                'exp_ret': hold_ret,
})

fig2, ax2 = plt.subplots(figsize = (8,5))
ext_ret.plot(
        x = 'day',
        y = 'exp_ret',
        ax=ax2,
        title = 'Extended Hold Returns',
)

ax2.set(xlabel = 'Holding Day', ylabel = 'Percent return')