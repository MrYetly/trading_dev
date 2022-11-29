#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 13:46:15 2021

@author: ivananich
"""

import pandas as pd
import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.regression.linear_model import OLS
from statsmodels.iolib.summary2 import summary_col
from matplotlib import pyplot as plt
from datetime import datetime
from gap_down_15_clean import clean_master


def check_returns(order_by = 0, plot = True):
    
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

def check_gap(plot = True):
    
    df = sample
    sub_df = df.sort_values('gap')[['ticker', 'init_date','prev_close', 'open','gap']]
    
    if plot == True:
        #create histogram
        return sub_df, sub_df.hist(
            column = 'gap',
            figsize = (8,5),
            bins = 50,
            )
        
    
    return sub_df

#Params (remember, with morning check)
final_day = 5
exit_day = 1
entry_day = 0
exit_time = 'open'
entry_time = 'open'
initial_val = 25000
#don't set checking=True if exit_day=1 and exit_time='open'
checking = False
check_range = range(1, exit_day)


#create new variables

print('Building master DF')
master = pd.read_csv('../../data/gap_down_15_2021-07-15 agg -253 5.csv')
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
earliest = datetime.strptime('2021-02-23', '%Y-%m-%d')
latest = datetime.strptime('2021-07-15', '%Y-%m-%d')
master = master.loc[master['init_date']>=earliest]
master = master.loc[master['init_date']<=latest]

app_1 = pd.read_csv('../../data/gap d 15 2021-02-22 agg -253 5.csv')
app_1 = app_1.rename(columns = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume',
                'init date': 'init_date',
            },
)

app_1['date'] = pd.to_datetime(app_1['date'])
app_1['init_date'] = pd.to_datetime(app_1['init_date'])
earliest = datetime.strptime('2021-01-04', '%Y-%m-%d')
latest = datetime.strptime('2021-02-22', '%Y-%m-%d')
app_1 = app_1.loc[app_1['init_date']>=earliest]
app_1 = app_1.loc[app_1['init_date']<=latest]


master = master.append(
    app_1,
    ignore_index=True,
)


app_2 = pd.read_csv('../../data/gap d 15 01112021 agg -253 5.csv')
app_2 = app_2.rename(columns = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume',
                'init date': 'init_date',
            },
)

app_2['date'] = pd.to_datetime(app_2['date'])
app_2['init_date'] = pd.to_datetime(app_2['init_date'])
earliest = datetime.strptime('2020-08-17', '%Y-%m-%d')
latest = datetime.strptime('2020-12-31', '%Y-%m-%d')
app_2 = app_2.loc[app_2['init_date']>=earliest]
app_2 = app_2.loc[app_2['init_date']<=latest]

master = master.append(
    app_2,
    ignore_index=True,
)


app_3 = pd.read_csv('../../data/gap d 15 01112021 check agg -253 5.csv')
app_3 = app_3.rename(columns = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume',
                'init date': 'init_date',
            },
)

app_3['date'] = pd.to_datetime(app_3['date'])
app_3['init_date'] = pd.to_datetime(app_3['init_date'])
earliest = datetime.strptime('2020-04-01', '%Y-%m-%d')
latest = datetime.strptime('2020-08-14', '%Y-%m-%d')
app_3 = app_3.loc[app_3['init_date']>=earliest]
app_3 = app_3.loc[app_3['init_date']<=latest]

master = master.append(
    app_3,
    ignore_index=True,
)

app_4 = pd.read_csv('../../data/gap_down_15_2021-10-17 agg -253 5.csv')
app_4 = app_4.rename(columns = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume',
                'init date': 'init_date',
            },
)

app_4['date'] = pd.to_datetime(app_4['date'])
app_4['init_date'] = pd.to_datetime(app_4['init_date'])
earliest = datetime.strptime('2021-07-16', '%Y-%m-%d')
latest = datetime.strptime('2021-10-08', '%Y-%m-%d')
app_4 = app_4.loc[app_4['init_date']>=earliest]
app_4 = app_4.loc[app_4['init_date']<=latest]

master = master.append(
    app_4,
    ignore_index=True,
)

#clean master dataframe
master = clean_master(master)

#build sample DF
print('building sample DF')

#start sample with day 0 trade signals
sample = master.loc[master.delta == 0].copy()


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
    day = master.loc[master.delta == delta][['init_date','ticker', exit_time]]
    day = day.rename(columns={exit_time:f'exit_{delta}'})
    sample = sample.merge(day, how='inner', on=['init_date','ticker'], validate='1:1')
    sample[f'return_{delta}'] = (sample[f'exit_{delta}'] - sample.entry)/sample.entry

#create gap column
prev_close = master.loc[master.delta == -1][['ticker', 'init_date', 'close']]
prev_close = prev_close.rename(columns = {'close': 'prev_close'})
sample = sample.merge(prev_close, how='inner', on=['init_date','ticker'], validate='1:1')
sample['gap'] = (sample.entry - sample.prev_close)/sample.prev_close

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
    sample[f'm_{label}'] = (sample.entry - sample[f'close_{label}'])/sample[f'close_{label}']



#drop any rows with blanks
print('sample with NA:', sample.shape)
sample = sample.dropna()
print('sampel without NA:', sample.shape)

#create constant for regression analysis
sample['constant'] = 1.0


#backtest strategy

print('backtesting')

#limit sample for analysis
sample = sample.loc[
        #leave out day with 54 trades
        (sample.init_date != '2021-02-23')
        #remove fed announcement day
        & (sample['init_date'] != np.datetime64('2020-06-11T00:00:00.000000000'))
        & (sample.entry >= 0.01)
        & (sample.init_date >= '2020-06-18')
        #& (sample.init_date <= '2020-11-18')
        & (sample.entry <= 5)
        & (sample.m_w <= 0)
        & (sample.m_m <= 0)
        & (sample.m_q <= 0)
        & (sample.m_hy <= 0)
        & (sample.m_y <= 0)
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

at_risk = .75
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
    
    
#dates seem to be shifted too far ahead
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
fig_1, ax_1 = plt.subplots(figsize = (8,6))
fig_2, ax_2 = plt.subplots(figsize = (8,6))
fig_3, ax_3 = plt.subplots(figsize = (8,6))
fig_4, ax_4 = plt.subplots(figsize = (8,6))
test_res.value.plot(
        ax=ax_1,
        title = 'Value of portfolio',
        grid = True,
)
test_res.pc_return.rolling(20).mean().plot(
        ax=ax_2,
        title = 'Return to Strategy',
        grid = True,
)
test_res.n_trades.plot(
        ax=ax_3,
        title = 'Number of Trades',
        grid = True,
)

test_res['log_value'] = np.log10(test_res.value)
test_res.log_value.plot(
        ax=ax_4,
        title = 'Value of portfolio',
        grid = True,
)

ax_1.set(xlabel = 'Trading Day', ylabel = 'Value of portfolio ($)')
ax_2.set(xlabel = 'Trading Day', ylabel = 'Percent return')
ax_3.set(xlabel = 'Trading Day', ylabel = 'Count')

    

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

#overall
h_o = 0.95
l_o = 0.05
ovr_qr = QuantReg(exit_df['final_ret'], exit_df[regressors])
res_ol = ovr_qr.fit(q=l_o)
res_ou = ovr_qr.fit(q=h_o)
res_omed = ovr_qr.fit(q=.5)

ovr_ols = OLS(exit_df['final_ret'], exit_df[regressors])
res_ools = ovr_ols.fit()


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
tab_ovr = summary_col(
        [res_ol, res_ou, res_omed, res_ools],
        model_names = [f'{l_o}', f'{h_o}', 'median', 'average'],
        info_dict={
        'N':lambda x: "{0:d}".format(int(x.nobs)),
        'Win rate':lambda x: "{:.2f}".format(win_rate),
}
)
tab_win.title = 'Analysis of Winners'
tab_loss.title = 'Analysis of Gap Losers'
tab_ovr.title = 'Analysis Overall'

print(tab_ovr)
print(tab_win)
print(tab_loss)

#with open(f'pump_dump_var_2_after_win.tex', 'w') as f:
#    f.write(tab_win.as_latex()[59:-25])
#with open(f'pump_dump_var_2_after_loss.tex', 'w') as f:
#    f.write(tab_loss.as_latex()[62:-25])
#extended hold analysis

'''
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


'''

