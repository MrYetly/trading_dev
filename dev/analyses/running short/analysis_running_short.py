#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 08:34:45 2021

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
entry_day = 0
exit_day = 5
exit_time = 'close'
entry_time = 'open'
intra_exit = pd.DataFrame(
        {
                'hour': [15,10],
                'minute' : [30,0],
        }
)
initial_val = 1000000
risk_ref_metric = 'high'
risk_ref_day = -1
risk_per_trade = 100
mom_ref_day = -1
mom_ref_metric = 'close'
#don't set checking=True if exit_day=1 and exit_time='open'
#checking is broken for now
checking = False
check_range = range(entry_day+1, exit_day)
stoploss = True

#create new variables
master_daily = pd.read_csv('../data/running short 05242021 agg -253 5.csv')
master_intra = pd.read_csv('../data/running short 05242021 agg pt1 intra 30 min.csv')
df = pd.read_csv('../data/running short 05242021 agg pt2 intra 30 min.csv')
master_intra = master_intra.append(df)
print('Building master_daily DF')
master_daily = master_daily.rename(columns = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume',
                'init date': 'init_date',
            },
)

master_intra = master_intra.rename(columns = {
                'init date': 'init_date',
            },
)

master_daily.date = pd.to_datetime(master_daily.date)
master_daily.init_date = pd.to_datetime(master_daily.init_date)
earliest = datetime.strptime('2020-12-21', '%Y-%m-%d')
latest = datetime.strptime('2021-05-17', '%Y-%m-%d')
master_daily = master_daily.loc[master_daily.init_date >= earliest]
master_daily = master_daily.loc[master_daily.init_date <= latest]

master_intra.time = pd.to_datetime(master_intra.time)
master_intra.init_date = pd.to_datetime(master_intra.init_date)
master_intra = master_intra.loc[master_intra.init_date >= earliest]
master_intra = master_intra.loc[master_intra.init_date <= latest]

#throw out innaccuracies
inaccuracies = [
        master_daily.loc[(master_daily.ticker == 'RELI') & (master_daily.init_date == '2021-02-09')],
]
for i in inaccuracies:
    master_daily = master_daily.drop(index=i.index)

#build sample DF
print('building sample DF')

#start sample with day 0 trade signals
sample = master_daily.loc[master_daily.delta == 0].copy()

#create entry price column
entries = master_daily.loc[master_daily.delta == entry_day][['ticker', 'init_date',entry_time]]
entries = entries.rename(columns = {entry_time:'entry'})
sample = sample.merge(entries, how='inner', on=['init_date','ticker'], validate='1:1')

#create lagged open and close columns
for i in range(1,5):
    lag_open = master_daily.loc[master_daily.delta == -i][['ticker', 'init_date','open']]
    lag_open = lag_open.rename(columns = {'open':f'open_lag{i}'})
    sample = sample.merge(lag_open, how='inner', on=['init_date','ticker'], validate='1:1')
    
    lag_close = master_daily.loc[master_daily.delta == -i][['ticker', 'init_date','close']]
    lag_close = lag_close.rename(columns = {'close':f'close_lag{i}'})
    sample = sample.merge(lag_close, how='inner', on=['init_date','ticker'], validate='1:1')

#create gap column
prev_close = master_daily.loc[master_daily.delta == -1][['ticker', 'init_date','close']]
prev_close = prev_close.rename(columns={'close':'prev_close'})
sample = sample.merge(prev_close, how='inner', on=['init_date','ticker'], validate='1:1')
sample['gap']= (sample.open - sample.prev_close)/sample.prev_close

#create risk_ref column
rr = master_daily.loc[master_daily.delta == risk_ref_day][['ticker', 'init_date',risk_ref_metric]]
rr = rr.rename(columns = {risk_ref_metric:'risk_ref'})
sample = sample.merge(rr, how='inner', on=['init_date','ticker'], validate='1:1')
sample['stop_ret'] = (sample.entry - sample.risk_ref)/sample.entry    
    
#create high of day columns for stoploss calculation
for delta in range(entry_day, final_day+1):
    day = master_daily.loc[master_daily.delta == delta][['init_date','ticker', 'high']]
    day = day.rename(columns={'high':f'high_{delta}'})
    sample = sample.merge(day, how='inner', on=['init_date','ticker'], validate='1:1')
    
    
#create high as of day columns for daily stoploss calculation
#ex: for exit at close, h_as_of_2 = max(high_0, high_1, high_2)
for delta in range(entry_day, final_day+1):
    if exit_time == 'open':
        
        if entry_day == 0 and delta == 0:
            continue
        
        #if exiting at open, high as of doesn't include high of last day
        high_range = range(entry_day, delta)
        
        
    else:
        high_range = range(entry_day, delta+1)

    highs = [f'high_{i}' for i in high_range]
    highs = sample[highs]
    sample[f'h_as_of_{delta}'] = highs.max(axis=1)
            
#create high as of time for day 0 intraday stoploss calculation
for i, t in intra_exit.iterrows():
    window = master_intra.loc[
            (master_intra.time <= master_intra.init_date + np.timedelta64(int(t.hour),'h') + np.timedelta64(int(t.minute),'m'))
            & (master_intra.time > master_intra.init_date + np.timedelta64(9,'h') + np.timedelta64(30,'m'))
            ][['init_date', 'ticker','high']]
    window = window.groupby(['init_date', 'ticker'])
    window = window.max().reset_index()
    window = window.rename(columns={'high':f'h_as_of_{t.hour}{t.minute}'})
    sample = sample.merge(window, how='inner', on=['init_date','ticker'], validate='1:1')
    
    
#calculate daily returns
if exit_time == 'open' or exit_time == 'close':
    for delta in range(entry_day+1, final_day+1):
        
        day = master_daily.loc[master_daily.delta == delta][['init_date','ticker', exit_time]]
        day = day.rename(columns={exit_time:f'exit_{delta}'})
        sample = sample.merge(day, how='inner', on=['init_date','ticker'], validate='1:1')
        sample[f'return_{delta}'] = (sample.entry - sample[f'exit_{delta}'])/sample.entry

        #if stopped out, set return = stopped return  
        if stoploss == True:
            
            #create stopped variable to track if stopped
            sample[f'stopped_{delta}'] = np.nan
            
            stopped = sample.loc[sample[f'h_as_of_{delta}'] >= sample.risk_ref]
            sample.loc[stopped.index, f'return_{delta}'] = sample.loc[stopped.index, 'stop_ret']
            
            #track stoppages
            sample.loc[stopped.index, f'stopped_{delta}'] = True
            not_stopped = sample.loc[sample[f'h_as_of_{delta}'] < sample.risk_ref]            
            sample.loc[not_stopped.index, f'stopped_{delta}'] = False
            
    #
    if entry_day == 0 and entry_time == 'open' and exit_time == 'close':
        
        day = master_daily.loc[master_daily.delta == 0][['init_date','ticker', 'close']]
        day = day.rename(columns={'close':f'exit_0'})
        sample = sample.merge(day, how='inner', on=['init_date','ticker'], validate='1:1')
        sample[f'return_0'] = (sample.entry - sample[f'exit_0'])/sample.entry
   
        #if stopped out, set return = stopped return  
        if stoploss == True:
            
            #create stopped variable to track if stopped
            sample[f'stopped_0'] = np.nan
            
            stopped = sample.loc[sample[f'h_as_of_0'] >= sample.risk_ref]
            sample.loc[stopped.index, f'return_0'] = sample.loc[stopped.index, 'stop_ret']
            
            #track stoppages
            sample.loc[stopped.index, f'stopped_0'] = True
            not_stopped = sample.loc[sample[f'h_as_of_0'] < sample.risk_ref]            
            sample.loc[not_stopped.index, f'stopped_0'] = False

#calculate intraday returns (each half hour's close is the price at the explicit time stamp)
#NOTE: on finviz the half hour's open is the price at the time stamp
for i, t in intra_exit.iterrows():
    time = master_intra.loc[
            master_intra.time == master_intra.init_date + np.timedelta64(int(t.hour),'h') + np.timedelta64(int(t.minute),'m')
            ][['init_date', 'ticker','close']]
    time = time.rename(columns={'close':f'exit_{t.hour}{t.minute}'})
    time = time.drop_duplicates()
    sample = sample.merge(time, how='inner', on=['init_date','ticker'], validate='1:1')
    sample[f'return_{t.hour}{t.minute}'] = (sample.entry - sample[f'exit_{t.hour}{t.minute}'])/sample.entry
    
    
    if stoploss == True:
        
        #create stopped variable to track if stopped
        sample[f'stopped_{t.hour}{t.minute}'] = np.nan
        
        stopped = sample.loc[sample[f'h_as_of_{t.hour}{t.minute}'] >= sample.risk_ref]
        sample.loc[stopped.index, f'return_{t.hour}{t.minute}'] = sample.loc[stopped.index, 'stop_ret']
        
        #track stoppages
        sample.loc[stopped.index, f'stopped_{t.hour}{t.minute}'] = True
        not_stopped = sample.loc[sample[f'h_as_of_{t.hour}{t.minute}'] < sample.risk_ref]            
        sample.loc[not_stopped.index, f'stopped_{t.hour}{t.minute}'] = False

#calculate momentum    
#create momentum reference column
mr = master_daily.loc[master_daily.delta == mom_ref_day][['ticker', 'init_date',mom_ref_metric]]
mr = mr.rename(columns = {mom_ref_metric:'mom_ref'})
sample = sample.merge(mr, how='inner', on=['init_date','ticker'], validate='1:1')    
    
#specify 'to-date' lags
td_dict = {
        #adjusted ytd lag b/c of -1 mom_ref_day
        'ytd': -252,
        'hytd': -126,
        'qtd': -63,
        'mtd': -20,
        'wtd': -5,
}

for period, lag in td_dict.items():
    td_prices = master_daily.loc[master_daily.delta == mom_ref_day+lag][['init_date', 'ticker', 'close']]
    td_prices = td_prices.rename(columns={'close':f'close_{period}'})
    #momentum w.r.t open
    sample = sample.merge(td_prices, how='left', on=['init_date','ticker'])
    sample[period] = (sample.mom_ref - sample[f'close_{period}'])/sample[f'close_{period}']

#drop any rows with blanks
print('sample with NA:', sample.shape)
sample = sample.dropna()
print('sampel without NA:', sample.shape)

#create constant for regression analysis
sample['constant'] = 1.0

#mometum indicator is 1 if all to-date perfromance metrics are positive/negative
sample['umomentum'] = sample[['wtd','mtd','qtd','hytd','ytd']].gt(0.75).all(axis=1)
sample['dmomentum'] = sample[['wtd','mtd','qtd','hytd','ytd']].lt(0).all(axis=1) 

#backtest strategy

print('backtesting')

#limit sample for analysis

#default
sample = sample.loc[ 
            (sample.entry >= 0.01)
]


#variation 1
#sample = sample.loc[ 
#            (sample.entry >= 0.01)
#            & (sample.gap <= -0.1)
#            & (sample.open >= 10)
#] 

#variation 2
#sample = sample.loc[ 
#            (sample.entry >= 0.01)
#            & (sample.close_lag4 > sample.open_lag4)
#            & (sample.close_lag3 > sample.open_lag3)
#            & (sample.close_lag2 > sample.open_lag2)
#            & (sample.close_lag1 > sample.open_lag1)
#            #& (sample.gap >= -0.1)
#            #& (sample.open_lag3 > sample.close_lag4)
#            #& (sample.open_lag2 > sample.close_lag3)
#            #& (sample.open_lag1 > sample.close_lag2)
#] 


sample['final_ret'] = np.nan
sample['position'] = np.nan
sample['quantity'] = np.nan
sample['at_risk'] = np.nan

exit_df = pd.DataFrame()

all_dates = list(sample.date.unique())
all_dates.sort()
    
inv_val = [initial_val,]
net_returns = [np.nan,]
pc_returns = [np.nan,]
n_trades = [np.nan,]
positions = [0,]
at_risk = [np.nan,]

cost = 0.01    

print('initial value:', inv_val[-1])

#filler DF for inplay
blank = pd.DataFrame(columns = sample.columns)

play_range = exit_day-entry_day

inplay = [blank for i in range(play_range)]
    

#add extra days
extra_dates = list(master_daily.date.unique())
extra_dates.sort()
ref = extra_dates.index(all_dates[-1])
extra_dates = extra_dates[ref+1:ref+1+play_range]


#refactor to make things more sample dataframe oriented
for d in (all_dates+extra_dates):
    
        
    #add new trades signalled today (day 0) into inplay list
    new_trades = sample.loc[sample.init_date == d].copy()
    inplay = [new_trades,]+inplay
    
    #get trades played today
    play = inplay[entry_day]
    n_trades.append(play.shape[0])
    
    #calculate position, constrained by unexposed capital available to invest
    if play.shape[0] == 0:
        total_position = 0
        play.position = 0
        play.at_risk = 0
    else:
        
        #calculate quantity based on level of risk divided by risk spread
        play.quantity = risk_per_trade/(play.risk_ref - play.entry)
        
        #floor quantity for integer amount
        play.quantity = play.quantity.apply(np.floor)
        
        #calcualte dollar value of position
        play.position = play.quantity*play.entry
        
        #calculate total value of risk for trade
        play.at_risk = (play.risk_ref - play.entry)*play.quantity
        
        #calculate total position, to check if liquidity constraint binds
        total_position = play.position.sum()
        
        #calculate liquidity constrant
        inv_unexposed = inv_val[-1] - sum(positions[-(play_range+1):-1])
        
        #calculate max share of portfolio exposed
        max_exposure = inv_val[-1]/(play_range+1)
        
        #ch3ck if constraints bind 
        if max_exposure > inv_unexposed:
            max_exposure = inv_unexposed
        if total_position > max_exposure:
            pc_adjust = max_exposure/total_position
            play.position = play.position * pc_adjust
            play.quantity = play.position/play.entry
            play.at_risk = (play.risk_ref - play.entry)*play.quantity
            total_position = play.position.sum()
    positions.append(total_position)
    at_risk.append(play.at_risk.sum())
    
    
    #get trades exited that day
    exiting = pd.DataFrame(columns = sample.columns)
    
    
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
    
    #handle stoplosses, risk_ref is stop level
    if stoploss == True:
        
        #don't have stop losses affect final day if exiting at open
        if (entry_time == 'open' and exit_time == 'open') or entry_time == 'close':
            stop_range = range(entry_day+1, exit_day+1)
        else:
            stop_range = range(entry_day, exit_day+1)
        
        for i in stop_range:
            
            if exit_time == 'close' or exit_time == 'open':
            
                stop_check = inplay[i]
                
                if stop_check.shape[0] != 0:
                    
                    stop_out = stop_check.loc[stop_check[f'stopped_{i}'] == False].copy()
                    
                    exiting = exiting.append(stop_out)
                    
                    stop_keep = stop_check.loc[stop_check[f'stopped_{i}'] == True]
                    if stop_keep.shape[0] == 0:
                        stop_keep = blank
                    inplay[i] = stop_keep
            
            else:
                
                stop_check = inplay[i]
                
                if stop_check.shape[0] != 0:
                    stop_out = stop_check.loc[stop_check[f'stopped_{exit_time}'] == True].copy()
                    exiting = exiting.append(stop_out)
                    
                    stop_keep = stop_check.loc[stop_check[f'stopped_{exit_time}'] == False]
                    if stop_keep.shape[0] == 0:
                        stop_keep = blank
                    inplay[i] = stop_keep
                
    lastday = inplay.pop().copy()
    
    if lastday.shape[0] != 0: 
        exiting = exiting.append(lastday)
    
    if exit_day == 0 and exit_time != 'close':
        exiting['final_ret'] = exiting[f'return_{exit_time}']
    else:
        exiting['final_ret'] = exiting[f'return_{exit_day}']
        
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
                'at_risk': at_risk,
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
tab_loss.title = f'Analysis of Losers'

print(tab_win)
print(tab_loss)

with open(f'win_exit_{exit_day}_entry_{entry_day}_risk_{risk_ref_metric}.tex', 'w') as f:
    f.write(tab_win.as_latex())
with open(f'loss_exit_{exit_day}_entry_{entry_day}_risk_{risk_ref_metric}.tex', 'w') as f:
    f.write(tab_loss.as_latex())




#extended hold analysis (returns may differ when stop loss used)


hold_ret = [sample[f'return_{i}'].mean() - cost for i in range(entry_day,exit_day+1)]
hold_day = list(range(entry_day,exit_day+1))

for i,t in intra_exit.iterrows():
    ret = sample[f'return_{t.hour}{t.minute}'].mean() - cost
    day = -(i+1)
    hold_ret = [ret,]+hold_ret
    hold_day = [day,]+hold_day

labels = ['day'+str(i)+exit_time for i in range(entry_day,exit_day+1)]
labels = ['10:00','15:30',]+labels

ext_ret = pd.DataFrame(
        {
                'day': hold_day,
                'exp_ret': hold_ret,
})

fig2, ax2 = plt.subplots(figsize = (8,5))
ext_ret.plot(
        x = 'day',
        y = 'exp_ret',
        ax=ax2,
        title = f'Average Returns for Each Exit, Costs = {cost*100}%, Stoploss = {stoploss}',
        ylim = (-.2,.2),
        grid = True,
)

ax2.set(xlabel = 'Holding Day', ylabel = 'Percent return')
ax2.set_xticks([-2, -1,0,1,2,3,4,5])
ax2.set_xticklabels(labels)

#fig2.savefig(f'avg_ret_risk_{risk_ref_metric}_var_1.pdf')

if stoploss == True:
    
    pc_stopped = [sample[f'stopped_{i}'].mean() for i in range(entry_day,exit_day+1)]
    for i,t in intra_exit.iterrows():
        stop = sample[f'stopped_{t.hour}{t.minute}'].mean()
        pc_stopped = [stop,]+pc_stopped
    
    ext_stop = pd.DataFrame(
        {
                'day': hold_day,
                'pc_stopped': pc_stopped,
    })
    
    fig3, ax3 = plt.subplots(figsize = (8,5))

    ext_stop.plot(
            x = 'day',
            y = 'pc_stopped',
            ax=ax3,
            title = f'Percent of Trades Stopped As of Exit',
            ylim = (0,1),
            grid = True,
    )
    
    ax3.set(xlabel = 'Holding Day', ylabel = 'Percent Stopped')
    ax3.set_xticks([-2,-1,0,1,2,3,4,5])
    ax3.set_xticklabels(labels)
    
    #fig3.savefig(f'pc_stop_risk_{risk_ref_metric}_var_1.pdf')
