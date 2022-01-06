#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 07:43:44 2021

@author: ianich
"""

import pandas as pd
import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.regression.linear_model import OLS
from statsmodels.iolib.summary2 import summary_col
from matplotlib import pyplot as plt
from clean_down_15_OTD import clean_master
import dt_functions.functions as fnc

#Params 
final_day = 5
exit_day = 2
entry_day = 1
exit_time = 'open'
entry_time = 'open'
initial_val = 25000
#don't set checking=True if exit_day=1 and exit_time='open'
checking = False
check_range = range(1, exit_day)

#import master
master = fnc.import_master(
    '../../data/down 15 OTD 05192021 agg -253 5.csv',
    start_date = '2021-03-01',
    end_date = '2021-05-12',
)

#run cleaning function
master = clean_master(master)

#build sample DF
print('building sample DF')

#start sample with day 0 trade signals
sample = master.loc[master.delta == 0].copy()

#create returns
sample = fnc.create_swing_returns(
    sample,
    master,
    entry_time = entry_time,
    entry_day = entry_day,
    exit_time = exit_time,
    start=1,
    end=final_day
    )

#create momentum indicators
sample = fnc.create_momentum_indicators(sample, master)

#calculate std of volume over past month
vol = master.loc[(master.delta < 0) & (master.delta >= -20)]
vol_std = vol.groupby(['ticker', 'init_date'], as_index = False).volume.std()
vol_std = vol_std.rename(columns={'volume':'v_std'})
sample = sample.merge(vol_std, how='left', on=['ticker', 'init_date'])

#calculate mean of volume over past month
vol_mean = vol.groupby(['ticker', 'init_date'], as_index = False).volume.mean()
vol_mean = vol_mean.rename(columns={'volume':'v_mean'})
sample = sample.merge(vol_mean, how='left', on=['ticker', 'init_date'])


#drop any rows with blanks
print('sample with NA:', sample.shape)
sample = sample.dropna()
print('sampel without NA:', sample.shape)

#add constant
sample['constant'] = 1.0

#backtest strategy

print('backtesting')

#limit sample for analysis
sample = sample.loc[
        (sample.entry >= 0.01)
        #& (sample.volume < sample.v_mean+sample.v_std)
        #& (sample.volume > sample.v_mean-sample.v_std)
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

at_risk = 0.75
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
        title = 'Value of portfolio',
        grid = True,
)
test_res.pc_return.plot(
        ax=ax[1],
        title = 'Return to Strategy',
        grid = True,
)
test_res.n_trades.plot(
        ax=ax[2],
        title = 'Number of Trades',
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