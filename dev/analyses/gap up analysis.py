#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 19:31:23 2022

@author: ivananich
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm


##############################################################################
#declare functions
##############################################################################
def get_ts(ticker, feature):
    ts = data.loc[data.ticker == ticker]
    ts = ts[[feature, 't']]
    ts.plot(y = feature, x = 't')
    return ts

def show(ascending):
    return sample.sort_values('pc_ret', ascending = ascending)[['ticker', 't', 'prev_c','o', 'exit', 'pc_ret']].head(20)


    
    
    

##############################################################################
#import ohlc and general tables and format
##############################################################################

data = pd.read_csv(
    #'../data/all of finviz/ohlc_2021-01-06_2022-01-04.csv',
    '../data/database/ohlc.csv',
    index_col = 'Unnamed: 0',
    low_memory = False,
)
exchanges = pd.read_csv(
    '../data/database/general.csv',
    usecols=['Ticker', 'exchange'],
    low_memory = False,
)

exchanges.rename(columns={'Ticker':'ticker'}, inplace=True)
exchanges.dropna(inplace = True)
exchanges.drop_duplicates(inplace = True)
data.t = pd.to_datetime(data.t)
data.sort_values(['ticker', 't'], inplace = True)
data.reset_index(inplace=True, drop = True)


##############################################################################
#calculate variables
##############################################################################

#assign exchange to ticker
data = data.merge(exchanges, on = 'ticker', how='left')

#calculate returns
data['exit'] = data.o.shift(-1)
data['ret'] = data.o - data.exit
mask_index = data.loc[data.ticker != data.ticker.shift(-1)].index
data.loc[mask_index, 'ret'] = np.nan
data['pc_ret'] = data.ret/data.o
    

#calculate gaps
data['prev_c'] = data.c.shift(1)
mask_index = data.loc[data.ticker != data.ticker.shift(1)].index
data.loc[mask_index, 'prev_c'] = np.nan
data['gap'] = (data.o - data.prev_c)/data.prev_c

#confirm gap
data['prev_t'] = data.t.shift(1)
mask_index = data.loc[data.ticker != data.ticker.shift(1)].index
data.loc[mask_index, 'prev_t'] = np.nan
data['gap_t'] = data.t - data.prev_t

##############################################################################
#create sample
##############################################################################


sample = data.loc[
    (data.gap >= 0.25)
    & (data.o <= 5)
    & (data.o >= .1)
    & (data.v > 1000000)
    #& (data.t >= '2022-01-01')
    & (data.gap_t <= pd.Timedelta(4, unit= 'D'))
]

#handle stocks with identifier letters
drop_nasdaq = sample.loc[(sample.ticker.str.len() >= 5) & (sample.exchange == 'nasdaq')]
sample.drop(drop_nasdaq.index, inplace = True)
drop_nyse = sample.loc[(sample.ticker.str.len() >= 4) & (sample.exchange == 'nyse')]
sample.drop(drop_nyse.index, inplace = True)
drop_amex = sample.loc[(sample.ticker.str.len() >= 4) & (sample.exchange == 'amex')]
sample.drop(drop_amex.index, inplace = True)

##############################################################################
#evaluate sample
##############################################################################

print('\n stocks \n',sample.pc_ret.describe(percentiles=[.01,0.25,.5,0.75,.99]))
stock_wins = sample.loc[sample.pc_ret >0].shape[0]
print(f'stock win rate: {stock_wins/sample.shape[0]}')

##############################################################################
#simulate strategy
##############################################################################

#simulation parameters 
init_value = 300
at_risk = 0.25


#run simulation
pc_ret_by_date = sample.groupby('t').pc_ret.mean()

#implement into full date range
earliest = pc_ret_by_date.index.min()
latest = pc_ret_by_date.index.max()
date_range = pd.date_range(start=earliest, end=latest)
pc_ret_by_date = pd.Series(pc_ret_by_date, index = date_range)


value_ts = []
value = init_value
for t, pc_ret in pc_ret_by_date.fillna(0).items():
    position = at_risk*value
    ret = position * pc_ret
    value = value + ret
    value_ts.append(value)
value_ts = pd.Series(value_ts, index = pc_ret_by_date.index)

#store results of simulation
results = pd.DataFrame(
    {
     'returns': pc_ret_by_date,
     'n_trades': sample.groupby('t').pc_ret.count(),
     'value': value_ts,
     },
    index = pc_ret_by_date.index,
)


#calculate moving average of returns
#Note window = 5 is a 4 lag window
results['returns_ma'] = results.returns.rolling(20, min_periods = 1).mean()



##############################################################################
#evaluate strategy
##############################################################################

#plot results
fig1, ax1 = plt.subplots(figsize = (8,6))
fig2, ax2 = plt.subplots(figsize = (8,6))
fig3, ax3 = plt.subplots(figsize = (8,6))
fig4, ax4 = plt.subplots(figsize = (8,6))


ax1.scatter(results.index, results.returns, s = 2)
results.returns_ma.plot(ax = ax1, grid = True, color='red')
ax2.bar(results.index, results.n_trades)
ax2.grid(True)
results.value.plot(ax = ax3)
results.value.plot(ax = ax4, logy=True)


print('\n strategy \n', results.returns.describe(percentiles=[.01,0.25,.5,0.75,.99]))
strategy_wins = results.loc[results.returns >0].shape[0]
print(f'strategy win rate: {strategy_wins/results.dropna().shape[0]}')
print(f'Quasi-sharp ratio: {results.returns.mean()/results.returns.std()}')

# #plot ACF and PACF of outcome variable

# #remember to drop NaNs
# outcome_ts = results.returns.dropna()

# ma_ACF, ma_ACF_ci = sm.tsa.acf(
#     outcome_ts, 
#     alpha = 0.05, 
#     nlags=40, 
#     fft=False,
# )
# fig4, ax4 = plt.subplots(figsize = (8,6))
# ax4.plot(ma_ACF, color = 'blue')
# ax4.plot(ma_ACF_ci, color='blue', linestyle='dashed', alpha=0.5)
# ax4.grid(True)

# ma_PACF, ma_PACF_ci = sm.tsa.pacf(
#     outcome_ts, 
#     alpha = 0.05, 
#     nlags = 40,
# )
# fig5, ax5 = plt.subplots(figsize = (8,6))
# ax5.plot(ma_PACF, color = 'blue')
# ax5.plot(ma_PACF_ci, color='blue', linestyle='dashed', alpha = 0.5)
# ax5.grid(True)


