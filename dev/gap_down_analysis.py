# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
#import ohlc and ticker tables and format
##############################################################################

data = pd.read_csv(
    '../data/all of finviz/ohlc_2021-01-06_2022-01-04.csv',
    index_col = 'Unnamed: 0',
)
tickers = pd.read_csv(
    '../data/database/tickers.csv',
    index_col = 'Unnamed: 0',
)

tickers.rename(columns={'0':'ticker'}, inplace=True)
data.t = pd.to_datetime(data.t)
data.sort_values(['ticker', 't'], inplace = True)
data.reset_index(inplace=True, drop = True)

##############################################################################
#calculate variables
##############################################################################


#calculate returns
data['exit'] = data.o.shift(-1)
data['ret'] = data.exit - data.o
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
    (data.gap <= -.15)
    & (data.o <= 5)
    & (data.o >= 1)
    & (data.gap_t <= pd.Timedelta(4, unit= 'D'))
    #& (data.pc_ret <= 2)
]

##############################################################################
#evaluate sample
##############################################################################

print('\n stocks \n',sample.pc_ret.describe())
stock_wins = sample.loc[sample.pc_ret >0].shape[0]
print(f'stock win rate: {stock_wins/sample.shape[0]}')

##############################################################################
#simulate strategy
##############################################################################

#simulation parameters 
init_value = 1000
at_risk = 0.25


#run simulation
pc_ret_by_date = sample.groupby('t').pc_ret.mean()
value_ts = []
value = init_value
for t, pc_ret in pc_ret_by_date.items():
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
results['returns_ma'] = results.returns.rolling(5).mean()



##############################################################################
#evaluate strategy
##############################################################################

#plot results
fig1, ax1 = plt.subplots(figsize = (8,6))
fig2, ax2 = plt.subplots(figsize = (8,6))
fig3, ax3 = plt.subplots(figsize = (8,6))

results.returns.plot(ax = ax1, grid = True)
results.returns_ma.plot(ax = ax1, grid = True)
results.n_trades.plot(ax = ax2)
results.value.plot(ax = ax3)

print('\n strategy \n', results.returns.describe())
strategy_wins = results.loc[results.returns >0].shape[0]
print(f'strategy win rate: {strategy_wins/results.shape[0]}')

#plot ACF and PACF of outcome variable

#remember to drop NaNs
outcome_ts = results.returns_ma.dropna()

ma_ACF, ma_ACF_ci = sm.tsa.acf(
    outcome_ts, 
    alpha = 0.05, 
    nlags=40, 
    fft=False,
)
fig4, ax4 = plt.subplots(figsize = (8,6))
ax4.plot(ma_ACF, color = 'blue')
ax4.plot(ma_ACF_ci, color='blue', linestyle='dashed', alpha=0.5)
ax4.grid(True)

ma_PACF, ma_PACF_ci = sm.tsa.pacf(
    outcome_ts, 
    alpha = 0.05, 
    nlags = 40,
)
fig5, ax5 = plt.subplots(figsize = (8,6))
ax5.plot(ma_PACF, color = 'blue')
ax5.plot(ma_PACF_ci, color='blue', linestyle='dashed', alpha = 0.5)
ax5.grid(True)


