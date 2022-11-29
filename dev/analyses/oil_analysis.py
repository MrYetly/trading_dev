# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


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

general = pd.read_csv(
    '../data/database/general.csv',
    usecols=['Ticker', 'exchange', 'Industry'],
    low_memory = False,
)
general.rename(columns={'Ticker':'ticker', 'Industry': 'industry'}, inplace=True)

general.dropna(inplace = True)
general.drop_duplicates(inplace = True)

data.t = pd.to_datetime(data.t)
data.sort_values(['ticker', 't'], inplace = True)
data.reset_index(inplace=True, drop = True)


##############################################################################
#calculate variables
##############################################################################

#assign exchange to ticker
data = data.merge(general, on = 'ticker', how='left')

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

#calculate standardized gaps
gap_std = data[['ticker', 'industry', 'gap']].groupby(['ticker', 'industry'], as_index=False).std()
gap_std.rename(columns={'gap':'gap_std'}, inplace=True)
data = data.merge(gap_std, on=['ticker', 'industry'], how='left')

gap_mean = data[['ticker', 'industry', 'gap']].groupby(['ticker', 'industry'], as_index=False).mean()
gap_mean.rename(columns={'gap':'gap_mean'}, inplace=True)
data = data.merge(gap_mean, on=['ticker', 'industry'], how='left')

data['gap_s'] = (data.gap-data.gap_mean)/data.gap_std

#confirm gap
data['prev_t'] = data.t.shift(1)
mask_index = data.loc[data.ticker != data.ticker.shift(1)].index
data.loc[mask_index, 'prev_t'] = np.nan
data['gap_t'] = data.t - data.prev_t


#find gap of signal index: 'BNO'
signal_gap = data.loc[data.ticker == 'BNO'][['t', 'gap_s', 'gap']]
signal_gap.rename(columns = {'gap_s': 'signal_gap_s', 'gap': 'signal_gap'}, inplace=True)
data = data.merge(signal_gap, on = 't', how='left')



##############################################################################
#create sample
##############################################################################
init_date = '2020-01-01'
end_date =  '2022-05-20'

sample = data.loc[
    (data.industry.str.contains('Oil') == True)
    & (data.o <= 3)
    & (data.o >= 0.1)
    & (data.v >= 500000)
    & (data.gap_s - data.signal_gap_s <= 0)
    & (data.t >= init_date)
    & (data.gap_t <= pd.Timedelta(4, unit= 'D'))
    & (data.gap_t > pd.Timedelta(0, unit='D'))
].copy()

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
#backtest strategy
##############################################################################

#simulation parameters 
init_value = 1200
at_risk = 0.4


#run simulation
pc_ret_by_date = sample.groupby('t').pc_ret.mean()

#implement into full date range
init_date = pc_ret_by_date.index.min()
end_date = pc_ret_by_date.index.max()
date_range = pd.date_range(start=init_date, end=end_date)
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

#get benchmark returns from investing in signal index
signal_value = data.loc[(data.ticker == 'BNO') & (data.t >= init_date)][['t', 'c']]
signal_value['bench_value'] = init_value*signal_value.c/signal_value.c.iloc[0]
signal_value.set_index('t', inplace = True)
results = pd.concat([results, signal_value], axis = 1, join = 'outer')

##############################################################################
#monte carlo simulate strategy
##############################################################################

monte_results = {}
for i in range(1000):
    
    value_ts = []
    value = init_value
    for t in range(100):
        position = at_risk*value
        pc_ret = results.returns.dropna().sample().iloc[0]
        ret = position * pc_ret
        value = value + ret
        value_ts.append(value)
    monte_results[str(i)] = value_ts
    
monte_results = pd.DataFrame(monte_results)
monte_results['q95_value'] = monte_results.quantile(0.95, axis=1)
monte_results['q5_value'] = monte_results.quantile(0.05, axis=1)
monte_results['median_value'] = monte_results.median(axis=1)

    
##############################################################################
#evaluate strategy
##############################################################################

#plot results
figsize = (8,6)
fig1, ax1 = plt.subplots(figsize = figsize)
fig2, ax2 = plt.subplots(figsize = figsize)
fig3, ax3 = plt.subplots(figsize = figsize)
fig4, ax4 = plt.subplots(figsize = figsize)
fig5, ax5 = plt.subplots(figsize = figsize)
fig6, ax6 = plt.subplots(figsize = figsize)

ax1.scatter(results.index, results.returns, s = 2)
results.returns_ma.plot(ax = ax1, grid = True, color='red')
ax2.bar(results.index, results.n_trades)
ax2.grid(True)
results[['value', 'bench_value']].plot(ax = ax3)
results[['value', 'bench_value']].plot(ax = ax4, logy=True)

#plot monte carlo sims
for col in monte_results.columns:
    if col != 'mean_value' and col != 'median_value':
        monte_results[col].plot(ax=ax5, alpha = 0.1, color = 'grey', legend=False)
monte_results[['q95_value', 'q5_value', 'median_value']].plot(ax=ax5)

for col in monte_results.columns:
    if col != 'q5_value' and col != 'median_value' and col != 'q95_value':
        monte_results[col].plot(ax=ax6, alpha = 0.01, color = 'grey', legend=False, logy=True)
monte_results[['q95_value', 'q5_value', 'median_value']].plot(ax=ax6, logy=True)


print('\n backtest \n', results.returns.describe(percentiles=[.01,0.25,.5,0.75,.99]))
strategy_wins = results.loc[results.returns >0].shape[0]
print(f'backtest win rate: {strategy_wins/results.dropna().shape[0]}')
print(f'backtest quasi-sharp ratio: {results.returns.mean()/results.returns.std()}')

print(f'Monte Carlo q=0.95: {monte_results.q95_value.iloc[-1]}')
print(f'Monte Carlo median: {monte_results.median_value.iloc[-1]}')
print(f'Monte Carlo q=0.5: {monte_results.q5_value.iloc[-1]}')


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


