# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


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
data.reset_index(inplace=True)



def get_ts(ticker, feature):
    ts = data.loc[data.ticker == ticker]
    ts = ts[[feature, 't']]
    ts.plot(y = feature, x = 't')
    return ts

#calculate day return 
data['return_otd'] = (data.c - data.o)/data.o

#calculate entry
data['entry'] = data.o.shift(-1)
mask_index = data.loc[data.ticker != data.ticker.shift(-1)].index
data.loc[mask_index, 'entry'] = np.nan

#confirm gap of entry
data['entry_t'] = data.t.shift(-1)
mask_index = data.loc[data.ticker != data.ticker.shift(-1)].index
data.loc[mask_index, 'entry_t'] = np.nan
data['entry_gap_t'] = data.entry_t - data.t

#calculate returns
for d in [2,3,4,5,6]:
    data[f'exit_{d}'] = data.o.shift(-d)
    data[f'return_{d}'] = data[f'exit_{d}'] - data.entry
    mask_index = data.loc[data.ticker != data.ticker.shift(-d)].index
    data.loc[mask_index, f'return_{d}'] = np.nan
    data[f'pc_ret_{d}'] = data[f'return_{d}']/data.entry

sample = data.loc[
    (data.return_otd >= .40)
    & (data.o <= 20)
    & (data.o >= 1)
    #& (data.v < 20_000_000)
    & (data.entry_gap_t <= pd.Timedelta(4, unit= 'D'))
    #& (data.pc_return <= 2)
]

exit_day = 6

print('\n stocks \n',sample[f'pc_ret_{exit_day}'].describe())

stock_wins = sample.loc[sample[f'pc_ret_{exit_day}'] >0].shape[0]
print(f'stock win rate: {stock_wins/sample.shape[0]}')

def show(ascending):
    return sample.sort_values(f'pc_ret_{exit_day}', ascending = ascending)[['ticker', 't', 'prev_c','o', 'o_1', 'pc_return']].head(20)

by_date_df = pd.DataFrame()
pc_return_by_date = sample.groupby('t')[f'pc_ret_{exit_day}'].mean()
n_trades_by_date = sample.groupby('t')[f'pc_ret_{exit_day}'].count()

#simulate returns
init_value = 1000
at_risk = 0.25
value_ts = []

value = init_value
for t, pc_return in pc_return_by_date.items():
    position = at_risk*value
    ret = position * pc_return
    value = value + ret
    value_ts.append(value)

value_ts = pd.Series(value_ts, index = pc_return_by_date.index)

by_date_df = pd.DataFrame(
    {
     'returns': pc_return_by_date,
     'n_trades': n_trades_by_date,
     'value': value_ts,
     },
    index = pc_return_by_date.index,
)

by_date_df['returns_ma'] = by_date_df.returns.rolling(20).mean()

fig1, ax1 = plt.subplots(figsize = (8,6))
fig2, ax2 = plt.subplots(figsize = (8,6))
fig3, ax3 = plt.subplots(figsize = (8,6))

by_date_df.returns.plot(ax = ax1, grid = True)
by_date_df.returns_ma.plot(ax = ax1, grid = True)
by_date_df.n_trades.plot(ax = ax2)
by_date_df.value.plot(ax = ax3)

print('\n strategy \n', by_date_df.returns.describe())

strategy_wins = by_date_df.loc[by_date_df.returns >0].shape[0]
print(f'strategy win rate: {strategy_wins/by_date_df.shape[0]}')

# pull more data, get 10 minute window after open for stocks in sample
#separate script to grab live prices?




