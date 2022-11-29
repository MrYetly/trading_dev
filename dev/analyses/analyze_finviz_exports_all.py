# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np


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
missing = data.ticker.unique()
missing = list(missing)
missing = set(missing)

check = tickers.ticker
check = list(check)
check = set(check)

data['return_1'] = data.c.diff(1)
mask = data.ticker != data.ticker.shift(1)
mask_index = data['return_1'][mask].index
data.loc[mask_index, 'return_1'] = np.nan
data['pc_return'] = data.return_1/(data.c-data.return_1)


bins = [i/100 for i in range(-30,30)]
data.pc_return.hist(bins=bins)

def get_ts(ticker, feature):
    ts = data.loc[data.ticker == ticker]
    ts = ts[[feature, 't']]
    ts.plot(y = feature, x = 't')
    return ts
    