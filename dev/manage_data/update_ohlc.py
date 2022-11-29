import pandas as pd

#import ohlc table
ohlc = pd.read_csv('../../data/database/ohlc.csv', index_col = 'Unnamed: 0')

#import new data
new = pd.read_csv('../../data/all of finviz/ohlc_2022-04-02_2022-05-23.csv', index_col = 'Unnamed: 0')

#set date column to datetime type
ohlc.t = pd.to_datetime(ohlc.t)
new.t = pd.to_datetime(new.t)

#append new data to existing
ohlc = ohlc.append(new, ignore_index = True)

#check for duplicat ticker-dates
ticker_t = ohlc[['ticker', 't']]
duplicates = ohlc.loc[ticker_t.duplicated()]
print(f'The following ticker-date duplicates were found: \n{duplicates}')
print(f'Unique dates of duplicates: \n{duplicates.t.unique()}')
print(f'Number of unique tickers of duplicates: {duplicates.ticker.nunique()}')
progress = input('Do you want to continue? [y/n])')

if progress == 'y':
    ohlc.drop_duplicates().to_csv('../../data/database/ohlc.csv')
