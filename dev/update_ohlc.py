import pandas as pd

#import ohlc table
ohlc = pd.read_csv('../data/database/ohlc.csv', index_col = 'Unnamed: 0')

#import new data
new = pd.read_csv('../data/all of finviz/ohlc_2021-01-06_2022-01-04.csv', index_col = 'Unnamed: 0')

#set date column to datetime type
ohlc.t = pd.to_datetime(ohlc.t)
new.t = pd.to_datetime(new.t)

#append new data to existing
ohlc = ohlc.append(new, ignore_index = True)

#check for duplicat ticker-dates
ticker_t = ohlc[['ticker', 't']]
ohlc_rows = ohlc.shape[0]
ticker_t_rows = ticker_t.drop_duplicates().shape[0]
print(f'total rows with append: {ohlc_rows}')
print(f'total rows with append, no duplicates ticker-dates: {ticker_t_rows}')

if ohlc_rows > ticker_t_rows:
    print('Duplicate ticker-dates found, new data not added to database')
else:
    print(ohlc)
    ohlc.to_csv('../data/database/ohlc.csv')
