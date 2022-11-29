import requests, os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from io import StringIO

###organize initial data

key = 'O_V9UBmN1CfQ03AJyJp6XMp0pgCYRzHN'



data_path = '../data/gap up 25 12302020/'
init_data = {}
filenames = os.listdir(data_path)
unique = set()
unique_limit = 5000000000

for f in filenames:
    if f[-4:] == '.csv':
        d = f[-4-8:-4]
        d = f'{d[-4:]}-{d[:2]}-{d[2:4]}'
        df = pd.read_csv(data_path+f)
        tickers = df['Symbol']
        tickers = tickers.dropna()
        tickers = list(tickers)
        
        #check max unique tickers before adding
        t = set(tickers)
        u = unique | t
        if len(u) > unique_limit:
            break
        else:
            for ticker in tickers:
                if init_data.get(ticker) == None:
                    init_data[ticker] = [d,]
                else:
                    init_data[ticker].append(d)
            unique = u

count = 0
for ticker, d_list in init_data.items():
    count += len(d_list)

print('total "trades:"',count, 'trade limit:', unique_limit)


###pull time series data and aggregate

count = 0
max_count = 1
price_data = pd.DataFrame()
for ticker, d_list in init_data.items():
    
    if count > max_count:
        print('max count hit')
        break
    
    #process days
    for d_o in d_list:
        
        count += 1
        print(count)
        
        if count > max_count:
            print('max count hit')
            break
        
        if count % 5 == 1 and count != 1:
            print('waiting...')
            time.sleep(60)
        
        d_o = datetime.strptime(d_o, '%Y-%m-%d').date()

        
        #pull time series
        price_url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/5/minute/{d_o}/{d_o}?unadjusted=true&sort=asc&apiKey=O_V9UBmN1CfQ03AJyJp6XMp0pgCYRzHN'
        try:
            r = requests.get(price_url)
            content = json.loads(r.content.decode('utf-8'))
            ts = content['results']
            ts = pd.DataFrame(ts)
            ts.t = pd.to_datetime(ts.t, unit = 'ms')
            print(ts.shape)
        except:
            print(f'Failed to pull prices for {ticker}!')
            continue
    
        #set inital date
'''        

        #find other days if necessary (referenced by delta from now)
        try:
            d_dict = {
                    '0': d_o,
            }
            ts_dates = list(ts.keys())
            ref = ts_dates.index(str(d_o))
            lags = list(range(-253,1))
            ahead = list(range(1,6))
            date_range = lags+ahead
            for d in date_range:
                if d == 0:
                    continue
                try:
                    d_i = ts_dates[ref-d]
                    d_i = datetime.strptime(d_i, '%Y-%m-%d').date()
                    d_dict[str(d)] = d_i
                
                except:
                    print(f'Error for {ticker} on {str(d_o)}, lag {d}')
                    continue
                
            for delta, d in d_dict.items():
                    row = pd.Series(ts[str(d)])
                    row['init date'] = str(d_dict['0'])
                    row['date'] = str(d)
                    row['ticker'] = ticker
                    row['delta'] =  delta
                    price_data = price_data.append(row, ignore_index=True,)
        except:
            print(f'Failed to process init date {d_o} for {ticker}')
            continue

print(price_data)
price_data.to_csv('../data/gap u 25 12302020 agg intra 5 min.csv')'''
