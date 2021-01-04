import requests, os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from io import StringIO

###organize initial data
data_path = '../data/gap up 25 12302020/'
init_data = {}
filenames = os.listdir(data_path)
#month slice reference dates
ref = [
                datetime.strptime('12/7/2020', '%m/%d/%Y').date(),
                datetime.strptime('11/5/2020', '%m/%d/%Y').date(),
                datetime.strptime('10/6/2020', '%m/%d/%Y').date(),
                datetime.strptime('9/8/2020', '%m/%d/%Y').date(),
                datetime.strptime('8/7/2020', '%m/%d/%Y').date(),
]
for f in filenames:
    if f[-4:] == '.csv':
        d = f[-4-8:-4]
        d = f'{d[-4:]}-{d[:2]}-{d[2:4]}'
        d = datetime.strptime(d, '%Y-%m-%d').date()
        df = pd.read_csv(data_path+f)
        tickers = df['Symbol']
        tickers = tickers.dropna()
        tickers = list(tickers)
        
        for ticker in tickers:
            for d_r in ref:
                if d >= d_r:
                    m = str(ref.index(d_r) +1)
                    break
            
            if init_data.get(ticker) == None:
                init_data[ticker] = {m:[d,]}
            elif init_data[ticker].get(m) == None:
                init_data[ticker][m] = [d,]
            else:
                init_data[ticker][m].append(d)

t_count = 0
ms_count = 0
for ticker, md_dict in init_data.items():
    ms_count += len(list(md_dict.keys()))
    for t_list in md_dict.values():
        t_count += len(t_list)

print('total "trades:"',t_count, 'total month-slices:', ms_count)


###pull time series data and aggregate

count = 0
max_count = 5
price_data = pd.DataFrame()
for ticker, md_dict in init_data.items():
     
    #if count > max_count:
     #       print('max count hit')
      #      break
        
    #pull time series by month slice
    for m, d_list in md_dict.items():
        count += 1
        print(count)
    
        #if count > max_count:
         #   print('max count hit')
          #  break
        
        if count % 5 == 1 and count != 1:
            print('waiting...')
            time.sleep(60)
    
        price_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={ticker}&interval=5min&slice=year1month{m}&apikey=HJYREXCXYL2536T7'
        try:
            r = requests.get(price_url)
            content = StringIO(r.content.decode('utf-8'))
            ts = pd.read_csv(content)
            ts['time'] = pd.to_datetime(ts['time'])
        except:
            print(f'Failed to pull prices for {ticker}!')
            continue
    
        #process days
        for d_o in d_list:
    
            #set window close
            d_f = d_o + timedelta(days = 1)
            d_o = np.datetime64(str(d_o))
            d_f = np.datetime64(str(d_f))
            
            try:
                intraday = ts.loc[(ts['time'] >= d_o) & (ts['time'] < d_f)].copy()
                intraday['init date'] = d_o
                intraday['ticker'] = ticker
                price_data = pd.concat([price_data, intraday], ignore_index = True, sort = True)
            except:
                print(f'Failed to pull intraday for {ticker} on {str(d_o)}')
                continue

print(price_data)
price_data.to_csv('../data/gap u 25 12302020 agg intra 5 min.csv')
