import requests, os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from io import StringIO

###organize initial data
data_path = '../data/high up 20 1 lt prev close lt 2 06072021/'
init_data = {}
filenames = os.listdir(data_path)

#month slice reference dates

#reference list of start date for each month-slice (ms 1 is most recent month)
ref = [
                datetime.strptime('5/26/2021', '%m/%d/%Y').date(),
                datetime.strptime('4/26/2021', '%m/%d/%Y').date(),
                datetime.strptime('3/29/2021', '%m/%d/%Y').date(),
                datetime.strptime('2/25/2021', '%m/%d/%Y').date(),
                #datetime.strptime('12/28/2020', '%m/%d/%Y').date(),
                #datetime.strptime('11/27/2020', '%m/%d/%Y').date(),
                #datetime.strptime('6/18/2020', '%m/%d/%Y').date(),
                #datetime.strptime('5/19/2020', '%m/%d/%Y').date(),
                #datetime.strptime('4/20/2020', '%m/%d/%Y').date(),
                #datetime.strptime('3/20/2020', '%m/%d/%Y').date(),
]

req_limit = 490
t_count = 0
ms_count = 0
limit_reached = False

for f in filenames:
    
    if limit_reached == True:
        print('request limit reached, first remaining file:', f)
        break
    
    if f[-4:] == '.csv':
        d = f[-4-8:-4]
        d = f'{d[-4:]}-{d[:2]}-{d[2:4]}'
        d = datetime.strptime(d, '%Y-%m-%d').date()
        #latest = datetime.strptime('8/7/2020', '%m/%d/%Y').date()
        #if d >= latest:
         #   continue
        df = pd.read_csv(data_path+f)
        tickers = df['Symbol']
        tickers = tickers.dropna()
        tickers = list(tickers)
        
        if ms_count + len(tickers) > req_limit:
            print('request limit potentially reached, first remaining file:', f)
            break
        else:
            for ticker in tickers:
                
                t_count += 1
                
                for d_r in ref:
                    if d >= d_r:
                        m = str(ref.index(d_r) +1)
                        break
                
                if init_data.get(ticker) == None:
                    init_data[ticker] = {m:[d,]}
                    ms_count += 1
                elif init_data[ticker].get(m) == None:
                    init_data[ticker][m] = [d,]
                    ms_count += 1
                else:
                    init_data[ticker][m].append(d)


print('total "trades:"',t_count, 'total month-slices:', ms_count)


###pull time series data and aggregate

count = 0
max_count = 5
price_data = pd.DataFrame()
for ticker, md_dict in init_data.items():
     
    #if count > max_count:
            #print('max count hit')
            #break
        
    #pull time series by month slice
    for m, d_list in md_dict.items():
        count += 1
        print(count)
    
        #if count > max_count:
            #print('max count hit')
            #break
        
        if count % 5 == 1 and count != 1:
            print('waiting...')
            time.sleep(60)
    
        price_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={ticker}&interval=5min&adjusted=false&slice=year1month{m}&apikey=HJYREXCXYL2536T7'
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
price_data.to_csv('../data/high up 20 1 lt prev close lt 2 06072021 intra 5 min.csv')
