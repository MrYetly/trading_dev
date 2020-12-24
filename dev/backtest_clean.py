import requests, os, json
import pandas as pd
from datetime import datetime, timedelta, date
import time

###organize initial data
data_path = '../data/backtest_test/'
init_data = {}
filenames = os.listdir(data_path)

unique = set()
unique_limit = 500
for f in filenames:
    if f[-4:] == '.csv':
        d = f[-4-8:-4]
        d = f'{d[-4:]}-{d[:2]}-{d[2:4]}'
        df = pd.read_csv(data_path+f)
        tickers = list(df['Ticker'])
        
        #check max unique tickers
        t = set(tickers)
        u = unique | t
        if len(u) > 500:
            break
        else:
            init_data[d] = tickers
            unique = u

count = 0
for d, tickers in init_data.items():
    count += len(tickers)

print('total tickers',count, 'unique tickers', len(unique), 'unique limit', unique_limit)


###pull time series data and aggregate

count = 0
max_count = 10
agg_data = pd.DataFrame()
for d, tickers in init_data.items():

    if count > max_count:
        print('broken dates')
        break

    #set initial date
    d_o = datetime.strptime(d, '%Y-%m-%d').date()

    for ticker in tickers:
        count += 1
        print(count)

        if count > max_count:
            print('broken tickers')
            break

        if count % 5 == 1 and count != 1:
            print('waiting...')
            time.sleep(60)

        #pull time series
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey=HJYREXCXYL2536T7'
        try:
            r = requests.get(url)
            api_data = json.loads(r.content.decode('utf-8'))
            ts = api_data['Time Series (Daily)']
        except:
            print(f'Failed to pull {ticker}!')
            continue
        

        #find next day
        found = False
        i = 1
        while found == False:
            d_i = d_o + timedelta(days = i)
            if str(d_i) in list(ts.keys()):
                found = True
            else:
                i += 1
        
        for d in [d_o, d_i]:
            row = pd.Series(ts[str(d)])
            row['date'] = str(d)
            row['ticker'] = ticker
            agg_data = agg_data.append(row, ignore_index=True,)
        
agg_data.to_csv('../data/backtest_data.csv')