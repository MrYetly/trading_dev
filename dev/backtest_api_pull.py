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
        
        #check max unique tickers before adding
        t = set(tickers)
        u = unique | t
        if len(u) > 500:
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

print('total "trades"',count, 'unique tickers', len(unique), 'unique limit', unique_limit)


###pull time series data and aggregate

count = 0
max_count = 5
agg_data = pd.DataFrame()
for ticker, d_list in init_data.items():
    
    count += 1
    print(count)

    #if count > max_count:
     #   print('broken dates')
      #  break
    
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
    
    #process days
    for d_o in d_list:

        #set inital date
        d_o = datetime.strptime(d_o, '%Y-%m-%d').date()

        #find next day
        try:
            found = False
            i = 1
            while found == False:
                d_i = d_o + timedelta(days = i)
                if str(d_i) in list(ts.keys()):
                    found = True
                else:
                    i += 1
            
            for j in [0, 1]:
                if j == 0:
                    d = d_o
                else:
                    d = d_i
                row = pd.Series(ts[str(d)])
                row['date'] = str(d)
                row['ticker'] = ticker
                row['day'] =  j
                agg_data = agg_data.append(row, ignore_index=True,)
        except:
            print(f'Error for {ticker} on {str(d_o)}')
            continue

agg_data.to_csv('../data/backtest_data.csv')