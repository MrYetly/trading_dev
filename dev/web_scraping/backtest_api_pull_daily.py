import requests, os, json
import pandas as pd
from datetime import datetime
import time

###organize initial data
data_path = '../data/up 20 OTD_2021-10-08/'
init_data = pd.DataFrame() 
filenames = os.listdir(data_path)
unique = set()
unique_limit = 490
check = False 

#create df to check against previous partial download
if check == True:
    past_path = '../data/4x_up_10_otm_2021-06-30 agg -253 20.csv'
    past_df = pd.read_csv(past_path)
    past_df = past_df.append(pd.read_csv('../data/4x_up_10_otm_2021-06-30 agg -253 20 part 2.csv'))
    past_check = past_df.loc[past_df.delta == 0][['ticker', 'init date', 'delta']]

#get all (ticker, init date) combinations, put into dataframe
for f in filenames:
    if f[-4:] == '.csv':
        d = f[-4-10:-4]
        #d = f'{d[-4:]}-{d[:2]}-{d[2:4]}'
            
        df = pd.read_csv(data_path+f)
        tickers = df['Symbol']
        tickers = tickers.dropna()
        tickers = list(tickers)
        
        for ticker in tickers:
            init_data = init_data.append({'ticker':ticker, 'init date':d}, ignore_index = True)
init_data = init_data.drop_duplicates()

#drop rows already downloaded
if check  == True:
    init_data = init_data.merge(past_check, how = 'left', on = ['ticker', 'init date'])
    init_data = init_data.loc[init_data.delta != 0]
    init_data = init_data.drop('delta', axis = 1)

#enforece unique_limit such that unique tickers  <= unique limit
unique_tickers = init_data['ticker'].drop_duplicates()
unique_tickers = pd.DataFrame(unique_tickers)
unique_tickers = unique_tickers.reset_index()
if unique_tickers.shape[0] > unique_limit:
    unique_tickers = unique_tickers.loc[:unique_limit-1]
unique_tickers['keep'] = True
init_data = init_data.merge(unique_tickers, how = 'left', on = 'ticker')
init_data = init_data.loc[init_data.keep == 1]
init_data = init_data.drop(['keep','index'], axis = 1)

print('total "trades:"',init_data.shape[0], 'unique tickers:', unique_tickers.shape[0], 'unique limit:', unique_limit)

count=0
#max_count = 5
price_data = pd.DataFrame()
for i, ticker in unique_tickers['ticker'].items():
    
    count += 1
    print(count)

    #if count > max_count:
    #    print('max count hit')
    #    break
    
    if count % 5 == 1 and count != 1:
        print('waiting...')
        time.sleep(60)

    #pull time series
    price_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey=HJYREXCXYL2536T7'
    try:
        r = requests.get(price_url)
        content = json.loads(r.content.decode('utf-8'))
        ts = content['Time Series (Daily)']
    except:
        print(f'Failed to pull prices for {ticker}!')
        continue
    
    #get ticker sub df
    sub_df = init_data.loc[init_data.ticker == ticker]

    #process days
    for j, d_o in sub_df['init date'].items():

        #set inital date
        d_o = datetime.strptime(d_o, '%Y-%m-%d').date()

        #find other days if necessary (referenced by delta from now)
        try:
            d_dict = {
                    '0': d_o,
            }
            ts_dates = list(ts.keys())
            ref = ts_dates.index(str(d_o))

            #set number of lags
            lags = list(range(-253,1))

            #set number of days ahead
            ahead = list(range(1,21))
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
price_data.to_csv('../data/up 20 OTD_2021-10-08 agg -253 5.csv')

