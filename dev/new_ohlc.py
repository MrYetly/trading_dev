import pandas as pd
from time import perf_counter, sleep
import finnhub

##############################################################
#update the table of time series for all tickers in a given time window
##############################################################

#Declare window to draw data for
init_date = '2021-01-06'
end_date = '2022-01-04'

#Get all tickers from tickers table
database_rel_path = '../data/database/'
tickers = pd.read_csv(
        database_rel_path+'tickers.csv', 
        index_col='Unnamed: 0',
)
tickers = tickers.rename(columns={'0':'ticker'})

#set initial and end timestamps
init_t = pd.to_datetime(init_date)
end_t = pd.to_datetime(end_date)
init_t = int(init_t.value / 10**9)
end_t = int(end_t.value / 10**9)

#create empyt data frame to hold new data
new_data = pd.DataFrame()

#instantiate API client
finnhub_client = finnhub.Client(api_key="c5mo56qad3iam7tung4g")

#set reference time
ref_time = perf_counter()
for i, row in tickers.iterrows():

    ticker = row.ticker 
    print(f'{i} out of {tickers.shape[0]}')

    #check API limit every 60 calls (limit 60 calls per minute)
    if (i+1) % 60 == 0:
        current_time = perf_counter()
        elapsed = current_time - ref_time
        #if time elapsed since last 60 marker is less than 60 seconds, wait until 60 seconds has passed before calling API again
        if elapsed <= 60:
            print(f'waiting ~{(60 - round(elapsed))} seconds')
            sleep(60 - elapsed+1)
        else:
            print(f'Time elapsed since last 60: {elapsed}')
        #reset reference time every 60 api calls
        ref_time = perf_counter()
    
    try:
        #attempt to pull data for ticker
        entry = finnhub_client.stock_candles(ticker, 'D', init_t, end_t)
        entry = pd.DataFrame(entry)
        entry.t = pd.to_datetime(entry.t, unit='s')
        entry['ticker']=ticker
        new_data = new_data.append(entry, ignore_index = True, )
    except Exception as e:
        print(f'Failed to pull data for {ticker}')
        print(e)
        continue

print(new_data)
new_data.to_csv(f'../data/all of finviz/ohlc_{init_date}_{end_date}.csv')


