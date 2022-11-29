import pandas as pd

protect = input('ARE YOU SURE YOU WANT TO RUN THIS. WILL RESET DATABASE:')


pw = 'BHADKEA44'

if protect != pw:
    exit()

master_rel_path = '../data/batch pulls/'
master_dates = [
        '2021-01-04',
        '2021-11-05',
        '2022-01-03',
        ]

master_dfs = {}
for d in master_dates:
    df = pd.read_csv(master_rel_path+f'master_{d}.csv')
    master_dfs[d] = df

### Create ticker table
tickers = pd.Series(dtype='float')
for d in master_dates:
    tickers = tickers.append(master_dfs[d]['Ticker'], ignore_index=True)

tickers.drop_duplicates(inplace=True)
tickers.reset_index(inplace=True, drop=True)
database_rel_path = '../data/database/'
tickers.to_csv(database_rel_path+'tickers.csv')
### create general data time series


for d in master_dates:
    df = master_dfs[d]
    df['date'] = d

general = master_dfs['2021-01-04'].copy()
general = general.append(master_dfs['2021-11-05'], ignore_index=True)
general = general.append(master_dfs['2022-01-03'], ignore_index=True)

general.to_csv(database_rel_path+'general.csv')


### create blank OHLC time series dataframe

ohlc = pd.DataFrame(
       columns = [ 
            'ticker',
            'o',
            'l',
            'c',
            'h',
            's',
            't',
            'v',
        ],
)

ohlc.to_csv(database_rel_path+'ohlc.csv')
