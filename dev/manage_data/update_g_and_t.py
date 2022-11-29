import pandas as pd
import numpy as np
import os


###################################################################
#establish paths
###################################################################

new_data_path = '../../data/batch pulls/'
database_path = '../../data/database/'


###################################################################
#Import current general table
###################################################################
general = pd.read_csv(database_path+'general.csv', low_memory=False)
general.date = pd.to_datetime(general.date)
print('Initial shape of general table: ', general.shape)
print('Initial start and end dates of general table: ',general.date.min(), general.date.max())

###################################################################
#get new batch pull files
###################################################################

#get date of last update of general dataframe
last_update_date = general.date.max()

#get all batch pull files
prefixes = [
        'nasdaq_',
        'amex_',
        'nyse_',
]
files = os.listdir(new_data_path)
batch_pulls = []
for p in prefixes:
    for f in files:
        if f.startswith(p):
            batch_pulls.append(f)

#organize batch pulls into dataframe 
batch_pulls = pd.DataFrame({'filename':batch_pulls})
batch_pulls['date'] = batch_pulls.filename.apply(lambda x: x[-14:-4])
batch_pulls.date = pd.to_datetime(batch_pulls.date)
batch_pulls['exchange'] = batch_pulls.filename.apply(lambda x: x[:-15])
batch_pulls = batch_pulls.sort_values(['exchange', 'date'])


#keep only new batchpulls
new_pulls = batch_pulls.loc[batch_pulls.date > last_update_date]


###################################################################
#assimilate new batch pull data into general dataframe, also update ticker table
###################################################################


#concatenate new batch pull data, batch by batch
for i, row in new_pulls.iterrows():
    new_data = pd.read_csv(new_data_path+row.filename)
    new_data['date'] = row.date
    new_data['exchange'] = row.exchange
    general = pd.concat(
            [general,new_data], 
            ignore_index = True, 
            sort = True,
    )

#drop duplicate ticker/dates
t_d_index = general.drop_duplicates(subset = ['Ticker', 'date'])
print('Final shape of general table: ', general.shape)
print('Final start and end dates of general table: ', general.date.min(), general.date.max())

general.to_csv(database_path+'general.csv')

tickers = general.Ticker.drop_duplicates()
tickers.to_csv(database_path+'tickers.csv')
print(tickers)

