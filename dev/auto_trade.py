from time import perf_counter
import os, json, requests
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from time import sleep
import numpy as np
import alpaca_trade_api as tradeapi
from bs4 import BeautifulSoup

################################################################################
#get secrets
################################################################################

secrets = pd.read_csv('dt_secrets.csv')



################################################################################
#set environment variables
################################################################################

#live
os.environ['APCA_API_KEY_ID'] = 'AK6HI8LGBE5OQV1B6BB2'
os.environ['APCA_API_SECRET_KEY'] = secrets.loc[secrets.key == 'live_alpaca_secret_key'].value

#paper
#os.environ['APCA_API_KEY_ID'] = 'PKQ68I3V62OJJD6CNJNV'
#os.environ['APCA_API_SECRET_KEY'] = secrets.loc[secrets.key == 'paper_alpaca_secret_key'].value

os.environ['APCA_API_BASE_URL'] = 'https://api.alpaca.markets'
api = tradeapi.REST()


#set directory
os.chdir('/users/ivananich/desktop/work book/dt_dev/dev/trade targets/')



################################################################################
#get target data from finviz
################################################################################


#open Session with username and password to access live data from Finviz Elite account
with requests.Session() as session:
    #set user-agent header to meet finviz header requirements
    headers = { 'User-Agent': 'Mozilla/5.0'}
    session.headers.update(headers)
    #post log-in data to log-in URL
    log_in_data = {
            'email': secrets.loc[secrets.key == 'finviz_email'].value, 
            'password': secrets.loc[secrets.key == 'finviz_pw'].value,
    }
    log_in = session.post('https://finviz.com/login_submit.ashx', data = log_in_data)

    #elicit splits for the day (elicit here to load session before getting targets)
    splits = input('Splits (list space separated or "no")? ')
    splits = splits.split()

    #start timer
    tick = perf_counter()

    #scrape finviz screener for target trades
    url = secrets.loc[secrets.key == 'finviz_url'].value
    r = session.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')

#get columns of table
def has_table_top(css_class):
    return css_class is not None and 'table-top' in css_class
#find all cells with column names
columns = soup.find_all('td', class_=has_table_top)
names = []
for col in columns:
    #handle ticker column name cell contents having two entries
    if len(col) > 1:
        name = col.contents[1]
    else:
        name = col.contents[0]
    names.append(name)


#get content of table
def has_screener_link(css_class):
    return css_class is not None and 'screener-link' in css_class
#find all cells that are in body of table
cells = soup.find_all('td', attrs={'class': 'screener-body-table-nw'})
data = []
row = []
count = 0 
for td in cells:
    count += 1
    a_list = td.find_all('a', class_=has_screener_link)
    for a in a_list:
        span_check = a.find('span')
        #handle possible presence of span in cell that conditionally colors number on website
        if span_check == None:
            cell = a.contents[0]
        else:
            cell = span_check.contents[0]
    row.append(cell)
    if count == len(names):
        data.append(row)
        row = []
        count = 0 


targets = pd.DataFrame(data, columns = names)


#if there are target buys available, process, format, and calculate
if targets.shape[0] > 0:

    #format dataframe
    targets = targets.rename(
            columns = {
                'Ticker': 'ticker',
                'Price': 'price',
                'Change': 'change',
            }
    )
    targets.ticker = targets.ticker.astype('str')
    targets.price = targets.price.astype('float')
    #remove percent sign from change column
    def remove_last_letter(x):
        return x[:-1]
    targets.change = targets.change.apply(remove_last_letter)
    targets.change = targets.change.astype('float')
    targets.change = targets.change/100
    targets.ticker = targets.ticker.str.strip()

    #calculate bids and insert
    def bid(x):
        if x.price <= 1:
            bid = x.price + 0.01
        else:
            bid = x.price + 0.02
        return bid
    targets['bid'] = targets.apply(lambda x: bid(x), axis = 1)
    
    #re-enforce stock qualifiers
    targets = targets.loc[(targets.price <= 3) & (targets.price >= 0.1)]
    targets = targets.loc[targets.change <= -.1]

    #handle splits
    found_splits = []
    for ticker in splits:
        if ticker in targets.ticker.values:
            found_splits.append(ticker)
            targets = targets.loc[targets.ticker != ticker]
    if len(found_splits) > 0:
        print(f'The following splits were found in the target trades and removed: {found_splits}')
    else:
        print('No splits were found in the target trades')


################################################################################
#Build dataframe of orders
################################################################################


#create blank order dataframe
orders = pd.DataFrame(
        {
            'symbol': [],
            'qty': [],
            'side': [],
            'type': [],
            'time_in_force': [],
            'limit_price': [],
            'price': [],
            'pos_size':[],
            'filled': [],
            'exchange': [],
        }
)

#handle buy orders, if there are buy targets
if targets.shape[0] > 0:

    #create order information for buy targets
    for i, target in targets.iterrows():
        #handle identifiers
        #skip if 5 letters long (length of NASDAQ special tickers, not normal companies)
        if len(target.ticker) >= 5:
            continue
        #check exchange to see if normal stock or stock w/ identifier
        id_check = api.get_asset(target.ticker)
        exchange = id_check.exchange
        if exchange != 'NASDAQ' and len(target.ticker) >= 4:
            continue
        order = {
                'symbol': target.ticker,
                'side': 'buy',
                'type': 'limit',
                'time_in_force': 'day',
                'limit_price': str(np.round(target.bid, decimals = 2)),
                'price': str(np.round(target.price, decimals = 2)),
                'exchange': exchange,
        }
        orders = orders.append(order, ignore_index = True)

    #get portfolio equity
    account = api.get_account()
    equity = account.equity
    equity = float(equity)
    print('Equity: ', equity)


    #set risk level
    at_risk = equity*0.25
    position = at_risk/orders.loc[orders.side == 'buy'].shape[0]
    orders.loc[orders.side == 'buy', 'pos_size'] = position
    orders.qty = np.round(position/orders.limit_price.astype('float'))
    orders.qty = orders.qty.astype('str')
    print('position size: ', position)
else:
    print('No buy orders will be processed becuase no stocks qualified for buying.')

#check for sell orders
open_positions = api.list_positions()
#if there are open positions to sell, iterate through and create order information for each
if len(open_positions) > 0:
    for position in open_positions:
        #get current price
        price = float(position.current_price)
        price = np.round(price, decimals = 2)
        #calculate limite price
        if price > 2:
            limit_price = price - 0.03
        elif price > 1:
            limit_price = price - 0.02
        else:
            limit_price = price - 0.01
        limit_price = np.round(limit_price, decimals = 2)
        #consolidate order information
        order = {
                'symbol': position.symbol,
                'qty': str(position.qty),
                'side': 'sell',
                'type': 'limit',
                'time_in_force': 'day',
                'limit_price': str(limit_price),
                'price': str(price),
        }
        #add order info to dataframe of order info
        orders = orders.append(order, ignore_index = True)
else:
    print('No open positions to sell.')



################################################################################
#send orders to API endpoint
################################################################################

#execute orders if there are any to execute
if orders.shape[0] > 0:
    #create list of failed orders to remove later
    failed_index = []
    for i, order in orders.iterrows():
        try:
            #hit API with order using order info from order info dataframe
            api.submit_order(
                symbol= order.symbol,
                qty=order.qty,
                side=order.side,
                type= order.type,
                time_in_force= order.time_in_force,
                limit_price= order.limit_price,
            )
        except Exception as e:
            print(f'Failed to order {order.symbol}: {e}')
            #if order fails for any reason, add index of order to failed orders list
            failed_index.append(i)


    #print time to process first executions (start time is 'tick' above)
    tock = perf_counter()
    print(f'First orders execution time: {tock - tick}')
    
    #remove failed orders from order dataframe
    orders.drop(failed_index, inplace = True)

    print('First execution:\n', orders)
else:
    exit()

date = pd.Timestamp.now()
date = date.strftime('%Y-%m-%d')

orders.to_csv(f'orders_{date}.csv')

