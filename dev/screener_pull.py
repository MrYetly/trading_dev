from finviz.screener import Screener
from finviz import get_stock
import pandas as pd
from datetime import date, timedelta
import os

today = date.today()
window = [(today - timedelta(days=i)).strftime("%m%d%Y") for i in range(9)]

ponds = {
    'DS_up_15': [
        "sh_price_u5",
        "sh_short_low",
        "ta_change_u15",
        "sh_float_o2",
        #'additional',
    ],
    'DS_master': [
        "sh_price_u5",
        "sh_avgvol_o50",
        #'additional',
    ]
}

columns = [str(i+1) for i in range(73)]

data_path = '../data/batch pulls/'

check = input("Are you sure you have NOT pulled today's data yet?")

print('tracking past pulls...')

###track past pulls: THIS MUST RUN BEFORE GETTING TODAY'S PULLS!!!

#organize all past files
filenames = os.listdir(data_path)
pulls = {name: {d:[] for d in window} for name in ponds.keys()}
for pond, d in pulls.items():
    for f in filenames:
        if pond in f:
            _date = f[len(pond)+1:len(pond)+9]
            if _date in window:
                d[_date].append(f)

left_out = []
#download and save to csv
for pond in ponds.keys():
    tracking = pulls[pond]
    for _date, f_list in tracking.items():
        if len(f_list) > 0 and len(f_list) < 5:
            latest = f'{pond}_{_date}_{len(f_list)}.csv'
            df_in = pd.read_csv(data_path+latest)
            tickers = set(df_in['Ticker'])
            scr = Screener(tickers = tickers, table='Custom', custom = columns)
            print(pond, _date, len(f_list)+1, scr._total_rows, 'out of', len(tickers))
            scr.to_csv(data_path+f'{pond}_{_date}_{len(f_list)+1}.csv')


'''print("Getting today's pulls...")
#get today's pulls
for name, filters in ponds.items():
    scr = Screener(filters = filters, table = 'Performance')
    print(pd.DataFrame(scr.data))
    #df.to_csv(data_path+f'{name}_{window[0]}_1.csv')'''