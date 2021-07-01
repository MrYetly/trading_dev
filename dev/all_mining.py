#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 16:07:54 2021

@author: ianich
"""

import pandas as pd
import numpy as np
import os

data_path = '../data/batch pulls/'
init_data = pd.DataFrame()
filenames = os.listdir(data_path)

for f in filenames:
    
    if f[-4:] == '.csv' and f[:6] == 'DS_all' and (('open' in f) or ('close' in f)):
        d = f[-4-8:-4]
        d = f'{d[-4:]}-{d[:2]}-{d[2:4]}'
        d = datetime.strptime(d, '%Y-%m-%d').date()
        df = pd.read_csv(data_path+f)
        df['init date'] = np.datetime64(d)
        if 'open' in f:
            EOD = 0
        if 'close' in f:
            EOD = 1
        df['EOD'] = EOD
        init_data = init_data.append(df)
        
BOD = init_data.loc[init_data.EOD == 0]
EOD = init_data.loc[init_data.EOD == 1]
_returns = pd.Series((EOD.Price - BOD.Price)/BOD.Price)
_returns = _returns.append(_returns)
init_data['return'] = _returns

#rerurn BOD to get returns
BOD = init_data.loc[init_data.EOD == 0]
winners = BOD.loc[BOD['return']>0]
losers = BOD.loc[BOD['return']<0]

avg_win = winners['return'].mean()
avg_loss = losers['return'].mean()
print('average win:', avg_win)
print('average loss:', avg_loss)
win_rate = winners.shape[0]/BOD.shape[0]
print('win rate:', win_rate)

