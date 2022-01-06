#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 16:05:44 2021

@author: ivananich
"""

def clean_master(master):
    #throw out innaccuracies
        
    inaccuracies = [
            master.loc[master['ticker'] == 'SNNAQ'],
            master.loc[master['ticker'] == 'MUDSW'],
            master.loc[master['ticker'] == 'SPHS'],
            master.loc[master['ticker'] == 'BFIIW'],
            master.loc[master['ticker'] == 'MYT'],
            master.loc[(master['ticker'] == 'EEIQ') & (master.init_date == '2021-03-25')],
            master.loc[(master['ticker'] == 'SUMR') & (master.init_date == '2021-03-17')],
    ]
    
    
    for i in inaccuracies:
        master = master.drop(index=i.index)
    
    #correct innaccuracies
    master.loc[
            (master.ticker == 'NOVN')
            & (master.init_date == '2021-05-25')
            & (master.delta == 0),
            ['open','close']
    ] = [9.799, 8.35]
    
    master.loc[
            (master.ticker == 'GHSI')
            & (master.init_date == '2021-02-26')
            & (master.delta == 0),
            ['open','close']
    ] = [3.07, 3.03]
    
    master.loc[
            (master.ticker == 'LMFA')
            & (master.init_date == '2021-05-06')
            & (master.delta == 0),
            ['open','close']
    ] = [3.98, 3.80]
    
    master.loc[
            (master.ticker == 'ONTX')
            & (master.init_date == '2021-05-19')
            & (master.delta == 0),
            ['open','close']
    ] = [13.37, 13.88]
    
    master.loc[
            (master.ticker == 'ONTX')
            & (master.init_date == '2021-05-19')
            & (master.delta == 1),
            ['open','close']
    ] = [12.23, 10.95]
    
    master.loc[
            (master.ticker == 'REPX')
            & (master.init_date == '2021-02-26')
            & (master.delta == 0),
            ['open','close']
    ] = [34.08, 29.64]
    
    master.loc[
            (master.ticker == 'REPX')
            & (master.init_date == '2021-02-26')
            & (master.delta == 1),
            ['open','close']
    ] = [32.09, 24.21]
    
    master.loc[
            (master.ticker == 'ZIVO')
            & (master.init_date == '2021-05-28')
            & (master.delta == -1),
            ['open','close']
    ] = [10.40, 9.60]
    
    master.loc[
            (master.ticker == 'ONTX')
            & (master.init_date == '2021-05-19')
            & (master.delta == -1),
            ['open','close']
    ] = [11.88, 16.20]
    
    master.loc[
            (master.ticker == 'NSPR')
            & (master.init_date == '2021-04-27')
            & (master.delta == -1),
            ['open','close']
    ] = [7.35, 7.35]
    
    master.loc[
            (master.ticker == 'REPX')
            & (master.init_date == '2021-02-26')
            & (master.delta == -1),
            ['open','close']
    ] = [50.64, 44.40]
    
    master.loc[
            (master.ticker == 'NOVN')
            & (master.init_date == '2021-05-25')
            & (master.delta == -1),
            ['open','close']
    ] = [13.1, 14]
    
    master.loc[
            (master.ticker == 'GHSI')
            & (master.init_date == '2021-02-26')
            & (master.delta == -1),
            ['open','close']
    ] = [4.52, 4.17]
    
    master.loc[
            (master.ticker == 'BTTR')
            & (master.init_date == '2021-06-29')
            & (master.delta == -1),
            ['open','close']
    ] = [7.68, 7.38]
    
    master.loc[
            (master.ticker == 'LMFA')
            & (master.init_date == '2021-05-06')
            & (master.delta == -1),
            ['open','close']
    ] = [5.25, 5.05]
    
    master.loc[
            (master.ticker == 'SNPX')
            & (master.init_date == '2021-05-19')
            & (master.delta == -1),
            ['open','close']
    ] = [5.40, 6.24]
    
    master.loc[
            (master.ticker == 'SBEV')
            & (master.init_date == '2021-06-11')
            & (master.delta == -1),
            ['open','close']
    ] = [6.24, 5.43]
    
    master.loc[
            (master.ticker == 'SUMR')
            & (master.init_date == '2021-03-17')
            & (master.delta == -1),
            ['open','close']
    ] = [6.24, 5.43]
    
    master.loc[
            (master.ticker == 'GLBS')
            & (master.init_date == '2020-10-20')
            & (master.delta == 0),
            ['open','close']
    ] = [8.59, 7.54]
    
    master.loc[
            (master.ticker == 'HJLI')
            & (master.init_date == '2020-11-27')
            & (master.delta == 0),
            ['open','close']
    ] = [7.58, 7.58]
    
    master.loc[
            (master.ticker == 'UCO')
            & (master.init_date == '2020-04-20')
            & (master.delta == 0),
            ['open','close']
    ] = [33.25, 33.75]
    
    master.loc[
            (master.ticker == 'NTEC')
            & (master.init_date == '2020-10-29')
            & (master.delta == 0),
            ['open','close']
    ] = [12.8, 12.3]
    
    master.loc[
            (master.ticker == 'NTEC')
            & (master.init_date == '2020-10-29')
            & (master.delta == 1),
            ['open','close']
    ] = [11.08, 9.16]
    
    master.loc[
            (master.ticker == 'NBRV')
            & (master.init_date == '2020-12-02')
            & (master.delta == 0),
            ['open','close']
    ] = [3.7, 3.94]
    
    master.loc[
            (master.ticker == 'HTBX')
            & (master.init_date == '2020-12-10')
            & (master.delta == 0),
            ['open','close']
    ] = [6.16, 6.51]
    
    master.loc[
            (master.ticker == 'EFOI')
            & (master.init_date == '2020-06-11')
            & (master.delta == 0),
            ['open','close']
    ] = [5.6, 5.9]
    
    master.loc[
            (master.ticker == 'FTSI')
            & (master.init_date == '2020-08-24')
            & (master.delta == 0),
            ['open','close']
    ] = [1.69, 5.26]
    
    master.loc[
            (master.ticker == 'FTSI')
            & (master.init_date == '2020-08-24')
            & (master.delta == 1),
            ['open','close']
    ] = [5.29, 7.28]
    
    master.loc[
            (master.ticker == 'SXTC')
            & (master.init_date == '2021-02-19')
            & (master.delta == 0),
            ['open','close']
    ] = [3.18, 3.14]
    
    master.loc[
            (master.ticker == 'XSPA')
            & (master.init_date == '2020-06-10')
            & (master.delta == 0),
            ['open','close']
    ] = [4.95, 5.01]
    
    master.loc[
            (master.ticker == 'XELA')
            & (master.init_date == '2021-01-25')
            & (master.delta == 0),
            ['open','close']
    ] = [2.51, 2.62]
    
    master.loc[
            (master.ticker == 'SHIP')
            & (master.init_date == '2020-06-26')
            & (master.delta == 0),
            ['open','close']
    ] = [3.04, 2.71]
    
    master.loc[
            (master.ticker == 'SHIP')
            & (master.init_date == '2020-06-26')
            & (master.delta == 1),
            ['open','close']
    ] = [2.63, 2.52]
    
    
    
    return master