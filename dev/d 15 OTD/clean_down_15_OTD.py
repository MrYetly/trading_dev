#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 10:02:07 2021

@author: ivananich
"""

def clean_master(master):
    #throw out innaccuracies
        
    inaccuracies = [
            master.loc[master['ticker'] == 'SNNAQ'],
            master.loc[master['ticker'] == 'SPHS'],
            master.loc[(master['ticker'] == 'REPX') & (master['init_date'] == '2021-03-01')],
            master.loc[(master['ticker'] == 'CMMB') & (master['init_date'] == '2021-03-10')],
    ]
    
    
    for i in inaccuracies:
        master = master.drop(index=i.index)
    
    # #correct innaccuracies
    # master.loc[
    #         (master.ticker == 'NOVN')
    #         & (master.init_date == '2021-05-25')
    #         & (master.delta == 0),
    #         ['open','close']
    # ] = [9.799, 8.35]
    
    return master