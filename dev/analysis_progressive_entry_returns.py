#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 09:30:39 2021

@author: ianich
"""


import pandas as pd
import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.regression.linear_model import OLS
from statsmodels.iolib.summary2 import summary_col
from matplotlib import pyplot as plt
from datetime import datetime



#relationship between increment and ratio?

#Params 
final_day = 5
entry_day = 0
#entry level = change w.r.t. open
entry_level = 0.2
increment = 0.05
inc_factor = 2
max_entries = 4
exit_day = 0
initial_val = 100000
#set time slot (remember, 24 hour clock)
t_start_h = 9
t_start_m = 30
t_end_h = 11
t_end_m = 0
#set window size in minutes
window = 30
#stoploss (still needs refinement)
#Note, must have max_entry if using a stop loss (else keep adding)
stop_loss = None
stop_win = 0.05
stop_method = 'cost'
#set time series bias for bracketed stop loss, must be 'win' or 'add'
bias = 'add'



#calculate average cost
def avg_cost_pc(n, e, i, f):
    
    if f == 0:
        return 1 + e
    
    num = sum([(e+i*j)*f**j for j in range(n)])
    denom = sum([f**j for j in range(n)])
    return (1 + num/denom)


#calculate weight 
def weights(n, f):
    
    if f == 0:
        return 1
    
    return sum([f**i for i in range(n)])


def find_entry_slot(row,n, ref_df):
    '''
    Find timeslot of earliest entry. Checks that low<=entry<=high.
    Note: all checks for exits (stoploss, profit taking) commence in 
    next time slot, so order of high,low in entry time slot does not matter.
    '''
    last_entry = row.open * (1+entry_level+increment*(n-1))
    
    times = ref_df.loc[
                (ref_df.ticker == row.ticker)
                & (ref_df.init_date == row.init_date)
                & (ref_df.high >= last_entry)
                & (ref_df.low <= last_entry)
    ]
    times = times.sort_values('time')
    earliest = times.iloc[0]

    return earliest.time

def handle_stop_loss(row, ref_df, loss, method = 'cost'):
    '''
    method must be either 'last' or 'cost'
    '''
    #find time series, beginning with time slot after last entry slot
    
    le_time = find_entry_slot(row, row.n_entries, ref_df)
    
    ts = ref_df.loc[
                (ref_df.ticker == row.ticker)
                & (ref_df.init_date == row.init_date)
                & (ref_df.time > le_time)
    ]
    
    #shape will be zero if entered in final time slot
    if ts.shape[0] == 0:
        return row.exit, None
    
    #sort in ascending order of time
    ts = ts.sort_values('time')

    ts = ts[['ticker', 'init_date', 'time','high']]
    
    if method == 'last':
        stop_level = row.open * (1+entry_level+increment*(max_entries-1)+loss)
    elif method == 'cost':
        cost_pc = avg_cost_pc(max_entries, entry_level, increment, inc_factor)
        stop_level = row.open * (cost_pc*(1+loss))

    losses = ts.loc[ts.high >= stop_level]

    
    if losses.shape[0] == 0:
        #include tuple for use in handle_stop_bracket()
        return row.exit, None
    else:
        #include tuple for use in handle_stop_bracket()
        return stop_level, losses.iloc[0].time
        
def handle_stop_win(row, ref_df, win, bias = 'win', method = 'cost'):
    '''
    Bias indicates if adds are assumed to occur before wins or vice versa
    '''
    for n in range(1, row.n_entries+1):
        
        ts = ref_df.loc[
                    (ref_df.ticker == row.ticker)
                    & (ref_df.init_date == row.init_date)
        ]
        #sort in ascending order of time
        ts = ts.sort_values('time')
        ts = ts[['ticker', 'init_date', 'time', 'low', 'high']] 
                
        #find time slot of entry
        entry_slot = find_entry_slot(row, n, intra)

        #if next add in same slot, continue to next add
        if n < row.n_entries:
            next_add_level = row.open * (1+entry_level+increment*(n))
            ts_entry = ts.loc[ts.time == entry_slot].iloc[0]
            if ts_entry.high >= next_add_level:
                continue
            
        #assume action can only be taken in slot after 
        ts = ts.loc [ts.time > entry_slot]
                
        if method == 'last':
            stop_level = row.open * (1+entry_level+increment*(n-1)-win)
        elif method == 'cost':
            avg_pc = avg_cost_pc(n, entry_level, increment, inc_factor)
            stop_level = row.open * (avg_pc*(1-win))
        
        wins = ts.loc[ts.low <= stop_level]
        
        
        if n < row.n_entries:
            adds = ts.loc[ts.high >= next_add_level]
        else:
            adds = pd.DataFrame()
        
        #check that a win and/or add exists
        
        if adds.shape[0] == 0 and wins.shape[0] == 0:
            #include tuple for use in handle_stop_bracket()
            return row.exit, row.n_entries, None 
        
        elif adds.shape[0] != 0 and wins.shape[0] == 0:
            
            continue
        
        elif adds.shape[0] == 0 and wins.shape[0] != 0:
            #include tuple for use in handle_stop_bracket()
            return stop_level, n, wins.iloc[0].time
        
        else:
            
            earliest_win = wins.iloc[0].time
            next_add = adds.iloc[0].time
            
            if earliest_win < next_add:
                #include tuple for use in handle_stop_bracket()
                return stop_level, n, wins.iloc[0].time
            
            elif earliest_win > next_add:
        
                continue
            
            else:
                
                print(f'bias hit: {row.ticker}, {row.init_date}')
                if bias == 'win':
                    #include tuple for use in handle_stop_bracket()
                    return stop_level, n, wins.iloc[0].time
                
                elif bias == 'add':
                    
                    continue
    
    return row.exit, row.n_entries, None    
    
def handle_stop_bracket(row, ref_df, win, loss, bias = 'win', method = 'cost'):           
    '''
    If stop loss and stop win occur in same time slot, losses are assumed to occur.
    '''
    win_level, n, win_time = handle_stop_win(row, ref_df, win, bias=bias,method=method)
    loss_level, loss_time = handle_stop_loss(row, ref_df, loss, method=method)

    
    if win_time == None and loss_time == None:
        return row.exit, row.n_entries
    elif win_time == None and loss_time != None:
        return loss_level, row.n_entries
    elif win_time != None and loss_time == None:
        return win_level, n
    else:
        if win_time < loss_time:
            return win_level, n
        if win_time > loss_time:
            return loss_level, row.n_entries
        else:
            return loss_level, row.n_entries

#create new variables
master = pd.read_csv('../data/high up 20 1 lt prev close lt 2 06072021 agg -253 5.csv')
print('Building master DF')
master = master.rename(columns = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume',
                'init date': 'init_date',
            },
)

master.date = pd.to_datetime(master.date)
master.init_date = pd.to_datetime(master.init_date)
earliest = datetime.strptime('2021-02-26', '%Y-%m-%d')
latest = datetime.strptime('2021-05-28', '%Y-%m-%d')
master = master.loc[master.init_date >= earliest]
master = master.loc[master.init_date <= latest]

intra = pd.read_csv('../data/high up 20 1 lt prev close lt 2 06072021 intra 5 min.csv')
intra = intra.rename(columns = {
                'init date': 'init_date',
            },
)
intra.time = pd.to_datetime(intra.time)
intra.init_date = pd.to_datetime(intra.init_date)
earliest = datetime.strptime('2021-02-26', '%Y-%m-%d')
latest = datetime.strptime('2021-05-28', '%Y-%m-%d')
intra = intra.loc[intra.init_date >= earliest]
intra = intra.loc[intra.init_date <= latest]

#throw out innaccuracies
inaccuracies = [
        master.loc[master['ticker'] == 'SPHS'],
        master.loc[master['ticker'] == 'SNNAQ'],
        #master.loc[master['ticker'] == 'EMMS'],
        #master.loc[master['ticker'] == 'HYMCW'],
]

for i in inaccuracies:
    master = master.drop(index=i.index)


####build sample DF############################################################
print('building sample DF')

#start sample
sample = master.loc[master.delta == 0].copy()

###handle intraday data


if window != None:
    #windowed trade
    intra['t_start'] = intra.init_date + pd.Timedelta(hours=t_start_h, minutes=t_start_m)
    intra['t_end'] = intra.init_date + pd.Timedelta(hours=t_end_h, minutes = t_end_m + window)
    intra = intra.loc[(intra.time >= intra.t_start) & (intra.time <= intra.t_end)]

    
    #find relevant high (must occur by t_end, cannot occur in t_end+window)
    t_high = intra.loc[intra.time <= intra.t_end-pd.Timedelta(minutes = window)]
    t_high = intra.groupby(['ticker', 'init_date'], as_index=False).max()
    t_high = t_high[['ticker', 'init_date', 'high']]
    t_high = t_high.rename(columns={'high':'t_high'})
    sample = sample.merge(t_high, how='inner', on=['init_date','ticker'], validate='1:1')
    
    
    #isolate data that qualifies
    sample = sample.loc[sample.t_high >= sample.open*(1+entry_level)]
    
    
    #find time slot of first entry
    sample['first_slot'] = sample.apply(
            lambda x: find_entry_slot(x,1, intra),
            axis=1
    )
    intra = intra.merge(
            sample[['ticker', 'init_date', 'first_slot']],
            how = 'left',
            on = ['ticker', 'init_date'],
    )
    
    #restrict intra data to rows with first_entry <= time slot <= first_entry + window
    intra = intra.loc[
            (intra.first_slot <= intra.time) 
            & (intra.time <= intra.first_slot + pd.Timedelta(minutes = window))
    ]
    
    #calculate possible number of entries within window
    #find relevant window high (redundant, should improve)
    w_high = intra.groupby(['ticker', 'init_date'], as_index=False).max()
    w_high = w_high[['ticker', 'init_date', 'high']]
    w_high = w_high.rename(columns={'high':'w_high'})
    sample = sample.merge(w_high, how='inner', on=['init_date','ticker'], validate='1:1')
    sample['rise'] = (sample.w_high - sample.open)/sample.open
    sample['n_entries'] = np.floor((sample.rise - entry_level) / increment) + 1
    if max_entries != None:
        #restrict by max entries allowed
        limit_break = sample.loc[sample.n_entries > max_entries]
        sample.loc[limit_break.index, 'n_entries'] = max_entries
    #clean up
    sample.n_entries = sample.n_entries.astype('int')
    sample = sample.loc[sample.n_entries>0]   
    
    #find exit (exit time as parameter)
    t_exit = intra.first_slot + pd.Timedelta(minutes=window)
    exit_ = intra.loc[intra.time == t_exit][['ticker', 'init_date', 'close']]
    exit_ = exit_.rename(columns={'close':'exit'})
    sample = sample.merge(exit_, how='inner', on=['init_date','ticker'], validate='1:1')

else:
    #hard exit trade
    #isolate time slot
    intra['t_start'] = intra.init_date + pd.Timedelta(hours=t_start_h, minutes=t_start_m)
    intra['t_end'] = intra.init_date + pd.Timedelta(hours=t_end_h, minutes = t_end_m)
    intra = intra.loc[(intra.time >= intra.t_start) & (intra.time <= intra.t_end)]
    
    
    #find relevant high
    t_high = intra.groupby(['ticker', 'init_date'], as_index=False).max()
    t_high = t_high[['ticker', 'init_date', 'high']]
    t_high = t_high.rename(columns={'high':'t_high'})
    sample = sample.merge(t_high, how='inner', on=['init_date','ticker'], validate='1:1')
    
    
    #isolate data that qualifies
    sample = sample.loc[sample.t_high >= sample.open*(1+entry_level)]
    
    
    #calculate possible number of entries
    sample['rise'] = (sample.t_high - sample.open)/sample.open
    sample['n_entries'] = np.floor((sample.rise - entry_level) / increment) + 1
    if max_entries != None:
        #restrict by max entries allowed
        limit_break = sample.loc[sample.n_entries > max_entries]
        sample.loc[limit_break.index, 'n_entries'] = max_entries
    #clean up
    sample.n_entries = sample.n_entries.astype('int')
    sample = sample.loc[sample.n_entries>0]
    
    
    #find exit (exit time as parameter)
    t_exit = intra.init_date + pd.Timedelta(hours=t_end_h, minutes=t_end_m)
    exit_ = intra.loc[intra.time == t_exit][['ticker', 'init_date', 'close']]
    exit_ = exit_.rename(columns={'close':'exit'})
    sample = sample.merge(exit_, how='inner', on=['init_date','ticker'], validate='1:1')
    
#enforce stops 

if (stop_win == None ) and (stop_loss != None):
    #stop loss
    sample.exit = sample.apply(
        lambda x: handle_stop_loss(x, intra, stop_loss, method = stop_method)[0],
        axis = 1,
    )        
    
elif (stop_win != None) and (stop_loss == None):
    #stop win
    sample.exit = sample.apply(
        lambda x: handle_stop_win(x, intra, stop_win,bias = bias, method = stop_method)[0],
        axis = 1,
    )
    sample.n_entries = sample.apply(
        lambda x: handle_stop_win(x, intra, stop_win,bias = bias, method = stop_method)[1],
        axis = 1,
    )

elif (stop_win != None) and (stop_loss != None):
    #bracketed stops
    sample.exit = sample.apply(
        lambda x: handle_stop_bracket(x, intra, stop_win, stop_loss, bias = bias, method = stop_method)[0],
        axis = 1,
    )
    sample.n_entries = sample.apply(
        lambda x: handle_stop_bracket(x, intra, stop_win, stop_loss, bias = bias, method = stop_method)[1],
        axis = 1,
    ) 

    
    
    
    
#calculate average cost
    
sample['avg_cost_pc'] = sample.apply(
        lambda x: avg_cost_pc(x.n_entries, entry_level, increment, inc_factor),
        axis=1,
)

sample['avg_cost'] = sample.avg_cost_pc * sample.open
      

#calculate returns

sample['pc_return'] = (sample.avg_cost - sample.exit)/sample.avg_cost



#create weight column

sample['weight'] = sample.apply(
        lambda x: weights(x.n_entries, inc_factor),
        axis = 1,
)


#drop any rows with blanks
print('sample with NA:', sample.shape)
sample = sample.dropna()
print('sampel without NA:', sample.shape)

#create constant for regression analysis
sample['constant'] = 1.0


#analyze winners and losers


exit_df_win = sample.loc[sample['pc_return'] > 0]
exit_df_loss = sample.loc[sample['pc_return'] < 0]
    
    
print('total trades:', sample.shape[0])
print('unique tickers:', sample['ticker'].unique().shape[0])
    
#analysis of strategy
win_rate = exit_df_win.shape[0]/sample.shape[0]
loss_rate = exit_df_loss.shape[0]/sample.shape[0]
print('# winners:', exit_df_win.shape[0])
print('# losers:', exit_df_loss.shape[0])
print('win rate:', win_rate)
print('loss rate:', loss_rate)
    
#mu, sigma = scipy.stats.norm.fit(exit_df['return'])
#print('mu:', mu, 'sigma:', sigma)
    
l = 0.25
h = 0.75
regressors = ['constant',]
#winners
win_qr = QuantReg(exit_df_win['pc_return'], exit_df_win[regressors])
res_wl = win_qr.fit(q=l)
res_wu = win_qr.fit(q=h)
res_wmed = win_qr.fit(q=.5)
    
win_ols = OLS(exit_df_win['pc_return'], exit_df_win[regressors])
res_wols = win_ols.fit()
    
#losers
#CHANGING SIGN OF LOSS
loss_qr = QuantReg(-exit_df_loss['pc_return'], exit_df_loss[regressors])
res_ll = loss_qr.fit(q=l)
res_lu = loss_qr.fit(q=h)
res_lmed = loss_qr.fit(q=.5)
    
loss_ols = OLS(-exit_df_loss['pc_return'], exit_df_loss[regressors])
res_lols = loss_ols.fit()

#overall
h_o = 0.9
l_o = 0.1
ovr_qr = QuantReg(sample['pc_return'], sample[regressors])
res_ol = ovr_qr.fit(q=l_o)
res_ou = ovr_qr.fit(q=h_o)
res_omed = ovr_qr.fit(q=.5)

ovr_ols = OLS(sample['pc_return'], sample[regressors])
res_ools = ovr_ols.fit()

#calculate expected final ret
w_avg = res_wols.params['constant']
l_avg = res_lols.params['constant']
exp_ret = win_rate*w_avg - loss_rate*l_avg
print('expected final ret:', exp_ret)
kelly = win_rate - (loss_rate)/(w_avg/l_avg)
print('kelly percentage:', kelly)
tab_win = summary_col(
        [res_wl, res_wu, res_wmed, res_wols],
        model_names = [f'{l}', f'{h}', 'median', 'average'],
        info_dict={
        'N':lambda x: "{0:d}".format(int(x.nobs)),
        'Win rate':lambda x: "{:.2f}".format(win_rate),
}
)
tab_loss = summary_col(
        [res_ll, res_lu, res_lmed, res_lols],
        model_names = [f'{l}', f'{h}', 'median', 'average'],
        info_dict={
        'N':lambda x: "{0:d}".format(int(x.nobs)),
        'Loss rate':lambda x: "{:.2f}".format(loss_rate),
}
)
tab_ovr = summary_col(
        [res_ol, res_ou, res_omed, res_ools],
        model_names = [f'{l_o}', f'{h_o}', 'median', 'average'],
        info_dict={
        'N':lambda x: "{0:d}".format(int(x.nobs)),
        'Win rate':lambda x: "{:.2f}".format(win_rate),
}
)
tab_win.title = 'Analysis of Winners'
tab_loss.title = 'Analysis of Losers'
tab_ovr.title = 'Analysis Overall'

print('\n', tab_ovr)
weight_avg = ((sample.pc_return * sample.weight)/sum(sample.weight)).sum()
print('\n Weighted expected return: ', weight_avg)
print('\n', tab_win)
print('\n', tab_loss)

with open(f'prog_entry_w30_stop_win_lim4_ovr.tex', 'w') as f:
    f.write(tab_ovr.as_latex()[56:-25])
with open(f'prog_entry_w30_stop_win_lim4_win.tex', 'w') as f:
    f.write(tab_win.as_latex()[59:-25])
with open(f'prog_entry_w30_stop_win_lim4_loss.tex', 'w') as f:
    f.write(tab_loss.as_latex()[58:-25])

#zinger figure for weighted expected returns vs max entry
#
#max_list = list(range(1, sample.n_entries.max()+1))
#output = pd.DataFrame()
#for e in max_list:
#    
#    e_df = sample.copy()
#    
#    #restrict by max entries allowed
#    limit_break = e_df.loc[e_df.n_entries > e]
#    e_df.loc[limit_break.index, 'n_entries'] = e
#    
#    #recalculate average cost
#    
#    e_df['avg_cost_pc'] = e_df.apply(
#            lambda x: avg_cost_pc(x.n_entries, entry_level, increment, inc_factor),
#            axis=1,
#    )
#    
#    e_df['avg_cost'] = e_df.avg_cost_pc * e_df.open
#
#    #recalculate weights
#    e_df['weight'] = e_df.apply(
#        lambda x: weights(x.n_entries, inc_factor),
#        axis = 1,
#    )
#    
#    #recalculate returns
#
#    e_df['pc_return'] = (e_df.avg_cost - e_df.exit)/e_df.avg_cost
#
#    #calculate expected return
#    
#    e_exp_ret = e_df.pc_return.mean()
#
#    #calculate weighted expected return
#    e_avg = ((e_df.pc_return * e_df.weight)/sum(e_df.weight)).sum()
#    
#    wins = e_df.loc[e_df.pc_return > 0]
#    losses = e_df.loc[e_df.pc_return < 0]
#    
#    #calculate win rate
#
#    e_win_rate = wins.shape[0]/e_df.shape[0]
#    
#    #calculate average win
#    
#    e_w_avg = wins.pc_return.mean()
#    
#    #calculate average loss 
#    
#    e_l_avg = losses.pc_return.mean()
#        
#    row = pd.Series(
#            {
#                    'max_entry': e,
#                    'exp_ret': e_exp_ret,
#                    'w_exp_ret': e_avg,
#                    'win_rate': e_win_rate,
#                    'avg_win': e_w_avg,
#                    'avg_loss': e_l_avg,
#            },
#    )
#    
#    output = output.append(row, ignore_index=True)
#
#fig, ax = plt.subplots(3, 1, figsize = (8,15))
#
#output.plot(
#        x = 'max_entry',
#        y = 'exp_ret',
#        ax=ax[0],
#        title = 'Expected Returns by Max Number of Entries',
#        ylim = (-.05, output.exp_ret.max() + 0.05),
#        grid = True,
#)
#ax[0].set(xlabel = 'Max Number of Entries Allowed', ylabel = 'Precent Return')
#
#output.plot(
#        x = 'max_entry',
#        y = 'win_rate',
#        ax=ax[1],
#        title = 'Win Rate by Max Number of Entries',
#        ylim = (0, 1),
#        grid = True,
#)
#ax[1].set(xlabel = 'Max Number of Entries Allowed', ylabel = 'Win Rate')
#
#output.plot(
#        x = 'max_entry',
#        y = ['avg_win', 'avg_loss'],
#        ax=ax[2],
#        title = 'Average Win and Average Loss by Max Number of Entries',
#        ylim = (output.avg_loss.min()-.05, output.avg_win.max() + 0.05),
#        grid = True,
#)
#ax[2].set(xlabel = 'Max Number of Entries Allowed', ylabel = 'Precent Return')
#
#for i in range(3):
#    ax[i].set_xticks(list(range(1,sample.n_entries.max()+1)))
#
#
#fig2, ax2 = plt.subplots(figsize = (8,5))
#
#sample.hist(
#        column = 'n_entries',
#        ax = ax2,
#        bins = sample.n_entries.max(),
#)
#
#ax2.set(xlabel = 'Amount of Entries', ylabel = 'Count')
##ax2.set_xticks(list(range(1,sample.n_entries.max()+1)))
#ax2.set_title('Histogram of Trades by Number of Entries')
#
#fig3, ax3 = plt.subplots(figsize = (8,5))
#
#output.plot(
#        x = 'max_entry',
#        y = 'w_exp_ret',
#        ax=ax3,
#        title = 'Weighted Expected Returns by Max Number of Entries',
#        ylim = (-.05, output.w_exp_ret.max() + 0.05),
#        grid = True,
#)
#ax3.set(xlabel = 'Max Number of Entries Allowed', ylabel = 'Precent Return')

##3 part histogram
#
#fig4, ax4 = plt.subplots(3, 1, figsize = (8, 15))
#
#for i in range(1,4):
#    
#    df = sample.loc[sample.n_entries == i]
#    
#    df.hist(
#        column = 'pc_return',
#        ax = ax4[i-1],
#        bins = np.arange(-0.5, 0.6, 0.1),
#        )
#    
#    ax4[i-1].set(xlabel = 'Percent Return', ylabel = 'Count')
#    ax4[i-1].set_title(f'Histogram of Percent Return for {i} Entries')
#    ax4[i-1].set_xticks(np.arange(-0.5, 0.6, 0.1))
#    ax4[i-1].set_ylim(0,)
#
#


#fig.savefig('prog_entry_3plot.pdf')
#fig2.savefig('prog_entry_exit4_hist.pdf')
#fig3.savefig('prog_entry_exit4_w_avg.pdf')
#fig4.savefig('prog_entry_lim3_hist_by_entry.pdf')

#number of trades with entries >=3
gtme = sample.loc[sample.n_entries >= max_entries].shape[0]

print(f'numer of trades with >= {max_entries} entries: ', gtme/sample.shape[0])