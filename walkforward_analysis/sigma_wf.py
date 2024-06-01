import pandas as pd
import numpy as np
import backtrader as bt
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from arch.unitroot import ADF
from arch.unitroot import KPSS
from arch.unitroot import PhillipsPerron
from sklearn.linear_model import LinearRegression
from itertools import combinations
import time 
import concurrent.futures
import strategy_sigma as strat_create

def run_backtest(start, end, spread, sigma_val):
    print(start, end, spread, sigma_val)
    crypto_data = strat_create.get_data(filename, start, end)
    btc = crypto_data['BTC-USD']
    eth = crypto_data['ETH-USD']
    ltc = crypto_data['LTC-USD']
    sol = crypto_data['SOL-USD']

    cerebro = bt.Cerebro()
    TestStrategy = strat_create.strat_creation(len(spread), sigma_val) #create the strategy with sigma value
    cerebro.addstrategy(TestStrategy)
    if 'BTC-USD' in spread:
        btc_bt = bt.feeds.PandasData(dataname=btc)
        cerebro.adddata(btc_bt)
    if 'ETH-USD' in spread:
        eth_bt = bt.feeds.PandasData(dataname=eth)
        cerebro.adddata(eth_bt)
    if 'LTC-USD' in spread:
        ltc_bt = bt.feeds.PandasData(dataname=ltc)
        cerebro.adddata(ltc_bt)
    if 'SOL-USD' in spread:
        sol_bt = bt.feeds.PandasData(dataname=sol)
        cerebro.adddata(sol_bt)

    initial_cash = 1000000
    cerebro.broker.set_cash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)

    results = cerebro.run()
    strategy = results[0]

    total_net_profit = cerebro.broker.getvalue() - initial_cash

    return total_net_profit


def walk_forward(start_date, end_date, period_length, spreads):
    #start 3/1, end 3/5
    #metrics
    metrics = {
        'spread': [], 'train_start': [], 'train_end': [], 'test_start': [], 'test_end': [],
        'sigma_val': [], 'Total Net Profit': []
    }
    sigma_vals = [1,1.5,2,2.5,3,3.5,4,4.5,5]
    #train start date
    train_start = start_date
    while train_start + timedelta(days=period_length*2) <= end_date: 
        train_end = train_start + timedelta(days=period_length) #end of train date
        test_start = train_end #start of test date
        test_end = test_start + timedelta(days=period_length) #end of test start
        best_sigma = None
        best_profit = -float('inf')
        for sigma_val in sigma_vals: #iterate through the sigmas
            train_profit = run_backtest(train_start, train_end, spreads, sigma_val) #run backtest on the training data
            if train_profit > best_profit and train_profit != 0.0: #update best profit and best sigma
                best_profit = train_profit
                best_sigma = sigma_val
        test_profit = run_backtest(test_start, test_end, spreads, best_sigma) #run backtest on test data with best sigma from training run
        metrics['spread'].append(spreads)
        metrics['train_start'].append(train_start)
        metrics['train_end'].append(train_end)
        metrics['test_start'].append(test_start)
        metrics['test_end'].append(test_end)
        metrics['sigma_val'].append(best_sigma)
        metrics['Total Net Profit'].append(test_profit)
        
        train_start += timedelta(days=period_length)

    return pd.DataFrame(metrics)

filename = 'multi_process/crypto_mar_april.csv'

if __name__ == "__main__":
    filename = 'multi_process/crypto_mar_april.csv'
    start_date = datetime(2024, 3, 1, 0, 0)
    end_date = datetime(2024, 4, 1, 0, 0)
    period_length = 5
    #spreads = [['BTC-USD', 'LTC-USD', 'SOL-USD'], ['BTC-USD', 'ETH-USD']]
    cryptos = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'SOL-USD']
    all_combinations = []
    for r in range(2, len(cryptos) + 1):
        all_combinations.extend(list(combinations(cryptos, r)))

    spreads = []
    for combo in all_combinations:
        spreads.append(list(combo))
    #spreads = [['BTC-USD', 'ETH-USD', 'LTC-USD', 'SOL-USD']]
    start = time.perf_counter()
    metrics = {
        'spread': [], 'train_start': [], 'train_end': [], 'test_start': [], 'test_end': [],
        'sigma_val': [], 'Total Net Profit': []
    }
    error_list = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(walk_forward, start_date, end_date, period_length, spread) for spread in spreads]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                for key in metrics:
                    metrics[key].extend(result[key])
            except Exception as e:
                print(f'Error: {e}')
                error_list.append(future)

    final = time.perf_counter()
    df = pd.DataFrame(metrics)
    print(df)

    print(f'Time duration: {round(final - start, 2)} seconds')
    df.to_csv('optimize_results/sigma/results_sigma_val_WF.csv')
    print(error_list)
