import pandas as pd
import numpy as np
import backtrader as bt
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from arch.unitroot import ADF
from arch.unitroot import KPSS
from arch.unitroot import PhillipsPerron
from sklearn.linear_model import LinearRegression
from itertools import combinations
import time 
import concurrent.futures
import strategy_sigma as strat_create

# Example usage:
filename = 'multi_process/crypto_mar_april.csv'
start_date = datetime(2024,3,1,0,0)
end_date = datetime(2024,3,10,0,0)

crypto_data = strat_create.get_data(filename, start_date, end_date)

btc = crypto_data['BTC-USD']
eth = crypto_data['ETH-USD']
ltc = crypto_data['LTC-USD']
sol = crypto_data['SOL-USD']

def run_backtest(spread, sigma_val):
    print(spread, sigma_val)
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

    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    results = cerebro.run()
    strategy = results[0]

    drawdown = strategy.analyzers.drawdown.get_analysis()
    total_net_profit = cerebro.broker.getvalue() - initial_cash
    max_drawdown = drawdown.max.moneydown

    return {
        'spread': spread,
        'sigma_val': sigma_val,
        'Total Net Profit': total_net_profit,
        'Max. Drawdown': max_drawdown
    }

#create all the spreads we want to calculate
cryptos = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'SOL-USD']
all_combinations = []
for r in range(2, len(cryptos) + 1):
    all_combinations.extend(list(combinations(cryptos, r)))

spreads = []
for combo in all_combinations:
    spreads.append(list(combo))

spreads = [['BTC-USD', 'ETH-USD', 'LTC-USD', 'SOL-USD']]
sigma_vals = [3,4]
metrics = {
    'spread': [],
    'sigma_val': [],
    'Total Net Profit': [],
    'Max. Drawdown': []
}

if __name__ == "__main__":
    start = time.perf_counter()
    error_list = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_backtest, spread, sigma_val) for spread in spreads for sigma_val in sigma_vals]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                metrics['spread'].append(result['spread'])
                metrics['sigma_val'].append(result['sigma_val'])
                metrics['Total Net Profit'].append(result['Total Net Profit'])
                metrics['Max. Drawdown'].append(result['Max. Drawdown'])
            except:
                print('error', future)
                error_list.append(future)
    final = time.perf_counter()
    df = pd.DataFrame(metrics)
    print(df.T)

    print(f'time duration: {round(final-start,2)}')
    df.to_csv('optimize_results/sigma/results_sigma_val.csv')
    
    print(error_list)