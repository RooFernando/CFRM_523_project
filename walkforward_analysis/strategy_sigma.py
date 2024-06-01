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

def get_data(filename, start_date, end_date):
    data = pd.read_csv(filename)
    data['date'] = pd.to_datetime(data['date'])
    data = data.drop('Unnamed: 0', axis=1)

    df = data.loc[(data['date'] >= start_date) & (data['date'] <= end_date)]
    df = df.drop_duplicates()
    df = df.sort_values(by='date')
    df = df.reset_index(drop=True)

    crypto_data = {}
    for ticker in ['BTC-USD', 'ETH-USD', 'LTC-USD', 'SOL-USD']:
        crypto_df = df.loc[df['ticker'] == ticker]
        crypto_df = crypto_df.sort_values(by='date')
        crypto_df = crypto_df.reset_index(drop=True)
        crypto_df = crypto_df.set_index('date')
        crypto_data[ticker] = crypto_df

    return crypto_data


def strat_creation(datas, sigma_val):
  # Create a Stratey
  class spread_strat(bt.Strategy):
      params = (
          ('history', 17*60),
      )

      def log(self, txt, dt=None):
          ''' Logging function for this strategy '''
          if dt is None:
              # Access the datetime index from the current line in the data series
              dt = self.datas[0].datetime.datetime(0)

          # Check if dt is still a float (the internal representation for Backtrader), and convert it if needed
          if isinstance(dt, float):
              # Convert backtrader float date to datetime
              dt = bt.num2date(dt)

          # Format datetime object to string
          dt_str = dt.strftime('%Y-%m-%d %H:%M:%S')
          print('%s, %s' % (dt_str, txt))

      def log_trade(self, action, size, asset, executed_price, status):
        self.trade_logs.append({
                'Date': self.datas[0].datetime.datetime(0).strftime('%Y-%m-%d %H:%M:%S'),
                'Action': action,
                'Size': size,
                'Asset': asset,
                'executed_price': executed_price,
                'status': status
            })
        return

      def close_all_positions(self):
        count = 0
        round_val = 5
        names = ['BTC', 'ETH', 'LTC', 'SOL']
        for data in self.datas:
          position = self.getposition(data)
          if position.size != 0:
              #action = 'SELL' if position.size > 0 else 'BUY'
              #self.log_trade(action, round(-1*position.size, round_val), data.close[0], names[count])
              self.order = self.close(data)
          count+=1
        return

      def __init__(self):
        # initiate the data needed for each asset
        if datas == 2:
          self.data0 = self.datas[0].close
          self.data1 = self.datas[1].close
        elif datas == 3:
          self.data0 = self.datas[0].close
          self.data1 = self.datas[1].close
          self.data2 = self.datas[2].close
        elif datas == 4:
          self.data0 = self.datas[0].close
          self.data1 = self.datas[1].close
          self.data2 = self.datas[2].close
          self.data3 = self.datas[3].close
        #constant for strat
        self.cointegrated = False
        self.upper_bound = None
        self.lower_bound = None
        self.model_built = False
        self.coefficients = None
        self.mean_ = None
        self.sigma = sigma_val
        self.spread_val = None
        self.model_date = self.datas[0].datetime.datetime(0)
        self.trade_logs = []
        self.order = None
        self.stop_loss_constant = 0.15
        self.stop_loss_spread_val = None
        self.postion_type = None
        self.trailing_spread_val = None
        self.trailing_constant = 0.05

        return

      def create_series_tickers(self, datas):
        if datas == 2:
          series1 = pd.Series(self.data0.get(size=self.params.history))
          series2 = pd.Series(self.data1.get(size=self.params.history))
          series1_diff = series1.diff().dropna()  # taking the diff of both series
          series2_diff = series2.diff().dropna()
          tickers = [series1_diff, series2_diff]
          series_tickers = [series1, series2]
        if datas == 3:
          series1 = pd.Series(self.data0.get(size=self.params.history))
          series2 = pd.Series(self.data1.get(size=self.params.history))
          series3 = pd.Series(self.data2.get(size=self.params.history))
          series1_diff = series1.diff().dropna()  # taking the diff of both series
          series2_diff = series2.diff().dropna()
          series3_diff = series3.diff().dropna()
          tickers = [series1_diff, series2_diff, series3_diff]
          series_tickers = [series1, series2, series3]
        if datas == 4:
          series1 = pd.Series(self.data0.get(size=self.params.history))
          series2 = pd.Series(self.data1.get(size=self.params.history))
          series3 = pd.Series(self.data2.get(size=self.params.history))
          series4 = pd.Series(self.data3.get(size=self.params.history))
          series1_diff = series1.diff().dropna()  # taking the diff of both series
          series2_diff = series2.diff().dropna()
          series3_diff = series3.diff().dropna()
          series4_diff = series4.diff().dropna()
          tickers = [series1_diff, series2_diff, series3_diff, series4_diff]
          series_tickers = [series1, series2, series3, series4]
        return tickers, series_tickers

      def create_model(self, series_tickers):
        lin_model = LinearRegression()
        X = np.column_stack([s.values for s in series_tickers[1:]])
        Y = series_tickers[0].values
        lin_model.fit(X,Y)  # Fit model to the differenced data
        self.coefficients = [1] + list(-1*lin_model.coef_)
        spread = series_tickers[0]
        for i in range(1,len(self.coefficients)):
          spread += self.coefficients[i] * series_tickers[i]
        self.mean_ = np.mean(spread)#mean
        self.upper_bound = self.mean_ + self.sigma*np.std(spread)
        self.lower_bound = self.mean_ - self.sigma*np.std(spread)
        self.model_built = True
        return

      def stationarity_test(self, tickers):
        for i in range(len(tickers)): #iterate through series and check for stationarity on the DIFFERENCED
          adf = ADF(tickers[i])
          pp = PhillipsPerron(tickers[i])
          kpss = KPSS(tickers[i])
          if adf.pvalue < 0.05 and pp.pvalue < 0.05 and kpss.pvalue > 0.10: #check weather pass all the tests
            self.cointegrated = True
          else:
            self.cointegrated = False
        return

      def calc_spread_val(self, datas):
        if datas == 2:
          self.spread_val = self.data0[0] + self.data1[0]*self.coefficients[1]
        elif datas == 3:
          self.spread_val = self.data0[0] + self.data1[0]*self.coefficients[1] + self.data2[0]*self.coefficients[2]
        elif datas == 4:
          self.spread_val = self.data0[0] + self.data1[0]*self.coefficients[1] + self.data2[0]*self.coefficients[2] + self.data3[0]*self.coefficients[3]

        return

      def clear_vars(self):
        self.cointegrated = False
        self.upper_bound = None
        self.lower_bound = None
        self.model_built = False
        self.coefficients = None
        self.mean_ = None
        self.spread_val = None
        self.stop_loss_spread_val = None
        self.position_type = None
        self.trailing_spread_val = None
        return

      def check_stop_loss(self):
        if self.spread_val < self.stop_loss_spread_val and self.position_type == "LONG":
          #self.log(f'Long position stopped out: {self.spread_val}')
          self.close_all_positions()
          #print('in stop')
          self.clear_vars()
        elif self.spread_val > self.stop_loss_spread_val and self.position_type == "SHORT":
          #self.log(f'Short position stopped out: {self.spread_val}')
          self.close_all_positions()
          #print('in stop')
          self.clear_vars()
        elif self.spread_val < self.trailing_spread_val and self.position_type == "LONG":
          #self.log(f'Trailing Long position stopped out: {self.spread_val}')
          self.close_all_positions()
          #print('in trailing stop')
          self.clear_vars()
        elif self.spread_val > self.trailing_spread_val and self.position_type == "SHORT":
          #self.log(f'Trailing Short position stopped out: {self.spread_val}')
          self.close_all_positions()
          #print('in trailing stop')
          self.clear_vars()
        #update trailing stop loss
        if self.position_type == "LONG":
          self.trailing_spread_val = self.spread_val - self.spread_val * self.trailing_constant if self.spread_val > 0 else self.spread_val + self.spread_val * self.trailing_constant
        if self.position_type == "SHORT":
          self.trailing_spread_val = self.spread_val + self.spread_val * self.trailing_constant if self.spread_val > 0 else self.spread_val - self.spread_val * self.trailing_constant
        return

      def next(self):
        round_val = 5
        current_datetime = self.datas[0].datetime.datetime(0)
        current_hour = current_datetime.hour
        current_minute = current_datetime.minute
        tickers, series_tickers = self.create_series_tickers(datas) #get the datas in pandas series
        if 4 <= current_hour < 8: #this is inherently rolling
          if len(self.data0) >= self.params.history: #check if there is sufficient data for looking back
            #######################################
            if self.model_built == False: #we only build the model once for the time period
              self.stationarity_test(tickers)
              #we want to define the model once and trade on that model for the time period
              if self.cointegrated == True and self.model_built == False:
                self.create_model(series_tickers)
                self.calc_spread_val(datas)
                #self.log(f'Upper bound {self.upper_bound}, Lower bound {self.lower_bound}, spread val: {self.spread_val}, coefs:{self.coefficients}')
            #if the model is already built we trade now
            elif self.model_built == True:
              #calculate the spread value
              self.calc_spread_val(datas)
              #identify if not in position
              if not self.position:
                #get into a long spread
                if self.spread_val < self.lower_bound:
                  self.order = self.buy(self.datas[0], size=1)
                  #self.log_trade('BUY', 1, self.datas[0][0], 'BTC')
                  for i in range(1,len(self.coefficients)):
                    if self.coefficients[i] > 0:
                      self.order = self.buy(self.datas[i], size=self.coefficients[i])
                    elif self.coefficients[i] < 0:
                      self.order = self.sell(self.datas[i], size = self.coefficients[i])
                  self.stop_loss_spread_val = self.spread_val - self.spread_val * self.stop_loss_constant if self.spread_val > 0 else self.spread_val + self.spread_val * self.stop_loss_constant
                  self.trailing_spread_val = self.stop_loss_spread_val
                  self.position_type = 'LONG'
                  #self.log(f'Buy Created:{self.spread_val}, stopout: {self.stop_loss_spread_val}')
                #get into a short
                elif self.spread_val > self.upper_bound:
                  self.order = self.sell(self.datas[0], size=1)
                  #self.log_trade('SELL', -1, self.datas[0][0], 'BTC')
                  for i in range(1,len(self.coefficients)):
                    if self.coefficients[i] > 0:
                      self.order = self.sell(self.datas[i], size = self.coefficients[i])
                    elif self.coefficients[i] < 0:
                      self.order = self.buy(self.datas[i], size = self.coefficients[i])
                  self.stop_loss_spread_val = self.spread_val + self.spread_val * self.stop_loss_constant if self.spread_val > 0 else self.spread_val - self.spread_val * self.stop_loss_constant
                  self.trailing_spread_val = self.stop_loss_spread_val
                  self.position_type = 'SHORT'
                  #self.log(f'Sell Created:{self.spread_val}, stopout: {self.stop_loss_spread_val}')

              #identify if in position
              elif self.position:
                position_size = self.position.size
                #check if currently in long and hit our exit (upper boundary) position size is inherently BTC which is in direction of spread
                if position_size > 0 and self.spread_val > self.upper_bound:
                    #self.log('Close existing Long position, %.2f' % self.spread_val)
                    self.close_all_positions()
                #check if currently in short and hit our exit (lower boundary)
                elif position_size < 0 and self.spread_val < self.lower_bound:
                    #self.log('Close existing Short position, %.2f' % self.spread_val)
                    self.close_all_positions()
                self.check_stop_loss()

        #reset our model after 6 and close out any positions
        if current_hour == 8 and current_minute == 0: #always close out trades
          self.clear_vars()
          if self.position:
            #self.log("Closing all positions: ")
            self.close_all_positions()

      def notify_order(self, order):
        # Check if an order has been completed
        action = None
        executed_price = None
        size = None
        status = None
        if order.status in [order.Submitted, order.Accepted]:
          return
        if order.status in [order.Completed]:
          if order.isbuy():
            #self.log(f'BUY Executed,  asset:{order.data._name}, size:{order.executed.size}, execution price: {order.executed.price}, Cost: {order.executed.value},')
            executed_price = order.executed.price
            size = order.executed.size
            asset = order.data._name
            action = "BUY"
            status = "Complete"
          elif order.issell():
            #self.log(f'BUY Executed,  asset:{order.data._name}, size:{order.executed.size}, execution price: {order.executed.price}, Cost: {order.executed.value},')
            executed_price = order.executed.price
            size = order.executed.size
            asset = order.data._name
            action = "SELL"
            status = "Complete"

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected')
            status = "REJECTED"
            action = None
          # Write down: no pending order
        self.log_trade(action, size, asset, executed_price, status)
        self.order = None
        return

      def stop(self):
        # Close all positions at the end of the strategy
        #self.log('Closing all positions at the end of the backtest')
        #if self.position:
        #  self.close_all_positions()
        #  self.close()
          #print('need to close')
        #self.trade_logs_df = pd.DataFrame(self.trade_logs)
        #self.trade_logs_df['value'] = self.trade_logs_df['Size']*self.trade_logs_df['executed_price']
        #print(self.trade_logs_df)
        #self.trade_logs_df.to_csv('output.csv')
        return

  return spread_strat