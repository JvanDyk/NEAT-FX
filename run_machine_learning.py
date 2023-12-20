from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from shared.machine_learning import BBANDS, MLTrainOnceStrategy, MLWalkForwardStrategy, get_clean_Xy
from operator import truediv
import MetaTrader5 as mt5
import sys
import pandas as pd
import numpy as np
from backtesting import Backtest
import pandas as pd
from backtesting.test import SMA

# MetaTrader 5 data scraping
ACCOUNT_NUMBER = 123456789 # Replace with your actual account number
PASSWORD = "[]" # Replace with your actual password

SERVER_NAME = "Deriv-Demo" # Replace with your broker servername
#SYMBOL = "Crash 500 Index" # Replace with symbol data trying to fetch from server
SYMBOL = "EURUSD"

TIMEFRAME = mt5.TIMEFRAME_M1 # The timeframe
START_POS = 0 # Starting position of the data from the present to the past
COUNT = 10000 # number of candle sticks / bars / rates

login_result = mt5.initialize(login = ACCOUNT_NUMBER, server = SERVER_NAME, password = PASSWORD)
if login_result:
    print("Login successful")
else:
    sys.exit("Login failed")
    quit()

# Download
rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, START_POS, COUNT)

# Convert to pandas data frame and set index
data = pd.DataFrame(rates)
data['time'] = pd.to_datetime(data['time'], unit='s')
data.set_index('time', inplace=True)

# Remove columns
data.drop(columns=['spread', 'real_volume'], inplace=True)

# Convert columns to float64
for col in ['open', 'high', 'low', 'close', 'tick_volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with any missing values
data.dropna(inplace=True)

# Rename columns
data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)


close = data.Close.values
sma10 = SMA(data.Close, 10)
sma20 = SMA(data.Close, 20)
sma50 = SMA(data.Close, 50)
sma100 = SMA(data.Close, 100)
upper, lower = BBANDS(data, 20, 2)

# Design matrix / independent features:

# Price-derived features
data['X_SMA10'] = (close - sma10) / close
data['X_SMA20'] = (close - sma20) / close
data['X_SMA50'] = (close - sma50) / close
data['X_SMA100'] = (close - sma100) / close

data['X_DELTA_SMA10'] = (sma10 - sma20) / close
data['X_DELTA_SMA20'] = (sma20 - sma50) / close
data['X_DELTA_SMA50'] = (sma50 - sma100) / close

# Indicator features
data['X_MOM'] = data.Close.pct_change(periods=2)
data['X_BB_upper'] = (upper - close) / close
data['X_BB_lower'] = (lower - close) / close
data['X_BB_width'] = (upper - lower) / close
data['X_Sentiment'] = ~data.index.to_series().between('2017-09-27', '2017-12-14')

# Some datetime features for good measure
data['X_day'] = data.index.dayofweek
data['X_hour'] = data.index.hour

data = data.dropna().astype(float)


X, y = get_clean_Xy(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

clf = KNeighborsClassifier(7)  # Model the output based on 7 "nearest" examples
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

_ = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred}).plot(figsize=(15, 2), alpha=.7)
print('Classification accuracy: ', np.mean(y_test == y_pred))



bt = Backtest(data, MLTrainOnceStrategy, cash=500, commission=.002, margin=.05)
stats = bt.run()
bt.plot()
print(stats)



bt = Backtest(data, MLWalkForwardStrategy, cash=500, commission=.002, margin=.05)
stats = bt.run()
bt.plot()
print(stats)

