from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from shared.basic_strategy import SmaCross, SmaCrossBasic
from shared.machine_learning_strategy import BBANDS, MLTrainOnceStrategy, MLWalkForwardStrategy, get_clean_Xy
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

bt = Backtest(data, SmaCrossBasic)
stats = bt.run()
bt.plot()
print(stats)

stats = bt.optimize(n1=range(5, 30, 5),
                    n2=range(10, 70, 5),
                    maximize='Equity Final [$]',
                    constraint=lambda param: param.n1 < param.n2)

print(stats)
