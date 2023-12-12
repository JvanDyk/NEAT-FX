import multiprocessing
from shared.neat_strategy_buy_sell import Strat
import neat
import multiprocessing
from datetime import datetime
from operator import truediv
import threading
import MetaTrader5 as mt5
import os
import random
import math
import sys
import asyncio
import pandas as pd
import gzip
import pickle
import random
import time
import numpy as np
from backtesting import Backtest, Strategy
import traceback
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from neat.nn.feed_forward import FeedForwardNetwork
from neat.nn.recurrent import RecurrentNetwork

# Set NEAT AI Configurations
NEAT_CONFIG_PATH = "shared/Feedforward.txt"

# MetaTrader 5 data scraping
ACCOUNT_NUMBER = 12345689 # Replace with your actual account number
PASSWORD = "[]" # Replace with your actual password

SERVER_NAME = "Demo" # Replace with your broker servername
SYMBOL = "EURUSD" # Replace with symbol data trying to fetch from server


TIMEFRAME = mt5.TIMEFRAME_M30 # The timeframe
START_POS = 0 # Starting position of the data from the present to the past
COUNT = 1000 # number of candle sticks / bars / rates

login_result = mt5.initialize(login = ACCOUNT_NUMBER, server = SERVER_NAME, password = PASSWORD)
if login_result:
    print("Login successful")
else:
    sys.exit("Login failed")
    quit()
# Download
rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, START_POS, COUNT)

# Convert to pandas data frame and set index
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)

# Remove columns
df.drop(columns=['spread', 'real_volume'], inplace=True)

# Convert columns to float64
for col in ['open', 'high', 'low', 'close', 'tick_volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with any missing values
df.dropna(inplace=True)

# Rename columns
df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)


BACKTESTS = []
NETS = []
GENOMES = []

def eval_genome(genome, config):
    global df
    # Init
    genome.fitness = 0

    # FeedForward
    net = FeedForwardNetwork.create(genome, config)

    # CTRNN
    #net = neat.ctrnn.CTRNN.create(genome, config, 0.01)

    # Recurrent
    #net = RecurrentNetwork.create(genome, config)

    data = df
    runsPerNet = 2
    fitnesses = []

    for runs in range(runsPerNet):

        backTest = Backtest(df, Strat, cash=500, commission=.002, margin=.05, trade_on_close=False, exclusive_orders=False, hedging=True)
        stats = backTest.run(_genome=genome, _net=net)
        fitnesses.append(genome.fitness)

    BACKTESTS.append(stats)
    NETS.append(net)
    GENOMES.append(genome)

    print("Returns: ", stats.values[6])

    return min(fitnesses)

def run(config_path):
    global pop
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    # Start running the simulation
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome, timeout=None)

    winner = pop.run(pe.evaluate, 50)

    ####################
    # Test final result
    ####################

    # Feedforward
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # CTRNN
    #winner_net = neat.ctrnn.CTRNN.create(winner, config, 0.01)
    # Recurrent
    #winner_net = RecurrentNetwork.create(winner, config)


    backTest =Backtest(df, Strat, cash=500, commission=.002, margin=.05, trade_on_close=False, exclusive_orders=False, hedging=True)
    stats = backTest.run(_genome=winner, _net=winner_net)

    backTest.plot()
    print(stats)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, NEAT_CONFIG_PATH)

    run(config_path)

def save_checkpoint(config, population, species_set, generation):
    """ Save the current simulation state. """

    with gzip.open("winner", 'w', compresslevel=5) as f:
        data = (generation, config, population, species_set, random.getstate())
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
