from shared.neat_strategy_buy_sell_default import Strat
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


TIMEFRAME = mt5.TIMEFRAME_M1 # The timeframe
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

 # Simulate traders <- eval_genomes runs the tests for each genome
def eval_genomes(genomes, config):
    global GENOMES, NETS, BACKTESTS, df

    for genome_id, genome in genomes:
        genome.fitness = 0

        # Create Network
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        #net = neat.nn.RecurrentNetwork.create(genome, config)
        #net = neat.ctrnn.CTRNN.create(genome, config, 0.01)

        # Runs Network on Backtesting environment and adjust genomes fitness
        backTest = Backtest(df, Strat, cash=500, commission=.002, margin=.05, trade_on_close=False, exclusive_orders=False, hedging=True)
        stats = backTest.run(_genome=genome, _net=net)
        print(stats)

        # Track each genome
        BACKTESTS.append(stats)
        NETS.append(net)
        GENOMES.append(genome)

    # Remove genomes that are unfit
    for i, g in enumerate(genomes):
        if(g[1].fitness <= 0):
            genomes.pop(i)

# Runs the population of genomes against the backtest
def run(config_path):

    # Setup configuration from shared/neat_strategy
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Creates checkpoint files of the population
    pop.add_reporter(neat.Checkpointer(5))

    # Start running the simulation
    winner = pop.run(eval_genomes, 10) # run the simulation for X amount of Generations for populations -> results in generation mutation

    # Show the performance of the winner
    print(winner)
    stats.save()
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    #winner_net = neat.nn.RecurrentNetwork.create(genome, config)
    #winner_net = neat.ctrnn.CTRNN.create(genome, config, 0.01)

    backTest = Backtest(df, Strat, cash=500, commission=.002, margin=.05, trade_on_close=False, exclusive_orders=False, hedging=True)
    stats = backTest.run(_genome=winner, _net=winner_net)
    backTest.plot()
    print(stats)

# Starts the program
if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, NEAT_CONFIG_PATH)

    run(config_path)

# Use for manual save
def save_checkpoint(config, population, species_set, generation):
    """ Save the current simulation state. """
    with gzip.open("winner", 'w', compresslevel=5) as f:
        data = (generation, config, population, species_set, random.getstate())
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
