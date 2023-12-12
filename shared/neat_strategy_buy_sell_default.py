import pandas as pd
from neat.nn.feed_forward import FeedForwardNetwork
from neat.nn.recurrent import RecurrentNetwork
from backtesting import Strategy, Backtest
from backtesting.lib import crossover, cross
import numpy as np
from backtesting.test import SMA
import neat
from backtesting import Backtest, Strategy
from backtesting.lib import TrailingStrategy

# Configuration
AI_INDEX = 0
N_TRAIN = 400

# Visit backtesting.py and check different strategies
class Strat(Strategy):
    _genome = any

    # Backward
    #_net = neat.ctrnn.CTRNN

    # Recurrent
    #_net = RecurrentNetwork

    # Feedforward
    _net = FeedForwardNetwork

    ma_periods = [5, 10, 20, 40, 80, 160, 320, 640]

    def init(self):

        # Define the moving average periods
        self.closeArr = self.data.Close
        self.openArr = self.data.Open
        self.highArr = self.data.High
        self.lowArr = self.data.Low

        self.sma1 = self.I(SMA, self.data.Close, 5)
        self.sma2 = self.I(SMA, self.data.Close, 10)
        self.sma3 = self.I(SMA, self.data.Close, 20)
        self.sma4 = self.I(SMA, self.data.Close, 40)
        self.sma5 = self.I(SMA, self.data.Close, 80)
        self.sma6 = self.I(SMA, self.data.Close, 160)
        self.sma7 = self.I(SMA, self.data.Close, 320)
        self._trade_open = -1



    def next(self):

        # Skip the training, in-sample data
        if len(self.data) < N_TRAIN:
            return


        # candle pattern
        p100 = self.highArr[-1] - self.lowArr[-1]
        highPer = 0
        lowPer = 0
        bodyPer = 0
        trend = 0
        uod = self.closeArr[-1] - self.openArr[-1]

        if uod > 0:
            highPer = self.highArr[-1] - self.closeArr[-1]
            lowPer = self.openArr[-1] - self.lowArr[-1]
            bodyPer = self.openArr[-1] - self.closeArr[-1]
            trend = -1
        else:
            highPer = self.highArr[-1] - self.openArr[-1]
            lowPer = self.closeArr[-1] - self.lowArr[-1]
            bodyPer = self.openArr[-1] - self.closeArr[-1]
            trend = -1

        if p100 == 0:
            return
        candlePattern = [(highPer / p100), (lowPer / p100), (bodyPer / p100), trend]



        open, high, low, close = self.data.Opem, self.data.High, self.data.Low, self.data.Close
        current_time = self.data.index[-1]
        volume = self.data.Volume

        period = 10
        slope = (close[-1] - close[-period]) / period

        current_price = self.data.Close[-1]
        if len(self.trades) > 0:
            profit_loss = (current_price - self.trades[0].entry_price) * self.trades[0].size
        else:
            profit_loss = 0


        c = pd.Series(self.data.Close)
        target_risk = 0.01
        volatility = np.std(c.pct_change().dropna()[-20:]) * np.sqrt(252)
        shares = target_risk * self.equity / volatility


        output = self._net.activate(
            [
                candlePattern,
                # 1 if self.sma5[-1] > self.sma6[-1] else -1,
                # 1 if self.sma4[-1] > self.sma5[-1] else -1,
                # 1 if self.sma3[-1] > self.sma4[-1] else -1,
                # 1 if self.sma2[-1] > self.sma3[-1] else -1,
                # 1 if self.sma1[-1] > self.sma2[-1] else -1,
                slope,
                profit_loss,
                volatility,
                volume,
                close[-15],
                close[-10],
                close[-5],
                open[-1],
                high[-1],
                low[-1],
                close[-1],
            ]
        )


        if output[0] > 0.5:
            if output[1] > 0.5 and not self.position:
                self.buy()
                self._trade_open=1
            elif output[2] > 0.5 and not self.position:
                self.sell()
                self._trade_open=-1
            elif output[3] > 0.5:
                pl = 0.00
                if self.position.is_long:
                    pl += close[-1] - self.trades[0].entry_price
                    if pl > 0:
                        self._genome.fitness += pl
                    else:
                        self._genome.fitness -= pl
                elif self.position.is_short:
                    pl += self.trades[0].entry_price - close[-1]
                    if pl > 0:
                        self._genome.fitness += pl
                    else:
                        self._genome.fitness -= pl

                self.position.close()
