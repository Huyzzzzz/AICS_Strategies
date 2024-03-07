# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union
from matplotlib.pyplot import close, grid, plot, savefig

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


class strategy_Of_An(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = '1h'

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.10

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return [("ICP/USDT", "1d")]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        # Calculate max & min trendline
        if not self.dp:
            return dataframe
        
        inf_tf = "1d"
        # Get the informative pair
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf) 

        # Calculate max line and min line in 1d timeframe
        informative['maxline_1d'] = gentrends(dataframe=informative, previous_candles=100)['Max Line']
        informative['minline_1d'] = gentrends(dataframe=informative, previous_candles=100)['Min Line']

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        dataframe.loc[
            (
                (dataframe['close_1d'] <= dataframe['maxline_1d']) &
                (dataframe['close_1d'].shift(1) > dataframe['maxline_1d'].shift(1))  
            ),
            ['enter_long', 'enter_tag']] = (1, 'max_line_cross')

        dataframe.loc[
            (
                (dataframe['close_1d'] >= dataframe['minline_1d']) &
                (dataframe['close_1d'].shift(1) < dataframe['minline_1d'].shift(1)) 
            ),
            ['enter_short', 'enter_tag']] = (1, 'min_line_cross')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        dataframe.loc[
            (
                # (qtpylib.crossed_above(dataframe['rsi'], self.sell_rsi.value)) &  # Signal: RSI crosses above sell_rsi
                # (dataframe['tema'] > dataframe['bb_middleband']) &  # Guard: tema above BB middle
                # (dataframe['tema'] < dataframe['tema'].shift(1)) &  # Guard: tema is falling
            ),
            'exit_long'] = 1
       
        dataframe.loc[
            (
                # (qtpylib.crossed_above(dataframe['rsi'], self.buy_rsi.value)) &  # Signal: RSI crosses above buy_rsi
                # (dataframe['tema'] <= dataframe['bb_middleband']) &  # Guard: tema below BB middle
                # (dataframe['tema'] > dataframe['tema'].shift(1)) &  # Guard: tema is raising
            ),
            'exit_short'] = 1
        return dataframe
    
    def find_top_k_max_elements(arr, k):
        """
        Finds the indices of the k largest values in a NumPy array.

        :param arr: Input NumPy array
        :param k: Number of largest elements to find
        :return: Array of indices corresponding to the k largest elements
        """
        indices = np.argpartition(arr, -k)[-k:]

        return indices

    def find_bottom_k_min_elements(arr, k):
        """
        Finds the indices of the k largest values in a NumPy array.

        :param arr: Input NumPy array
        :param k: Number of largest elements to find
        :return: Array of indices corresponding to the k largest elements
        """
        indices = np.argpartition(arr, k)[:k]

        return indices

    def find_best_fit_line(points):
        """
        Finds the best-fit line passing through three given points.

        :param points: List of three points [(x1, y1), (x2, y2), (x3, y3)]
        :return: Tuple (slope, y-intercept) representing the best-fit line equation
        """
        x_values, y_values = zip(*points)

        # Fit a linear regression model
        A = np.vstack([x_values, np.ones(len(x_values))]).T
        slope, y_intercept = np.linalg.lstsq(A, y_values, rcond=None)[0]

        return slope, y_intercept

    def gentrends(dataframe, previous_candles=100):
        """
        Returns a Pandas dataframe with support and resistance lines.

        :param dataframe: incomming data matrix
        :param field: for which column would you like to generate the trendline
        :param window: How long the trendlines should be. If window < 1, then it
                    will be taken as a percentage of the size of the data
        :param charts: Boolean value saying whether to print chart to screen
        """
        # x = dataframe[field][-100:]
        # x = np.array(x)
        df_high = np.array(dataframe["high_1d"][-previous_candles:])
        df_low = np.array(dataframe["low_1d"][-previous_candles:])

        top_three_high = find_top_k_max_elements(df_high, 3) # indices of three max
        bottom_three_low = find_bottom_k_min_elements(df_low, 3) # indices of three min

        # Three max points
        max_point0 = (top_three_high[0], df_high[top_three_high[0]])
        max_point1 = (top_three_high[1], df_high[top_three_high[1]])
        max_point2 = (top_three_high[2], df_high[top_three_high[2]])
        max_points = [max_point0, max_point1, max_point2]
        
        # Three min points
        min_point0 = (bottom_three_low[0], df_low[bottom_three_low[0]])
        min_point1 = (bottom_three_low[1], df_low[bottom_three_low[1]])
        min_point2 = (bottom_three_low[2], df_low[bottom_three_low[2]])
        min_points = [min_point0, min_point1, min_point2]

        # Create & extend the lines
        max_slope, max_intercept = find_best_fit_line(max_points)
        min_slope, min_intercept = find_best_fit_line(min_points)

        maxline = np.array([(max_intercept + max_slope * max) for max in range(len(df_high))])
        minline = np.array([(min_intercept + min_slope * min) for min in range(len(df_high))])

        # OUTPUT
        dataframe.loc[dataframe.index[-previous_candles:], "maxline"] = maxline
        dataframe.loc[dataframe.index[-previous_candles:], "minline"] = minline

        trends = np.transpose(np.array((df_high, df_low, maxline, minline)))
        trends = pd.DataFrame(
            trends, index=np.arange(0, len(df_high)), columns=["High", "Low", "Max Line", "Min Line"]
        )

        return trends