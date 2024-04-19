# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


class Duy_strategy(IStrategy):
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
    timeframe = '5m'

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 1
    }

    stoploss = -0.5

    # Trailing stoploss
    trailing_stop = False

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 400


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
        return [
            ("BTC/USDT:USDT", "1h")
        ]

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
        
        if not self.dp:
            # Don't do anything if DataProvider is not available.
            return dataframe

        """
        - Parameters used for populate indicators
        """
        # Find lower & upper trendlines 
        slicing_window_size = 50
        distance = 25

        # Find supports/resistance and bottom/peak
        rollsize = 20

        #### TREND ####
        inf_tf = '1h'
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)    
        # Get the 200 hour EMA
        informative['ema200'] = ta.EMA(informative, timeperiod=200)
        
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        #### AREA OF VALUE & ENTRY TRIGER ####
        # Tìm ra các điểm max, min theo từng cụm 12 nến
        max_of_high_prices, max_indices_of_high_prices, min_of_low_prices, min_indices_of_low_prices = self.process_data(dataframe, number_candle_to_process=12)
        
        # Upper trendline
        max_values_dataframe = pd.DataFrame(data=max_of_high_prices.values, index=max_indices_of_high_prices, columns=['max_high'])
        max_values_dataframe = self.upper_trendlines(dataframe=max_values_dataframe, slicing_window=slicing_window_size, distance=distance)

        # Lower trendline
        min_values_dataframe = pd.DataFrame(data=min_of_low_prices.values, index=min_indices_of_low_prices, columns=['min_low'])
        min_values_dataframe = self.lower_trendlines(dataframe=min_values_dataframe, slicing_window=slicing_window_size, distance=distance)

        # Merge in to dataframe
        dataframe['maxslope']= None
        dataframe.loc[max_values_dataframe.index, 'maxslope'] = max_values_dataframe['maxslope']

        dataframe['max_y_intercept'] = None
        dataframe.loc[max_values_dataframe.index, 'max_y_intercept'] = max_values_dataframe['max_y_intercept']

        dataframe['minslope']= None
        dataframe.loc[min_values_dataframe.index, 'minslope'] = min_values_dataframe['minslope']

        dataframe['min_y_intercept'] = None
        dataframe.loc[min_values_dataframe.index, 'min_y_intercept'] = min_values_dataframe['min_y_intercept']

        # Group candles by year, month, day, and hour
        grouped = dataframe.groupby([dataframe['date'].dt.year, dataframe['date'].dt.month, dataframe['date'].dt.day, dataframe['date'].dt.hour])

        # Apply the function to each group and concatenate the results
        dataframe = grouped.apply(self.fill_non_none).reset_index(drop=True)

        # Shift values of maxslope, minslope, max_y_intercept, min_y_intercept of previous hour to prior hour
        dataframe['previous_maxslope'] = dataframe['maxslope'].shift(12)
        dataframe['previous_minslope'] = dataframe['minslope'].shift(12)
        dataframe['previous_max_y_intercept'] = dataframe['max_y_intercept'].shift(12)
        dataframe['previous_min_y_intercept'] = dataframe['min_y_intercept'].shift(12)

        # Detect resistances and supports
        supports, resistances = self.supports_and_resistances(dataframe, rollsize, field_for_support='low', field_for_resistance='high')
        dataframe['Support'] = None
        dataframe.loc[supports.index, "Support"] = supports.values
        dataframe['Resistance'] = None
        dataframe.loc[resistances.index, "Resistance"] = resistances.values
        
 
        # Xét từng nến để tìm các đỉnh/đáy, vùng kháng cự/ hỗ trợ gần nhất
        dataframe['nearest_support'] = -1
        dataframe['nearest_resistance'] = -1
        dataframe['nearest_peak'] = -1
        dataframe['nearest_bottom'] = -1


        # Duyệt qua từng hàng trong DataFrame
        for i in range(len(dataframe)):

            # Lấy giá đóng cửa của nến hiện tại
            close_price = dataframe.loc[i, 'close']

            # Tìm chỉ số (indice) của mức hỗ trợ và mức kháng cự gần nhất
            nearest_bottom = supports[supports.index < i]
            if len(nearest_bottom) != 0:
                dataframe.loc[i, 'nearest_bottom'] = nearest_bottom[nearest_bottom.index.max()]

            nearest_peak = resistances[resistances.index < i]
            if len(nearest_peak) != 0:
                dataframe.loc[i, 'nearest_peak'] = nearest_peak[nearest_peak.index.max()]
            

            # Tìm mức hỗ trợ gần nhất (nhỏ hơn giá đóng cửa của nến)
            nearest_support_index = nearest_bottom[nearest_bottom.values < close_price].index.max()
            if not pd.isna(nearest_support_index):
                dataframe.loc[i, 'nearest_support'] = nearest_bottom[nearest_support_index]
            
            # Tìm mức kháng cự gần nhất (lớn hơn giá đóng cửa của nến)
            nearest_resistance_index = nearest_peak[nearest_peak.values > close_price].index.max()
            if not pd.isna(nearest_resistance_index):
                dataframe.loc[i, 'nearest_resistance'] = nearest_peak[nearest_resistance_index]
        

        # Xét từng nến để tìm các đỉnh/đáy, vùng kháng cự/ hỗ trợ gần nhất
        dataframe['previous_nearest_support'] = dataframe['nearest_support'].shift(1)
        dataframe['previous_nearest_resistance'] = dataframe['nearest_resistance'].shift(1)
        dataframe['previous_nearest_peak'] = dataframe['nearest_peak'].shift(1)
        dataframe['previous_nearest_bottom'] = dataframe['nearest_bottom'].shift(1)

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
                (
                    dataframe['close'] > dataframe['previous_maxslope'] * dataframe['close'].index + dataframe['previous_max_y_intercept']
                ) &
                (   
                    dataframe['close_1h'] > dataframe['ema200_1h']
                ) 
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'enter_long')
        
        
        dataframe.loc[
            (
                (
                    dataframe['close'] < dataframe['previous_minslope'] * dataframe['close'].index + dataframe['previous_min_y_intercept']
                ) &
                (   
                    dataframe['close_1h'] < dataframe['ema200_1h']
                ) 
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'enter_short')

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
                (
                    dataframe['previous_nearest_resistance'] != -1
                ) &
                (
                    dataframe['previous_nearest_bottom'] != -1
                ) &
                (
                    (
                        dataframe['close'] >= dataframe['previous_nearest_resistance']
                    ) | 
                    (
                        dataframe['close'] <= dataframe['previous_nearest_bottom']
                    ) 
                )
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'exit_long')
       
        dataframe.loc[
            (
                (
                    dataframe['previous_nearest_support'] != -1
                ) &
                (
                    dataframe['previous_nearest_peak'] != -1
                ) &
                ( 
                    (
                        dataframe['close'] <= dataframe['previous_nearest_support'] 
                    ) | 
                    (   
                        dataframe['close'] >= dataframe['previous_nearest_peak']
                    ) 
                )
            ),
            ['exit_short', 'exit_tag']
        ] = (1, 'exit_short')
        
        dataframe.to_csv('/home/duchao/freqtrade/user_data/strategies/Duy_exit_trend_4days.csv')

        return dataframe
    
    """
    Put other function for our own strategy in here
    """

    # Define a function to fill non-None values within each group
    def fill_non_none(self, group):
        non_none_values = group.dropna(subset=['maxslope', 'max_y_intercept', 'minslope', 'min_y_intercept'], how='all')
        if not non_none_values.empty:
            if non_none_values['maxslope'].notnull().any():
                group['maxslope'] = non_none_values['maxslope'].dropna().iloc[0]
            if non_none_values['max_y_intercept'].notnull().any():
                group['max_y_intercept'] = non_none_values['max_y_intercept'].dropna().iloc[0]
            if non_none_values['minslope'].notnull().any():
                group['minslope'] = non_none_values['minslope'].dropna().iloc[0]
            if non_none_values['min_y_intercept'].notnull().any():
                group['min_y_intercept'] = non_none_values['min_y_intercept'].dropna().iloc[0]
        return group


    def find_second_peak(self, window_data, distance):
        true_index = pd.Series(window_data.index.copy(), name='true_index')
        max_high = pd.Series(window_data.values.copy(), name='max_high')
        df_window_data = pd.DataFrame({'true_index': true_index, 'max_high': max_high})

        peak1_idx = df_window_data['max_high'].idxmax()
        if peak1_idx + distance >= len(df_window_data):    
            peak2_idx = df_window_data.loc[: (peak1_idx - distance + 1), 'max_high'].idxmax()
        else:
            peak2_idx = df_window_data.loc[(peak1_idx + distance) :, 'max_high'].idxmax()
        return df_window_data.loc[peak2_idx, 'true_index']


    def find_second_bottom(self, window_data, distance):
        true_index = pd.Series(window_data.index.copy(), name='true_index')
        min_low = pd.Series(window_data.values.copy(), name='min_low')
        df_window_data = pd.DataFrame({'true_index': true_index, 'min_low': min_low})

        bottom1_idx = df_window_data.min_low.idxmin()
        if bottom1_idx + distance >= len(df_window_data):    
            bottom2_idx = df_window_data.loc[: (bottom1_idx - distance + 1), 'min_low'].idxmin()
        else:
            bottom2_idx = df_window_data.loc[(bottom1_idx + distance) :, 'min_low'].idxmin()
        return df_window_data.loc[bottom2_idx, 'true_index']


    def upper_trendlines(self, dataframe: pd.DataFrame, slicing_window=100, distance=50):
        """
        Return a Pandas dataframe with resistance lines.

        :param dataframe: incoming data matrix
        :param slicing_window: number of candles for slicing window
        :param distance: Number of candles between two maximum points
        """
        dataframe['peak1_idx'] = dataframe['max_high'].rolling(window=slicing_window).apply(lambda x: x.idxmax())

        dataframe['peak2_idx'] = dataframe['max_high'].rolling(window=slicing_window).apply(self.find_second_peak, args=(distance, ))
    
        dataframe['maxslope'] = None
        dataframe.loc[dataframe['peak1_idx'][slicing_window - 1:].index, 'maxslope'] = (np.array(dataframe.loc[dataframe['peak1_idx'][slicing_window - 1:].astype(int), 'max_high']) - 
                                                    np.array(dataframe.loc[dataframe['peak2_idx'][slicing_window - 1:].astype(int), 'max_high'])) / (np.array(dataframe['peak1_idx'][slicing_window - 1:]) - 
                                                                                                                                                np.array(dataframe['peak2_idx'][slicing_window - 1:])) # Slope between max points
        
        dataframe['max_y_intercept'] = None
        dataframe.loc[dataframe['peak1_idx'][slicing_window - 1:].index, 'max_y_intercept'] = (np.array(dataframe.loc[dataframe['peak1_idx'][slicing_window - 1:].astype(int), 'max_high']) - 
                                                            np.array(dataframe.loc[dataframe['peak1_idx'][slicing_window - 1:].index, 'maxslope']) * np.array(dataframe['peak1_idx'][slicing_window - 1:])) # y-intercept for upper trendline

        return dataframe


    def lower_trendlines(self, dataframe: pd.DataFrame, slicing_window=100, distance=50):
        """
        Return a Pandas dataframe with support lines.

        :param dataframe: incoming data matrix
        :param slicing_window: number of candles for slicing window
        :param distance: Number of candles between two minimum points
        """
        # print(dataframe)
        dataframe['bottom1_idx'] = dataframe['min_low'].rolling(window=slicing_window).apply(lambda x: x.idxmin())

        dataframe['bottom2_idx'] = dataframe['min_low'].rolling(window=slicing_window).apply(self.find_second_bottom, args=(distance, ))
    
        dataframe['minslope'] = None
        dataframe.loc[dataframe['bottom1_idx'][slicing_window - 1:].index, 'minslope'] = (np.array(dataframe.loc[dataframe['bottom1_idx'][slicing_window - 1:].astype(int), 'min_low']) - 
                                                    np.array(dataframe.loc[dataframe['bottom2_idx'][slicing_window - 1:].astype(int), 'min_low'])) / (np.array(dataframe['bottom1_idx'][slicing_window - 1:]) - 
                                                                                                                                                np.array(dataframe['bottom2_idx'][slicing_window - 1:])) # Slope between min points
        
        dataframe['min_y_intercept'] = None
        
        dataframe.loc[dataframe['bottom1_idx'][slicing_window - 1:].index, 'min_y_intercept'] = (np.array(dataframe.loc[dataframe['bottom1_idx'][slicing_window - 1:].astype(int), 'min_low']) - 
                                                            np.array(dataframe.loc[dataframe['bottom1_idx'][slicing_window - 1:].index, 'minslope']) * np.array(dataframe['bottom1_idx'][slicing_window - 1:])) # y-intercept for lower trendline
        
        return dataframe


    def process_data(self, dataframe, number_candle_to_process=1):
        
        """    
        STEP 1: Tìm ra các nến có data format "x:00:00+00:00"
        """
        dataframe['date'] = pd.to_datetime(dataframe['date'], format='%Y-%m-%d %H:%M:%S%z')
        dataframe['start_candle'] = dataframe['date'].dt.minute == 0
        start_candle = dataframe[dataframe['start_candle']].index

        """
        STEP 2: Lọc ra các vị trí nến có đủ number_candle_to_process nến đằng sau
        """
        qualified_start_candle = start_candle[start_candle + number_candle_to_process <= len(dataframe)]
        dataframe['qualified_start_candle'] = False
        dataframe.loc[qualified_start_candle, 'qualified_start_candle'] = True

        """
        STEP 3: Tìm ra các max high và min low theo từng cụm 12 nến tình từ vị trí nến bắt đầu
        """
        max_of_high_prices = dataframe['high'].rolling(window=number_candle_to_process, min_periods=1).max().shift(-(number_candle_to_process - 1)).loc[qualified_start_candle]
        max_indices_of_high_prices  = dataframe['high'].rolling(window=number_candle_to_process, min_periods=1).apply(lambda x: x.idxmax()).shift(-(number_candle_to_process - 1)).loc[qualified_start_candle].astype(int)

        min_of_low_prices = dataframe['low'].rolling(window=number_candle_to_process, min_periods=1).min().shift(-(number_candle_to_process - 1)).loc[qualified_start_candle]
        min_indices_of_low_prices = dataframe['low'].rolling(window=number_candle_to_process, min_periods=1).apply(lambda x: x.idxmin()).shift(-(number_candle_to_process - 1)).loc[qualified_start_candle].astype(int)
    
        return max_of_high_prices, max_indices_of_high_prices, min_of_low_prices, min_indices_of_low_prices
    
    def supports_and_resistances(self, dataframe, rollsize, field_for_support='low', field_for_resistance='high'): 
        diffs1 = abs(dataframe['high'].diff().abs().iloc[1:]) 

        diffs2 = abs(dataframe['low'].diff().abs().iloc[1:]) 


        mean_deviation_ressistance = diffs1.mean() 

        mean_deviation_support = diffs2.mean() 
        supports = dataframe[dataframe.low == dataframe[field_for_support].rolling(rollsize, center=True).min()].low 
        resistances = dataframe[dataframe.high == dataframe[field_for_resistance].rolling(rollsize, center=True).max()].high 
        supports = supports[abs(supports.diff()) > mean_deviation_support] 
        resistances = resistances[abs(resistances.diff()) > mean_deviation_ressistance] 
        return supports,resistances 
