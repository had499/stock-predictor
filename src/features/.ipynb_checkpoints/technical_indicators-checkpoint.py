import pandas as pd
import numpy as np
import ta

def add_technical_indicators(df,short=True,medium=True,long=True):
    """
    Adds short-, medium-, and long-term technical indicators to a DataFrame.
    Target: next-day return.
    
    Parameters:
        df (pd.DataFrame): DataFrame with columns ['Open','High','Low','Close','Volume'] 
                           and a datetime index.
    """

    # --- Target ---
    df['returns'] = df['Close'].pct_change()
    df['returns_tmrw'] = df['returns'].shift(-1)  # tomorrow's return as target

    
    if short:
        # --- Lagged returns (short-term) ---
        for lag in range(1, 4):
            df[f'short_returns_{lag}lag'] = df['returns'].shift(lag)
    
        # --- Spread (intraday range) ---
        df['short_spread_1lag'] = (df['High'] - df['Low']).shift(1)
        
        # --- Volatility ---
        # Short-term
        df['short_volat_5rol'] = df['returns'].rolling(5).std().shift(1) * np.sqrt(252)

          # --- Volume features ---
        df['short_volume_pct_chg'] = df['Volume'].pct_change().shift(1)
        df['short_volume_5rol'] = df['Volume'].rolling(5).mean().shift(1)
            
        df['short_volume_zscore'] = (
            (df['Volume'].shift(1) - df['short_volume_5rol']) /
            df['Volume'].rolling(5).std().shift(1)
        )

            # --- Momentum indicators ---
        # Short-term
        for window in [2, 5, 7]:
            df[f'short_rsi_{window}'] = ta.momentum.RSIIndicator(df['Close'], window=window).rsi().shift(1)
        stoch_short = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=5)
        df['short_stoch'] = stoch_short.stoch().shift(1)
        df['short_stoch_signal'] = stoch_short.stoch_signal().shift(1)
        df['short_williams'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close'], lbp=5).williams_r().shift(1)
        # Short-term
        for window in [5, 10]:
            df[f'short_sma_{window}'] = ta.trend.SMAIndicator(df['Close'], window=window).sma_indicator().shift(1)
            df[f'short_ema_{window}'] = ta.trend.EMAIndicator(df['Close'], window=window).ema_indicator().shift(1)
    
        # --- Volatility indicators ---
        # Bollinger Bands
        bb_short = ta.volatility.BollingerBands(df['Close'], window=10)
        df['short_bb_hband'] = bb_short.bollinger_hband().shift(1)
        df['short_bb_lband'] = bb_short.bollinger_lband().shift(1)
        df['short_bb_pband'] = bb_short.bollinger_pband().shift(1)
    
        # ATR
        df['short_atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=5).average_true_range().shift(1)
      

    if medium:
        # Medium-term
        df['med_volat_10rol'] = df['returns'].rolling(10).std().shift(1) * np.sqrt(252)
        
        # Medium-term
        df['med_rsi_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi().shift(1)
        stoch_med = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14)
        df['med_stoch'] = stoch_med.stoch().shift(1)
        df['med_stoch_signal'] = stoch_med.stoch_signal().shift(1)
        df['med_williams'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close'], lbp=14).williams_r().shift(1)
        
        df['med_volume_20rol'] = df['Volume'].rolling(20).mean().shift(1)
        # MACD (medium-term)
        macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['med_macd'] = macd.macd().shift(1)
        df['med_macd_signal'] = macd.macd_signal().shift(1)
        df['med_macd_diff'] = macd.macd_diff().shift(1)
        # Medium-term
        for window in [20, 50]:
            df[f'med_sma_{window}'] = ta.trend.SMAIndicator(df['Close'], window=window).sma_indicator().shift(1)
            df[f'med_ema_{window}'] = ta.trend.EMAIndicator(df['Close'], window=window).ema_indicator().shift(1)

        bb_med = ta.volatility.BollingerBands(df['Close'], window=20)
        df['med_bb_hband'] = bb_med.bollinger_hband().shift(1)
        df['med_bb_lband'] = bb_med.bollinger_lband().shift(1)
        df['med_bb_pband'] = bb_med.bollinger_pband().shift(1)
        df['med_atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range().shift(1)

    if long:
        # Long-term
        df['long_volat_20rol'] = df['returns'].rolling(20).std().shift(1) * np.sqrt(252)
    
            # Long-term rolling volume
        df['long_volume_50rol'] = df['Volume'].rolling(50).mean().shift(1)
            
        df['long_volume_zscore'] = (df['Volume'].shift(1) - df['long_volume_50rol']) / df['Volume'].rolling(50).std().shift(1)
    
    
        # Long-term
        for window in [100, 200]:
            df[f'long_sma_{window}'] = ta.trend.SMAIndicator(df['Close'], window=window).sma_indicator().shift(1)


    return df
