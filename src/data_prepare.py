import os 
import numpy as np
import pandas as pd

def data_load(file_path) -> pd.DataFrame :

    with open(file_path,'r') as f:
        df = pd.read_csv(file_path)

    print("-"*50)
    print(f"df.columns : {df.columns}")
    df['date'] = pd.to_datetime(df['date'], utc=True)
    print("-"*50)
    print(f"df.info : {df.info()}")
    df['date'] = df['date'].dt.date
    print("-"*50)
    print(f"df.columns : {df.columns}")

    if 'date' in df.columns:
        df.set_index('date', inplace=True)
        print("-"*50)
        print(f"df.columns : {df.columns}")
    return df

def daily_returns(df) -> pd.DataFrame :
    df = df.copy()
    df['returns'] = df['close'].pct_change() * 100
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1)) * 100
    return df

def create_technical_features(df, window_sizes=[5, 10, 20, 50]) -> pd.DataFrame:
    df = df.copy()

    # Price-based features
    df['price_range'] = (df['high'] - df['low']) / df['close'] * 100
    df['price_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    
    # Volatility features
    for window in window_sizes:
        df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
        df[f'realized_vol_{window}'] = (df['returns']**2).rolling(window=window).mean() * np.sqrt(252)
    
    # Moving averages
    for window in window_sizes:
        df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
    
    # Relative strength index
    for window in [14, 21]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
    
    # Moving average convergence divergence
    exp12 = df['close'].ewm(span=12).mean()
    exp26 = df['close'].ewm(span=26).mean()
    df['macd'] = exp12 - exp26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # Bollinger Bands
    for window in [20, 50]:
        sma = df['close'].rolling(window=window).mean()
        std = df['close'].rolling(window=window).std()
        df[f'bb_upper_{window}'] = sma + (std * 2)
        df[f'bb_lower_{window}'] = sma - (std * 2)
        df[f'bb_position_{window}'] = (df['close'] - sma) / (2 * std)
    
    # Volume features
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # Lagged returns
    for lag in [1, 2, 3, 5, 10]:
        df[f'return_lag_{lag}'] = df['returns'].shift(lag)
    
    # Rolling risk metrics
    for window in [22, 66, 252]:  # 1 month, 3 months, 1 year
        df[f'rolling_var_95_{window}'] = df['returns'].rolling(window=window).quantile(0.05)
        df[f'rolling_cvar_95_{window}'] = df['returns'].rolling(window=window).apply(
            lambda x: x[x <= x.quantile(0.05)].mean()
        )
    
    return df


if __name__ ==  "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir,'../datasets/JPMC_stock.csv')
    df = data_load(file_path=file_path)
    print("-"*50)
    print(f"df.shape : {df.shape}")
    print("-"*50)
    print(f"df.isna().sum() : {df.isna().sum()[df.isna().sum()>0]}")

    

    df = daily_returns(df=df)

    df = create_technical_features(df=df)

    dir_path = os.path.join(cur_dir,'../datasets/feature')
    os.makedirs(dir_path,exist_ok=True)
    df.to_csv(f"{dir_path}/featured_jpmc_stocks.csv",index=False)
    print("-"*50)
    print(f"df.columns : {df.columns}")
