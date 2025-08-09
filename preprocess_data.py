import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import pandas_ta as ta
from binance.client import Client
from binance.exceptions import BinanceAPIException
import keys
import time
import shutil
import multiprocessing as mp
import json
from datetime import datetime, timedelta

# --- Configuration ---
SYMBOLS = pd.read_csv('symbols.csv').iloc[:, 0].tolist()
TIME燜RAMES = ['1h', '4h']
PRIMARY_TIMEFRAME = '1h'
SEQUENCE_LENGTH = 60
PROFIT_LOSS_RATIO = 1.5
STOP_LOSS_PCT = 0.02
DATA_DIR = 'data'
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# --- Helper Functions ---

def get_client():
    """Initializes and returns the Binance client with a timeout."""
    return Client(keys.BINANCE_API_KEY, keys.BINANCE_API_SECRET, requests_params={'timeout': 10})

def download_data_for_symbol(symbol, timeframe):
    """Downloads and saves historical kline data for a single symbol and timeframe."""
    client = get_client()
    start_date = datetime.now() - timedelta(days=500)
    start_str = start_date.strftime("%d %b, %Y")
    
    # This function is executed in a subprocess, so exceptions must be handled
    # to avoid crashing the pool worker.
    try:
        klines = client.get_historical_klines(symbol, timeframe, start_str)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        
        symbol_dir = os.path.join(RAW_DIR, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        
        filepath = os.path.join(symbol_dir, f'{timeframe}.parquet')
        df.to_parquet(filepath)
        return True
    except Exception as e:
        # Propagate exception to be caught by the main process
        raise e

def download_all_raw_data():
    """
    Downloads raw kline data. If it fails due to a Binance API error (e.g., location block),
    it will warn the user and proceed, assuming data is provided manually.
    """
    print("--- Attempting to download fresh raw data from Binance ---")
    
    tasks = [(symbol, timeframe) for symbol in SYMBOLS for timeframe in TIME燜RAMES]
            
    try:
        # Use a process pool for parallel downloads
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = list(tqdm(pool.starmap(download_data_for_symbol, tasks), total=len(tasks), desc="Downloading Raw Kline Data"))
        print("--- Raw data download complete. ---")
    except BinanceAPIException as e:
        print("\n" + "="*80)
        print("WARNING: Could not download data from Binance due to an API error.")
        print(f"ERROR: {e}")
        print("This is likely due to a geographic restriction.")
        print("The script will now proceed assuming you have MANUALLY provided the raw data files.")
        print("Please ensure the 'data/raw/' directory is populated with the required parquet files.")
        print("Example structure:")
        print("  - data/raw/BTCUSDT/1h.parquet")
        print("  - data/raw/BTCUSDT/4h.parquet")
        print("  - etc. for all your symbols.")
        print("="*80 + "\n")
    except Exception as e:
        print(f"\nAn unexpected error occurred during data download: {e}")
        print("Proceeding with locally available data if any.")

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates technical indicators for a given DataFrame."""
    df.ta.rsi(append=True)
    df.ta.macd(append=True)
    df.ta.bbands(append=True)
    df.ta.atr(append=True)
    df.dropna(inplace=True)
    return df

def create_labels(df: pd.DataFrame, future_window: int = 20) -> pd.DataFrame:
    """
    Creates binary labels (win/loss) for each timestep.
    A 'win' (1) is when the price hits the take profit target before the stop loss.
    A 'loss' (0) is when the price hits the stop loss target first.
    """
    df['future_high'] = df['high'].rolling(window=future_window).max().shift(-future_window)
    df['future_low'] = df['low'].rolling(window=future_window).min().shift(-future_window)
    
    take_profit_long = df['close'] * (1 + STOP_LOSS_PCT * PROFIT_LOSS_RATIO)
    stop_loss_long = df['close'] * (1 - STOP_LOSS_PCT)
    
    win_long = (df['future_high'] >= take_profit_long) & (df['future_low'] > stop_loss_long)
    loss_long = (df['future_low'] <= stop_loss_long)
    
    df['label'] = np.where(win_long, 1, np.where(loss_long, 0, np.nan))
    
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)
    
    return df

def create_sequences(df: pd.DataFrame, feature_cols: list) -> (np.ndarray, np.ndarray, np.ndarray):
    """Creates sequences of features and corresponding labels."""
    X, y, timestamps = [], [], []
    for i in range(len(df) - SEQUENCE_LENGTH):
        seq = df.iloc[i:i + SEQUENCE_LENGTH]
        label = df.iloc[i + SEQUENCE_LENGTH]['label']
        timestamp = df.index[i + SEQUENCE_LENGTH]
        
        first_close = seq['close'].iloc[0]
        first_volume = seq['volume'].iloc[0] if 'volume' in seq.columns and seq['volume'].iloc[0] > 0 else 1

        normalized_seq_df = seq[feature_cols].copy()
        
        price_cols = [col for col in feature_cols if any(c in col for c in ['open', 'high', 'low', 'close', 'BBL', 'BBM', 'BBU', 'ATRr'])]
        for col in price_cols:
            if first_close > 0:
                normalized_seq_df[col] = (normalized_seq_df[col] / first_close) - 1
        
        if 'volume' in feature_cols:
            normalized_seq_df['volume'] = (normalized_seq_df['volume'] / first_volume) - 1

        for col in ['RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']:
            if col in feature_cols:
                normalized_seq_df[col] = normalized_seq_df[col] / 100.0

        normalized_seq_df.fillna(0, inplace=True)

        X.append(normalized_seq_df.to_numpy())
        y.append(label)
        timestamps.append(timestamp)
        
    return np.array(X), np.array(y), np.array(timestamps)

def process_symbol(symbol: str):
    """Processes the data for a single symbol."""
    try:
        primary_path = os.path.join(RAW_DIR, symbol, f'{PRIMARY_TIMEFRAME}.parquet')
        if not os.path.exists(primary_path):
            # This will be logged by the main loop
            return None

        df_primary = pd.read_parquet(primary_path)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_primary[col] = pd.to_numeric(df_primary[col])
        df_primary['timestamp'] = pd.to_datetime(df_primary['timestamp'], unit='ms')
        df_primary.set_index('timestamp', inplace=True)
        
        df_primary = calculate_features(df_primary.copy())

        df_merged = df_primary.copy()
        for timeframe in TIME燜RAMES:
            if timeframe == PRIMARY_TIMEFRAME:
                continue
            
            htf_path = os.path.join(RAW_DIR, symbol, f'{timeframe}.parquet')
            if not os.path.exists(htf_path):
                print(f"Warning: Missing {timeframe} data for {symbol}. Skipping multi-timeframe features for it.")
                continue

            df_htf = pd.read_parquet(htf_path)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_htf[col] = pd.to_numeric(df_htf[col])
            df_htf['timestamp'] = pd.to_datetime(df_htf['timestamp'], unit='ms')
            df_htf.set_index('timestamp', inplace=True)
            
            df_htf = calculate_features(df_htf.copy())
            df_htf.columns = [f'{col}_{timeframe}' for col in df_htf.columns]
            
            df_merged = pd.merge_asof(df_merged.sort_index(), df_htf.sort_index(), on='timestamp', direction='backward')

        df_merged.dropna(inplace=True)
        
        df_labeled = create_labels(df_merged.copy())
        
        feature_cols = [col for col in df_labeled.columns if col not in ['label', 'future_high', 'future_low']]
        
        X, y, timestamps = create_sequences(df_labeled, feature_cols)
        
        if len(X) > 0:
            processed_symbol_dir = os.path.join(PROCESSED_DIR, symbol)
            os.makedirs(processed_symbol_dir, exist_ok=True)
            save_path = os.path.join(processed_symbol_dir, f'features_{PRIMARY_TIMEFRAME}.npz')
            np.savez_compressed(save_path, features=X, labels=y, timestamps=timestamps)
            return feature_cols
    except FileNotFoundError as e:
        print(f"Could not process symbol {symbol} because a data file was not found: {e}")
    except Exception as e:
        print(f"Failed to process symbol {symbol}: {e}")
    return None

def main():
    """Main function to run the data preprocessing pipeline."""
    start_time = time.time()
    
    # Ensure raw data directory is cleared to force a fresh download
    if os.path.exists(RAW_DIR):
        print("--- Clearing old raw data to ensure fresh download ---")
        shutil.rmtree(RAW_DIR)
    os.makedirs(RAW_DIR)

    # Ensure processed data directory is also cleared
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)
    os.makedirs(PROCESSED_DIR)
    
    # Always download fresh raw data
    download_all_raw_data()
    
    print("\n--- Processing Data and Generating Features ---")
    feature_columns = None
    
    symbols_to_process = [s for s in SYMBOLS if os.path.exists(os.path.join(RAW_DIR, s))]
    if not symbols_to_process:
        print("No raw data found to process. Please provide data manually in the 'data/raw' directory.")
        return

    with mp.Pool(processes=mp.cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_symbol, symbols_to_process), total=len(symbols_to_process), desc="Processing Symbols"):
            if result and feature_columns is None:
                feature_columns = result
    
    if feature_columns:
        feature_columns_path = os.path.join(PROCESSED_DIR, 'feature_columns.json')
        with open(feature_columns_path, 'w') as f:
            json.dump(feature_columns, f)
        print(f"\nSaved {len(feature_columns)} feature columns to {feature_columns_path}")
        
    total_time = time.time() - start_time
    print(f"\n--- Data Preprocessing Complete in {total_time:.2f} seconds ---")

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
        
    main()
