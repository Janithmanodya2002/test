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
from sklearn.preprocessing import StandardScaler
import joblib

# --- Configuration ---
class Config:
    """
    Holds all configuration parameters for the data preprocessing script.
    """
    SYMBOLS = pd.read_csv('symbols.csv').iloc[:, 0].tolist()
    TIMEFRAMES = ['1h', '4h']
    PRIMARY_TIMEFRAME = '1h'
    SEQUENCE_LENGTH = 60
    FUTURE_WINDOW_OPTIONS = [10, 20, 30, 40]  # New: list of windows to generate data for
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
    """
    Downloads and saves historical kline data for a single symbol and timeframe,
    ensuring correct data types before saving.
    """
    print(f"[Download] Starting: {symbol} ({timeframe})")
    client = get_client()
    start_date = datetime.now() - timedelta(days=500)
    start_str = start_date.strftime("%d %b, %Y")
    
    try:
        klines = client.get_historical_klines(symbol, timeframe, start_str)
        if not klines:
            print(f"[Download] No data returned for {symbol} ({timeframe}). Skipping.")
            return False

        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        
        # --- Correct Data Types at the Source ---
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        numeric_cols = [
            'open', 'high', 'low', 'close', 'volume', 
            'quote_asset_volume', 'taker_buy_base_asset_volume', 
            'taker_buy_quote_asset_volume'
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # number_of_trades should be an integer
        df['number_of_trades'] = pd.to_numeric(df['number_of_trades'], errors='coerce').astype('Int64')

        # --- Log Dropped Rows ---
        initial_rows = len(df)
        # Drop any rows where key numeric data couldn't be parsed
        df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            print(f"[Download] Dropped {dropped_rows} rows from {symbol} ({timeframe}) due to missing values.")
        
        symbol_dir = os.path.join(Config.RAW_DIR, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        
        filepath = os.path.join(symbol_dir, f'{timeframe}.parquet')
        df.to_parquet(filepath)
        
        print(f"[Download] Success: {symbol} ({timeframe}) - {len(df)} rows saved to {filepath}")
        return True
    except Exception as e:
        print(f"[Download] FAILED for {symbol} ({timeframe}). Error: {e}")
        # To prevent silent failures in multiprocessing, it's better to raise
        raise e

def download_all_raw_data():
    """
    Downloads raw kline data. If it fails due to a Binance API error (e.g., location block),
    it will warn the user and proceed, assuming data is provided manually.
    """
    print("--- Attempting to download fresh raw data from Binance ---")
    
    tasks = [(symbol, timeframe) for symbol in Config.SYMBOLS for timeframe in Config.TIMEFRAMES]
            
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
    df.ta.obv(append=True) # Added On-Balance Volume
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
    
    take_profit_long = df['close'] * (1 + Config.STOP_LOSS_PCT * Config.PROFIT_LOSS_RATIO)
    stop_loss_long = df['close'] * (1 - Config.STOP_LOSS_PCT)
    
    win_long = (df['future_high'] >= take_profit_long) & (df['future_low'] > stop_loss_long)
    loss_long = (df['future_low'] <= stop_loss_long)
    
    df['label'] = np.where(win_long, 1, np.where(loss_long, 0, np.nan))
    
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)
    
    return df

def create_sequences(df: pd.DataFrame, feature_cols: list) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Creates sequences from a DataFrame that is already scaled.
    """
    X, y, timestamps = [], [], []
    
    # The feature data is already scaled, so we just need to create sequences
    feature_data = df[feature_cols].to_numpy()
    
    for i in range(len(df) - Config.SEQUENCE_LENGTH):
        # The sequence is a view into the feature_data array
        seq = feature_data[i:i + Config.SEQUENCE_LENGTH]
        
        # The label corresponds to the end of the sequence
        label = df.iloc[i + Config.SEQUENCE_LENGTH]['label']
        timestamp = df.index[i + Config.SEQUENCE_LENGTH]

        X.append(seq)
        y.append(label)
        timestamps.append(timestamp)
        
    return np.array(X), np.array(y), np.array(timestamps)

def process_symbol(args):
    """Processes the data for a single symbol."""
    symbol, future_window = args
    # Added detailed logging for symbol-by-symbol progress
    print(f"[Process] Starting: {symbol}")
    try:
        primary_path = os.path.join(Config.RAW_DIR, symbol, f'{Config.PRIMARY_TIMEFRAME}.parquet')
        if not os.path.exists(primary_path):
            # This will be logged by the main loop
            print(f"[Process] SKIPPED: {symbol} - Missing primary data file: {primary_path}")
            return None

        df_primary = pd.read_parquet(primary_path)
        df_primary = df_primary[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_primary[col] = pd.to_numeric(df_primary[col], errors='coerce')
        df_primary['timestamp'] = pd.to_datetime(df_primary['timestamp'], unit='ms')
        df_primary.set_index('timestamp', inplace=True)
        
        df_primary = calculate_features(df_primary.copy())

        df_merged = df_primary.copy()
        for timeframe in Config.TIMEFRAMES:
            if timeframe == Config.PRIMARY_TIMEFRAME:
                continue
            
            htf_path = os.path.join(Config.RAW_DIR, symbol, f'{timeframe}.parquet')
            if not os.path.exists(htf_path):
                print(f"[Process] Warning: Missing {timeframe} data for {symbol}. Skipping multi-timeframe features for it.")
                continue

            df_htf = pd.read_parquet(htf_path)
            df_htf = df_htf[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_htf[col] = pd.to_numeric(df_htf[col], errors='coerce')
            df_htf['timestamp'] = pd.to_datetime(df_htf['timestamp'], unit='ms')
            df_htf.set_index('timestamp', inplace=True)
            
            df_htf = calculate_features(df_htf.copy())
            df_htf.columns = [f'{col}_{timeframe}' for col in df_htf.columns]
            
            df_merged = pd.merge_asof(df_merged.sort_index(), df_htf.sort_index(), on='timestamp', direction='backward')

        df_merged.dropna(inplace=True)
        
        df_labeled = create_labels(df_merged.copy(), future_window=future_window)
        
        feature_cols = [col for col in df_labeled.columns if col not in ['label', 'future_high', 'future_low']]
        
        # Ensure all feature columns are numeric before scaling.
        for col in feature_cols:
            df_labeled[col] = pd.to_numeric(df_labeled[col], errors='coerce')
        df_labeled.dropna(subset=feature_cols, inplace=True)

        if df_labeled.empty:
            print(f"[Process] SKIPPED: {symbol} - No data left after cleaning.")
            return None

        # --- Create and Save Sequences (Unscaled) ---
        # The scaling will now be handled globally in the ML pipeline
        # to prevent data leakage and ensure consistency.
        X, y, timestamps = create_sequences(df_labeled, feature_cols)
        
        if len(X) > 0:
            processed_symbol_dir = os.path.join(Config.PROCESSED_DIR, symbol)
            os.makedirs(processed_symbol_dir, exist_ok=True)
            # New filename format includes the future window
            save_path = os.path.join(processed_symbol_dir, f'features_{Config.PRIMARY_TIMEFRAME}_fw{future_window}.npz')
            np.savez_compressed(save_path, features=X, labels=y, timestamps=timestamps)
            print(f"[Process] Success: {symbol} (fw={future_window}) - Created {len(X)} unscaled sequences. Saved to {save_path}")
            return feature_cols
        else:
            print(f"[Process] SKIPPED: {symbol} - No sequences generated after processing.")
            return None
            
    except FileNotFoundError as e:
        print(f"[Process] FAILED: {symbol} - A data file was not found: {e}")
    except Exception as e:
        import traceback
        print(f"[Process] FAILED: {symbol} - An unexpected error occurred: {e}")
        traceback.print_exc()
    return None

def main():
    """Main function to run the data preprocessing pipeline."""
    start_time = time.time()
    
    # --- Step 1: Handle Raw Data ---
    # Check if raw data directory exists and is not empty
    if not os.path.exists(Config.RAW_DIR) or not os.listdir(Config.RAW_DIR):
        print("--- Raw data not found. Starting download. ---")
        os.makedirs(Config.RAW_DIR, exist_ok=True)
        download_all_raw_data()
    else:
        print("--- Raw data found. Skipping download. ---")

    # --- Step 2: Handle Processed Data ---
    # Check if processed data directory exists and is not empty
    if not os.path.exists(Config.PROCESSED_DIR) or not os.listdir(Config.PROCESSED_DIR):
        print("\n--- Processed data not found. Starting feature generation. ---")
        os.makedirs(Config.PROCESSED_DIR, exist_ok=True)
        
        feature_columns = None
        # Prepare arguments for multiprocessing
        symbols_to_process = [s for s in Config.SYMBOLS if os.path.exists(os.path.join(Config.RAW_DIR, s))]
        if not symbols_to_process:
            print("No raw data fund to process. Please provide data manually in 'data/raw' or run with a clean state.")
            return
        
        tasks = []
        for fw in Config.FUTURE_WINDOW_OPTIONS:
            for symbol in symbols_to_process:
                tasks.append((symbol, fw))

        with mp.Pool(processes=mp.cpu_count()) as pool:
            for result in tqdm(pool.imap_unordered(process_symbol, tasks), total=len(tasks), desc="Processing Symbols for all Future Windows"):
                if result and feature_columns is None:
                    feature_columns = result # Capture feature columns from the first successful run
        
        if feature_columns:
            feature_columns_path = os.path.join(Config.PROCESSED_DIR, 'feature_columns.json')
            with open(feature_columns_path, 'w') as f:
                json.dump(feature_columns, f)
            print(f"\nSaved {len(feature_columns)} feature columns to {feature_columns_path}")
    else:
        print("\n--- Processed data found. Skipping feature generation. ---")
        
    total_time = time.time() - start_time
    print(f"\n--- Data Preprocessing Script Finished in {total_time:.2f} seconds ---")

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
        
    main()
