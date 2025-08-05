import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

# --- Configuration from ML.py ---
LOOKBACK_CANDLES = 100
SWING_WINDOW = 5
SEQUENCE_LENGTH = 60
QUICK_TEST = False  # Set to True to run a quick test on a small subset of data
QUICK_TEST_DATA_SIZE = 1000
QUICK_TEST_SYMBOL_COUNT = 1


def get_swing_points(klines_df, window=5):
    """Identify swing points from kline data."""
    highs = klines_df['high'].to_numpy()
    lows = klines_df['low'].to_numpy()
    timestamps = klines_df.index.to_numpy()

    swing_highs = []
    swing_lows = []

    for i in range(window, len(highs) - window):
        is_swing_high = highs[i] == np.max(highs[i-window:i+window+1])
        if is_swing_high:
            swing_highs.append((timestamps[i], highs[i]))

        is_swing_low = lows[i] == np.min(lows[i-window:i+window+1])
        if is_swing_low:
            swing_lows.append((timestamps[i], lows[i]))

    return swing_highs, swing_lows

def get_trend(swing_highs, swing_lows):
    """Determine the trend based on swing points."""
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "undetermined"

    last_high = swing_highs[-1][1]
    prev_high = swing_highs[-2][1]
    last_low = swing_lows[-1][1]
    prev_low = swing_lows[-2][1]

    if last_high > prev_high and last_low > prev_low:
        return "uptrend"
    elif last_high < prev_high and last_low < prev_low:
        return "downtrend"
    else:
        return "undetermined"

def get_fib_retracement(p1, p2, trend):
    """Calculate Fibonacci retracement levels."""
    price_range = abs(p1 - p2)
    if trend == "downtrend":
        entry_price = p1 - (price_range * 0.618) # Target the 61.8% level
    else: # Uptrend
        entry_price = p1 + (price_range * 0.618) # Target the 61.8% level
    return entry_price


def process_file(filepath):
    """
    Processes a single raw data file to generate and save features and labels.
    """
    # Disable GPU in worker process
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    symbol = filepath.split(os.sep)[-2]
    output_dir = os.path.join('data', 'processed', symbol)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'features.npz')
    if os.path.exists(output_path):
        print(f"Features for {symbol} already exist. Skipping.")
        return

    features, labels = [], []
    try:
        df = pd.read_parquet(filepath)
        df.columns = df.columns.str.lower()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        if QUICK_TEST:
            df = df.head(QUICK_TEST_DATA_SIZE)

        for i in range(LOOKBACK_CANDLES, len(df) - 1):
            strategy_klines = df.iloc[i - LOOKBACK_CANDLES : i]
            swing_highs, swing_lows = get_swing_points(strategy_klines, window=SWING_WINDOW)
            trend = get_trend(swing_highs, swing_lows)
            signal = None

            if trend == "downtrend" and len(swing_highs) > 1 and len(swing_lows) > 1:
                last_swing_high, last_swing_low = swing_highs[-1][1], swing_lows[-1][1]
                entry_price = get_fib_retracement(last_swing_high, last_swing_low, trend)
                if df['close'].iloc[i-1] > entry_price:
                    sl, tp = last_swing_high, entry_price - (last_swing_high - entry_price)
                    signal = {'side': 'short', 'entry': entry_price, 'sl': sl, 'tp': tp}
            elif trend == "uptrend" and len(swing_highs) > 1 and len(swing_lows) > 1:
                last_swing_high, last_swing_low = swing_highs[-1][1], swing_lows[-1][1]
                entry_price = get_fib_retracement(last_swing_low, last_swing_high, trend)
                if df['close'].iloc[i-1] < entry_price:
                    sl, tp = last_swing_low, entry_price + (entry_price - last_swing_low)
                    signal = {'side': 'long', 'entry': entry_price, 'sl': sl, 'tp': tp}

            if signal:
                label = None
                for j in range(i, len(df)):
                    future_high, future_low = df['high'].iloc[j], df['low'].iloc[j]
                    if signal['side'] == 'long':
                        if future_high >= signal['tp']: label = 1; break
                        if future_low <= signal['sl']: label = 0; break
                    else:
                        if future_low <= signal['tp']: label = 1; break
                        if future_high >= signal['sl']: label = 0; break
                
                if label is not None:
                    feature_klines = df.iloc[i - SEQUENCE_LENGTH : i]
                    first_candle = feature_klines.iloc[0]
                    if first_candle['close'] > 0 and first_candle['volume'] > 0:
                        normalized_features = feature_klines[['open', 'high', 'low', 'close', 'volume']].copy()
                        for col in ['open', 'high', 'low', 'close']:
                            normalized_features[col] = (normalized_features[col] / first_candle['close']) - 1
                        normalized_features['volume'] = (normalized_features['volume'] / first_candle['volume']) - 1
                        features.append(normalized_features.to_numpy())
                        labels.append(np.array([label], dtype=np.float32))

        if features:
            np.savez_compressed(output_path, features=np.array(features), labels=np.array(labels))
            print(f"Saved {len(features)} features for {symbol} to {output_path}")
        else:
            print(f"No features generated for {symbol}")

    except Exception as e:
        print(f"Error processing {symbol}: {e}")

def main():
    """
    Main function to run the data preprocessing pipeline.
    """
    path_pattern = os.path.join('data', 'raw', '*', '*.parquet')
    files = glob.glob(path_pattern)
    if not files:
        raise FileNotFoundError(f"No parquet files found at {path_pattern}. Make sure the data is in the correct directory.")

    if QUICK_TEST:
        files = files[:QUICK_TEST_SYMBOL_COUNT]

    print(f"Found {len(files)} symbol data files to process.")

    num_cores = min(mp.cpu_count(), 4)
    with mp.Pool(processes=num_cores) as pool:
        for _ in tqdm(pool.imap_unordered(process_file, files), total=len(files), desc="Preprocessing Data"):
            pass

    print("\n--- Data Preprocessing Finished ---")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
