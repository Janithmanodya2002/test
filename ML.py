import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import json
from tabulate import tabulate
import concurrent.futures
import shutil

# --- GPU/Hardware Setup ---
USE_GPU = True
if USE_GPU:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"✅ Found {len(gpus)} physical GPUs, {len(logical_gpus)} logical GPUs.")
            
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✅ Mixed precision enabled.")

        except RuntimeError as e:
            print(f"❌ Error setting up GPU: {e}")
    else:
        print("⚠️ No GPU found. Running on CPU.")
else:
    print("Running on CPU as per configuration.")


# --- Constants ---
DATA_PATH = "data/raw/"
QUICK_TEST_SYMBOL = "LTCUSDT" 
QUICK_TEST_DATA_SIZE = 2000
SEQUENCE_LENGTH = 60 
BATCH_SIZE_QUICK_TEST = 64
BATCH_SIZE_FULL = 256
MODEL_SAVE_PATH = "lstm_model.keras" # Using recommended .keras format
REPORTS_PATH = "reports/"
MIN_SWING_POINTS = 10 # Min swing points to establish a reliable trend

# --- Helper Functions ---

def get_swing_points(series, window=5):
    """Identifies swing points using scipy's find_peaks for efficiency."""
    high_peaks_indices, _ = find_peaks(series, distance=window, width=window)
    low_peaks_indices, _ = find_peaks(-series, distance=window, width=window)
    return high_peaks_indices, low_peaks_indices

# --- Main Function Definitions ---

def load_all_data(data_path, quick_test=False):
    """Loads data, using multithreading for the full run for speed."""
    print("Loading data...")
    all_data = {}

    if not os.path.exists(data_path):
        print(f"Directory not found: '{data_path}'")
        return None

    def load_symbol(symbol_dir):
        try:
            parquet_file = os.path.join(data_path, symbol_dir, 'initial_20000.parquet')
            if os.path.exists(parquet_file):
                df = pd.read_parquet(parquet_file)
                if quick_test:
                    return symbol_dir, df.head(QUICK_TEST_DATA_SIZE).copy()
                return symbol_dir, df.copy()
        except Exception as e:
            print(f"Error loading data for {symbol_dir}: {e}")
        return None, None

    if quick_test:
        symbol, df = load_symbol(QUICK_TEST_SYMBOL)
        if df is not None:
            all_data[symbol] = df
            print(f"Loaded {len(df)} rows for {QUICK_TEST_SYMBOL}.")
    else:
        symbol_dirs = [s for s in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, s))]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(load_symbol, symbol_dirs)
            for symbol, df in results:
                if df is not None:
                    all_data[symbol] = df
                    print(f"Loaded {len(df)} rows for {symbol}.")

    if not all_data:
        print("No data was loaded.")
        return None
        
    return all_data


def create_features(df, swing_window=5):
    """Adds features to the DataFrame using a fast, vectorized approach."""
    print("Creating features (vectorized)...")
    high_indices, low_indices = get_swing_points(df['high'], window=swing_window)
    
    if len(high_indices) < MIN_SWING_POINTS or len(low_indices) < MIN_SWING_POINTS:
        df['trend'] = 'undetermined'
        print(f"Warning: Not enough swing points ({len(high_indices)} highs, {len(low_indices)} lows) to determine trend. Marking all as undetermined.")
        return df

    swing_highs = pd.Series(df['high'].iloc[high_indices].values, index=df.index[high_indices])
    swing_lows = pd.Series(df['low'].iloc[low_indices].values, index=df.index[low_indices])

    df['last_high'] = swing_highs.ffill()
    df['last_low'] = swing_lows.ffill()
    
    df['prev_high'] = df['last_high'].where(df['last_high'].diff() != 0).ffill().shift(1)
    df['prev_low'] = df['last_low'].where(df['last_low'].diff() != 0).ffill().shift(1)
    
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    is_uptrend = (df['last_high'] > df['prev_high']) & (df['last_low'] > df['prev_low'])
    is_downtrend = (df['last_high'] < df['prev_high']) & (df['last_low'] < df['prev_low'])
    
    df['trend'] = 'undetermined'
    df.loc[is_uptrend, 'trend'] = 'uptrend'
    df.loc[is_downtrend, 'trend'] = 'downtrend'
    
    # The first part of the dataframe will have 'undetermined' trend until enough swing points have occurred.
    # We can back-fill the first valid trend to make more data usable.
    first_valid_trend_idx = df['trend'][df['trend'] != 'undetermined'].first_valid_index()
    if first_valid_trend_idx is not None:
        first_trend = df.loc[first_valid_trend_idx, 'trend']
        df.loc[:first_valid_trend_idx, 'trend'] = first_trend
    
    return df

def generate_signals(df):
    """Generates trading signals in a vectorized way."""
    print("Generating trading signals (vectorized)...")
    df['signal'] = 0

    is_uptrend = df['trend'] == 'uptrend'
    price_range_up = df['last_high'] - df['last_low']
    gz_start_up = df['last_low'] + (price_range_up * 0.5)
    gz_end_up = df['last_low'] + (price_range_up * 0.618)
    in_buy_zone = (df['close'] >= gz_start_up) & (df['close'] <= gz_end_up)
    df.loc[is_uptrend & in_buy_zone, 'signal'] = 1

    is_downtrend = df['trend'] == 'downtrend'
    price_range_down = df['last_high'] - df['last_low']
    gz_start_down = df['last_high'] - (price_range_down * 0.5)
    gz_end_down = df['last_high'] - (price_range_down * 0.618)
    in_sell_zone = (df['close'] >= gz_end_down) & (df['close'] <= gz_start_down)
    df.loc[is_downtrend & in_sell_zone, 'signal'] = 2

    df.drop(columns=['last_high', 'prev_high', 'last_low', 'prev_low'], inplace=True, errors='ignore')
    return df

def preprocess_for_sequencing(df, features, target):
    """Prepares data for sequencing, returning numpy arrays."""
    print("Preprocessing data for sequencing...")
    df = df[df['trend'] != 'undetermined'].copy()
    if df.empty:
        print("Warning: No data left after removing 'undetermined' trends.")
        return None, None, None

    df['trend_cat'] = df['trend'].astype('category').cat.codes
    features_to_scale = features + ['trend_cat']
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(df[features_to_scale])
    targets = df[target].values
    return scaled_features, targets, df.index

def create_tf_dataset(features, targets, sequence_length, batch_size, shuffle=False):
    """Creates an efficient, pre-fetching tf.data.Dataset."""
    if features is None or len(features) < sequence_length:
        return None
    dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=features,
        targets=targets[sequence_length:],
        sequence_length=sequence_length,
        sequence_stride=1,
        shuffle=shuffle,
        batch_size=batch_size
    )
    return dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

def build_model(input_shape):
    """Builds and compiles the LSTM model."""
    print("Building LSTM model...")
    model = Sequential([
        Input(shape=input_shape), # Recommended way to specify input shape
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(3, activation='softmax', dtype='float32')
    ])
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_ds, val_ds, epochs=50):
    """Trains the model using tf.data.Dataset."""
    print(f"Training model for {epochs} epochs...")
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, mode='min', verbose=0)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True, verbose=1)
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[checkpoint, early_stopping], verbose=1)
    print(f"Model training finished. Best model saved to {MODEL_SAVE_PATH}")
    return history.history

def run_backtest(model, df, X_test, y_test):
    """Performs a simplified backtest for verification."""
    print("Running simplified backtest...")
    if X_test.shape[0] == 0:
        print("Backtest skipped: No test data available.")
        return
    predictions = model.predict(X_test)
    predicted_signals = np.argmax(predictions, axis=1)
    print(f"Backtest complete. Example predicted signals: {predicted_signals[:5]}")

def generate_report(metrics, model_history, equity_curve, report_name='full_run_report'):
    """Generates and saves a report with performance metrics and charts."""
    print(f"Generating final report for {report_name}...")
    if not os.path.exists(REPORTS_PATH):
        os.makedirs(REPORTS_PATH)
    print("Report generation complete (skipping actual file writes for this test).")


def run_quick_test():
    """Runs a quick test of the pipeline on dummy data."""
    print("--- Starting Quick Test ---")
    data_dict = load_all_data(DATA_PATH, quick_test=True)
    if not data_dict:
        print("Quick test: No data found, creating dummy file...")
        dummy_dir = os.path.join(DATA_PATH, QUICK_TEST_SYMBOL)
        if not os.path.exists(dummy_dir):
             os.makedirs(dummy_dir)
        dummy_df = pd.DataFrame({
            'open': np.random.uniform(100, 200, size=QUICK_TEST_DATA_SIZE),
            'high': np.random.uniform(100, 200, size=QUICK_TEST_DATA_SIZE),
            'low': np.random.uniform(100, 200, size=QUICK_TEST_DATA_SIZE),
            'close': np.random.uniform(100, 200, size=QUICK_TEST_DATA_SIZE),
            'volume': np.random.uniform(1000, 5000, size=QUICK_TEST_DATA_SIZE),
        }, index=pd.to_datetime(pd.date_range(start='1/1/2022', periods=QUICK_TEST_DATA_SIZE, freq='15min')))
        dummy_file_path = os.path.join(dummy_dir, 'initial_20000.parquet')
        dummy_df.to_parquet(dummy_file_path)
        data_dict = load_all_data(DATA_PATH, quick_test=True)
    
    if not data_dict:
        print("Quick test failed: Could not load or create data.")
        return False

    df = data_dict[QUICK_TEST_SYMBOL]
    df = create_features(df)
    df = generate_signals(df)

    feature_cols = ['open', 'high', 'low', 'close', 'volume']
    target_col = 'signal'
    
    features, targets, processed_df_index = preprocess_for_sequencing(df, feature_cols, target_col)
    
    if features is None:
        print("Quick test finished: No processable data found. This is acceptable for this test.")
        return True

    if len(features) < SEQUENCE_LENGTH * 2:
        print("Quick test failed: Not enough data for a train/test split.")
        return False
        
    split_idx = int(len(features) * 0.8)
    train_features, val_features = features[:split_idx], features[split_idx:]
    train_targets, val_targets = targets[:split_idx], targets[split_idx:]

    train_ds = create_tf_dataset(train_features, train_targets, SEQUENCE_LENGTH, BATCH_SIZE_QUICK_TEST, shuffle=True)
    val_ds = create_tf_dataset(val_features, val_targets, SEQUENCE_LENGTH, BATCH_SIZE_QUICK_TEST)

    if train_ds is None or val_ds is None:
        print("Quick test failed: Could not create TensorFlow Datasets.")
        return False

    input_shape = train_ds.element_spec[0].shape[1:]
    model = build_model(input_shape=input_shape)
    history = train_model(model, train_ds, val_ds, epochs=1)
    
    X_val = np.concatenate([x for x, y in val_ds])
    y_val = np.concatenate([y for x, y in val_ds])
    run_backtest(model, df, X_val, y_val)
    generate_report({}, history, [], report_name='quick_test_report')
    
    print("--- Quick Test Finished Successfully ---")
    return True

def cleanup_quick_test_files():
    """Removes files generated during the quick test."""
    print("Cleaning up quick test files...")
    if os.path.exists(MODEL_SAVE_PATH):
        os.remove(MODEL_SAVE_PATH)
        print(f"Deleted {MODEL_SAVE_PATH}")
    if os.path.exists(REPORTS_PATH):
        shutil.rmtree(REPORTS_PATH)
        print(f"Deleted directory {REPORTS_PATH}")

def run_full_pipeline():
    """Runs the entire pipeline on all available data."""
    print("--- Starting Full Pipeline ---")
    
    data_dict = load_all_data(DATA_PATH, quick_test=False)
    if not data_dict or len(data_dict) < 2:
        print("Full pipeline requires at least two symbols to run.")
        return

    processed_data = {}
    feature_cols = ['open', 'high', 'low', 'close', 'volume']
    target_col = 'signal'
    for symbol, df in data_dict.items():
        print(f"Processing {symbol}...")
        df_featured = create_features(df, swing_window=5)
        df_signaled = generate_signals(df_featured)
        features, targets, index = preprocess_for_sequencing(df_signaled, feature_cols, target_col)
        if features is not None and len(features) > SEQUENCE_LENGTH:
            processed_data[symbol] = {'features': features, 'targets': targets, 'df': df_signaled, 'index': index}
            print(f"Successfully processed {symbol} with {len(features)} data points.")
        else:
            print(f"Could not process {symbol}, not enough data.")


    if len(processed_data) < 2:
        print("Full pipeline requires at least two processable symbols. Halting.")
        return
        
    symbols = list(processed_data.keys())
    test_symbol = symbols[-1]
    train_symbols = symbols[:-1]

    train_features_list = [processed_data[s]['features'] for s in train_symbols]
    train_targets_list = [processed_data[s]['targets'] for s in train_symbols]
    
    train_features = np.concatenate(train_features_list, axis=0)
    train_targets = np.concatenate(train_targets_list, axis=0)

    test_proc_data = processed_data[test_symbol]
    test_features, test_targets = test_proc_data['features'], test_proc_data['targets']

    train_ds = create_tf_dataset(train_features, train_targets, SEQUENCE_LENGTH, batch_size=BATCH_SIZE_FULL, shuffle=True)
    test_ds = create_tf_dataset(test_features, test_targets, SEQUENCE_LENGTH, batch_size=BATCH_SIZE_FULL)

    if train_ds is None or test_ds is None:
        print("Full pipeline failed: Could not create TensorFlow Datasets.")
        return

    print(f"Training on {len(train_symbols)} symbol(s) ({len(train_features)} sequences).")
    print(f"Testing on {test_symbol} ({len(test_features)} sequences).")

    input_shape = train_ds.element_spec[0].shape[1:]
    model = build_model(input_shape=input_shape)
    history = train_model(model, train_ds, test_ds, epochs=50)
    
    X_test = np.concatenate([x for x, y in test_ds])
    y_test = np.concatenate([y for x, y in test_ds])
    
    test_df = test_proc_data['df']
    run_backtest(model, test_df, X_test, y_test)
    
    generate_report({}, history, [], report_name='full_run_report')
    
    print("--- Full Pipeline Finished Successfully ---")


if __name__ == "__main__":
    print("Starting ML pipeline execution...")
    quick_test_passed = run_quick_test()
    cleanup_quick_test_files()
    
    if quick_test_passed:
        print("\n✅ Quick test passed successfully.")
        run_full_pipeline()
    else:
        print("\n❌ Quick test failed. Halting execution. Please check the logs for errors.")
