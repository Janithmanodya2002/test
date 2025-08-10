import os
import subprocess
import sys

def install_dependencies():
    """
    Installs all required libraries from requirements.txt.
    """
    print("--- Installing dependencies from requirements.txt ---")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("--- Dependencies installed successfully ---")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        sys.exit(1)

# Install dependencies before other imports
install_dependencies()

# Set TensorFlow logging level to suppress all but error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import setuptools before tensorflow to solve 'distutils' error on Python 3.12+
import setuptools
import pandas as pd
import numpy as np
import tensorflow as tf
from absl import logging
logging.set_verbosity(logging.ERROR)
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, MultiHeadAttention, GlobalAveragePooling1D, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import glob
import shutil
import multiprocessing as mp
import time
import json
from tabulate import tabulate
import datetime
import pandas_ta as ta
from binance.client import Client
import keys
import keras_tuner as kt
import joblib

# --- New Data Loading and Analysis Functions ---

def load_symbols():
    """Loads symbols from symbols.csv."""
    return pd.read_csv('symbols.csv').iloc[:, 0].tolist()

def download_data(symbol, timeframe='1h', limit=1000):
    """Downloads historical kline data for a given symbol and timeframe."""
    client = Client(keys.BINANCE_API_KEY, keys.BINANCE_API_SECRET)
    klines = client.get_historical_klines(symbol, timeframe, limit=limit)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    return df[['open', 'high', 'low', 'close', 'volume']]

# --- Configuration ---
class Config:
    """
    A static class to hold all configuration variables for the ML pipeline.
    This makes it easy to manage and adjust parameters from one place.
    """
    # --- Run Mode ---
    # Set to True to run a quick test on a small subset of data
    QUICK_TEST = True
    QUICK_TEST_DATA_SIZE = 1000
    QUICK_TEST_EPOCHS = 5
    QUICK_TEST_SYMBOL_COUNT = 1

    # --- Strategy and Feature Generation ---
    # These should ideally match the configuration used in the original strategy
    PRIMARY_TIMEFRAME = '1h'
    LOOKBACK_CANDLES = 100
    SWING_WINDOW = 5
    FUTURE_WINDOW_OPTIONS = [10, 20, 30, 40]
    
    # --- Model and Training Parameters ---
    SEQUENCE_LENGTH = 60  # Number of timesteps in each sample
    QUICK_TEST_BATCH_SIZE = 32
    FULL_RUN_BATCH_SIZE = 128 # Larger batch size for better GPU utilization
    EPOCHS = 50
    LEARNING_RATE = 0.001

    # --- Backtesting Parameters ---
    STARTING_BALANCE = 10000
    RISK_PER_TRADE = 0.02 # Default risk percentage per trade
    
    # --- Data Directories ---
    DATA_DIR = 'data'
    RAW_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# --- Classes and Functions from main.py (for backtesting and reporting) ---

class TradeResult:
    """A class to hold the results of a single backtested trade."""
    def __init__(self, symbol, side, entry_price, exit_price, entry_timestamp, exit_timestamp, status, pnl_usd, pnl_pct, reason_for_entry, reason_for_exit, confidence):
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.entry_timestamp = entry_timestamp
        self.exit_timestamp = exit_timestamp
        self.status = status
        self.pnl_usd = pnl_usd
        self.pnl_pct = pnl_pct
        self.reason_for_entry = reason_for_entry
        self.reason_for_exit = reason_for_exit
        self.confidence = confidence
        self.balance = 0.0 # Will be updated during backtest

def get_swing_points(klines_df, window=10):
    """
    Identify swing points from kline data, enhanced with technical indicators.
    """
    # Add indicators
    klines_df.ta.rsi(append=True)
    klines_df.ta.macd(append=True)
    klines_df.ta.bbands(append=True)
    
    # Clean up NaN values
    klines_df.dropna(inplace=True)

    highs = klines_df['high'].to_numpy()
    lows = klines_df['low'].to_numpy()
    timestamps = klines_df.index.to_numpy()
    rsi = klines_df['RSI_14'].to_numpy()
    
    swing_highs = []
    swing_lows = []

    for i in range(window, len(highs) - window):
        is_swing_high = highs[i] == np.max(highs[i-window:i+window+1])
        # Condition: RSI is overbought (e.g., > 70) to confirm swing high
        if is_swing_high and rsi[i] > 65:
            swing_highs.append((timestamps[i], highs[i]))

        is_swing_low = lows[i] == np.min(lows[i-window:i+window+1])
        # Condition: RSI is oversold (e.g., < 30) to confirm swing low
        if is_swing_low and rsi[i] < 35:
            swing_lows.append((timestamps[i], lows[i]))

    return swing_highs, swing_lows

def get_trend(swing_highs, swing_lows):
    """Determine the trend based on a sequence of swing points (HH, HL, LH, LL)."""
    if len(swing_highs) < 3 or len(swing_lows) < 3:
        return "undetermined", []

    highs = [h[1] for h in swing_highs]
    lows = [l[1] for l in swing_lows]

    # Identify last two highs and lows
    last_high = highs[-1]
    prev_high = highs[-2]
    third_high = highs[-3]
    last_low = lows[-1]
    prev_low = lows[-2]
    third_low = lows[-3]

    points = []
    # Uptrend: Higher Highs (HH) and Higher Lows (HL)
    if last_high > prev_high and prev_high > third_high and last_low > prev_low and prev_low > third_low:
        points = [
            {'type': 'HL', 'price': prev_low, 'timestamp': swing_lows[-2][0]},
            {'type': 'HH', 'price': last_high, 'timestamp': swing_highs[-1][0]},
            {'type': 'HL', 'price': last_low, 'timestamp': swing_lows[-1][0]},
        ]
        return "uptrend", points

    # Downtrend: Lower Highs (LH) and Lower Lows (LL)
    elif last_high < prev_high and prev_high < third_high and last_low < prev_low and prev_low < third_low:
        points = [
            {'type': 'LH', 'price': prev_high, 'timestamp': swing_highs[-2][0]},
            {'type': 'LL', 'price': last_low, 'timestamp': swing_lows[-1][0]},
            {'type': 'LH', 'price': last_high, 'timestamp': swing_highs[-1][0]},
        ]
        return "downtrend", points

    else:
        return "undetermined", []

def get_fib_retracement(p1, p2, trend):
    """Calculate Fibonacci retracement levels."""
    price_range = abs(p1 - p2)
    if trend == "downtrend":
        entry_price = p1 - (price_range * 0.618) # Target the 61.8% level
    else: # Uptrend
        entry_price = p1 + (price_range * 0.618) # Target the 61.8% level
    return entry_price

def analyze_market():
    """
    Main function to analyze the market for all symbols and timeframes.
    """
    symbols = load_symbols()
    timeframes = ['1h', '4h', '1d']
    lookback_periods = [10, 20, 30]

    results = []

    for symbol in symbols:
        for timeframe in timeframes:
            for lookback in lookback_periods:
                print(f"Analyzing {symbol} on {timeframe} timeframe with a lookback of {lookback}...")
                try:
                    df = download_data(symbol, timeframe)
                    swing_highs, swing_lows = get_swing_points(df.copy(), window=lookback)
                    trend, points = get_trend(swing_highs, swing_lows)

                    if trend != "undetermined":
                        result = {
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'lookback': lookback,
                            'trend': trend,
                            'swing_points': points
                        }
                        results.append(result)
                        print(f"  -> Trend Found: {trend.upper()}")
                except Exception as e:
                    print(f"Could not analyze {symbol} on {timeframe} with lookback {lookback}. Error: {e}")

    # For now, just print the results. In a real scenario, you might save this to a file or database.
    print("\n--- Analysis Complete ---")
    for res in results:
        print(json.dumps(res, indent=4, default=str))


def calculate_performance_metrics(backtest_trades, starting_balance):
    """Calculate performance metrics from a list of trades."""
    num_trades = len(backtest_trades)
    if num_trades == 0:
        return {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'win_rate': 0, 'average_win': 0, 'average_loss': 0,
            'profit_factor': 0, 'max_drawdown': 0, 'net_pnl_usd': 0,
            'net_pnl_pct': 0, 'expectancy': 0
        }

    wins = sum(1 for trade in backtest_trades if trade.status == 'win')
    losses = num_trades - wins
    win_rate = (wins / num_trades) * 100

    total_win_amount = sum(trade.pnl_usd for trade in backtest_trades if trade.status == 'win')
    total_loss_amount = sum(trade.pnl_usd for trade in backtest_trades if trade.status == 'loss')

    avg_win = total_win_amount / wins if wins > 0 else 0
    avg_loss = total_loss_amount / losses if losses > 0 else 0

    profit_factor = total_win_amount / abs(total_loss_amount) if total_loss_amount != 0 else float('inf')

    net_pnl_usd = total_win_amount + total_loss_amount
    net_pnl_pct = (net_pnl_usd / starting_balance) * 100

    expectancy = (win_rate / 100 * avg_win) - ((losses / num_trades) * abs(avg_loss))

    # Drawdown calculation
    balance_over_time = np.array([starting_balance] + [trade.balance for trade in backtest_trades])
    running_max = np.maximum.accumulate(balance_over_time)
    drawdown_series = (running_max - balance_over_time) / running_max
    max_drawdown = np.max(drawdown_series)

    return {
        'total_trades': num_trades, 'winning_trades': wins, 'losing_trades': losses,
        'win_rate': win_rate, 'average_win': avg_win, 'average_loss': avg_loss,
        'profit_factor': profit_factor, 'max_drawdown': max_drawdown * 100,
        'net_pnl_usd': net_pnl_usd, 'net_pnl_pct': net_pnl_pct, 'expectancy': expectancy,
        'drawdown_series': drawdown_series # Return the series for plotting
    }

def generate_summary_report(metrics, output_dir, starting_balance=10000):
    """Generate a human-readable summary of the backtest results."""
    headers = ["Metric", "Value"]
    table = [
        ["Starting Balance", f"${starting_balance:,.2f}"],
        ["Ending Balance", f"${metrics['net_pnl_usd'] + starting_balance:,.2f}"],
        ["Total Profit", f"${metrics['net_pnl_usd']:,.2f} ({metrics['net_pnl_pct']:.2f}%)"],
        ["Total Trades", metrics['total_trades']],
        ["Winning Trades", metrics['winning_trades']],
        ["Losing Trades", metrics['losing_trades']],
        ["Win Rate", f"{metrics['win_rate']:.2f}%"],
        ["Average Win", f"${metrics['average_win']:,.2f}"],
        ["Average Loss", f"${metrics['average_loss']:,.2f}"],
        ["Profit Factor", f"{metrics['profit_factor']:.2f}"],
        ["Max Drawdown", f"{metrics['max_drawdown']:.2f}%"],
        ["Expectancy", f"${metrics['expectancy']:,.2f}"]
    ]

    report = "Backtesting Summary\n"
    report += "===================\n\n"
    report += "Overall Performance:\n"
    report += "--------------------\n"
    report += tabulate(table, headers=headers, tablefmt="grid")

    report_path = os.path.join(output_dir, "backtest_summary.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Backtest summary saved to {report_path}")

def generate_equity_curve(backtest_trades, output_dir, starting_balance=10000):
    """Generate and save a plot of the equity curve."""
    if not backtest_trades:
        return
    balance_over_time = [starting_balance] + [trade.balance for trade in backtest_trades]
    plt.figure(figsize=(10, 6))
    plt.plot(balance_over_time)
    plt.title('Equity Curve')
    plt.xlabel('Trade Number')
    plt.ylabel('Balance (USD)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'equity_curve.png'))
    plt.close()

def generate_pnl_histogram(backtest_trades, output_dir):
    """Generates and saves a histogram of P&L for all trades."""
    if not backtest_trades:
        return
    pnl_values = [trade.pnl_usd for trade in backtest_trades]
    plt.figure(figsize=(10, 6))
    plt.hist(pnl_values, bins=50, edgecolor='black')
    plt.title('Distribution of P&L per Trade')
    plt.xlabel('P&L (USD)')
    plt.ylabel('Number of Trades')
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(output_dir, 'pnl_distribution.png'))
    plt.close()

def generate_drawdown_chart(drawdown_series, output_dir):
    """Generates and saves a plot of the drawdown series."""
    if drawdown_series is None or len(drawdown_series) == 0:
        return
    plt.figure(figsize=(10, 6))
    plt.fill_between(range(len(drawdown_series)), -drawdown_series * 100, color='red', alpha=0.3)
    plt.plot(range(len(drawdown_series)), -drawdown_series * 100, color='red')
    plt.title('Drawdown Over Time')
    plt.xlabel('Trade Number')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'drawdown_chart.png'))
    plt.close()

def generate_full_backtest_report(backtest_trades, output_dir, starting_balance=10000):
    """Generates all reports for the backtest."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not backtest_trades:
        print("No trades to generate a report for.")
        return

    metrics = calculate_performance_metrics(backtest_trades, starting_balance)
    generate_summary_report(metrics, output_dir, starting_balance)
    generate_equity_curve(backtest_trades, output_dir, starting_balance)
    generate_pnl_histogram(backtest_trades, output_dir)
    generate_drawdown_chart(metrics.get('drawdown_series'), output_dir)

    df = pd.DataFrame([vars(t) for t in backtest_trades])
    df.to_csv(os.path.join(output_dir, 'backtest_trades.csv'), index=False)
    print(f"Backtest trades saved to {os.path.join(output_dir, 'backtest_trades.csv')}")


# --- ML Specific Functions ---

def get_chronological_sample_map_and_labels(future_window, symbol_limit=None):
    """
    Scans processed data files for a SPECIFIC future_window to build a
    memory-efficient map of samples sorted chronologically by their timestamp.
    """
    print(f"--- Scanning data for future_window = {future_window} ---")
    path_pattern = os.path.join(Config.PROCESSED_DIR, '*', f'*_fw{future_window}.npz')
    files = glob.glob(path_pattern)
    if not files:
        raise FileNotFoundError(f"No processed feature files found at {path_pattern}. Run preprocess_data.py first.")

    if symbol_limit:
        # To make quick test faster, limit the number of files to scan
        files = files[:symbol_limit * 2]  # x2 to account for timeframes

    # --- Pass 1: Scan all timestamps and create a map of sample locations ---
    all_timestamps = []
    sample_location_map = []  # Stores (file_path, index_in_file) for each sample before sorting
    
    print("--- Scanning timestamps from all files (Pass 1/2) ---")
    for file_path in tqdm(files, desc="Scanning Timestamps"):
        # Use a context manager to ensure files are closed
        with np.load(file_path, allow_pickle=True) as data:
            timestamps = data['timestamps']
            num_samples = len(timestamps)
            all_timestamps.append(timestamps)
            for i in range(num_samples):
                sample_location_map.append((file_path, i))

    if not all_timestamps:
        return [], np.array([])

    timestamps_flat = np.concatenate(all_timestamps)
    
    # Get the indices that would sort the timestamps array globally
    print("--- Sorting all sample timestamps... ---")
    sorted_indices = np.argsort(timestamps_flat)
    
    # Reorder the sample location map to be in chronological order
    chronological_sample_map = [sample_location_map[i] for i in sorted_indices]
    
    # --- Pass 2: Scan all labels and sort them according to the timestamp sort order ---
    # This is necessary for calculating class weights and for reports.
    all_labels = []
    print("--- Scanning labels from all files (Pass 2/2) ---")
    for file_path in tqdm(files, desc="Scanning Labels"):
        with np.load(file_path) as data:
            all_labels.append(data['labels'])
    
    labels_flat = np.concatenate(all_labels)
    sorted_labels = labels_flat[sorted_indices]

    print(f"Found and sorted {len(chronological_sample_map)} total samples across {len(files)} files.")
    return chronological_sample_map, sorted_labels

def data_generator(sample_map, indices_to_yield):
    """
    A generator that yields single (features, label) samples for a given set of indices.
    This is designed to be wrapped by tf.data.Dataset.from_generator().
    Args:
        sample_map: The chronologically sorted list of (file_path, index_in_file).
        indices_to_yield: The specific indices from the map to load (e.g., from a train/val split).
    """
    # Group indices by file to minimize file I/O
    file_to_indices = {}
    for i in indices_to_yield:
        file_path, index_in_file = sample_map[i]
        if file_path not in file_to_indices:
            file_to_indices[file_path] = []
        file_to_indices[file_path].append(index_in_file)

    # Iterate through the files that contain the required indices
    for file_path, indices_in_file in file_to_indices.items():
        # Load the data for the current file once, allowing pickle for object arrays.
        with np.load(file_path, allow_pickle=True) as data:
            features_array = data['features']
            labels_array = data['labels']
            
            # Yield the required samples one by one from the loaded data
            for i in indices_in_file:
                yield (features_array[i].astype(np.float32), labels_array[i].astype(np.int32))

def generate_training_report(history, y_true, y_pred_probs, output_dir):
    """Generates and saves a report of the model's training and performance."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    if 'loss' in history.history:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.savefig(os.path.join(output_dir, "training_history.png"))
    plt.close()

    # Classification report
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    # Add zero_division=0 to prevent warnings when a class has no predictions
    unique_labels = np.unique(y_true)
    if len(unique_labels) == 1:
        # If only one class is present in the true labels, specify it for the report
        target_names = [f'Class {int(unique_labels[0])}']
        labels = unique_labels
        report = classification_report(y_true, y_pred, target_names=target_names, labels=labels, zero_division=0)
    else:
        # Default case when both classes are present
        report = classification_report(y_true, y_pred, target_names=['Loss', 'Win'], zero_division=0)

    conf_matrix = confusion_matrix(y_true, y_pred)
    
    test_loss, test_acc = -1, -1
    if 'val_loss' in history.history:
        test_loss = history.history['val_loss'][-1]
        test_acc = history.history['val_accuracy'][-1]

    report_str = "--- Model Training Report ---\n\n"
    report_str += f"Final Validation Accuracy: {test_acc:.4f}\n"
    report_str += f"Final Validation Loss: {test_loss:.4f}\n\n"
    report_str += "Classification Report (on Test Set):\n"
    report_str += report + "\n\n"
    report_str += "Confusion Matrix (on Test Set):\n"
    report_str += str(conf_matrix) + "\n"
    
    report_path = os.path.join(output_dir, "training_summary.txt")
    with open(report_path, "w") as f:
        f.write(report_str)

    print(f"Training report saved to {output_dir}")


def build_model_for_tuning(hp, input_shape):
    """Builds the model for hyperparameter tuning."""
    inputs = tf.keras.Input(shape=input_shape)
    
    hp_l2_reg = hp.Choice('l2_reg', values=[1e-2, 1e-3, 1e-4])

    hp_units_1 = hp.Int('units_1', min_value=64, max_value=256, step=64)
    x = Bidirectional(LSTM(units=hp_units_1, return_sequences=True, kernel_regularizer=l2(hp_l2_reg)))(inputs)
    hp_dropout_1 = hp.Float('dropout_1', min_value=0.2, max_value=0.6, step=0.1)
    x = Dropout(hp_dropout_1)(x)

    hp_units_2 = hp.Int('units_2', min_value=32, max_value=128, step=32)
    x = Bidirectional(LSTM(units=hp_units_2, return_sequences=True, kernel_regularizer=l2(hp_l2_reg)))(x)

    hp_num_heads = hp.Int('num_heads', min_value=2, max_value=8, step=2)
    # For self-attention, query, value, and key are the same.
    x = MultiHeadAttention(num_heads=hp_num_heads, key_dim=hp_units_2)(x, x)
    
    x = GlobalAveragePooling1D()(x)

    hp_dropout_2 = hp.Float('dropout_2', min_value=0.2, max_value=0.6, step=0.1)
    x = Dropout(hp_dropout_2)(x)

    hp_dense_units = hp.Int('dense_units', min_value=32, max_value=128, step=32)
    x = Dense(units=hp_dense_units, activation='relu', kernel_regularizer=l2(hp_l2_reg))(x)
    
    outputs = Dense(1, activation='sigmoid', dtype='float32')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])

    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def run_hyperparameter_search(train_dataset, val_dataset, epochs, future_window, steps_per_epoch, validation_steps, is_quick_test=False):
    """Runs hyperparameter search using Keras Tuner."""
    print("--- Starting Hyperparameter Search ---")

    input_shape = train_dataset.element_spec[0].shape[1:]
    build_fn = lambda hp: build_model_for_tuning(hp, input_shape)

    tuner = kt.Hyperband(
        build_fn,
        objective='val_accuracy',
        max_epochs=epochs,
        factor=3,
        directory='hyperparameter_tuning',
        project_name=f'lstm_trader_tuning_fw{future_window}',
        overwrite=False
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    search_epochs = 3 if is_quick_test else epochs
    print(f"Tuner search running for a max of {search_epochs} epochs.")
    tuner.search(train_dataset, epochs=search_epochs, validation_data=val_dataset, callbacks=[stop_early], steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Return the tuner and the best hyperparameters
    return tuner, best_hps


def generate_walk_forward_report(fold_metrics, best_hps, output_dir):
    """Generates a detailed report of the walk-forward validation results."""
    report_str = "--- Walk-Forward Validation Report ---\n\n"

    report_str += "--- Best Hyperparameters Found ---\n"
    report_str += json.dumps(best_hps.values, indent=4)
    report_str += "\n\n"

    report_str += "--- Fold-by-Fold Performance ---\n"
    for i, metrics in enumerate(fold_metrics):
        report_str += f"\nFold {i + 1}:\n"
        report_str += f"  - Validation Loss: {metrics['loss']:.4f}\n"
        report_str += f"  - Validation Accuracy: {metrics['accuracy']:.4f}\n"
        if '1' in metrics['report']:
            report_str += f"  - Win Precision: {metrics['report']['1']['precision']:.4f}\n"
            report_str += f"  - Win Recall: {metrics['report']['1']['recall']:.4f}\n"
            report_str += f"  - Win F1-Score: {metrics['report']['1']['f1-score']:.4f}\n"

    report_str += "\n\n--- Average Performance ---\n"
    avg_loss = np.mean([m['loss'] for m in fold_metrics])
    avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
    # Handle cases where a class might not be predicted in a fold
    avg_precision = np.mean([m['report'].get('1', {}).get('precision', 0) for m in fold_metrics if '1' in m['report']])
    avg_recall = np.mean([m['report'].get('1', {}).get('recall', 0) for m in fold_metrics if '1' in m['report']])
    avg_f1 = np.mean([m['report'].get('1', {}).get('f1-score', 0) for m in fold_metrics if '1' in m['report']])

    report_str += f"  - Average Validation Loss: {avg_loss:.4f}\n"
    report_str += f"  - Average Validation Accuracy: {avg_accuracy:.4f}\n"
    report_str += f"  - Average Win Precision: {avg_precision:.4f}\n"
    report_str += f"  - Average Win Recall: {avg_recall:.4f}\n"
    report_str += f"  - Average Win F1-Score: {avg_f1:.4f}\n"

    report_path = os.path.join(output_dir, "walk_forward_summary.txt")
    with open(report_path, "w") as f:
        f.write(report_str)
    
    print(f"Walk-forward validation summary saved to {report_path}")


def create_and_train_model(model, train_dataset, val_dataset, test_dataset, output_dir, training_report_dir, epochs, class_weight=None):
    """Build, compile, and train the LSTM model using tf.data.Dataset."""
    print("Training final model with best hyperparameters...")
    
    model.summary()

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        verbose=1,
        class_weight=class_weight,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
    )

    print("Model training complete.")
    
    model_path = os.path.join(output_dir, "lstm_trader.keras")
    model.save(model_path)
    
    print("Generating training report on test data...")
    # To generate the report, we need to get the labels from the test_dataset
    y_true = np.concatenate([y for x, y in test_dataset], axis=0)
    y_pred_probs = model.predict(test_dataset)
    generate_training_report(history, y_true, y_pred_probs, training_report_dir)

    return model_path

def get_signal_and_features(df, i, feature_columns):
    """
    Checks for a trade signal at a specific index `i` and returns the signal
    and the UN-SCALED features for the model if a signal is found.
    Scaling is handled globally in the main pipeline.
    """
    if i < Config.LOOKBACK_CANDLES or i >= len(df):
        return None, None

    # 1. Generate Signal
    strategy_klines = df.iloc[i - Config.LOOKBACK_CANDLES : i]
    swing_highs, swing_lows = get_swing_points(strategy_klines.copy(), window=Config.SWING_WINDOW)
    trend, _ = get_trend(swing_highs, swing_lows)
    
    signal = None
    np_close = df['close'].to_numpy()

    if trend == "downtrend" and len(swing_highs) > 1 and len(swing_lows) > 1:
        last_swing_high, last_swing_low = swing_highs[-1][1], swing_lows[-1][1]
        entry_price = get_fib_retracement(last_swing_high, last_swing_low, trend)
        if np_close[i-1] > entry_price:
            sl, tp = last_swing_high, entry_price - (last_swing_high - entry_price)
            signal = {'side': 'short', 'entry': entry_price, 'sl': sl, 'tp': tp}
    elif trend == "uptrend" and len(swing_highs) > 1 and len(swing_lows) > 1:
        last_swing_high, last_swing_low = swing_highs[-1][1], swing_lows[-1][1]
        entry_price = get_fib_retracement(last_swing_low, last_swing_high, trend)
        if np_close[i-1] < entry_price:
            sl, tp = last_swing_low, entry_price + (entry_price - last_swing_low)
            signal = {'side': 'long', 'entry': entry_price, 'sl': sl, 'tp': tp}

    if not signal:
        return None, None

    # 2. Extract Features (Unscaled)
    feature_klines = df.iloc[i - Config.SEQUENCE_LENGTH : i].copy()
    
    # Calculate all technical indicators that were used in preprocessing
    feature_klines.ta.rsi(append=True)
    feature_klines.ta.macd(append=True)
    feature_klines.ta.bbands(append=True)
    feature_klines.ta.atr(append=True)
    feature_klines.ta.obv(append=True) # Added On-Balance Volume
    feature_klines.dropna(inplace=True)
    
    if len(feature_klines) < Config.SEQUENCE_LENGTH:
        return None, None # Not enough data after indicator calculation

    # Ensure the columns match what the model was trained on
    feature_values = feature_klines[feature_columns]
    
    return signal, feature_values.to_numpy()


def simulate_trades(df, symbol, potential_trades, predictions):
    """Simulates trades based on predictions and returns a list of TradeResult objects."""
    local_trades = []
    np_high = df['high'].to_numpy()
    np_low = df['low'].to_numpy()

    for (signal, trade_idx), prediction in zip(potential_trades, predictions):
        if prediction[0] > 0.45:
            exit_price, status, reason = (None, None, None)
            for j in range(trade_idx, len(df)):
                future_high, future_low = np_high[j], np_low[j]
                if signal['side'] == 'long':
                    if future_high >= signal['tp']: exit_price, status, reason = signal['tp'], 'win', 'TP Hit'; break
                    if future_low <= signal['sl']: exit_price, status, reason = signal['sl'], 'loss', 'SL Hit'; break
                else:
                    if future_low <= signal['tp']: exit_price, status, reason = signal['tp'], 'win', 'TP Hit'; break
                    if future_high >= signal['sl']: exit_price, status, reason = signal['sl'], 'loss', 'SL Hit'; break
            
            if status:
                trade = TradeResult(
                    symbol=symbol, side=signal['side'], entry_price=signal['entry'], exit_price=exit_price,
                    entry_timestamp=df.index[trade_idx], exit_timestamp=df.index[j], status=status,
                    pnl_usd=0, pnl_pct=0,
                    reason_for_entry=f"ML Signal (Pred: {prediction[0]:.2f})",
                    reason_for_exit=reason,
                    confidence=prediction[0]
                )
                local_trades.append(trade)
    return local_trades

def backtest_symbol_worker(args):
    """
    A worker function for multiprocessing. It finds potential trades and extracts
    features for a given symbol, but does NOT perform predictions.
    It returns the features and trade info to the main process.
    """
    filepath, primary_timeframe, feature_columns = args
    symbol = os.path.basename(os.path.dirname(filepath))
    
    try:
        df = pd.read_parquet(filepath)
        df.columns = df.columns.str.lower()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        features_for_symbol = []
        trades_for_symbol = []

        for i in range(Config.LOOKBACK_CANDLES, len(df)):
            signal, features = get_signal_and_features(df, i, feature_columns)
            if signal and features is not None:
                features_for_symbol.append(features)
                # We pass the dataframe itself to be used later in simulation
                trades_for_symbol.append({'signal': signal, 'index': i, 'dataframe': df, 'symbol': symbol})

        return features_for_symbol, trades_for_symbol

    except FileNotFoundError:
        print(f"Could not process {symbol}. Required files not found.")
        return [], []
    except Exception as e:
        print(f"Error processing {symbol} in worker: {e}")
        return [], []

def run_ml_backtest(model_path, data_files, output_dir, starting_balance=Config.STARTING_BALANCE, risk_per_trade=Config.RISK_PER_TRADE):
    """
    Run a memory-optimized backtest using the trained ML model.
    The model is loaded once, and feature extraction is done in parallel.
    """
    print("--- Starting Optimized ML-Powered Backtest ---")

    # 1. Load the model, the scaler, and feature columns
    print("Loading model, scaler, and feature columns...")
    model = tf.keras.models.load_model(model_path)
    scaler_path = os.path.join(os.path.dirname(model_path), 'global_scaler.joblib')
    scaler = joblib.load(scaler_path)
    
    feature_columns_path = os.path.join(Config.PROCESSED_DIR, 'feature_columns.json')
    with open(feature_columns_path, 'r') as f:
        feature_columns = json.load(f)

    # 2. Use multiprocessing to extract features in parallel
    print("Extracting features from all symbols in parallel...")
    num_cores = min(mp.cpu_count(), 4)
    pool_args = [(file, Config.PRIMARY_TIMEFRAME, feature_columns) for file in data_files]
    
    all_features = []
    all_potential_trades = []
    with mp.Pool(processes=num_cores) as pool:
        for features, trades in tqdm(pool.imap_unordered(backtest_symbol_worker, pool_args), total=len(data_files), desc="Extracting Features"):
            if features:
                all_features.extend(features)
                all_potential_trades.extend(trades)

    if not all_features:
        print("No potential trades found across all symbols. Backtest complete.")
        return

    # 3. Scale features and perform batch prediction
    print(f"Found {len(all_potential_trades)} potential trades. Scaling features and performing batch prediction...")
    
    # Reshape for scaling: from (n_samples, seq_len, n_features) to (n_samples * seq_len, n_features)
    n_samples, seq_len, n_features = np.array(all_features).shape
    reshaped_features = np.array(all_features).reshape(-1, n_features)
    
    scaled_reshaped_features = scaler.transform(reshaped_features)
    
    # Reshape back to sequence form
    scaled_features = scaled_reshaped_features.reshape(n_samples, seq_len, n_features)
    
    predictions = model.predict(scaled_features, batch_size=Config.FULL_RUN_BATCH_SIZE, verbose=0)
    
    # 4. Simulate trades and generate results
    print("Simulating trades with predictions...")
    all_trades = []
    for i, trade_info in enumerate(all_potential_trades):
        prediction = predictions[i]
        if prediction[0] < 0.45: # Confidence threshold
            continue

        df = trade_info['dataframe']
        signal = trade_info['signal']
        trade_idx = trade_info['index']
        symbol = trade_info['symbol']
        
        # Simulate the single trade
        np_high = df['high'].to_numpy()
        np_low = df['low'].to_numpy()
        exit_price, status, reason = (None, None, None)

        for j in range(trade_idx, len(df)):
            future_high, future_low = np_high[j], np_low[j]
            if signal['side'] == 'long':
                if future_high >= signal['tp']: exit_price, status, reason = signal['tp'], 'win', 'TP Hit'; break
                if future_low <= signal['sl']: exit_price, status, reason = signal['sl'], 'loss', 'SL Hit'; break
            else: # short
                if future_low <= signal['tp']: exit_price, status, reason = signal['tp'], 'win', 'TP Hit'; break
                if future_high >= signal['sl']: exit_price, status, reason = signal['sl'], 'loss', 'SL Hit'; break
        
        if status:
            trade = TradeResult(
                symbol=symbol, side=signal['side'], entry_price=signal['entry'], exit_price=exit_price,
                entry_timestamp=df.index[trade_idx], exit_timestamp=df.index[j], status=status,
                pnl_usd=0, pnl_pct=0,
                reason_for_entry=f"ML Signal (Pred: {prediction[0]:.2f})",
                reason_for_exit=reason,
                confidence=prediction[0]
            )
            all_trades.append(trade)
            
    # 5. Sort trades and calculate PnL
    all_trades.sort(key=lambda t: t.entry_timestamp)
    
    balance = starting_balance
    for trade in all_trades:
        risk_multiplier = 0.5 + abs(trade.confidence - 0.5)
        dynamic_risk = risk_per_trade * risk_multiplier
        risk_amount = balance * dynamic_risk
        
        sl_pct = abs(trade.entry_price - (trade.exit_price if trade.status == 'loss' else trade.entry_price * (1-0.02))) / trade.entry_price
        if sl_pct > 0:
            position_size_usd = risk_amount / sl_pct
            if trade.side == 'short':
                trade.pnl_usd = (trade.entry_price - trade.exit_price) / trade.entry_price * position_size_usd
            else: # long
                trade.pnl_usd = (trade.exit_price - trade.entry_price) / trade.entry_price * position_size_usd
            
            balance += trade.pnl_usd
            trade.balance = balance
            trade.pnl_pct = (trade.pnl_usd / position_size_usd) * 100 if position_size_usd > 0 else 0
        else:
            trade.pnl_usd = 0
            trade.pnl_pct = 0
            trade.balance = balance
            
    print(f"Backtest complete. Total trades executed: {len(all_trades)}")
    generate_full_backtest_report(all_trades, output_dir, starting_balance)


def run_pipeline(is_quick_test: bool, future_window: int):
    """
    Encapsulates the entire ML pipeline for a given future_window.
    """
    start_time = time.time()

    # --- Configuration based on mode ---
    run_mode = "Quick Test" if is_quick_test else "Full Run"
    # Create a unique output folder for each future_window run
    output_folder = f"quick_test_output_fw{future_window}" if is_quick_test else f"full_run_output_fw{future_window}"
    
    epochs = Config.QUICK_TEST_EPOCHS if is_quick_test else Config.EPOCHS
    symbol_limit = Config.QUICK_TEST_SYMBOL_COUNT if is_quick_test else None
    batch_size = Config.QUICK_TEST_BATCH_SIZE if is_quick_test else Config.FULL_RUN_BATCH_SIZE
    n_splits = 2 if is_quick_test else 5 # Fewer splits for a quick test
    
    model_output_dir = os.path.join(output_folder, "model")
    backtest_report_dir = os.path.join(output_folder, "backtest_report")
    training_report_dir = os.path.join(output_folder, "training_report")

    # --- Setup Directories ---
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    for d in [model_output_dir, backtest_report_dir, training_report_dir]:
        os.makedirs(d)

    print(f"\n{'='*80}")
    print(f"--- Starting {run_mode} for future_window={future_window} ---")
    print(f"{'='*80}")

    try:
        # Step 1: Build a memory-efficient map of the data samples for the given future_window
        print(f"\n[Step 1/4] Scanning data for future_window={future_window}...")
        sample_map, sorted_labels = get_chronological_sample_map_and_labels(future_window, symbol_limit)
        
        if len(sample_map) < 200:
            print(f"Error: Not enough data for future_window={future_window}. Skipping.")
            return None, None # Return None for results if skipped

        # Get one sample to determine the shape and type for tf.data.Dataset.
        temp_features, temp_label = next(data_generator(sample_map, [0]))
        output_signature = (tf.TensorSpec(shape=temp_features.shape, dtype=temp_features.dtype), tf.TensorSpec(shape=(), dtype=temp_label.dtype))
        del temp_features, temp_label
        
        # --- Walk-Forward Validation Setup ---
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics = []
        best_hps = None
        scaler = None
        
        final_model, final_history, final_y_val, final_y_pred_probs = None, None, None, None

        print(f"\n[Step 2/4] Starting Walk-Forward Validation...")
        for fold, (train_index, val_index) in enumerate(tscv.split(np.arange(len(sample_map)))):
            print(f"\n--- Fold {fold + 1}/{n_splits} ---")
            
            y_train, y_val = sorted_labels[train_index], sorted_labels[val_index]
            print(f"Train size: {len(train_index)}, Validation size: {len(val_index)}")
            
            train_dataset_raw = tf.data.Dataset.from_generator(lambda: data_generator(sample_map, train_index), output_signature=output_signature)
            val_dataset_raw = tf.data.Dataset.from_generator(lambda: data_generator(sample_map, val_index), output_signature=output_signature)

            if scaler is None:
                print("Fitting global feature scaler on the first training fold...")
                training_features_for_scaling = [x for x, y in train_dataset_raw]
                if not training_features_for_scaling:
                    print("Error: No training data to fit scaler."); return None, None
                n_samples, seq_len, n_features = np.array(training_features_for_scaling).shape
                reshaped_features = np.array(training_features_for_scaling).reshape(-1, n_features)
                scaler = StandardScaler().fit(reshaped_features)
                scaler_path = os.path.join(model_output_dir, 'global_scaler.joblib')
                joblib.dump(scaler, scaler_path)
                print(f"Global scaler saved to {scaler_path}")

            def scale_features(features, label):
                def _scale(features):
                    shape = features.shape
                    reshaped = tf.reshape(features, [-1, shape[-1]])
                    scaled = scaler.transform(reshaped.numpy())
                    return tf.reshape(scaled, shape)
                scaled_features = tf.py_function(_scale, [features], tf.float32)
                
                # Restore the shape information lost by tf.py_function
                scaled_features.set_shape(train_dataset_raw.element_spec[0].shape)
                
                return scaled_features, label

            train_dataset = train_dataset_raw.map(scale_features, num_parallel_calls=tf.data.AUTOTUNE).repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)
            val_dataset = val_dataset_raw.map(scale_features, num_parallel_calls=tf.data.AUTOTUNE).repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)

            steps_per_epoch = len(train_index) // batch_size
            validation_steps = len(val_index) // batch_size

            neg, pos = np.sum(y_train == 0), np.sum(y_train == 1)
            class_weight = {(1 / neg) * (len(y_train) / 2.0): 0, (1 / pos) * (len(y_train) / 2.0): 1} if neg > 0 and pos > 0 else None

            if best_hps is None:
                hps_path = os.path.join(training_report_dir, 'best_hyperparameters.json')
                if os.path.exists(hps_path):
                    print(f"--- Loading best hyperparameters from {hps_path} ---")
                    with open(hps_path, 'r') as f:
                        hps_values = json.load(f)
                    best_hps = kt.HyperParameters.from_config(hps_values)
                    print("Loaded hyperparameters:")
                    print(best_hps.values)
                    # We still need a tuner object to build the model from hps.
                    # The project_name should be consistent.
                    input_shape = train_dataset.element_spec[0].shape[1:]
                    build_fn = lambda hp: build_model_for_tuning(hp, input_shape)
                    tuner = kt.Hyperband(
                        build_fn,
                        objective='val_accuracy',
                        max_epochs=epochs,
                        factor=3,
                        directory='hyperparameter_tuning',
                        project_name=f'lstm_trader_tuning_fw{future_window}',
                        overwrite=False)
                else:
                    print("--- No saved hyperparameters found. Running search... ---")
                    tuner, best_hps = run_hyperparameter_search(train_dataset, val_dataset, epochs, future_window, steps_per_epoch, validation_steps, is_quick_test)
                    print(f"\n--- Best Hyperparameters Found for fw={future_window} ---")
                    print(best_hps.values)
                    # Save the best HPs to a file for future runs
                    with open(hps_path, 'w') as f:
                        json.dump(best_hps.get_config(), f, indent=4)
                    print(f"Best hyperparameters saved to {hps_path}")

            model = tuner.hypermodel.build(best_hps)
            
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                class_weight=class_weight,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)],
                verbose=1
            )
            
            loss, acc = model.evaluate(val_dataset, verbose=0)
            y_pred_probs = model.predict(val_dataset)
            y_pred = (y_pred_probs > 0.5).astype(int)
            report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
            
            fold_metrics.append({'loss': loss, 'accuracy': acc, 'report': report})
            print(f"Fold {fold + 1} Validation Accuracy: {acc:.4f}")
            
            if fold == n_splits - 1:
                final_model = model
                final_history = history
                final_y_val = y_val
                final_y_pred_probs = y_pred_probs

        print(f"\n--- Walk-Forward Validation Complete for fw={future_window} ---")
        generate_walk_forward_report(fold_metrics, best_hps, training_report_dir)
        
        if final_history:
            print("\nGenerating training history report for the final model...")
            generate_training_report(final_history, final_y_val, final_y_pred_probs, training_report_dir)

        trained_model_path = os.path.join(model_output_dir, "lstm_trader.keras")
        final_model.save(trained_model_path)
        print(f"Final model for fw={future_window} saved to {trained_model_path}")

        print(f"\n[Step 3/4] Running ML-powered backtest...")
        raw_files_path = os.path.join('data', 'raw', '*', '*.parquet')
        raw_data_files = glob.glob(raw_files_path)
        if symbol_limit: raw_data_files = raw_data_files[:symbol_limit]
        
        run_ml_backtest(trained_model_path, raw_data_files, backtest_report_dir, starting_balance=Config.STARTING_BALANCE)

        total_time = time.time() - start_time
        print(f"\n[Step 4/4] Pipeline for fw={future_window} finished in {total_time:.2f} seconds.")
        
        # Read the summary to get the final PnL to return for comparison
        summary_path = os.path.join(backtest_report_dir, 'backtest_summary.txt')
        with open(summary_path, 'r') as f:
            summary_content = f.read()
        
        return summary_content, best_hps.values

    except Exception as e:
        print(f"\n--- Pipeline FAILED for future_window={future_window} ---")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """
    Main function to orchestrate the ML pipeline.
    It now iterates through different future_window values to find the best one.
    """
    # --- Check if preprocessing is needed ---
    processed_path = os.path.join('data', 'processed')
    if not os.path.exists(processed_path) or not os.listdir(processed_path):
        print("--- Processed data not found. Running preprocessing script. ---")
        import subprocess
        subprocess.run(['python3', 'preprocess_data.py'], check=True)
    else:
        print("--- Processed data found. Skipping preprocessing step. ---")

    all_results = {}

    for fw in Config.FUTURE_WINDOW_OPTIONS:
        # Run the entire pipeline for one future_window value
        summary, hps = run_pipeline(is_quick_test=Config.QUICK_TEST, future_window=fw)
        if summary:
            all_results[fw] = {'summary': summary, 'hyperparameters': hps}

    # --- Final Summary ---
    print("\n\n" + "="*80)
    print("--- Hyperparameter Search Across All Future Windows Complete ---")
    print("="*80)
    
    best_fw = None
    best_pnl = -float('inf')

    for fw, result in all_results.items():
        print(f"\n--- Results for future_window = {fw} ---")
        # A simple way to parse PnL from the summary string
        try:
            pnl_line = [line for line in result['summary'].split('\n') if "Total Profit" in line][0]
            pnl_str = pnl_line.split('$')[1].split(' ')[0].replace(',', '')
            pnl = float(pnl_str)
            print(f"  - Final PnL: ${pnl:,.2f}")
            if pnl > best_pnl:
                best_pnl = pnl
                best_fw = fw
        except (IndexError, ValueError) as e:
            print(f"  - Could not parse PnL from summary. Error: {e}")
        print(f"  - Best Hyperparameters: {result['hyperparameters']}")

    print("\n--- Overall Best Configuration ---")
    if best_fw:
        print(f"Best future_window: {best_fw} with a PnL of ${best_pnl:,.2f}")
    else:
        print("No successful runs to determine the best configuration.")

    print("\n--- All Processes Finished ---")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ML trading pipeline.")
    parser.add_argument('mode', nargs='?', default='train', help="Mode to run: 'train' or 'analyze'. Default is 'train'.")
    args = parser.parse_args()

    if args.mode == 'analyze':
        analyze_market()
    elif args.mode == 'train':
        # Set the multiprocessing start method to 'spawn'
        # This is crucial for CUDA compatibility to prevent initialization errors in child processes.
        # It must be called once at the entry point of the script, before any other CUDA or multiprocessing code.
        try:
            mp.set_start_method('spawn', force=True)
            print("Set multiprocessing start method to 'spawn'.")
        except RuntimeError:
            pass # It may have been set already.

        # Check for GPU
        print("--- Checking for GPU ---")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"tf.config.list_physical_devices('GPU') returned: {gpus}")
        
        if gpus:
            try:
                print("GPU detected. Attempting to configure memory growth and mixed precision.")
                # Enable mixed precision for performance
                from tensorflow.keras import mixed_precision
                mixed_precision.set_global_policy('mixed_float16')

                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(f"Successfully configured {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs.")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(f"Error during GPU configuration: {e}")
        else:
            print("No GPU detected. The script will run on CPU.")
            # Also print available CPUs for confirmation
            cpus = tf.config.list_physical_devices('CPU')
            print(f"Available CPUs: {cpus}")

        main()
    else:
        print(f"Unknown mode: {args.mode}. Please use 'train' or 'analyze'.")
