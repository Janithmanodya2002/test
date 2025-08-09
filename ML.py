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
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, TimeSeriesSplit
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
# Set to True to run a quick test on a small subset of data
QUICK_TEST = False
QUICK_TEST_DATA_SIZE = 1000
QUICK_TEST_EPOCHS = 5
QUICK_TEST_SYMBOL_COUNT = 1

# Configuration for the strategy and feature generation (from main.py)
# These should ideally match the configuration used in the original strategy
LOOKBACK_CANDLES = 100
SWING_WINDOW = 5

# Model and Training Parameters
SEQUENCE_LENGTH = 60  # Number of timesteps in each sample
QUICK_TEST_BATCH_SIZE = 32
FULL_RUN_BATCH_SIZE = 128 # Larger batch size for better GPU utilization
EPOCHS = 50
LEARNING_RATE = 0.001

# --- Classes and Functions from main.py (for backtesting and reporting) ---

class TradeResult:
    """A class to hold the results of a single backtested trade."""
    def __init__(self, symbol, side, entry_price, exit_price, entry_timestamp, exit_timestamp, status, pnl_usd, pnl_pct, reason_for_entry, reason_for_exit):
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
    balance_over_time = [starting_balance] + [trade.balance for trade in backtest_trades]
    peak = balance_over_time[0]
    max_drawdown = 0
    for balance in balance_over_time:
        if balance > peak:
            peak = balance
        drawdown = (peak - balance) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return {
        'total_trades': num_trades, 'winning_trades': wins, 'losing_trades': losses,
        'win_rate': win_rate, 'average_win': avg_win, 'average_loss': avg_loss,
        'profit_factor': profit_factor, 'max_drawdown': max_drawdown * 100,
        'net_pnl_usd': net_pnl_usd, 'net_pnl_pct': net_pnl_pct, 'expectancy': expectancy
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

    df = pd.DataFrame([vars(t) for t in backtest_trades])
    df.to_csv(os.path.join(output_dir, 'backtest_trades.csv'), index=False)
    print(f"Backtest trades saved to {os.path.join(output_dir, 'backtest_trades.csv')}")


# --- ML Specific Functions ---

def get_chronological_sample_map_and_labels(symbol_limit=None):
    """
    Scans all processed data files to build a memory-efficient map of samples
    sorted chronologically by their timestamp.
    Returns a map of (file_path, index_in_file) and an array of sorted labels.
    This avoids loading all features into RAM at once.
    """
    path_pattern = os.path.join('data', 'processed', '*', '*.npz')
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
                yield (features_array[i], labels_array[i])

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


def run_hyperparameter_search(train_dataset, val_dataset, epochs, is_quick_test=False):
    """Runs hyperparameter search using Keras Tuner."""
    print("--- Starting Hyperparameter Search ---")
    
    input_shape = train_dataset.element_spec[0].shape[1:]

    def build_model(hp):
        """Builds the model for hyperparameter tuning."""
        model = Sequential()
        model.add(Input(shape=input_shape))

        hp_units_1 = hp.Int('units_1', min_value=64, max_value=256, step=64)
        model.add(Bidirectional(LSTM(units=hp_units_1, return_sequences=True)))
        hp_dropout_1 = hp.Float('dropout_1', min_value=0.1, max_value=0.4, step=0.1)
        model.add(Dropout(hp_dropout_1))

        hp_units_2 = hp.Int('units_2', min_value=32, max_value=128, step=32)
        model.add(Bidirectional(LSTM(units=hp_units_2, return_sequences=True)))

        hp_num_heads = hp.Int('num_heads', min_value=2, max_value=8, step=2)
        model.add(MultiHeadAttention(num_heads=hp_num_heads, key_dim=hp_units_2))
        
        model.add(GlobalAveragePooling1D())

        hp_dropout_2 = hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)
        model.add(Dropout(hp_dropout_2))

        hp_dense_units = hp.Int('dense_units', min_value=32, max_value=128, step=32)
        model.add(Dense(units=hp_dense_units, activation='relu'))
        
        model.add(Dense(1, activation='sigmoid', dtype='float32'))

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])

        model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=epochs,
        factor=3,
        directory='hyperparameter_tuning',
        project_name='lstm_trader_tuning'
    )

    # Clear tuning results from previous runs
    if os.path.exists('hyperparameter_tuning'):
        shutil.rmtree('hyperparameter_tuning')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    
    search_epochs = 3 if is_quick_test else epochs
    print(f"Tuner search running for a max of {search_epochs} epochs.")
    tuner.search(train_dataset, epochs=search_epochs, validation_data=val_dataset, callbacks=[stop_early])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Build the model with the optimal hyperparameters
    model = tuner.hypermodel.build(best_hps)
    return model, best_hps


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

def generate_signals_for_symbol(df):
    """Generates all potential trade signals for a given symbol's DataFrame."""
    potential_trades = []
    np_close = df['close'].to_numpy()

    for i in range(LOOKBACK_CANDLES, len(df) - 1):
        strategy_klines = df.iloc[i - LOOKBACK_CANDLES : i]
        swing_highs, swing_lows = get_swing_points(strategy_klines, window=SWING_WINDOW)
        trend = get_trend(swing_highs, swing_lows)
        
        signal = None
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
        
        if signal:
            potential_trades.append((signal, i))
            
    return potential_trades

def get_features_for_signals(df, potential_trades):
    """Extracts features for a list of potential trades."""
    features_to_predict = []
    for signal, i in potential_trades:
        feature_klines = df.iloc[i - SEQUENCE_LENGTH : i]
        first_candle = feature_klines.iloc[0]
        if first_candle['close'] > 0 and first_candle['volume'] > 0:
            normalized_features = feature_klines[['open', 'high', 'low', 'close', 'volume']].copy()
            for col in ['open', 'high', 'low', 'close']:
                normalized_features[col] = (normalized_features[col] / first_candle['close']) - 1
            normalized_features['volume'] = (normalized_features['volume'] / first_candle['volume']) - 1
            features_to_predict.append(normalized_features.to_numpy())
    return np.array(features_to_predict)

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
                    reason_for_exit=reason
                )
                local_trades.append(trade)
    return local_trades

def backtest_symbol(args):
    """
    Backtesting logic for a single symbol. Designed to be called by a multiprocessing pool.
    """
    # Disable GPU in worker process
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # Set env var BEFORE importing tensorflow in the new process
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from absl import logging
    logging.set_verbosity(logging.ERROR)
    import tensorflow as tf

    filepath, model_path = args
    symbol = filepath.split(os.sep)[-2]
    
    # Each process needs to load the model. This is memory-intensive but necessary for parallelism.
    model = tf.keras.models.load_model(model_path)
    
    local_trades = []
    
    try:
        df = pd.read_parquet(filepath)
        df.columns = df.columns.str.lower()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Convert to numpy for faster access
        np_close = df['close'].to_numpy()
        np_high = df['high'].to_numpy()
        np_low = df['low'].to_numpy()
        
        i = LOOKBACK_CANDLES
        while i < len(df) - 1:
            strategy_klines = df.iloc[i - LOOKBACK_CANDLES : i]
            swing_highs, swing_lows = get_swing_points(strategy_klines, window=SWING_WINDOW)
            trend = get_trend(swing_highs, swing_lows)
            
            signal = None
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

            if signal:
                feature_klines = df.iloc[i - SEQUENCE_LENGTH : i]
                first_candle = feature_klines.iloc[0]
                if first_candle['close'] > 0 and first_candle['volume'] > 0:
                    normalized_features = feature_klines[['open', 'high', 'low', 'close', 'volume']].copy()
                    for col in ['open', 'high', 'low', 'close']:
                        normalized_features[col] = (normalized_features[col] / first_candle['close']) - 1
                    normalized_features['volume'] = (normalized_features['volume'] / first_candle['volume']) - 1
                    feature_array = np.array([normalized_features.to_numpy()])
                    
                # Collect all features for batch prediction
                features_to_predict = []
                potential_trades = []

                if first_candle['close'] > 0 and first_candle['volume'] > 0:
                    normalized_features = feature_klines[['open', 'high', 'low', 'close', 'volume']].copy()
                    for col in ['open', 'high', 'low', 'close']:
                        normalized_features[col] = (normalized_features[col] / first_candle['close']) - 1
                    normalized_features['volume'] = (normalized_features['volume'] / first_candle['volume']) - 1
                    features_to_predict.append(normalized_features.to_numpy())
                    potential_trades.append((signal, i))
            i += 1
        
        # Batch prediction
        if features_to_predict:
            predictions = model.predict(np.array(features_to_predict), verbose=0)
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
                            reason_for_exit=reason
                        )
                        local_trades.append(trade)
            i += 1
    except Exception as e:
        print(f"Error backtesting {symbol}: {e}")
    
    return local_trades

def run_ml_backtest(model_path, data_files, output_dir, starting_balance=10000, risk_per_trade=0.02):
    """Run a backtest using the trained ML model to filter trades in parallel."""
    print("Starting parallel ML-powered backtest simulation...")
    
    num_cores = min(mp.cpu_count(), 4)
    pool_args = [(file, model_path) for file in data_files]
    
    all_trades_lists = []
    # Use tqdm to show progress for the overall backtesting process (per symbol)
    with mp.Pool(processes=num_cores) as pool:
        for result in tqdm(pool.imap_unordered(backtest_symbol, pool_args), total=len(data_files), desc="Overall Backtest Progress"):
            all_trades_lists.append(result)

    # Flatten the list of lists into a single list of trades
    all_trades = [trade for sublist in all_trades_lists for trade in sublist]
    
    # Sort trades by entry timestamp to build a coherent equity curve
    all_trades.sort(key=lambda t: t.entry_timestamp)
    
    # Calculate PnL and equity curve sequentially
    balance = starting_balance
    for trade in all_trades:
        risk_amount = balance * risk_per_trade
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


def run_pipeline(is_quick_test: bool):
    """
    Encapsulates the entire ML pipeline from data loading to backtesting,
    now featuring Walk-Forward Validation.
    """
    start_time = time.time()

    # --- Configuration based on mode ---
    run_mode = "Quick Test" if is_quick_test else "Full Run"
    output_folder = "quick_test_output" if is_quick_test else "full_run_output"
    epochs = QUICK_TEST_EPOCHS if is_quick_test else EPOCHS
    symbol_limit = QUICK_TEST_SYMBOL_COUNT if is_quick_test else None
    batch_size = QUICK_TEST_BATCH_SIZE if is_quick_test else FULL_RUN_BATCH_SIZE
    n_splits = 2 if is_quick_test else 5 # Fewer splits for a quick test
    
    model_output_dir = os.path.join(output_folder, "model")
    backtest_report_dir = os.path.join(output_folder, "backtest_report")
    training_report_dir = os.path.join(output_folder, "training_report")

    # --- Setup Directories ---
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    for d in [model_output_dir, backtest_report_dir, training_report_dir]:
        os.makedirs(d)

    print(f"--- Starting {run_mode} with Walk-Forward Validation ({n_splits} splits) ---")

    try:
        # Step 1: Build a memory-efficient map of the data samples
        print(f"\n[Step 1/4] Scanning and sorting all data samples...")
        sample_map, sorted_labels = get_chronological_sample_map_and_labels(symbol_limit)
        
        if len(sample_map) < 200: # Need enough data for multiple splits
            print("Error: Not enough data for walk-forward validation.")
            return False

        # Get one sample to determine the shape and type for tf.data.Dataset.
        # This is needed for the generator's output signature.
        temp_features, temp_label = next(data_generator(sample_map, [0]))
        output_signature = (
            tf.TensorSpec(shape=temp_features.shape, dtype=temp_features.dtype),
            tf.TensorSpec(shape=(), dtype=temp_label.dtype)
        )
        # Clean up temporary variables
        del temp_features
        del temp_label
        
        # --- Walk-Forward Validation Setup ---
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics = []
        best_hps = None
        
        final_model, final_history, final_y_val, final_y_pred_probs = None, None, None, None

        print(f"\n[Step 2/4] Starting Walk-Forward Validation...")
        # We split a dummy array of indices, not the actual data X, to save memory
        for fold, (train_index, val_index) in enumerate(tscv.split(np.arange(len(sample_map)))):
            print(f"\n--- Fold {fold + 1}/{n_splits} ---")
            
            # We don't load all X and y into memory. We get the labels for the fold from the pre-scanned array.
            y_train, y_val = sorted_labels[train_index], sorted_labels[val_index]
            
            print(f"Train size: {len(train_index)}, Validation size: {len(val_index)}")
            
            # --- Create tf.data.Dataset for the current fold using the memory-efficient generator ---
            train_dataset = tf.data.Dataset.from_generator(
                lambda: data_generator(sample_map, train_index),
                output_signature=output_signature
            ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            val_dataset = tf.data.Dataset.from_generator(
                lambda: data_generator(sample_map, val_index),
                output_signature=output_signature
            ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

            # Calculate class weights for the current training fold using the pre-scanned labels
            neg, pos = np.sum(y_train == 0), np.sum(y_train == 1)
            class_weight = {(1 / neg) * (len(y_train) / 2.0): 0, (1 / pos) * (len(y_train) / 2.0): 1} if neg > 0 and pos > 0 else None

            # --- Hyperparameter Search (only on the first, smallest fold) ---
            if best_hps is None:
                print("\nRunning hyperparameter search on the first fold...")
                tuner = run_hyperparameter_search(train_dataset, val_dataset, epochs, is_quick_test)
                best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                
                print(f"\n--- Best Hyperparameters Found ---")
                print(best_hps.values)
                hps_path = os.path.join(training_report_dir, 'best_hyperparameters.json')
                with open(hps_path, 'w') as f: json.dump(best_hps.values, f, indent=4)
                print(f"Best hyperparameters saved to {hps_path}")

            # --- Build and Train Model for the current fold ---
            print("\nBuilding and training model for the current fold...")
            # Build a new model with the best HPs for each fold to avoid data leakage
            model = tuner.hypermodel.build(best_hps)
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                class_weight=class_weight,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)],
                verbose=1
            )
            
            # --- Evaluate and store metrics ---
            loss, acc = model.evaluate(val_dataset, verbose=0)
            y_pred_probs = model.predict(val_dataset)
            y_pred = (y_pred_probs > 0.5).astype(int)
            report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
            
            fold_metrics.append({'loss': loss, 'accuracy': acc, 'report': report})
            print(f"Fold {fold + 1} Validation Accuracy: {acc:.4f}")
            
            # The model from the final fold is used for the final backtest
            if fold == n_splits - 1:
                final_model = model
                final_history = history
                final_y_val = y_val
                final_y_pred_probs = y_pred_probs

        # --- Aggregating and Reporting Walk-Forward Results ---
        print(f"\n--- Walk-Forward Validation Complete ---")
        generate_walk_forward_report(fold_metrics, best_hps, training_report_dir)
        
        # --- Generate report for the final model's training history ---
        if final_history:
            print("\nGenerating training history report for the final model...")
            generate_training_report(final_history, final_y_val, final_y_pred_probs, training_report_dir)

        # Save the final model
        trained_model_path = os.path.join(model_output_dir, "lstm_trader.keras")
        final_model.save(trained_model_path)
        print(f"Final model trained on the last fold saved to {trained_model_path}")

        # Step 3: Run ML-powered backtest
        print(f"\n[Step 3/4] Running ML-powered backtest on the final model...")
        raw_files_path = os.path.join('data', 'raw', '*', '*.parquet')
        raw_data_files = glob.glob(raw_files_path)
        if symbol_limit:
            raw_data_files = raw_data_files[:symbol_limit]
        
        run_ml_backtest(trained_model_path, raw_data_files, backtest_report_dir, starting_balance=10000)

        # Step 4: Finalization
        print(f"\n[Step 4/4] {run_mode} finished.")
        total_time = time.time() - start_time
        print(f"Total execution time for {run_mode}: {total_time:.2f} seconds.")

    except Exception as e:
        print(f"\n--- {run_mode} FAILED ---")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n--- {run_mode} Finished Successfully ---")
    return True


def main():
    """Main function to orchestrate the quick test and full run."""
    # --- Check if preprocessing is needed ---
    processed_path = os.path.join('data', 'processed')
    # If the processed data directory does not exist or is empty, run the preprocessing script
    if not os.path.exists(processed_path) or not os.listdir(processed_path):
        print("--- Processed data not found. Running preprocessing script. ---")
        import subprocess
        # This will trigger the download and processing logic within preprocess_data.py
        subprocess.run(['python3', 'preprocess_data.py'], check=True)
    else:
        print("--- Processed data found. Skipping preprocessing step. ---")

    # We need to declare the global here before any access.
    global QUICK_TEST
    
    # Hold the initial value of the flag.
    run_quick_test_first = QUICK_TEST

    if run_quick_test_first:
        print("--- Running Quick Test ---")
        # Ensure the global flag is set correctly for the pipeline
        QUICK_TEST = True
        success = run_pipeline(is_quick_test=True)

        if not success:
            print("\n--- Quick Test FAILED. Halting execution. ---")
            return

        print("\n--- Quick Test PASSED. Proceeding to Full Run. ---")
        
        quick_test_folder = "quick_test_output"
        if os.path.exists(quick_test_folder):
            print(f"Cleaning up {quick_test_folder} directory...")
            shutil.rmtree(quick_test_folder)

    # Proceed to the full run
    print("\n--- Preparing for Full Run ---")
    # Ensure the global flag is set to False for the full run
    QUICK_TEST = False
    run_pipeline(is_quick_test=False)

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
