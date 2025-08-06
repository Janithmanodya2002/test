import os
# Set TensorFlow logging level to suppress all but error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import tensorflow as tf
from absl import logging
logging.set_verbosity(logging.ERROR)
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import glob
import shutil
import multiprocessing as mp
import time
import json
from tabulate import tabulate
import datetime
import keys
from binance.client import Client
from pandas_ta import rsi, macd, bbands
from scipy.signal import find_peaks

# --- Configuration ---
# Set to True to run a quick test on a small subset of data
QUICK_TEST = False
QUICK_TEST_DATA_SIZE = 1000
QUICK_TEST_EPOCHS = 5
QUICK_TEST_SYMBOL_COUNT = 1

# --- New Enhanced Configuration ---
# Timeframes and lookbacks for signal detection
TIMEFTRAMES = ['15m', '1h', '4h']
LOOKBACK_VALUES = [5, 10, 15] # Different lookback windows for swing detection
DOWNLOAD_LOOKBACK_DAYS = 90 # How many days of data to download for each symbol

# Configuration for the strategy and feature generation (from main.py)
# These should ideally match the configuration used in the original strategy
LOOKBACK_CANDLES = 100
SWING_WINDOW = 5 # This will be replaced by LOOKBACK_VALUES but kept for compatibility for now

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

def detect_swing_points_and_trend(df, lookback_window):
    """
    Identifies swing points and market trend using technical indicators and peak detection.
    
    Returns:
        - df (pd.DataFrame): DataFrame augmented with indicators.
        - swing_points (list): List of classified swing points (HH, HL, LH, LL).
        - trend (str): The overall market trend ('Uptrend', 'Downtrend', 'Ranging').
    """
    # 1. Calculate Technical Indicators
    df['RSI_14'] = rsi(df['close'])
    macd_df = macd(df['close'])
    df = df.join(macd_df)
    bbands_df = bbands(df['close'])
    df = df.join(bbands_df)
    df.dropna(inplace=True)

    # 2. Find Peaks and Troughs using scipy.signal.find_peaks
    # The 'prominence' parameter is crucial for filtering out minor peaks/troughs.
    # A simple heuristic for prominence could be a fraction of the price range.
    price_range = df['high'].max() - df['low'].min()
    prominence_filter = price_range * 0.05 # Require a 5% price change for a peak to be significant

    high_peaks, _ = find_peaks(df['high'], prominence=prominence_filter, width=lookback_window)
    low_troughs, _ = find_peaks(-df['low'], prominence=prominence_filter, width=lookback_window)

    # 3. Refine and Classify Swing Points
    swing_points = []
    
    # Combine and sort all detected points by index
    all_points = sorted(
        [(i, 'high', df['high'].iloc[i]) for i in high_peaks] + 
        [(i, 'low', df['low'].iloc[i]) for i in low_troughs],
        key=lambda x: x[0]
    )

    # Filter out consecutive highs or lows
    if not all_points:
        return df, [], "undetermined"
        
    filtered_points = [all_points[0]]
    for i in range(1, len(all_points)):
        if all_points[i][1] != filtered_points[-1][1]:
            filtered_points.append(all_points[i])

    last_high = None
    last_low = None
    
    for i, type, price in filtered_points:
        point_info = {'index': i, 'type': '', 'price': price}
        
        if type == 'high':
            if last_high and price > last_high['price']:
                point_info['type'] = 'HH' # Higher High
            else:
                point_info['type'] = 'LH' # Lower High or initial high
            last_high = point_info
        else: # low
            if last_low and price > last_low['price']:
                point_info['type'] = 'HL' # Higher Low
            else:
                point_info['type'] = 'LL' # Lower Low or initial low
            last_low = point_info
            
        swing_points.append(point_info)

    # 4. Determine Market Trend
    trend = "Ranging"
    hh_count = sum(1 for p in swing_points[-4:] if p['type'] == 'HH')
    hl_count = sum(1 for p in swing_points[-4:] if p['type'] == 'HL')
    ll_count = sum(1 for p in swing_points[-4:] if p['type'] == 'LL')
    lh_count = sum(1 for p in swing_points[-4:] if p['type'] == 'LH')

    # Basic trend detection based on the last few swing points
    if hh_count >= 1 and hl_count >= 1:
        trend = "Uptrend"
    elif ll_count >= 1 and lh_count >= 1:
        trend = "Downtrend"
        
    return df, swing_points, trend

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

def get_symbols(filename="symbols.csv"):
    """Reads symbols from a CSV file."""
    try:
        df = pd.read_csv(filename, header=None)
        return df[0].tolist()
    except FileNotFoundError:
        print(f"Error: {filename} not found. Please create it with a list of symbols.")
        return []

def download_kline_data(symbol, timeframe, lookback_days):
    """Downloads historical k-line data from Binance."""
    client = Client(keys.BINANCE_API_KEY, keys.BINANCE_API_SECRET)
    start_str = f"{lookback_days} days ago UTC"
    
    print(f"Downloading {timeframe} klines for {symbol} for the last {lookback_days} days...")
    
    try:
        klines = client.get_historical_klines(symbol, timeframe, start_str)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        
        # Convert columns to appropriate types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
            
        return df
    except Exception as e:
        print(f"Error downloading data for {symbol} on {timeframe}: {e}")
        return pd.DataFrame()


def build_training_data(symbol_limit=None):
    """
    Builds a complete training dataset by downloading data, detecting signals,
    and generating features and labels.

    This is the core of the new data pipeline. It iterates through symbols, 
    timeframes, and lookback values to generate a diverse and robust dataset.
    The process is as follows:
    1. Download data for a symbol/timeframe.
    2. Analyze data with various lookback windows to find trends and swing points.
    3. Identify potential trade setups based on a defined strategy (e.g., entry on a
       pullback in a confirmed trend).
    4. Label the setup as a "win" or "loss" by looking into the future data.
    5. Generate a feature vector for the model, including OHLCV and technical indicators.
    6. Repeat for all combinations and aggregate the data.
    """
    all_features = []
    all_labels = []
    
    symbols = get_symbols()
    if symbol_limit:
        symbols = symbols[:symbol_limit]

    # Define the columns that will be used as features for the model
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 'BBB_5_2.0', 'BBP_5_2.0'
    ]

    for symbol in tqdm(symbols, desc="Building Training Data"):
        for timeframe in TIMEFTRAMES:
            df = download_kline_data(symbol, timeframe, DOWNLOAD_LOOKBACK_DAYS)
            if df.empty:
                continue

            for lookback in LOOKBACK_VALUES:
                df_with_indicators, swing_points, trend = detect_swing_points_and_trend(df.copy(), lookback)

                # This is where you'd define your trading logic to generate signals
                # For this example, let's create a simple signal:
                # "In an uptrend, after a Higher Low (HL), look for a long entry."
                
                for i in range(1, len(swing_points)):
                    # Check for a Higher Low in an uptrend
                    if trend == "Uptrend" and swing_points[i]['type'] == 'HL':
                        signal_idx = swing_points[i]['index']
                        
                        # Ensure we have enough data for a feature sequence
                        if signal_idx < SEQUENCE_LENGTH:
                            continue

                        # Define SL/TP for labeling
                        sl = swing_points[i]['price'] * 0.98 # 2% stop loss
                        tp = swing_points[i]['price'] * 1.04 # 4% take profit

                        # Look ahead for outcome
                        label = None
                        for j in range(signal_idx + 1, len(df_with_indicators)):
                            future_high = df_with_indicators['high'].iloc[j]
                            future_low = df_with_indicators['low'].iloc[j]
                            if future_high >= tp:
                                label = 1 # Win
                                break
                            if future_low <= sl:
                                label = 0 # Loss
                                break
                        
                        # If a win/loss was determined, create the feature set
                        if label is not None:
                            feature_df = df_with_indicators.iloc[signal_idx - SEQUENCE_LENGTH : signal_idx]
                            
                            # Normalize features
                            first_candle = feature_df.iloc[0]
                            normalized_df = feature_df[feature_columns].copy()
                            for col in ['open', 'high', 'low', 'close', 'volume'] + [c for c in feature_columns if 'BB' in c]:
                                if first_candle[col] > 0:
                                    normalized_df[col] = (normalized_df[col] / first_candle[col]) - 1
                            
                            # For other indicators like RSI, MACD, normalization might differ
                            # For simplicity, we'll just scale them
                            for col in ['RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']:
                                if col in normalized_df.columns:
                                    normalized_df[col] = normalized_df[col] / 100
                            
                            normalized_df.fillna(0, inplace=True) # Handle any potential NaNs after normalization

                            all_features.append(normalized_df.to_numpy())
                            all_labels.append(np.array([label], dtype=np.float32))

    if not all_features:
        return np.array([]), np.array([])
        
    return np.array(all_features), np.concatenate(all_labels)

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


def create_and_train_model(train_dataset, val_dataset, test_dataset, output_dir, training_report_dir, epochs, class_weight=None):
    """Build, compile, and train the LSTM model using tf.data.Dataset."""
    print("Building LSTM model...")
    
    # Define the input shape from the dataset's element spec
    input_shape = train_dataset.element_spec[0].shape[1:]
    
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid', dtype='float32')
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()

    print("Training model...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        verbose=1,
        class_weight=class_weight
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

# --- Placeholder for main.py compatibility ---
def process_and_save_kline_data(klines, symbol):
    """
    This function is a placeholder to maintain compatibility with main.py.
    The new pipeline builds data directly and doesn't save intermediate files.
    """
    print(f"Note: Data for {symbol} received. In the new pipeline, data processing is integrated.")
    pass

# Note: The following functions are now obsolete as their logic is handled
# by the new `build_training_data` and `detect_swing_points_and_trend` functions.
# - generate_signals_for_symbol
# - get_features_for_signals
# - simulate_trades
# - backtest_symbol

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
    Encapsulates the entire ML pipeline from data loading to backtesting.
    Returns True on success, False on failure.
    """
    start_time = time.time()

    # --- Configuration based on mode ---
    run_mode = "Quick Test" if is_quick_test else "Full Run"
    output_folder = "quick_test_output" if is_quick_test else "full_run_output"
    epochs = QUICK_TEST_EPOCHS if is_quick_test else EPOCHS
    symbol_limit = QUICK_TEST_SYMBOL_COUNT if is_quick_test else None
    batch_size = QUICK_TEST_BATCH_SIZE if is_quick_test else FULL_RUN_BATCH_SIZE
    
    model_output_dir = os.path.join(output_folder, "model")
    backtest_report_dir = os.path.join(output_folder, "backtest_report")
    training_report_dir = os.path.join(output_folder, "training_report")

    # --- Setup Directories ---
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    for d in [model_output_dir, backtest_report_dir, training_report_dir]:
        os.makedirs(d)

    print(f"--- Starting {run_mode} ---")

    try:
        # Step 1: Build Training Data
        print(f"\n[Step 1/4] Building training data for {run_mode}...")
        X, y = build_training_data(symbol_limit)
        
        dataset_size = len(X)
        print(f"Found {dataset_size} samples.")
        
        if dataset_size < 20:
            print("Error: Not enough data generated to proceed with training.")
            return False

        # Calculate class weights to handle imbalance
        neg = np.sum(y == 0)
        pos = np.sum(y == 1)
        total = neg + pos
        
        class_weight = None
        if neg > 0 and pos > 0:
            weight_for_0 = (1 / neg) * (total / 2.0)
            weight_for_1 = (1 / pos) * (total / 2.0)
            class_weight = {0: weight_for_0, 1: weight_for_1}
            print(f"Calculated class weights for imbalanced data: {class_weight}")

        # Create a tf.data.Dataset from the in-memory numpy arrays
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(buffer_size=dataset_size).cache()

        # Splitting the dataset
        train_size = int(0.7 * dataset_size)
        val_size = int(0.2 * dataset_size)
        
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size).take(val_size)
        test_dataset = dataset.skip(train_size + val_size)

        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        print(f"Dataset split: {train_size} train, {val_size} validation, {dataset_size - train_size - val_size} test samples.")

        # Step 2: Build and Train the Model
        print(f"\n[Step 2/4] Training model for {run_mode}... (Epochs: {epochs})")
        # The training function now takes datasets instead of numpy arrays
        trained_model_path = create_and_train_model(train_dataset, val_dataset, test_dataset, model_output_dir, training_report_dir, epochs, class_weight)
        print(f"Model for {run_mode} saved to {trained_model_path}")

        # Step 3: Run ML-powered backtest
        print(f"\n[Step 3/4] Running ML-powered backtest for {run_mode}...")
        # Since we download data directly, we can pass symbol names to the backtest
        symbols = get_symbols()
        if symbol_limit:
            symbols = symbols[:symbol_limit]
        
        # The backtest function will need to be adapted to this new data flow
        # For now, we'll comment this out as the user requested not to run/test.
        # run_ml_backtest(trained_model_path, symbols, backtest_report_dir, starting_balance=10000)
        print("Skipping backtest as per user instruction.")

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
    # The script is now self-contained for data downloading and processing.
    # The check for preprocessed data is no longer needed.

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
    # Set the multiprocessing start method to 'spawn'
    # This is crucial for CUDA compatibility to prevent initialization errors in child processes.
    # It must be called once at the entry point of the script, before any other CUDA or multiprocessing code.
    try:
        mp.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        pass # It may have been set already.

    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable mixed precision for performance
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')

            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs detected and configured.")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        print("No GPU detected. The script will run on CPU.")

    main()
