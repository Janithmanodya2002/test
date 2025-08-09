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
import pandas_ta as ta
from binance.client import Client
import keys

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

def load_processed_data(symbol_limit=None):
    """Load all preprocessed feature files from the data/processed directory."""
    path_pattern = os.path.join('data', 'processed', '*', '*.npz')
    files = glob.glob(path_pattern)
    if not files:
        raise FileNotFoundError(f"No processed feature files found at {path_pattern}. Run preprocess_data.py first.")

    if symbol_limit:
        files = files[:symbol_limit]

    all_features = []
    all_labels = []
    for file in tqdm(files, desc="Loading Processed Features"):
        data = np.load(file)
        all_features.append(data['features'])
        all_labels.append(data['labels'])
    
    if not all_features:
        return np.array([]), np.array([])

    return np.concatenate(all_features), np.concatenate(all_labels)

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
        # Step 1 & 2: Load Preprocessed Data
        print(f"\n[Step 1-2/5] Loading preprocessed data for {run_mode}...")
        X, y = load_processed_data(symbol_limit)
        
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

        # Step 3: Build and Train the Model
        print(f"\n[Step 3/5] Training model for {run_mode}... (Epochs: {epochs})")
        # The training function now takes datasets instead of numpy arrays
        trained_model_path = create_and_train_model(train_dataset, val_dataset, test_dataset, model_output_dir, training_report_dir, epochs, class_weight)
        print(f"Model for {run_mode} saved to {trained_model_path}")

        # Step 4: Run ML-powered backtest
        print(f"\n[Step 4/5] Running ML-powered backtest for {run_mode}...")
        # Load raw data file paths for the backtest simulation
        raw_files_path = os.path.join('data', 'raw', '*', '*.parquet')
        raw_data_files = glob.glob(raw_files_path)
        if symbol_limit:
            raw_data_files = raw_data_files[:symbol_limit]
        
        run_ml_backtest(trained_model_path, raw_data_files, backtest_report_dir, starting_balance=10000)

        # Step 5: Finalization
        print(f"\n[Step 5/5] {run_mode} finished.")
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
    # Check if preprocessing is needed
    processed_path = os.path.join('data', 'processed')
    if not os.path.exists(processed_path) or not os.listdir(processed_path):
        print("--- Processed data not found. Running preprocessing script. ---")
        import subprocess
        subprocess.run(['python3', 'preprocess_data.py'], check=True)

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
    else:
        print(f"Unknown mode: {args.mode}. Please use 'train' or 'analyze'.")
