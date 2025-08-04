import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import json
from tabulate import tabulate

# --- Constants ---
DATA_PATH = "data/raw/"
QUICK_TEST_SYMBOL = "LTCUSDT" # Using one symbol for the quick test
QUICK_TEST_DATA_SIZE = 1000 # Using a smaller dataset for the quick test
SEQUENCE_LENGTH = 60 # Number of time steps to look back
MODEL_SAVE_PATH = "lstm_model.h5"
REPORTS_PATH = "reports/"

# --- Helper Functions (ported from main.py) ---

def get_swing_points(df, window=5):
    """
    Identify swing points from a DataFrame.
    """
    highs = df['high']
    lows = df['low']
    
    swing_highs = []
    swing_lows = []

    for i in range(window, len(df) - window):
        is_swing_high = True
        for j in range(1, window + 1):
            if highs.iloc[i] < highs.iloc[i-j] or highs.iloc[i] < highs.iloc[i+j]:
                is_swing_high = False
                break
        if is_swing_high:
            swing_highs.append((df.index[i], highs.iloc[i]))

        is_swing_low = True
        for j in range(1, window + 1):
            if lows.iloc[i] > lows.iloc[i-j] or lows.iloc[i] > lows.iloc[i+j]:
                is_swing_low = False
                break
        if is_swing_low:
            swing_lows.append((df.index[i], lows.iloc[i]))
            
    return swing_highs, swing_lows

def get_trend(swing_highs, swing_lows):
    """
    Determine the trend based on swing points.
    """
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
    """
    Calculate Fibonacci retracement levels.
    """
    price_range = abs(p1 - p2)
    
    if trend == "downtrend":
        golden_zone_start = p1 - (price_range * 0.5)
        golden_zone_end = p1 - (price_range * 0.618)
    else: # Uptrend
        golden_zone_start = p1 + (price_range * 0.5)
        golden_zone_end = p1 + (price_range * 0.618)

    entry_price = (golden_zone_start + golden_zone_end) / 2
    return entry_price


# --- Main Function Definitions ---

def load_all_data(data_path, quick_test=False):
    """
    Scans the data directory and loads all parquet files into a dictionary of DataFrames.
    In quick_test mode, it only loads a small part of one symbol's data.
    """
    print("Loading data...")
    all_data = {}

    if not os.path.exists(data_path):
        print(f"Error: Data directory not found at '{data_path}'")
        return None

    if quick_test:
        print(f"Quick test mode: Loading sample data for {QUICK_TEST_SYMBOL}")
        symbol_path = os.path.join(data_path, QUICK_TEST_SYMBOL)
        parquet_file = os.path.join(symbol_path, 'initial_20000.parquet')
        if os.path.exists(parquet_file):
            df = pd.read_parquet(parquet_file)
            all_data[QUICK_TEST_SYMBOL] = df.head(QUICK_TEST_DATA_SIZE).copy()
            print(f"Loaded {len(all_data[QUICK_TEST_SYMBOL])} rows for {QUICK_TEST_SYMBOL}.")
        else:
            print(f"Warning: Parquet file not found for {QUICK_TEST_SYMBOL} at {parquet_file}")
    else:
        print("Full run mode: Loading all available data.")
        symbols = [s for s in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, s))]
        for symbol in symbols:
            print(f"Loading data for {symbol}...")
            parquet_file = os.path.join(data_path, symbol, 'initial_20000.parquet')
            if os.path.exists(parquet_file):
                df = pd.read_parquet(parquet_file)
                all_data[symbol] = df.copy()
                print(f"Loaded {len(df)} rows for {symbol}.")
            else:
                print(f"Warning: Parquet file not found for {symbol} at {parquet_file}")

    if not all_data:
        print("No data was loaded. Exiting.")
        return None
        
    return all_data


def create_features(df, lookback_candles=100, swing_window=5):
    """
    Adds technical indicators and other features to the DataFrame.
    This is based on the strategy in main.py.
    """
    print("Creating features (this may take a while)...")
    df['trend'] = 'undetermined'
    
    for i in range(lookback_candles, len(df)):
        window_df = df.iloc[i-lookback_candles:i]
        swing_highs, swing_lows = get_swing_points(window_df, swing_window)
        trend = get_trend(swing_highs, swing_lows)
        df.loc[df.index[i], 'trend'] = trend
        
    return df

def generate_signals(df, lookback_candles=100, swing_window=5):
    """
    Applies the Fibonacci trading strategy to generate buy/sell/hold signals.
    (0: Hold, 1: Buy, 2: Sell)
    """
    print("Generating trading signals (this may take a while)...")
    df['signal'] = 0  # 0 for Hold
    
    for i in range(lookback_candles, len(df)):
        current_trend = df.loc[df.index[i], 'trend']
        current_price = df.loc[df.index[i], 'close']
        
        if current_trend in ['uptrend', 'downtrend']:
            window_df = df.iloc[i-lookback_candles:i]
            swing_highs, swing_lows = get_swing_points(window_df, swing_window)
            
            if current_trend == 'uptrend' and len(swing_highs) > 1 and len(swing_lows) > 1:
                last_swing_high = swing_highs[-1][1]
                last_swing_low = swing_lows[-1][1]
                price_range = last_swing_high - last_swing_low
                if price_range == 0: continue
                
                golden_zone_start = last_swing_low + (price_range * 0.5)
                golden_zone_end = last_swing_low + (price_range * 0.618)

                if golden_zone_start <= current_price <= golden_zone_end:
                    df.loc[df.index[i], 'signal'] = 1 # 1 for Buy

            elif current_trend == 'downtrend' and len(swing_highs) > 1 and len(swing_lows) > 1:
                last_swing_high = swing_highs[-1][1]
                last_swing_low = swing_lows[-1][1]
                price_range = last_swing_high - last_swing_low
                if price_range == 0: continue
                
                golden_zone_start = last_swing_high - (price_range * 0.5)
                golden_zone_end = last_swing_high - (price_range * 0.618)

                if golden_zone_end <= current_price <= golden_zone_start:
                    df.loc[df.index[i], 'signal'] = 2 # 2 for Sell
    return df

def preprocess_data(df, features, target, sequence_length):
    """
    Scales the data and creates sequences for the LSTM model.
    """
    print("Preprocessing data...")
    
    df = df[df['trend'] != 'undetermined'].copy()
    if df.empty:
        print("Warning: No data left after removing 'undetermined' trends.")
        return np.array([]), np.array([])

    df['trend_cat'] = df['trend'].astype('category').cat.codes
    
    features_to_scale = features + ['trend_cat']
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(df[features_to_scale])
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_features)):
        X.append(scaled_features[i-sequence_length:i])
        y.append(df[target].iloc[i])
        
    if not X:
        print("Warning: Could not create any sequences from the data.")
        return np.array([]), np.array([])
        
    return np.array(X), np.array(y)


def build_model(input_shape):
    """
    Builds, compiles, and returns the LSTM model architecture.
    """
    print("Building LSTM model...")
    # Placeholder implementation
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(3, activation='softmax') # 3 outputs: Hold, Buy, Sell
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

class BacktestTrade:
    def __init__(self, entry_price, side, timestamp):
        self.entry_price = entry_price
        self.side = side # 'long' or 'short'
        self.status = 'open'
        self.exit_price = None
        self.pnl_pct = None
        self.entry_timestamp = timestamp
        self.exit_timestamp = None

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Trains the LSTM model and saves the best version.
    """
    print("Training model...")
    
    # Callback to save the best model
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, mode='min', verbose=0)
    
    # Callback for early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True, verbose=1)
    
    history = model.fit(X_train, y_train, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_data=(X_val, y_val), 
                        callbacks=[checkpoint, early_stopping],
                        verbose=1)
                        
    print(f"Model training finished. Best model saved to {MODEL_SAVE_PATH}")
    return history.history

def run_backtest(model, df, X_test, y_test, initial_balance=10000, sl_pct=0.02, tp_pct=0.04):
    """
    Performs a more detailed financial backtest.
    """
    print("Running detailed backtest...")
    
    if X_test.shape[0] == 0:
        print("Backtest skipped: No test data available.")
        return {}, [], []

    predictions = model.predict(X_test)
    predicted_signals = np.argmax(predictions, axis=1)
    
    # Align predictions with the original dataframe
    test_df = df.iloc[-len(X_test):].copy()
    test_df['predicted_signal'] = predicted_signals
    
    balance = initial_balance
    equity_curve = [initial_balance]
    trades = []
    current_trade = None

    for i in range(len(test_df)):
        row = test_df.iloc[i]
        signal = row['predicted_signal']
        current_price = row['close']
        timestamp = row.name

        # Handle exits
        if current_trade:
            if current_trade.side == 'long' and current_price <= current_trade.entry_price * (1 - sl_pct):
                current_trade.exit_price = current_price
                current_trade.status = 'sl_hit'
            elif current_trade.side == 'short' and current_price >= current_trade.entry_price * (1 + sl_pct):
                current_trade.exit_price = current_price
                current_trade.status = 'sl_hit'
            elif current_trade.side == 'long' and current_price >= current_trade.entry_price * (1 + tp_pct):
                current_trade.exit_price = current_price
                current_trade.status = 'tp_hit'
            elif current_trade.side == 'short' and current_price <= current_trade.entry_price * (1 - tp_pct):
                current_trade.exit_price = current_price
                current_trade.status = 'tp_hit'

            if current_trade.status != 'open':
                if current_trade.side == 'long':
                    current_trade.pnl_pct = (current_trade.exit_price - current_trade.entry_price) / current_trade.entry_price
                else: # short
                    current_trade.pnl_pct = (current_trade.entry_price - current_trade.exit_price) / current_trade.entry_price
                
                balance *= (1 + current_trade.pnl_pct)
                equity_curve.append(balance)
                current_trade.exit_timestamp = timestamp
                trades.append(current_trade)
                current_trade = None

        # Handle entries
        if not current_trade:
            if signal == 1: # Buy
                current_trade = BacktestTrade(entry_price=current_price, side='long', timestamp=timestamp)
            elif signal == 2: # Sell
                current_trade = BacktestTrade(entry_price=current_price, side='short', timestamp=timestamp)

    # Calculate final metrics
    num_trades = len(trades)
    wins = [t for t in trades if t.pnl_pct > 0]
    win_rate = len(wins) / num_trades if num_trades > 0 else 0
    pnl_values = [t.pnl_pct for t in trades]
    total_pnl_pct = sum(pnl_values)

    equity_series = pd.Series(equity_curve)
    peak = equity_series.expanding(min_periods=1).max()
    drawdown = (equity_series - peak) / peak
    max_drawdown = drawdown.min()

    metrics = {
        'total_trades': num_trades,
        'win_rate_pct': win_rate * 100,
        'total_pnl_pct': total_pnl_pct * 100,
        'max_drawdown_pct': abs(max_drawdown) * 100 if pd.notna(max_drawdown) else 0,
        'final_equity': balance
    }
    
    print("Backtest complete.")
    return metrics, equity_curve, trades

def generate_report(metrics, model_history, equity_curve, report_name='full_run_report'):
    """
    Generates and saves a detailed report with performance metrics and charts.
    """
    print("Generating final report...")
    if not os.path.exists(REPORTS_PATH):
        os.makedirs(REPORTS_PATH)

    # Plot Training History
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(model_history.get('accuracy'), label='Train Accuracy')
    plt.plot(model_history.get('val_accuracy'), label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(model_history.get('loss'), label='Train Loss')
    plt.plot(model_history.get('val_loss'), label='Validation Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.savefig(os.path.join(REPORTS_PATH, f'{report_name}_training_history.png'))
    plt.close()
    print("Saved training history plot.")

    # Plot Equity Curve
    if equity_curve:
        plt.figure(figsize=(10, 6))
        plt.plot(equity_curve)
        plt.title('Equity Curve')
        plt.xlabel('Trade Number')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.savefig(os.path.join(REPORTS_PATH, f'{report_name}_equity_curve.png'))
        plt.close()
        print("Saved equity curve plot.")

    # Save JSON Report
    report = {
        'backtest_metrics': metrics,
        'training_history': {k: [float(v) for v in val] for k, val in model_history.items()}
    }
    report_path = os.path.join(REPORTS_PATH, f'{report_name}.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Full report saved to {report_path}")

    # Print Summary to Console
    print(f"\n--- {report_name.replace('_', ' ').title()} Report Summary ---")
    if metrics:
        headers = list(metrics.keys())
        table = [[f"{v:.2f}" if isinstance(v, float) else v for v in metrics.values()]]
        print(tabulate(table, headers=headers, tablefmt='grid'))
    else:
        print("No backtest metrics to display.")
    print("---------------------------------\n")


def run_quick_test():
    """
    Runs the entire pipeline on a small subset of data to verify functionality.
    Returns True if successful, False otherwise.
    """
    print("--- Starting Quick Test ---")
    
    # 1. Load data, creating a dummy file if necessary
    data_dict = load_all_data(DATA_PATH, quick_test=True)
    if not data_dict:
        print("Quick test: No data found, creating dummy file...")
        if not os.path.exists(os.path.join(DATA_PATH, QUICK_TEST_SYMBOL)):
             os.makedirs(os.path.join(DATA_PATH, QUICK_TEST_SYMBOL))
        dummy_df = pd.DataFrame({
            'open': np.random.uniform(100, 200, size=QUICK_TEST_DATA_SIZE),
            'high': np.random.uniform(100, 200, size=QUICK_TEST_DATA_SIZE),
            'low': np.random.uniform(100, 200, size=QUICK_TEST_DATA_SIZE),
            'close': np.random.uniform(100, 200, size=QUICK_TEST_DATA_SIZE),
            'volume': np.random.uniform(1000, 5000, size=QUICK_TEST_DATA_SIZE),
        }, index=pd.to_datetime(pd.date_range(start='1/1/2022', periods=QUICK_TEST_DATA_SIZE, freq='15min')))
        dummy_file_path = os.path.join(DATA_PATH, QUICK_TEST_SYMBOL, 'initial_20000.parquet')
        dummy_df.to_parquet(dummy_file_path)
        data_dict = load_all_data(DATA_PATH, quick_test=True)
        if not data_dict:
            print("Quick test failed: Could not load or create data.")
            return False

    df = data_dict[QUICK_TEST_SYMBOL]
    
    # 2. Feature Engineering & 3. Signal Generation
    df = create_features(df)
    df = generate_signals(df)

    # 4. Preprocessing
    feature_cols = ['open', 'high', 'low', 'close', 'volume']
    target_col = 'signal'
    X, y = preprocess_data(df, feature_cols, target_col, SEQUENCE_LENGTH)
    
    if X.shape[0] < 10:
        print("Quick test failed: Not enough data for a train/test split.")
        return False
    print(f"Created {X.shape[0]} sequences.")

    # 5. Build and Train Model
    y_unique = np.unique(y)
    if len(y_unique) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=1, batch_size=8)
    
    # 6. Backtest
    backtest_metrics, equity_curve, _ = run_backtest(model, df, X_test, y_test)
    
    # 7. Report
    generate_report(backtest_metrics, history, equity_curve, report_name='quick_test_report')
    
    print("--- Quick Test Finished Successfully ---")
    return True


def run_full_pipeline():
    """
    Runs the entire pipeline on all available data.
    """
    print("--- Starting Full Pipeline ---")
    
    # 1. Load all data
    data_dict = load_all_data(DATA_PATH, quick_test=False)
    if not data_dict or len(data_dict) < 2:
        print("Full pipeline requires at least two symbols to run (one for training, one for testing).")
        print("Please ensure 'data/raw' contains at least two symbol subdirectories with parquet files.")
        return

    # 2. Process all dataframes first
    processed_data = {}
    feature_cols = ['open', 'high', 'low', 'close', 'volume']
    target_col = 'signal'
    for symbol, df in data_dict.items():
        print(f"Processing {symbol}...")
        df_featured = create_features(df)
        df_signaled = generate_signals(df_featured)
        X, y = preprocess_data(df_signaled, feature_cols, target_col, SEQUENCE_LENGTH)
        if X.shape[0] > 0:
            processed_data[symbol] = {'X': X, 'y': y, 'df': df_signaled}

    if len(processed_data) < 2:
        print("Full pipeline requires at least two processable symbols.")
        return
        
    # 3. Split data by symbol for train/test
    symbols = list(processed_data.keys())
    test_symbol = symbols[-1]
    train_symbols = symbols[:-1]

    X_train_list = [processed_data[s]['X'] for s in train_symbols]
    y_train_list = [processed_data[s]['y'] for s in train_symbols]
    
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    test_data = processed_data[test_symbol]
    X_test, y_test, test_df = test_data['X'], test_data['y'], test_data['df']
    
    print(f"Training on {len(train_symbols)} symbol(s) ({X_train.shape[0]} sequences).")
    print(f"Testing on {test_symbol} ({X_test.shape[0]} sequences).")

    # 4. Build and Train Model
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
    
    # 5. Backtest on the hold-out test symbol
    backtest_metrics, equity_curve, _ = run_backtest(model, test_df, X_test, y_test)
    
    # 6. Report
    generate_report(backtest_metrics, history, equity_curve, report_name='full_run_report')
    
    print("--- Full Pipeline Finished Successfully ---")


if __name__ == "__main__":
    print("Starting ML pipeline execution...")
    
    # First, run the quick test to verify the environment and pipeline structure.
    quick_test_passed = run_quick_test()
    
    if quick_test_passed:
        print("\n✅ Quick test passed successfully. Proceeding to the full pipeline.")
        run_full_pipeline()
    else:
        print("\n❌ Quick test failed. Halting execution. Please check the logs for errors.")
