# ml.py

import os
import glob
import time
import json
import datetime
import logging
import warnings
import shutil
import random
import sys
import io
from multiprocessing import Pool, cpu_count

import pyarrow
import fastparquet
import pandas as pd
import pandas_ta as ta
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tabulate import tabulate
import matplotlib.pyplot as plt

# --- Reproducibility ---
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds()

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# These will be loaded from configuration.csv, but we set defaults here.
CONFIG = {
    "quick_test": True,
    "quick_test_symbols": 1,
    "quick_test_data_fraction": 0.1,
    "quick_test_epochs": 5,
    "full_test_epochs": 100,
    "sequence_length": 100,
    "batch_size": 128,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "model_path": "trader_model.pth",
    "report_dir": "ml_report",
    "data_dir": "data",
    "leverage": 10,
    "risk_per_trade": 1,
    "starting_balance": 10000,
    "lookback_candles": 100,
    "swing_window": 5,
    "debug_no_ml": False,
    "run_overfit_test_only": True,
}

# --- GPU Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Functions copied from main.py for backtesting and reporting ---

class TradeResult:
    def __init__(self, symbol, side, entry_price, exit_price, entry_timestamp, exit_timestamp, status, pnl_usd, pnl_pct, drawdown, reason_for_entry, reason_for_exit, fib_levels):
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.entry_timestamp = entry_timestamp
        self.exit_timestamp = exit_timestamp
        self.status = status
        self.pnl_usd = pnl_usd
        self.pnl_pct = pnl_pct
        self.drawdown = drawdown
        self.reason_for_entry = reason_for_entry
        self.reason_for_exit = reason_for_exit
        self.fib_levels = fib_levels
        self.balance = 0 # Will be populated during backtest

def get_swing_points(klines, window=5):
    """
    Identify swing points from kline data.
    """
    highs = np.array([float(k[2]) for k in klines])
    lows = np.array([float(k[3]) for k in klines])
    
    swing_highs = []
    swing_lows = []

    # This ensures we have enough data points on either side of the candle to be a swing point
    for i in range(window, len(highs) - window):
        is_swing_high = all(highs[i] >= highs[i-j] and highs[i] >= highs[i+j] for j in range(1, window + 1))
        if is_swing_high:
            swing_highs.append((klines[i][0], highs[i]))

        is_swing_low = all(lows[i] <= lows[i-j] and lows[i] <= lows[i+j] for j in range(1, window + 1))
        if is_swing_low:
            swing_lows.append((klines[i][0], lows[i]))
            
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
        # For downtrend, p1 is swing high, p2 is swing low
        entry_price = p1 - (price_range * 0.618)
    else: # Uptrend
        # For uptrend, p1 is swing low, p2 is swing high
        entry_price = p1 + (price_range * 0.618)
    
    return entry_price

def calculate_quantity(balance, risk_per_trade, sl_price, entry_price, leverage):
    """
    Calculate the order quantity for backtesting. Simplified from main.py.
    """
    risk_amount = balance * (risk_per_trade / 100)
    sl_percentage = abs(entry_price - sl_price) / entry_price
    if sl_percentage == 0:
        return 0
    
    position_size = risk_amount / sl_percentage
    quantity = position_size / entry_price
    
    # In a real scenario, you'd adjust for lot size, but we'll ignore for this backtest
    return quantity

def calculate_performance_metrics(backtest_trades, starting_balance):
    """
    Calculate performance metrics from a list of trades.
    """
    num_trades = len(backtest_trades)
    if num_trades == 0:
        return {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'win_rate': 0,
            'average_win': 0, 'average_loss': 0, 'profit_factor': 0, 'max_drawdown': 0,
            'net_pnl_usd': 0, 'net_pnl_pct': 0, 'expectancy': 0
        }

    wins = sum(1 for trade in backtest_trades if trade.status == 'win')
    losses = num_trades - wins
    win_rate = (wins / num_trades) * 100 if num_trades > 0 else 0
    
    total_win_amount = sum(trade.pnl_usd for trade in backtest_trades if trade.status == 'win')
    total_loss_amount = sum(trade.pnl_usd for trade in backtest_trades if trade.status == 'loss')
    
    avg_win = total_win_amount / wins if wins > 0 else 0
    avg_loss = total_loss_amount / losses if losses > 0 else 0
    
    profit_factor = total_win_amount / abs(total_loss_amount) if total_loss_amount != 0 else float('inf')
    
    net_pnl_usd = total_win_amount + total_loss_amount
    net_pnl_pct = (net_pnl_usd / starting_balance) * 100
    
    expectancy = (win_rate/100 * avg_win) - ((losses/num_trades) * abs(avg_loss)) if num_trades > 0 else 0

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

def analyze_strategy_behavior(backtest_trades):
    """
    Analyze the performance of the strategy based on different conditions.
    """
    # Performance by hour
    hourly_performance = {}
    for trade in backtest_trades:
        hour = datetime.datetime.fromtimestamp(trade.entry_timestamp/1000).hour
        if hour not in hourly_performance:
            hourly_performance[hour] = {'wins': 0, 'losses': 0, 'total': 0}
        hourly_performance[hour]['total'] += 1
        if trade.status == 'win':
            hourly_performance[hour]['wins'] += 1
        else:
            hourly_performance[hour]['losses'] += 1
            
    # Performance by trend
    trend_performance = {'uptrend': {'wins': 0, 'losses': 0, 'total': 0}, 'downtrend': {'wins': 0, 'losses': 0, 'total': 0}}
    for trade in backtest_trades:
        if "uptrend" in trade.reason_for_entry:
            trend_performance['uptrend']['total'] += 1
            if trade.status == 'win':
                trend_performance['uptrend']['wins'] += 1
            else:
                trend_performance['uptrend']['losses'] += 1
        elif "downtrend" in trade.reason_for_entry:
            trend_performance['downtrend']['total'] += 1
            if trade.status == 'win':
                trend_performance['downtrend']['wins'] += 1
            else:
                trend_performance['downtrend']['losses'] += 1

    return {
        'hourly_performance': hourly_performance,
        'trend_performance': trend_performance
    }

def generate_equity_curve(backtest_trades, starting_balance, save_path):
    balance_over_time = [starting_balance] + [trade.balance for trade in backtest_trades]
    plt.figure(figsize=(10, 6))
    plt.plot(balance_over_time)
    plt.title('Equity Curve')
    plt.xlabel('Trade Number')
    plt.ylabel('Balance (USD)')
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def generate_drawdown_curve(backtest_trades, starting_balance, save_path):
    balance_over_time = [starting_balance] + [trade.balance for trade in backtest_trades]
    peak = balance_over_time[0]
    drawdowns = []
    for balance in balance_over_time:
        if balance > peak:
            peak = balance
        drawdown = (peak - balance) / peak
        drawdowns.append(drawdown * 100)
        
    plt.figure(figsize=(10, 6))
    plt.plot(drawdowns, color='red')
    plt.title('Drawdown Curve')
    plt.xlabel('Trade Number')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def generate_win_loss_distribution(backtest_trades, save_path):
    if not backtest_trades: return
    wins = sum(1 for trade in backtest_trades if trade.status == 'win')
    losses = len(backtest_trades) - wins
    labels = 'Wins', 'Losses'
    sizes = [wins, losses]
    colors = ['#26A69A', '#EF5350']
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Win/Loss Distribution')
    plt.axis('equal')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def generate_returns_histogram(backtest_trades, save_path):
    if not backtest_trades: return
    returns = [trade.pnl_pct for trade in backtest_trades]
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=50, color='blue', alpha=0.7)
    plt.title('Trade Returns Histogram')
    plt.xlabel('Return (%)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def generate_csv_report(backtest_trades, save_path):
    if not backtest_trades: return
    df = pd.DataFrame([vars(t) for t in backtest_trades])
    df.to_csv(save_path, index=False)
    logger.info(f"Backtest trades saved to {save_path}")

def generate_json_report(backtest_trades, metrics, strategy_analysis, save_path):
    if not backtest_trades: return
    report = {
        'metrics': metrics,
        'strategy_analysis': strategy_analysis,
        'trades': [vars(t) for t in backtest_trades]
    }
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=4)
    logger.info(f"Backtest report saved to {save_path}")

def generate_summary_report(metrics, strategy_analysis, config, starting_balance, save_path):
    headers = ["Metric", "Value"]
    table = [
        ["Starting Balance", f"${starting_balance:,.2f}"],
        ["Ending Balance", f"${metrics.get('net_pnl_usd', 0) + starting_balance:,.2f}"],
        ["Total Profit", f"${metrics.get('net_pnl_usd', 0):,.2f}"],
        ["Total Trades", metrics.get('total_trades', 0)],
        ["Winning Trades", metrics.get('winning_trades', 0)],
        ["Losing Trades", metrics.get('losing_trades', 0)],
        ["Win Rate", f"{metrics.get('win_rate', 0):.2f}%"],
        ["Profit Factor", f"{metrics.get('profit_factor', 0):.2f}"],
        ["Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%"]
    ]
    
    report = "Backtesting Summary\n"
    report += "===================\n\n"
    report += tabulate(table, headers=headers, tablefmt="grid")
    
    # Print the report to the console
    logger.info("\n" + report)
    
    with open(save_path, "w") as f:
        f.write(report)
    logger.info(f"Human-readable summary saved to {save_path}")

def generate_backtest_report(backtest_trades, config, starting_balance, report_dir):
    """
    Generate a detailed report from the backtest results.
    """
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    if not backtest_trades:
        logger.warning("No trades to generate a report for.")
        return

    metrics = calculate_performance_metrics(backtest_trades, starting_balance)
    strategy_analysis = analyze_strategy_behavior(backtest_trades)
    
    generate_summary_report(metrics, strategy_analysis, config, starting_balance, os.path.join(report_dir, "summary.txt"))
    generate_equity_curve(backtest_trades, starting_balance, os.path.join(report_dir, "equity_curve.png"))
    generate_drawdown_curve(backtest_trades, starting_balance, os.path.join(report_dir, "drawdown_curve.png"))
    generate_win_loss_distribution(backtest_trades, os.path.join(report_dir, "win_loss_distribution.png"))
    generate_returns_histogram(backtest_trades, os.path.join(report_dir, "returns_histogram.png"))
    generate_csv_report(backtest_trades, os.path.join(report_dir, "trades.csv"))
    generate_json_report(backtest_trades, metrics, strategy_analysis, os.path.join(report_dir, "report.json"))
    
    logger.info(f"Backtest report generated in directory: {report_dir}")


# --- ML Data Preparation ---

def validate_data(samples):
    """Logs statistics about the dataset for validation."""
    if not samples:
        logger.warning("No samples to validate.")
        return

    labels = [s[1] for s in samples]
    wins = sum(labels)
    losses = len(labels) - wins
    win_pct = (wins / len(labels)) * 100
    
    logger.info("--- Data Validation ---")
    logger.info(f"Total Samples: {len(samples)}")
    logger.info(f"Label Balance: {wins} Wins ({win_pct:.2f}%) vs. {losses} Losses ({100-win_pct:.2f}%)")
    logger.info(f"First 20 labels: {labels[:20]}")
    
    logger.info("Spot-checking 5 random samples (last 5 closing prices and label):")
    for i in range(5):
        idx = random.randint(0, len(samples) - 1)
        features, label = samples[idx]
        # Get the last 5 closing prices (index 3 of the features)
        last_5_closes = features[-5:, 3] 
        logger.info(f"  Sample {idx+1}: Closes={last_5_closes}, Label={label}")
    logger.info("-----------------------")

def add_technical_indicators(df):
    """Adds technical indicators to the DataFrame."""
    df['return'] = df['close'].pct_change()
    df['ema_10'] = ta.ema(df['close'], length=10)
    df['ema_50'] = ta.ema(df['close'], length=50)
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df.dropna(inplace=True)
    return df

def _process_symbol_data(symbol_file):
    """
    Worker function to load a symbol's data, generate trade signals, and label them.
    This is designed to be run in a multiprocessing pool.
    """
    try:
        symbol = os.path.basename(os.path.dirname(symbol_file))
        logger.info(f"Processing data for {symbol}...")
        
        df = pd.read_parquet(symbol_file)
        
        # Drop the non-numeric 'session' column if it exists
        if 'session' in df.columns:
            df.drop(columns=['session'], inplace=True)

        # First, clean the core OHLCV columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
        
        # Now, calculate technical indicators on the cleaned data
        df = add_technical_indicators(df)

        # Final check for any NaNs or infs introduced by indicators
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'return', 'ema_10', 'ema_50', 'rsi', 'atr']
        df.dropna(subset=feature_cols, inplace=True)
        df = df[np.isfinite(df[feature_cols]).all(axis=1)]

        if CONFIG["quick_test"]:
            fraction = CONFIG["quick_test_data_fraction"]
            df = df.head(int(len(df) * fraction))

        klines = df.to_numpy()
        samples = []
        
        # Iterate through the data to find trade opportunities and label them
        for i in range(CONFIG["lookback_candles"] + CONFIG["sequence_length"], len(klines) - 1):
            # The sequence of data the model will see
            sequence_start = i - CONFIG["sequence_length"]
            sequence_end = i
            data_sequence = klines[sequence_start:sequence_end]
            
            # The data used to determine the signal
            signal_klines = klines[i - CONFIG["lookback_candles"]:i]
            
            swing_highs, swing_lows = get_swing_points(signal_klines, CONFIG["swing_window"])
            trend = get_trend(swing_highs, swing_lows)
            
            signal = None
            if trend == "downtrend" and len(swing_highs) > 1 and len(swing_lows) > 1:
                last_swing_high = swing_highs[-1][1]
                last_swing_low = swing_lows[-1][1]
                entry_price = get_fib_retracement(last_swing_high, last_swing_low, trend)
                current_price = float(signal_klines[-1][4]) # last close
                
                # Check if the signal is valid
                if current_price > entry_price:
                    sl = last_swing_high
                    tp = entry_price - (sl - entry_price) # TP1
                    signal = {'side': 'short', 'entry': entry_price, 'sl': sl, 'tp': tp}
            
            elif trend == "uptrend" and len(swing_highs) > 1 and len(swing_lows) > 1:
                last_swing_high = swing_highs[-1][1]
                last_swing_low = swing_lows[-1][1]
                entry_price = get_fib_retracement(last_swing_low, last_swing_high, trend)
                current_price = float(signal_klines[-1][4])
                
                if current_price < entry_price:
                    sl = last_swing_low
                    tp = entry_price + (entry_price - sl) # TP1
                    signal = {'side': 'long', 'entry': entry_price, 'sl': sl, 'tp': tp}
            
            # If a signal was generated, look into the future to label it
            if signal:
                label = 0 # Default to loss
                for future_kline in klines[i:]:
                    future_high = float(future_kline[2])
                    future_low = float(future_kline[3])
                    
                    if signal['side'] == 'long':
                        if future_high >= signal['tp']:
                            label = 1 # Win
                            break
                        if future_low <= signal['sl']:
                            label = 0 # Loss
                            break
                    elif signal['side'] == 'short':
                        if future_low <= signal['tp']:
                            label = 1 # Win
                            break
                        if future_high >= signal['sl']:
                            label = 0 # Loss
                            break
                
                # We have a sequence and a label, add it to our samples
                # Features are OHLCV + new indicators
                features = data_sequence[:, 1:11].astype(np.float32)
                samples.append((features, label))

        logger.info(f"Finished processing {symbol}. Found {len(samples)} samples.")
        return samples
    except Exception as e:
        logger.error(f"Error processing file {symbol_file}: {e}")
        return []

class ManualMinMaxScaler:
    """A manual implementation of a min-max scaler."""
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit_transform(self, data):
        self.min_ = np.min(data, axis=0)
        self.max_ = np.max(data, axis=0)
        # Add a small epsilon to avoid division by zero if a feature is constant
        epsilon = 1e-8
        return (data - self.min_) / (self.max_ - self.min_ + epsilon)

    def transform(self, data):
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        epsilon = 1e-8
        return (data - self.min_) / (self.max_ - self.min_ + epsilon)

class TradingDataset(Dataset):
    """PyTorch dataset for loading and preparing trading data."""
    def __init__(self, samples, scaler=None):
        if not samples:
            self.samples = []
            self.scaler = scaler
            return
            
        features, labels = zip(*samples)
        
        n_samples, seq_len, n_features = len(features), features[0].shape[0], features[0].shape[1]
        features_reshaped = np.reshape(features, (n_samples * seq_len, n_features))
        
        if scaler is None:
            self.scaler = ManualMinMaxScaler()
            features_scaled = self.scaler.fit_transform(features_reshaped)
        else:
            self.scaler = scaler
            features_scaled = self.scaler.transform(features_reshaped)
        
        features_scaled = np.reshape(features_scaled, (n_samples, seq_len, n_features))
        
        self.features = torch.tensor(features_scaled, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features) if hasattr(self, 'features') else 0

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx].unsqueeze(-1)

def manual_train_test_split(samples, test_size=0.2, random_state=42):
    """A manual implementation of train_test_split with stratification."""
    if random_state is not None:
        random.seed(random_state)
    
    # Separate samples by label
    class_0 = [s for s in samples if s[1] == 0]
    class_1 = [s for s in samples if s[1] == 1]
    
    # Shuffle each class list
    random.shuffle(class_0)
    random.shuffle(class_1)
    
    # Calculate split indices
    split_0 = int(len(class_0) * (1 - test_size))
    split_1 = int(len(class_1) * (1 - test_size))
    
    # Create train and validation sets
    train_samples = class_0[:split_0] + class_1[:split_1]
    val_samples = class_0[split_0:] + class_1[split_1:]
    
    # Shuffle the final sets
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    
    return train_samples, val_samples

def prepare_all_data():
    """Scans data directory, loads all data, and prepares it for the model."""
    logger.info("--- Preparing Data ---")
    search_path = os.path.join(CONFIG["data_dir"], 'raw', '**', '*.parquet')
    data_files = glob.glob(search_path, recursive=True)
    
    if not data_files:
        raise FileNotFoundError(f"No Parquet files found in the '{CONFIG['data_dir']}/raw/' directory structure.")
        
    if CONFIG["quick_test"]:
        data_files = data_files[:CONFIG["quick_test_symbols"]]
    
    logger.info(f"Found {len(data_files)} symbol files to process.")
    
    with Pool(processes=min(4, cpu_count())) as pool:
        results = pool.map(_process_symbol_data, data_files)

    logger.info("--- Samples Generated Per Symbol ---")
    for symbol_file, samples in zip(data_files, results):
        logger.info(f"{os.path.basename(symbol_file)} → {len(samples)} samples")
    logger.info("------------------------------------")

    all_samples = [sample for result in results for sample in result]
    
    if not all_samples:
        raise ValueError("No training samples could be generated from the data. Check data files and strategy logic.")
        
    logger.info(f"Total samples generated: {len(all_samples)}")
    
    if not CONFIG.get("quick_test", False):
        validate_data(all_samples)
    
    train_samples, val_samples = manual_train_test_split(all_samples, test_size=0.2, random_state=42)
    
    train_dataset = TradingDataset(train_samples)
    val_dataset = TradingDataset(val_samples, scaler=train_dataset.scaler) # Use same scaler for validation
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=min(4, cpu_count()), pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=min(4, cpu_count()), pin_memory=True)
    
    logger.info("--- Data Preparation Complete ---")
    return train_loader, val_loader, train_dataset.scaler


# --- LSTM Model Definition ---

class TraderLSTM(nn.Module):
    """LSTM model to predict trade outcomes."""
    def __init__(self, input_size=10, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(TraderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the output from the last time step
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Pass to fully connected layer
        out = self.fc(out)
        
        return out


# --- Training and Evaluation ---

def train_model(model, train_loader, val_loader, epochs, model_path):
    """Function to train the LSTM model with advanced techniques."""
    # Calculate pos_weight for imbalanced dataset
    labels = train_loader.dataset.labels.numpy()
    neg = np.sum(labels == 0)
    pos = np.sum(labels == 1)
    
    if pos > 0:
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(DEVICE)
        logger.info(f"Using positive weight for loss function: {pos_weight.item():.2f}")
    else:
        # If there are no positive samples, weight is irrelevant but should be 1 to avoid errors.
        pos_weight = torch.tensor([1.0], dtype=torch.float32).to(DEVICE)
        logger.warning("No positive samples found in training data. Using default pos_weight of 1.0.")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG.get("weight_decay", 0))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    for epoch in range(epochs):
        # Training
        model.train()
        running_train_loss = 0.0
        for i, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(sequences)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
                # Apply sigmoid to logits to get probabilities for prediction
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = running_val_loss / len(val_loader)
        accuracy = 100 * correct / total
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(accuracy)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        logger.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {accuracy:.2f}%, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        scheduler.step(avg_val_loss)
        
        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path} (best validation loss: {best_val_loss:.4f})")
            
    # Log in-sample logit range after training
    model.eval()
    all_logits = []
    with torch.no_grad():
        for seqs, _ in train_loader:
            all_logits.extend(model(seqs.to(DEVICE)).cpu().numpy().flatten())
    if all_logits:
        logger.info(f"In-sample logits after training: min={min(all_logits):.4f}, max={max(all_logits):.4f}, avg={np.mean(all_logits):.4f}")

    return history

def generate_model_report(history, report_dir):
    """Generates a report on the model's training performance."""
    logger.info("--- Generating Model Report ---")
    os.makedirs(report_dir, exist_ok=True)
    
    # Plotting training & validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(report_dir, 'loss_curve.png'), bbox_inches='tight')
    plt.close()
    
    # Plotting validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(os.path.join(report_dir, 'accuracy_curve.png'), bbox_inches='tight')
    plt.close()
    
    # Summary text file
    best_epoch = np.argmin(history['val_loss'])
    summary = f"""
Model Training Summary
======================
Total Epochs: {len(history['train_loss'])}
Best Epoch: {best_epoch + 1}

Final Metrics:
--------------
Final Training Loss: {history['train_loss'][-1]:.4f}
Final Validation Loss: {history['val_loss'][-1]:.4f}
Final Validation Accuracy: {history['val_acc'][-1]:.2f}%

Best Model Metrics (Epoch {best_epoch + 1}):
------------------------------------
Validation Loss: {history['val_loss'][best_epoch]:.4f}
Validation Accuracy: {history['val_acc'][best_epoch]:.2f}%
"""
    with open(os.path.join(report_dir, 'model_summary.txt'), 'w') as f:
        f.write(summary)
        
    logger.info("--- Model Training Summary ---")
    logger.info("\n" + summary)
    logger.info(f"Model report saved in {report_dir}")


def generate_confidence_histogram(confidences, save_path):
    """Generates and saves a histogram of model confidence scores."""
    if not confidences:
        logger.warning("No confidence scores to generate a histogram for.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=50, color='purple', alpha=0.7)
    plt.title('Model Logit Distribution')
    plt.xlabel('Logit')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Confidence histogram saved to {save_path}")


# --- ML-Based Backtesting ---

def run_ml_backtest(model, scaler, report_dir):
    """Runs a backtest using the original strategy filtered by the ML model."""
    logger.info("--- Starting ML-Based Backtest ---")
    model.to(DEVICE)
    model.eval()

    search_path = os.path.join(CONFIG["data_dir"], 'raw', '**', '*.parquet')
    data_files = glob.glob(search_path, recursive=True)
    if not data_files:
        logger.warning("No data files found for backtesting.")
        return

    backtest_trades = []
    balance = CONFIG['starting_balance']
    all_confidences = []
    total_raw_signals = 0
    total_ml_signals = 0

    for symbol_file in data_files:
        symbol = os.path.basename(os.path.dirname(symbol_file))
        logger.info(f"Backtesting on {symbol}...")
        
        raw_signals = 0
        ml_signals = 0
        confidences_logged = 0
        trades_before_symbol = len(backtest_trades)
        
        df = pd.read_parquet(symbol_file)
        if 'session' in df.columns:
            df.drop(columns=['session'], inplace=True)

        # First, clean the core OHLCV columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
        
        # Now, calculate technical indicators on the cleaned data
        df = add_technical_indicators(df)

        # Final check for any NaNs or infs introduced by indicators
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'return', 'ema_10', 'ema_50', 'rsi', 'atr']
        df.dropna(subset=feature_cols, inplace=True)
        df = df[np.isfinite(df[feature_cols]).all(axis=1)]

        klines = df.to_numpy()

        for i in range(CONFIG["lookback_candles"] + CONFIG["sequence_length"], len(klines) - 1):
            signal_klines = klines[i - CONFIG["lookback_candles"]:i]
            
            swing_highs, swing_lows = get_swing_points(signal_klines, CONFIG["swing_window"])
            trend = get_trend(swing_highs, swing_lows)
            
            signal = None
            if trend == "downtrend" and len(swing_highs) > 1 and len(swing_lows) > 1:
                last_swing_high, last_swing_low = swing_highs[-1][1], swing_lows[-1][1]
                entry_price = get_fib_retracement(last_swing_high, last_swing_low, trend)
                if float(signal_klines[-1][4]) > entry_price:
                    sl, tp = last_swing_high, entry_price - (last_swing_high - entry_price)
                    signal = {'side': 'short', 'entry': entry_price, 'sl': sl, 'tp': tp, 'reason': 'Fib retracement in downtrend'}
            
            elif trend == "uptrend" and len(swing_highs) > 1 and len(swing_lows) > 1:
                last_swing_high, last_swing_low = swing_highs[-1][1], swing_lows[-1][1]
                entry_price = get_fib_retracement(last_swing_low, last_swing_high, trend)
                if float(signal_klines[-1][4]) < entry_price:
                    sl, tp = last_swing_low, entry_price + (entry_price - last_swing_low)
                    signal = {'side': 'long', 'entry': entry_price, 'sl': sl, 'tp': tp, 'reason': 'Fib retracement in uptrend'}

            if signal:
                raw_signals += 1
                logit = 1.0 # Default logit for debug mode

                if not CONFIG.get("debug_no_ml", False):
                    # ML Model Prediction Step
                    sequence_start = i - CONFIG["sequence_length"]
                    data_sequence = klines[sequence_start:i, 1:11].astype(np.float32)
                    
                    # Scale the sequence using the pre-fitted scaler
                    scaled_sequence = scaler.transform(data_sequence)
                    sequence_tensor = torch.tensor(scaled_sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        logit = model(sequence_tensor).item()
                
                all_confidences.append(logit)
                if confidences_logged < 5:
                    logger.info(f"Sample logit for {symbol} @ index {i}: {logit:.3f} (data shape: {scaled_sequence.shape}, data sum: {scaled_sequence.sum():.2f})")
                    confidences_logged += 1

                # A logit > 0 corresponds to a probability > 0.5
                if logit > 0:
                    ml_signals += 1
                    # Simulate the trade
                    exit_price, status, exit_timestamp = (None, None, None)
                    for j in range(i, len(klines)):
                        future_kline = klines[j]
                        high, low = float(future_kline[2]), float(future_kline[3])
                        
                        if signal['side'] == 'long':
                            if high >= signal['tp']:
                                exit_price, status, exit_timestamp = signal['tp'], 'win', future_kline[0]
                                break
                            elif low <= signal['sl']:
                                exit_price, status, exit_timestamp = signal['sl'], 'loss', future_kline[0]
                                break
                        elif signal['side'] == 'short':
                            if low <= signal['tp']:
                                exit_price, status, exit_timestamp = signal['tp'], 'win', future_kline[0]
                                break
                            elif high >= signal['sl']:
                                exit_price, status, exit_timestamp = signal['sl'], 'loss', future_kline[0]
                                break
                    
                    if j <= i:
                        logger.error(f"Exit index did not advance for {symbol} at index {i}; continuing to avoid infinite loop.")
                        continue

                    if status:
                        quantity = calculate_quantity(balance, CONFIG['risk_per_trade'], signal['sl'], signal['entry'], CONFIG['leverage'])
                        pnl_usd = (exit_price - signal['entry']) * quantity if signal['side'] == 'long' else (signal['entry'] - exit_price) * quantity
                        pnl_pct = (pnl_usd / (signal['entry'] * quantity)) * 100 if (signal['entry'] * quantity) != 0 else 0
                        
                        balance += pnl_usd
                        
                        trade = TradeResult(
                            symbol=symbol, side=signal['side'], entry_price=signal['entry'],
                            exit_price=exit_price, entry_timestamp=klines[i][0], exit_timestamp=exit_timestamp,
                            status=status, pnl_usd=pnl_usd, pnl_pct=pnl_pct, drawdown=0, # Drawdown simplified
                            reason_for_entry=signal['reason'], reason_for_exit=f'{status.capitalize()} hit', fib_levels=[]
                        )
                        trade.balance = balance
                        backtest_trades.append(trade)
                        i = j # Skip to the end of the concluded trade
        
        executed_trades = len(backtest_trades) - trades_before_symbol
        logger.info(f"{symbol}: {raw_signals} raw signals, {ml_signals} ML-filtered, {executed_trades} executed trades.")
        total_raw_signals += raw_signals
        total_ml_signals += ml_signals
    
    logger.info(f"--- Global Backtest Summary ---")
    logger.info(f"Total Raw Signals: {total_raw_signals}")
    logger.info(f"Total ML-Filtered Signals: {total_ml_signals}")
    logger.info(f"Total Trades Executed: {len(backtest_trades)}")
    generate_backtest_report(backtest_trades, CONFIG, CONFIG['starting_balance'], report_dir)
    generate_confidence_histogram(all_confidences, os.path.join(report_dir, "confidence_histogram.png"))


# --- Main Execution Flow ---

def cleanup_quick_test_files():
    """Removes files generated during the quick test."""
    logger.info("Cleaning up quick test files...")
    quick_model_path = CONFIG["model_path"] + ".quick"
    if os.path.exists(quick_model_path):
        os.remove(quick_model_path)
        
    logger.info("Cleanup complete.")

def quick_test():
    """Runs a quick test of the entire pipeline with a subset of data."""
    logger.info("--- Starting Quick Test ---")
    CONFIG["quick_test"] = True
    
    # 1. Prepare Data
    train_loader, val_loader, scaler = prepare_all_data()
    
    # 2. Define and Train Model
    model = TraderLSTM().to(DEVICE)
    quick_model_path = CONFIG["model_path"] + ".quick"
    history = train_model(model, train_loader, val_loader, CONFIG["quick_test_epochs"], quick_model_path)
    
    # Quick test is only for verifying the pipeline runs, no need to run a backtest here.
    
    logger.info("--- Quick Test Passed ---")
    cleanup_quick_test_files()
    return True

def full_run():
    """Runs the full training and backtesting process."""
    logger.info("--- Starting Full Run ---")
    CONFIG["quick_test"] = False

    # 1. Prepare Data
    train_loader, val_loader, scaler = prepare_all_data()

    # 2. Define and Train Model
    model = TraderLSTM().to(DEVICE)
    history = train_model(model, train_loader, val_loader, CONFIG["full_test_epochs"], CONFIG["model_path"])
    
    # 3. Generate Model Report
    generate_model_report(history, report_dir=os.path.join(CONFIG["report_dir"], "model_training_report"))
    
    # 4. Run Final Backtest
    logger.info("Loading best model for final backtest...")
    try:
        model.load_state_dict(torch.load(CONFIG["model_path"]))
    except FileNotFoundError:
        logger.error(f"Could not find saved model at {CONFIG['model_path']}. Halting.")
        return
    except Exception as e:
        logger.error(f"An error occurred while loading the model: {e}", exc_info=True)
        return
        
    run_ml_backtest(model, scaler, report_dir=os.path.join(CONFIG["report_dir"], "final_backtest_report"))

    logger.info("--- Full Run Complete ---")


def run_overfit_test():
    """Trains the model on a single batch to ensure it can overfit."""
    logger.info("--- Starting Single-Batch Overfit Test ---")
    # Use quick_test settings to load data faster for the test
    CONFIG["quick_test"] = True
    train_loader, _, _ = prepare_all_data()
    
    if not train_loader:
        logger.error("Data loader is empty. Cannot perform overfit test.")
        return

    # Get a single batch
    try:
        sequences, labels = next(iter(train_loader))
    except StopIteration:
        logger.error("Could not get a batch from the data loader. Cannot perform overfit test.")
        return
        
    sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
    
    # Log batch details
    pos_count = int(labels.sum())
    neg_count = labels.size(0) - pos_count
    logger.info(f"Overfit batch size: {sequences.shape}, Positives: {pos_count}, Negatives: {neg_count}")
    
    # Eliminate regularization and use high LR for the test
    model = TraderLSTM(dropout=0.0).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2) # High learning rate

    logger.info("Training on a single batch for 50 epochs...")
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Log gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().norm().item() ** 2
        total_norm = total_norm ** 0.5
        
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"Overfit Epoch [{epoch+1}/50], Loss: {loss.item():.4f}, Grad Norm: {total_norm:.4f}")
            
    logger.info("--- Overfit Test Complete ---")
    # A successful test will show the loss decreasing to near-zero.


def main():
    """Main orchestration function."""
    logger.info(f"Using device: {DEVICE}")
    
    if CONFIG.get("run_overfit_test_only", True):
        run_overfit_test()
        return

    # Create main report directory if it doesn't exist
    if not os.path.exists(CONFIG["report_dir"]):
        os.makedirs(CONFIG["report_dir"])

    # Determine if we start with a quick test
    run_quick_test_first = True # This is the default behavior as requested

    if run_quick_test_first:
        try:
            if quick_test():
                logger.info("\nQuick test successful. Starting full run...\n")
                full_run()
            else:
                # This else block might not be reachable if quick_test raises an exception
                logger.warning("Quick test returned False. Halting.")
        except Exception as e:
            logger.error(f"\n--- An error occurred during execution ---", exc_info=True)
            # Perform cleanup even if it fails
            cleanup_quick_test_files()
            logger.info("Halting execution due to an error.")
            sys.exit(1)
    else:
        # If quick test is disabled, go straight to full run
        full_run()

if __name__ == "__main__":
    main()
