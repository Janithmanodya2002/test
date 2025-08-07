import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import subprocess
import sys
import pandas as pd
import time
from binance.client import Client
import keys
import asyncio
import telegram
from telegram.ext import ApplicationBuilder, CommandHandler
import numpy as np
import json
import pytz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import matplotlib.patches as mpatches
import io
import threading
import requests
import logging
from binance.exceptions import BinanceAPIException
import ML
from tensorflow.keras.models import load_model
from pandas_ta import rsi, macd, bbands
try:
    import mplfinance as mpf
except ImportError:
    print("mplfinance not found. Please install it by running: pip install mplfinance")
    exit()
from tabulate import tabulate

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

install_dependencies()

def is_session_valid(client):
    """
    Check if the current Binance session is valid.
    """
    try:
        client.futures_account()
        return True
    except BinanceAPIException as e:
        print(f"Session check failed: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during session check: {e}")
        return False

class TradeResult:
    def __init__(self, symbol, side, entry_price, exit_price, entry_timestamp, exit_timestamp, status, pnl_usd, pnl_pct, drawdown, reason_for_entry, reason_for_exit, fib_levels, ml_prediction=None, ml_confidence=None):
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
        self.ml_prediction = ml_prediction
        self.ml_confidence = ml_confidence

def generate_fib_chart(symbol, klines, trend, swing_high, swing_low, entry_price, sl, tp1, tp2):
    """
    Generate a detailed candlestick chart with Fibonacci levels, entry, SL, and TP.
    """
    df = pd.DataFrame(klines, columns=['dt', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['dt'] = pd.to_datetime(df['dt'], unit='ms')
    df.set_index('dt', inplace=True)
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)

    # Chart Styling
    mc = mpf.make_marketcolors(up='#26A69A', down='#EF5350', wick={'up':'#26A69A', 'down':'#EF5350'}, volume='in', ohlc='i')
    s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc, gridcolor='lightgrey', facecolor='white')

    # Plot
    fig, axlist = mpf.plot(df, type='candle', style=s,
                          figsize=(8, 4.5),
                          returnfig=True,
                          volume=False) # We will plot volume ourselves if needed

    ax = axlist[0]
    ax.set_title(f'{symbol} 15m - Fib Entry/SL/TP', fontsize=16, weight='bold')
    ax.set_ylabel('Price (USDT)', fontsize=10)
    ax.tick_params(axis='x', labelsize=10, labelrotation=45)
    ax.tick_params(axis='y', labelsize=10)

    # Fibonacci Levels
    fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 1.0]
    price_range = swing_high - swing_low
    if trend == 'downtrend':
        fib_prices = [swing_high - level * price_range for level in fib_levels]
        golden_zone_start = swing_high - (price_range * 0.5)
        golden_zone_end = swing_high - (price_range * 0.618)
    else: # uptrend
        fib_prices = [swing_low + level * price_range for level in fib_levels]
        golden_zone_start = swing_low + (price_range * 0.5)
        golden_zone_end = swing_low + (price_range * 0.618)

    ax.axhspan(golden_zone_start, golden_zone_end, color='gold', alpha=0.2)

    for level, price in zip(fib_levels, fib_prices):
        ax.axhline(y=price, color='#455A64', linestyle='--', linewidth=1.2)
        ax.text(df.index[-1], price, f' {level*100:.1f}% - {price:.2f}', color='#455A64', va='center', ha='left', fontsize=9)

    # Current Price
    current_price = df['close'].iloc[-1]
    ax.axhline(y=current_price, color='#000000', linestyle='-', linewidth=1)
    ax.text(df.index[-1], current_price, f' PRICE {current_price:.2f}', color='#000000', va='center', ha='left', fontsize=9, weight='bold')

    # Entry/SL/TP Zones
    entry_high = entry_price * 1.005
    entry_low = entry_price * 0.995
    ax.axhspan(entry_low, entry_high, color='green', alpha=0.2)
    x_mid = df.index[0] + (df.index[-1] - df.index[0]) / 2
    ax.text(x_mid, (entry_high+entry_low)/2, f'ENTRY {entry_price:.2f}', color='white', va='center', ha='center', fontsize=10)

    sl_high = sl * 1.005
    sl_low = sl * 0.995
    ax.axhspan(sl_low, sl_high, color='red', alpha=0.2)
    ax.text(x_mid, (sl_high+sl_low)/2, f'SL {sl:.2f}', color='white', va='center', ha='center', fontsize=10)

    tp_high = tp1 * 1.005
    tp_low = tp1 * 0.995
    ax.axhspan(tp_low, tp_high, color='blue', alpha=0.2)
    ax.text(x_mid, (tp_high+tp_low)/2, f'TP {tp1:.2f}', color='white', va='center', ha='center', fontsize=10)

    # Entry/SL/TP Lines
    ax.axhline(y=entry_price, color='green', linestyle='-', linewidth=1.2)
    ax.axhline(y=sl, color='red', linestyle='-', linewidth=1.2)
    ax.axhline(y=tp1, color='blue', linestyle='-', linewidth=1.2)

    # Legend
    entry_patch = mpatches.Patch(color='green', label='ENTRY')
    sl_patch = mpatches.Patch(color='red', label='SL')
    tp_patch = mpatches.Patch(color='blue', label='TP')
    ax.legend(handles=[entry_patch, sl_patch, tp_patch], loc='lower left')

    # Save to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100) # dpi=100 and figsize=(8,4.5) gives 800x450
    buf.seek(0)

    return buf

def get_binance_server_time(client):
    """
    Get the current server time from Binance.
    """
    try:
        server_time = client.get_server_time()
        return server_time['serverTime']
    except Exception as e:
        print(f"Error getting Binance server time: {e}")
        return None

def get_public_ip():
    """
    Get the public IP address.
    """
    try:
        response = requests.get('https://api.ipify.org?format=json')
        response.raise_for_status()
        ip_data = response.json()
        return ip_data['ip']
    except requests.exceptions.RequestException as e:
        print(f"Error getting public IP address: {e}")
        return None

def get_swing_points(klines, window=5):
    """
    Identify swing points from kline data.
    """
    highs = np.array([float(k[2]) for k in klines])
    lows = np.array([float(k[3]) for k in klines])

    swing_highs = []
    swing_lows = []

    for i in range(window, len(highs) - window):
        is_swing_high = True
        for j in range(1, window + 1):
            if highs[i] < highs[i-j] or highs[i] < highs[i+j]:
                is_swing_high = False
                break
        if is_swing_high:
            swing_highs.append((klines[i][0], highs[i]))

        is_swing_low = True
        for j in range(1, window + 1):
            if lows[i] > lows[i-j] or lows[i] > lows[i+j]:
                is_swing_low = False
                break
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
        golden_zone_start = p1 - (price_range * 0.5)
        golden_zone_end = p1 - (price_range * 0.618)
    else: # Uptrend
        golden_zone_start = p1 + (price_range * 0.5)
        golden_zone_end = p1 + (price_range * 0.618)

    entry_price = (golden_zone_start + golden_zone_end) / 2

    return entry_price

def calculate_quantity(client, symbol_info, risk_per_trade, sl_price, entry_price, leverage, risk_amount_usd=0, use_fixed_risk_amount=False):
    """
    Calculate the order quantity based on risk and leverage.
    """
    try:
        # Get account balance
        account_info = client.futures_account()
        balance = float(account_info['totalWalletBalance'])

        # Calculate the maximum position size allowed by leverage
        max_position_size = balance * leverage

        # Calculate the desired position size based on risk
        risk_amount = 0
        if use_fixed_risk_amount:
            risk_amount = risk_amount_usd
        else:
            risk_amount = balance * (risk_per_trade / 100)
        sl_percentage = abs(entry_price - sl_price) / entry_price
        if sl_percentage == 0:
            return 0

        trade_position_size = risk_amount / sl_percentage

        # Use the smaller of the two position sizes
        final_position_size = min(trade_position_size, max_position_size)

        # Calculate quantity
        quantity = final_position_size / entry_price

        # Adjust for symbol's precision
        step_size = None
        for f in symbol_info['filters']:
            if f['filterType'] == 'LOT_SIZE':
                step_size = float(f['stepSize'])
                break
        if step_size is None:
            print(f"Could not find LOT_SIZE filter for {symbol_info['symbol']}")
            return None

        quantity = (quantity // step_size) * step_size

        return quantity
    except Exception as e:
        print(f"Error calculating quantity for {symbol_info['symbol']}: {e}")
        return None

def get_atr(klines, period=14):
    """
    Calculate the Average True Range (ATR).
    """
    highs = np.array([float(k[2]) for k in klines])
    lows = np.array([float(k[3]) for k in klines])
    closes = np.array([float(k[4]) for k in klines])

    tr1 = highs - lows
    tr2 = np.abs(highs - np.roll(closes, 1))
    tr3 = np.abs(lows - np.roll(closes, 1))

    tr = np.amax([tr1, tr2, tr3], axis=0)

    atr = np.zeros(len(tr))
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

    return atr


def get_klines(client, symbol, interval='15m', limit=100, start_str=None, end_str=None):
    """
    Get historical kline data from Binance.
    """
    try:
        klines = client.get_historical_klines(symbol=symbol, interval=interval, start_str=start_str, end_str=end_str)
        return klines
    except BinanceAPIException as e:
        print(f"Error fetching klines for {symbol}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred fetching klines for {symbol}: {e}")
        return None

def calculate_performance_metrics(backtest_trades, starting_balance):
    """
    Calculate performance metrics from a list of trades.
    """
    num_trades = len(backtest_trades)
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

    expectancy = (win_rate/100 * avg_win) - ( (losses/num_trades) * abs(avg_loss)) if num_trades > 0 else 0

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
        'total_trades': num_trades,
        'winning_trades': wins,
        'losing_trades': losses,
        'win_rate': win_rate,
        'average_win': avg_win,
        'average_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown * 100,
        'net_pnl_usd': net_pnl_usd,
        'net_pnl_pct': net_pnl_pct,
        'expectancy': expectancy
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

def generate_drawdown_curve(backtest_trades, starting_balance):
    """
    Generate and save a plot of the drawdown curve.
    """
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
    plt.savefig('backtest/drawdown_curve.png')
    plt.close()

def generate_win_loss_distribution(backtest_trades):
    """
    Generate and save a plot of the win/loss distribution.
    """
    wins = sum(1 for trade in backtest_trades if trade.status == 'win')
    losses = len(backtest_trades) - wins
    labels = 'Wins', 'Losses'
    sizes = [wins, losses]
    colors = ['#26A69A', '#EF5350']

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Win/Loss Distribution')
    plt.axis('equal')
    plt.savefig('backtest/win_loss_distribution.png')
    plt.close()

def generate_returns_histogram(backtest_trades):
    """
    Generate and save a histogram of trade returns.
    """
    returns = [trade.pnl_pct for trade in backtest_trades]
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=50, color='blue', alpha=0.7)
    plt.title('Trade Returns Histogram')
    plt.xlabel('Return (%)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('backtest/returns_histogram.png')
    plt.close()

def generate_csv_report(backtest_trades):
    """
    Generate a CSV report from the backtest results.
    """
    df = pd.DataFrame([{k: v for k, v in vars(t).items() if k != 'balance'} for t in backtest_trades]) # Exclude balance for clarity
    df.to_csv('backtest/backtest_trades.csv', index=False)
    print("Backtest trades saved to backtest/backtest_trades.csv")

def generate_json_report(backtest_trades, metrics, strategy_analysis):
    """
    Generate a JSON report from the backtest results.
    """
    report = {
        'metrics': metrics,
        'strategy_analysis': strategy_analysis,
        'trades': [{k: v for k, v in vars(t).items() if k != 'balance'} for t in backtest_trades]
    }
    with open('backtest/backtest_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    print("Backtest report saved to backtest/backtest_report.json")

def generate_summary_report(backtest_trades, metrics, strategy_analysis, config, starting_balance):
    """
    Generate a human-readable summary of the backtest results.
    """
    headers = ["Metric", "Value"]
    table = [
        ["Starting Balance", f"${starting_balance:,.2f}"],
        ["Ending Balance", f"${metrics['net_pnl_usd'] + starting_balance:,.2f}"],
        ["Total Profit", f"${metrics['net_pnl_usd']:,.2f}"],
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
    report += "Configuration:\n"
    report += "--------------\n"
    report += f"Risk per trade: {config['risk_per_trade']}%\n"
    report += f"Leverage: {config['leverage']}x\n"
    report += f"ATR Value: {config['atr_value']}\n"
    report += f"Lookback Candles: {config['lookback_candles']}\n"
    report += f"Swing Window: {config['swing_window']}\n\n"

    report += "Overall Performance:\n"
    report += "--------------------\n"
    report += tabulate(table, headers=headers, tablefmt="grid")
    report += "\n\n"

    report += "Strategy Behavior Insights:\n"
    report += "-------------------------\n"
    report += "\nHourly Performance:\n"
    hourly_table = [["Hour", "Wins", "Losses", "Win Rate"]]
    for hour, data in sorted(strategy_analysis['hourly_performance'].items()):
        win_rate = (data['wins'] / data['total']) * 100 if data['total'] > 0 else 0
        hourly_table.append([f"{hour:02d}", data['wins'], data['losses'], f"{win_rate:.2f}%"])
    report += tabulate(hourly_table, headers="firstrow", tablefmt="grid")
    report += "\n\n"

    report += "Trend Performance:\n"
    trend_table = [["Trend", "Wins", "Losses", "Win Rate"]]
    for trend, data in strategy_analysis['trend_performance'].items():
        win_rate = (data['wins'] / data['total']) * 100 if data['total'] > 0 else 0
        trend_table.append([trend.capitalize(), data['wins'], data['losses'], f"{win_rate:.2f}%"])
    report += tabulate(trend_table, headers="firstrow", tablefmt="grid")

    with open("backtest/backtest_summary.txt", "w") as f:
        f.write(report)

    print("Human-readable summary saved to backtest/backtest_summary.txt")

def generate_equity_curve(backtest_trades, starting_balance):
    """
    Generate and save a plot of the equity curve.
    """
    balance_over_time = [starting_balance] + [trade.balance for trade in backtest_trades]
    plt.figure(figsize=(10, 6))
    plt.plot(balance_over_time)
    plt.title('Equity Curve')
    plt.xlabel('Trade Number')
    plt.ylabel('Balance (USD)')
    plt.grid(True)
    plt.savefig('backtest/equity_curve.png')
    plt.close()

def generate_backtest_report(backtest_trades, config, starting_balance):
    """
    Generate a detailed report from the backtest results.
    """
    if not os.path.exists('backtest'):
        os.makedirs('backtest')

    if not backtest_trades:
        print("No trades to generate a report for.")
        return

    metrics = calculate_performance_metrics(backtest_trades, starting_balance)
    strategy_analysis = analyze_strategy_behavior(backtest_trades)

    report = f"""
Backtesting Report
==================

Configuration:
--------------
Risk per trade: {config['risk_per_trade']}%
Leverage: {config['leverage']}x
ATR Value: {config['atr_value']}
Lookback Candles: {config['lookback_candles']}
Swing Window: {config['swing_window']}

Results:
--------
Starting Balance: ${starting_balance:,.2f}
Ending Balance: ${metrics['net_pnl_usd'] + starting_balance:,.2f}
Total Profit: ${metrics['net_pnl_usd']:,.2f}
Total Trades: {metrics['total_trades']}
Winning Trades: {metrics['winning_trades']}
Losing Trades: {metrics['losing_trades']}
Win Rate: {metrics['win_rate']:.2f}%
Average Win: ${metrics['average_win']:,.2f}
Average Loss: ${metrics['average_loss']:,.2f}
Profit Factor: {metrics['profit_factor']:.2f}
Max Drawdown: {metrics['max_drawdown']:.2f}%
Expectancy: ${metrics['expectancy']:,.2f}

Strategy Behavior Insights:
-------------------------
"""
    report += "\nHourly Performance:\n"
    for hour, data in sorted(strategy_analysis['hourly_performance'].items()):
        win_rate = (data['wins'] / data['total']) * 100 if data['total'] > 0 else 0
        report += f"  Hour {hour:02d}: {data['wins']} wins, {data['losses']} losses, {win_rate:.2f}% win rate\n"

    report += "\nTrend Performance:\n"
    for trend, data in strategy_analysis['trend_performance'].items():
        win_rate = (data['wins'] / data['total']) * 100 if data['total'] > 0 else 0
        report += f"  {trend.capitalize()}: {data['wins']} wins, {data['losses']} losses, {win_rate:.2f}% win rate\n"

    report += """
Trade Log:
----------
"""
    for trade in backtest_trades:
        report += f"Timestamp: {datetime.datetime.fromtimestamp(trade.entry_timestamp/1000).strftime('%Y-%m-%d %H:%M:%S')}, Symbol: {trade.symbol}, Side: {trade.side}, Entry: {trade.entry_price:.8f}, Exit: {trade.exit_price:.8f}, Status: {trade.status}, PnL: ${trade.pnl_usd:,.2f} ({trade.pnl_pct:.2f}%), Drawdown: {trade.drawdown:.2f}%\n"
        if trade.ml_prediction is not None:
            prediction_map = {0: "Loss", 1: "TP1 Win", 2: "TP2 Win"}
            ml_info = f"    ML Prediction: {prediction_map.get(trade.ml_prediction, 'Unknown')} (Conf: {trade.ml_confidence:.1%})\n"
            report += ml_info

    # Save the report to a text file
    with open("backtest/backtest_report.txt", "w") as f:
        f.write(report)

    print("Backtest report generated: backtest/backtest_report.txt")
    generate_equity_curve(backtest_trades, starting_balance)
    generate_drawdown_curve(backtest_trades, starting_balance)
    generate_win_loss_distribution(backtest_trades)
    generate_returns_histogram(backtest_trades)
    generate_csv_report(backtest_trades)
    generate_json_report(backtest_trades, metrics, strategy_analysis)
    generate_summary_report(backtest_trades, metrics, strategy_analysis, config, starting_balance)

def run_backtest(client, symbols, days_to_backtest, config, symbols_info):
    """
    Run the backtesting simulation and generate data for the ML model.
    """
    print("Starting backtest...")

    # Load the ML model
    try:
        model_data = joblib.load('models/trading_model.joblib')
        loaded_ml_model = model_data['model']
        ml_feature_columns = model_data['feature_columns']
        print("ML model loaded for backtesting.")
    except FileNotFoundError:
        print("ML model not found. Backtest will run without ML filtering.")
        ml_model = None
    except Exception as e:
        print(f"Error loading ML model: {e}. Backtest will run without ML filtering.")
        ml_model = None

    end_date = datetime.datetime.now(pytz.utc)
    start_date = end_date - datetime.timedelta(days=days_to_backtest)

    all_klines = {}
    for symbol in symbols:
        print(f"Fetching historical data for {symbol}...")
        klines = get_klines(client, symbol, interval=Client.KLINE_INTERVAL_15MINUTE,
                              start_str=start_date.strftime("%Y-%m-%d %H:%M:%S"),
                              end_str=end_date.strftime("%Y-%m-%d %H:%M:%S"))
        if klines:
            all_klines[symbol] = klines
            # --- ML Data Generation ---
            print(f"  - Saving data for ML pipeline for {symbol}...")
            ML.process_and_save_kline_data(klines, symbol)
            # --------------------------

    print("Backtest data fetched.")

    backtest_trades = []
    balance = config['starting_balance']

    for symbol in all_klines:
        print(f"Backtesting {symbol}...")
        klines = all_klines[symbol]
        for i in range(config['lookback_candles'], len(klines)):
            current_klines = klines[i-config['lookback_candles']:i]
            swing_highs, swing_lows = get_swing_points(current_klines, config['swing_window'])
            trend = get_trend(swing_highs, swing_lows)

            if trend == "downtrend" and len(swing_highs) > 1 and len(swing_lows) > 1:
                last_swing_high = swing_highs[-1][1]
                last_swing_low = swing_lows[-1][1]
                entry_price = get_fib_retracement(last_swing_high, last_swing_low, trend)
                sl = last_swing_high
                tp1 = entry_price - (sl - entry_price)
                tp2 = entry_price - (sl - entry_price) * 2

                ml_prediction = None
                ml_confidence = None
                if loaded_ml_model:
                    setup_info = {
                        'swing_high_price': last_swing_high, 'swing_low_price': last_swing_low,
                        'entry_price': entry_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'side': 'short'
                    }
                    ml_prediction, ml_confidence = get_model_prediction([dict(zip(['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'], k)) for k in current_klines], setup_info, loaded_ml_model, ml_feature_columns)
                    if ml_prediction == 0 or (ml_confidence is not None and ml_confidence < config.get('model_confidence_threshold', 0.0)): # Apply threshold if configured for backtest
                        continue

                # Simulate trade entry
                if float(current_klines[-1][4]) > entry_price:
                    entry_timestamp = current_klines[-1][0]
                    quantity = calculate_quantity(client, symbols_info[symbol], config['risk_per_trade'], sl, entry_price, config['leverage'])
                    if quantity is None or quantity == 0:
                        continue

                    # Simulate trade exit
                    exit_timestamp = 0
                    exit_price = 0
                    status = ''
                    reason_for_exit = ''

                    for j in range(i, len(klines)):
                        future_kline = klines[j]
                        high_price = float(future_kline[2])
                        low_price = float(future_kline[3])

                        if low_price <= tp1:
                            exit_price = tp1
                            status = 'win'
                            reason_for_exit = 'TP1 Hit'
                            exit_timestamp = future_kline[0]
                            break
                        elif high_price >= sl:
                            exit_price = sl
                            status = 'loss'
                            reason_for_exit = 'SL Hit'
                            exit_timestamp = future_kline[0]
                            break

                    if status == '':
                        continue

                    pnl_usd = (entry_price - exit_price) * quantity if status == 'win' else (entry_price - exit_price) * quantity
                    pnl_pct = (pnl_usd / (entry_price * quantity)) * 100
                    balance += pnl_usd

                    trade = TradeResult(
                        symbol=symbol,
                        side='short',
                        entry_price=entry_price,
                        exit_price=exit_price,
                        entry_timestamp=entry_timestamp,
                        exit_timestamp=exit_timestamp,
                        status=status,
                        pnl_usd=pnl_usd,
                        pnl_pct=pnl_pct,
                        drawdown=0, # Simplified for now
                        reason_for_entry=f"Fib retracement in downtrend",
                        reason_for_exit=reason_for_exit,
                        fib_levels=[0, 0.236, 0.382, 0.5, 0.618, 1.0] # Simplified for now
 ,
 ml_prediction=ml_prediction,
 ml_confidence=ml_confidence)
                    trade.balance = balance
                    backtest_trades.append(trade)
                    i += (j - i) # Move to the next candle after the trade is closed

            elif trend == "uptrend" and len(swing_highs) > 1 and len(swing_lows) > 1:
                last_swing_high = swing_highs[-1][1]
                last_swing_low = swing_lows[-1][1]
                entry_price = get_fib_retracement(last_swing_low, last_swing_high, trend)
                sl = last_swing_low
                tp1 = entry_price + (entry_price - last_swing_low)
                tp2 = entry_price + (entry_price - last_swing_low) * 2

                ml_prediction = None
                ml_confidence = None
                if loaded_ml_model:
                    setup_info = {
                        'swing_high_price': last_swing_high, 'swing_low_price': last_swing_low,
                        'entry_price': entry_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'side': 'long'
                    }
                    ml_prediction, ml_confidence = get_model_prediction(
                        [dict(zip(['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'], k)) for k in current_klines],
                        setup_info, loaded_ml_model, ml_feature_columns)
                    if ml_prediction == 0 or (ml_confidence is not None and ml_confidence < config.get('model_confidence_threshold', 0.0)): # Apply threshold if configured for backtest
                        continue

                # Simulate trade entry
                if float(current_klines[-1][4]) < entry_price:
                    entry_timestamp = current_klines[-1][0]
                    quantity = calculate_quantity(client, symbols_info[symbol], config['risk_per_trade'], sl, entry_price, config['leverage'])
                    if quantity is None or quantity == 0:
                        continue

                    # Simulate trade exit
                    exit_timestamp = 0
                    exit_price = 0
                    status = ''
                    reason_for_exit = ''

                    for j in range(i, len(klines)):
                        future_kline = klines[j]
                        high_price = float(future_kline[2])
                        low_price = float(future_kline[3])

                        if high_price >= tp1:
                            exit_price = tp1
                            status = 'win'
                            reason_for_exit = 'TP1 Hit'
                            exit_timestamp = future_kline[0]
                            break
                        elif low_price <= sl:
                            exit_price = sl
                            status = 'loss'
                            reason_for_exit = 'SL Hit'
                            exit_timestamp = future_kline[0]
                            break

                    if status == '':
                        continue

                    pnl_usd = (exit_price - entry_price) * quantity if status == 'win' else (exit_price - entry_price) * quantity
                    pnl_pct = (pnl_usd / (entry_price * quantity)) * 100
                    balance += pnl_usd

                    trade = TradeResult(
                        symbol=symbol,
                        side='long',
                        entry_price=entry_price,
                        exit_price=exit_price,
                        entry_timestamp=entry_timestamp,
                        exit_timestamp=exit_timestamp,
                        status=status,
                        pnl_usd=pnl_usd,
                        pnl_pct=pnl_pct,
                        drawdown=0, # Simplified for now
                        reason_for_entry=f"Fib retracement in uptrend",
                        reason_for_exit=reason_for_exit,
                        fib_levels=[0, 0.236, 0.382, 0.5, 0.618, 1.0] # Simplified for now
 ,
 ml_prediction=ml_prediction,
 ml_confidence=ml_confidence)
                    trade.balance = balance
                    backtest_trades.append(trade)
                    i += (j - i) # Move to the next candle after the trade is closed

    print(f"Backtest complete. Found {len(backtest_trades)} potential trades.")
    return backtest_trades

def update_trade_report(trades, backtest_mode=False):
    """
    Update the trade report JSON file.
    """
    if not backtest_mode:
        with open('trades.json', 'w') as f:
            json.dump(trades, f, indent=4, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else o)

async def place_new_order(client, symbol_info, side, order_type, quantity, price=None, stop_price=None, reduce_only=None, position_side=None, is_closing_order=False):
    symbol = symbol_info['symbol']
    p_prec = int(symbol_info['pricePrecision'])
    q_prec = int(symbol_info['quantityPrecision'])

    params = {
        "symbol": symbol,
        "side": side.upper(),
        "type": order_type.upper(),
        "quantity": f"{quantity:.{q_prec}f}"
    }

    if position_side:
        params["positionSide"] = position_side.upper()

    if order_type.upper() in ["LIMIT", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"]:
        if price is None:
            print(f"Price needed for {order_type} on {symbol}")
            return None, "Price needed"
        params.update({
            "price": f"{price:.{p_prec}f}",
            "timeInForce": "GTC"
        })

    if order_type.upper() in ["STOP_MARKET", "TAKE_PROFIT_MARKET", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"]:
        if stop_price is None:
            print(f"Stop price needed for {order_type} on {symbol}")
            return None, "Stop price needed"
        params["stopPrice"] = f"{stop_price:.{p_prec}f}"
        if is_closing_order:
            params["closePosition"] = "true"
            params.pop("reduceOnly", None)
        elif reduce_only is not None:
            params["reduceOnly"] = str(reduce_only).lower()
    elif reduce_only is not None:
        params["reduceOnly"] = str(reduce_only).lower()

    try:
        loop = asyncio.get_running_loop()
        order = await loop.run_in_executor(None, lambda: client.futures_create_order(**params))

        print(f"Order PLACED: {order['symbol']} ID {order['orderId']} "
              f"{order.get('positionSide','N/A')} {order['side']} "
              f"{order['type']} {order['origQty']} @ "
              f"{order.get('price','MARKET')} SP:{order.get('stopPrice','N/A')} "
              f"CP:{order.get('closePosition','false')} "
              f"RO:{order.get('reduceOnly','false')} "
              f"AvgP:{order.get('avgPrice','N/A')} "
              f"Status:{order['status']}")
        return order, None
    except BinanceAPIException as e:
        error_msg = (f"ORDER FAILED for {symbol} {side} {quantity} "
                     f"{order_type}: {str(e)}")
        print(error_msg)
        return None, str(e)
    except Exception as e:
        error_msg = (f"UNEXPECTED ORDER FAILED for {symbol} {side} {quantity} "
                     f"{order_type}: {str(e)}")
        print(error_msg)
        return None, str(e)

async def send_telegram_alert(bot, message):
    """A simple helper function to send a Telegram message."""
    if not bot:
        logging.warning(f"Telegram bot not available. Message not sent: {message}")
        return
    try:
        await bot.send_message(chat_id=keys.telegram_chat_id, text=message)
    except Exception as e:
        logging.error(f"Failed to send Telegram alert: {e}")

async def send_start_message(bot, backtest_mode=False, current_session=None):
    if backtest_mode:
        return
    try:
        message = "🤖 Bot started!"
        if current_session:
            message += f"\n📈 Current Session: {current_session}"
        else:
            message += "\n😴 Outside of all trading sessions."
        await bot.send_message(chat_id=keys.telegram_chat_id, text=message)
    except Exception as e:
        print(f"Error sending start message: {e}")

async def send_backtest_summary(bot, metrics, backtest_trades, starting_balance):
    """
    Send a summary of the backtest results to Telegram.
    """
    summary_text = f"""
*Backtest Summary*
-------------------
*Total Trades:* {metrics['total_trades']}
*Win Rate:* {metrics['win_rate']:.2f}%
*Net PnL:* ${metrics['net_pnl_usd']:,.2f} ({metrics['net_pnl_pct']:.2f}%)
*Profit Factor:* {metrics['profit_factor']:.2f}
*Max Drawdown:* {metrics['max_drawdown']:.2f}%
"""
    try:
        await bot.send_message(chat_id=keys.telegram_chat_id, text=summary_text, parse_mode='Markdown')
        with open('backtest/equity_curve.png', 'rb') as photo:
            await bot.send_photo(chat_id=keys.telegram_chat_id, photo=photo, caption="Equity Curve")
        with open('backtest/backtest_trades.csv', 'rb') as document:
            await bot.send_document(chat_id=keys.telegram_chat_id, document=document, filename='backtest_trades.csv')
    except Exception as e:
        print(f"Error sending backtest summary to Telegram: {e}")

async def send_backtest_complete_message(bot):
    """
    Send a message to Telegram to indicate that the backtest is complete.
    """
    try:
        await bot.send_message(chat_id=keys.telegram_chat_id, text="✅ Backtest is done.")
    except Exception as e:
        print(f"Error sending backtest complete message: {e}")

async def send_market_analysis_image(bot, chat_id, image_buffer, caption, backtest_mode=False):
    """
    Send the market analysis image to Telegram.
    """
    if backtest_mode:
        return
    try:
        image_buffer.seek(0)
        await bot.send_photo(chat_id=chat_id, photo=image_buffer, caption=caption)
    except Exception as e:
        print(f"Error sending market analysis image: {e}")
        # Fallback to text message
        await bot.send_message(chat_id=chat_id, text=f"Error generating chart. {caption}")

async def op_command(update, context):
    """
    Send the latest chart for all open trades.
    """
    with trades_lock:
        open_trades = [trade for trade in trades if trade['status'] in ['running', 'tp1_hit', 'tp2_hit']]

    if not open_trades:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="No open trades.")
        return

    client = Client(keys.api_mainnet, keys.secret_mainnet)

    for trade in open_trades:
        symbol = trade['symbol']
        klines = get_klines(client, symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=chart_image_candles)
        if klines:
            swing_highs, swing_lows = get_swing_points(klines, 5)
            if not swing_highs or not swing_lows:
                continue

            last_swing_high = swing_highs[-1][1]
            last_swing_low = swing_lows[-1][1]

            image_buffer = generate_fib_chart(symbol, klines, trade['side'], last_swing_high, last_swing_low, trade['entry_price'], trade['sl'], trade['tp1'], trade['tp2'])
            current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
            caption = f"Open Trade: {symbol}\nSide: {trade['side']}\nEntry: {trade['entry_price']:.8f}\nCurrent Price: {current_price:.8f}\nSL: {trade['sl']:.8f}\nTP1: {trade['tp1']:.8f}"
            await context.bot.send_photo(chat_id=update.effective_chat.id, photo=image_buffer, caption=caption)

async def cancel_order(client, symbol, order_id):
    """Cancel an open order."""
    if order_id is None:
        return True, None
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: client.futures_cancel_order(symbol=symbol, orderId=order_id))
        print(f"Order CANCELED: {symbol} ID {order_id}")
        return True, None
    except BinanceAPIException as e:
        if e.code == -2011:
            print(f"Order {order_id} for {symbol} not found to cancel. It might have been filled or already cancelled.")
            return True, None
        error_msg = f"Failed to cancel order {order_id} for {symbol}: {str(e)}"
        print(error_msg)
        return False, str(e)
    except Exception as e:
        error_msg = f"Unexpected error canceling order {order_id} for {symbol}: {str(e)}"
        print(error_msg)
        return False, str(e)

async def order_status_monitor(client, application, backtest_mode=False, live_mode=False, symbols_info=None, is_hedge_mode=False):
    """
    Continuously monitor the status of open and pending trades using order status polling.
    """
    if backtest_mode:
        return
    print("Order status monitor started.")
    bot = application.bot
    loop = asyncio.get_running_loop()

    while True:
        try:
            active_trades = []
            with trades_lock:
                active_trades = [t for t in trades if t['status'] in ['pending', 'running']]

            if not active_trades:
                await asyncio.sleep(5)
                continue

            print(f"Monitor: Checking {len(active_trades)} active trades.")
            for trade in active_trades:
                symbol = trade['symbol']
                symbol_info = symbols_info[symbol]
                pos_side = 'LONG' if trade['side'] == 'long' else 'SHORT' if is_hedge_mode else None

                if trade['status'] == 'pending':
                    if time.time() * 1000 - trade['timestamp'] > 4 * 60 * 60 * 1000:
                        await send_telegram_alert(bot, f"⚠️ TRADE INVALIDATED ⚠️\nSymbol: {symbol}\nSide: {trade['side']}\nReason: Order expired (4 hours)")
                        trade['status'] = 'rejected'
                        if symbol in virtual_orders: del virtual_orders[symbol]
                        continue

                    current_price = float((await loop.run_in_executor(None, lambda: client.get_symbol_ticker(symbol=symbol)))['price'])
                    if (trade['side'] == 'long' and current_price >= trade['entry_price']) or \
                       (trade['side'] == 'short' and current_price <= trade['entry_price']):

                        if live_mode:
                            order, err = await place_new_order(client, symbol_info, 'BUY' if trade['side'] == 'long' else 'SELL', 'MARKET', trade['quantity'], position_side=pos_side)
                            if err:
                                await send_telegram_alert(bot, f"⚠️ MARKET ORDER FAILED for {symbol}: {err}")
                                trade['status'] = 'rejected'
                                if symbol in virtual_orders: del virtual_orders[symbol]
                                continue

                            entry_order_id = order['orderId']
                            start_time = time.time()
                            filled = False
                            while time.time() - start_time < 10: # 10 second timeout
                                try:
                                    entry_order = await loop.run_in_executor(None, lambda: client.futures_get_order(symbol=symbol, orderId=entry_order_id))
                                    if entry_order['status'] == 'FILLED':
                                        order = entry_order
                                        filled = True
                                        break
                                except Exception as e:
                                    await send_telegram_alert(bot, f"Warning: Could not check entry order status for {symbol}. Error: {e}")
                                await asyncio.sleep(0.5)

                            if not filled:
                                await send_telegram_alert(bot, f"CRITICAL: Market entry order for {symbol} was not filled in time. Cancelling trade.")
                                await cancel_order(client, symbol, entry_order_id)
                                trade['status'] = 'rejected'
                                if symbol in virtual_orders: del virtual_orders[symbol]
                                continue

                            trade['entry_price'] = float(order['avgPrice'])
                            sl_tp_side = 'SELL' if trade['side'] == 'long' else 'BUY'

                            sl_order, sl_err = await place_new_order(client, symbol_info, sl_tp_side, 'STOP_MARKET', trade['quantity'], stop_price=trade['sl'], is_closing_order=True, position_side=pos_side)
                            if sl_err:
                                await send_telegram_alert(bot, f"CRITICAL: Failed to place SL for {symbol}. Closing position.")
                                await place_new_order(client, symbol_info, sl_tp_side, 'MARKET', trade['quantity'], is_closing_order=True, position_side=pos_side)
                                trade['status'] = 'error'
                                if symbol in virtual_orders: del virtual_orders[symbol]
                                continue
                            trade['sl_order_id'] = sl_order['orderId']

                            step_size = next((float(f['stepSize']) for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)

                            qty_tp1 = (trade['quantity'] * 0.5 // step_size) * step_size if step_size else 0
                            qty_tp2 = trade['quantity'] - qty_tp1

                            if qty_tp1 > 0:
                                tp_order, tp_err = await place_new_order(client, symbol_info, sl_tp_side, 'TAKE_PROFIT_MARKET', qty_tp1, stop_price=trade['tp1'], is_closing_order=True, position_side=pos_side)
                                if not tp_err:
                                    trade['tp_order_id'] = tp_order['orderId']
                                else:
                                    await send_telegram_alert(bot, f"Warning: Failed to place TP1 for {symbol}. SL is active. Error: {tp_err}")

                            if qty_tp2 > 0:
                                tp2_order, tp2_err = await place_new_order(client, symbol_info, sl_tp_side, 'TAKE_PROFIT_MARKET', qty_tp2, stop_price=trade['tp2'], is_closing_order=True, position_side=pos_side)
                                if tp2_err:
                                    await send_telegram_alert(bot, f"Warning: Failed to place TP2 for {symbol}. Error: {tp2_err}")

                        trade['status'] = 'running'
                        await send_telegram_alert(bot, f"✅ TRADE TRIGGERED & PROTECTED ✅\nSymbol: {symbol}\nEntry: {trade['entry_price']:.8f}\nSide: {trade['side']}\nSL: {trade['sl']:.8f}\nTP1: {trade['tp1']:.8f}")

                elif trade['status'] == 'running':
                    sl_order = await loop.run_in_executor(None, lambda: client.futures_get_order(symbol=symbol, orderId=trade['sl_order_id']))
                    if sl_order['status'] == 'FILLED':
                        await send_telegram_alert(bot, f"🛑 STOP LOSS HIT 🛑\nSymbol: {symbol}\nSide: {trade['side']}\nPrice: {sl_order['avgPrice']}")
                        await cancel_order(client, symbol, trade['tp_order_id'])
                        trade['status'] = 'sl_hit'
                        if symbol in virtual_orders: del virtual_orders[symbol]
                        continue

                    if trade['tp_order_id']:
                        tp_order = await loop.run_in_executor(None, lambda: client.futures_get_order(symbol=symbol, orderId=trade['tp_order_id']))
                        if tp_order['status'] == 'FILLED':
                            await send_telegram_alert(bot, f"🎉 TAKE PROFIT 1 HIT 🎉\nSymbol: {symbol}\nSide: {trade['side']}\nPrice: {tp_order['avgPrice']}\nClosing 50% of the position.")
                            await cancel_order(client, symbol, trade['sl_order_id'])

                            trade['status'] = 'tp1_hit'
                            trade['sl'] = trade['entry_price']

                            step_size = next((float(f['stepSize']) for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                            remaining_quantity = (trade['quantity'] * 0.5 // step_size) * step_size if step_size else 0
                            trade['quantity'] = remaining_quantity

                            if live_mode and remaining_quantity > 0:
                                sl_tp_side = 'SELL' if trade['side'] == 'long' else 'BUY'

                                new_sl_order, sl_err = await place_new_order(client, symbol_info, sl_tp_side, 'STOP_MARKET', remaining_quantity, stop_price=trade['sl'], reduce_only=True, position_side=pos_side)
                                if sl_err:
                                    await send_telegram_alert(bot, f"CRITICAL: Failed to place new SL for {symbol} after TP1. Closing position.")
                                    await place_new_order(client, symbol_info, sl_tp_side, 'MARKET', remaining_quantity, reduce_only=True, position_side=pos_side)
                                    trade['status'] = 'error'
                                    if symbol in virtual_orders: del virtual_orders[symbol]
                                    continue
                                trade['sl_order_id'] = new_sl_order['orderId']

                                # Placeholder for TP2 logic
                                trade['tp_order_id'] = None
                            continue

                    current_price = float((await loop.run_in_executor(None, lambda: client.get_symbol_ticker(symbol=symbol)))['price'])
                    if (trade['side'] == 'long' and current_price >= trade['entry_price'] * 1.20) or \
                       (trade['side'] == 'short' and current_price <= trade['entry_price'] * 0.80):
                        new_sl = trade['entry_price']
                        if (trade['side'] == 'long' and new_sl > trade['sl']) or \
                           (trade['side'] == 'short' and new_sl < trade['sl']):

                            if live_mode:
                                sl_tp_side = 'SELL' if trade['side'] == 'long' else 'BUY'
                                ok, err = await cancel_order(client, symbol, trade['sl_order_id'])
                                if ok:
                                    new_sl_order, sl_err = await place_new_order(client, symbol_info, sl_tp_side, 'STOP_MARKET', trade['quantity'], stop_price=new_sl, reduce_only=True, position_side=pos_side)
                                    if not sl_err:
                                        trade['sl'] = new_sl
                                        trade['sl_order_id'] = new_sl_order['orderId']
                                        await send_telegram_alert(bot, f"🔒 STOP LOSS UPDATED 🔒\nSymbol: {symbol}\nSide: {trade['side']}\nNew SL: {new_sl:.8f}")
                                    else:
                                        await send_telegram_alert(bot, f"Warning: Failed to update SL for {symbol}. Original SL may be active or filled.")

            with trades_lock:
                update_trade_report(trades)

        except Exception as e:
            print(f"Error in order status monitor: {e}")

        await asyncio.sleep(5)


# Global variables
virtual_orders = {}
trades = []
leverage = 0
chart_image_candles = 0
trades_lock = threading.Lock()
rejected_symbols = {}
ml_model = None
ml_feature_columns = None
model_confidence_threshold = 0.7 # Default value

rejected_symbols = {} # To store symbols recently rejected by ML

def generate_live_features(klines, feature_columns, sequence_length=60):
    """
    Generates features for a live setup to be used for model prediction.
    """
    if len(klines) < sequence_length:
        return None

    # Convert klines to a DataFrame
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])

    # Calculate indicators
    df['RSI_14'] = rsi(df['close'])
    macd_df = macd(df['close'])
    df = df.join(macd_df)
    bbands_df = bbands(df['close'])
    df = df.join(bbands_df)
    df.dropna(inplace=True)

    if len(df) < sequence_length:
        return None

    # Get the last sequence
    feature_df = df.iloc[-sequence_length:]
    
    # Normalize features
    first_candle = feature_df.iloc[0]
    normalized_df = feature_df[feature_columns].copy()
    for col in ['open', 'high', 'low', 'close', 'volume'] + [c for c in feature_columns if 'BB' in c]:
        if first_candle[col] > 0:
            normalized_df[col] = (normalized_df[col] / first_candle[col]) - 1
    
    for col in ['RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']:
        if col in normalized_df.columns:
            normalized_df[col] = normalized_df[col] / 100
    
    normalized_df.fillna(0, inplace=True)

    # Reshape for the model
    feature_array = np.array([normalized_df.to_numpy()])
    
    return feature_array



def log_ml_decision(symbol, side, prediction, confidence, outcome='pending'):
    """Logs the ML model's decision and the trade outcome."""
    log_file = 'ml_decisions.csv'
    log_entry = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'symbol': symbol,
        'side': side,
        'prediction': prediction,
        'confidence': confidence,
        'outcome': outcome
    }
    
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = pd.DataFrame([log_entry])
        writer.to_csv(f, header=not file_exists, index=False)

def start_order_monitor(client, application, backtest_mode, live_mode, symbols_info, is_hedge_mode):
    """Wrapper to run the async order_status_monitor in a separate thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(order_status_monitor(client, application, backtest_mode, live_mode, symbols_info, is_hedge_mode))


async def main():
    """
    Main function to run the Binance trading bot.
    """
    print("Starting bot...")

    # Get and print public IP address
    public_ip = get_public_ip()
    if public_ip:
        print(f"Public IP Address: {public_ip}")
    else:
        print("Could not determine public IP address.")

    bot = telegram.Bot(token=keys.telegram_bot_token)

    # Load configuration
    global leverage, chart_image_candles, ml_model, ml_feature_columns, model_confidence_threshold
    try:
        config_df = pd.read_csv('configuration.csv').iloc[0]
        config = config_df.to_dict()
        risk_per_trade = config['risk_per_trade']
        risk_amount_usd = config['risk_amount_usd']
        use_fixed_risk_amount = config['use_fixed_risk_amount']
        leverage = config['leverage']
        atr_value = int(config['atr_value'])
        lookback_candles = int(config['lookback_candles'])
        swing_window = int(config['swing_window'])
        starting_balance = int(config['starting_balance'])
        chart_image_candles = int(config['chart_image_candles'])
        max_open_positions = int(config['max_open_positions'])
        model_confidence_threshold = config.get('model_confidence_threshold', 0.7) # Use .get for safe access
        print("Configuration loaded.")
    except FileNotFoundError:
        print("Error: configuration.csv not found.")
        return
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Load the ML model
    try:
        model_path = config.get('model_path', 'full_run_output/model/lstm_trader.keras')
        ml_model = load_model(model_path)
        feature_columns_path = os.path.join(os.path.dirname(model_path), "feature_columns.json")
        with open(feature_columns_path, 'r') as f:
            ml_feature_columns = json.load(f)
        print(f"ML model loaded from {model_path}. Signal confidence threshold is {model_confidence_threshold * 100}%.")
    except FileNotFoundError:
        print(f"ML model not found at {model_path}. The bot will run with rule-based logic only.")
        ml_model = None
    except Exception as e:
        print(f"Error loading ML model: {e}. The bot will run with rule-based logic only.")
        ml_model = None

    # Load symbols
    try:
        symbols = pd.read_csv('symbols.csv', header=None)[0].tolist()
        print("Symbols loaded.")
    except FileNotFoundError:
        print("Error: symbols.csv not found.")
        return
    except Exception as e:
        print(f"Error loading symbols: {e}")
        return

    # Set up the Telegram bot
    application = ApplicationBuilder().token(keys.telegram_bot_token).build()
    op_handler = CommandHandler('op', op_command)
    application.add_handler(op_handler)

    # Get user input for mode
    while True:
        mode = input("Select (1)Live / (2)Signal / (3)Backtest: ")
        if mode in ['1', '2', '3']:
            break
        else:
            print("Invalid input. Please select 1, 2, or 3.")

    client = None
    symbols_info = None
    is_hedge_mode = False
    live_mode = False
    backtest_mode = False

    if mode == '1' or mode == '2':
        live_mode = mode == '1'
        print(f"Running in {'Live' if live_mode else 'Signal'} mode.")
        try:
            client = Client(keys.api_mainnet, keys.secret_mainnet)
            print("Binance client initialized.")

            server_time = get_binance_server_time(client)
            current_session = "Asia" if 0 <= datetime.datetime.fromtimestamp(server_time / 1000, tz=pytz.utc).hour < 9 else "London" if 7 <= datetime.datetime.fromtimestamp(server_time / 1000, tz=pytz.utc).hour < 16 else "New York" if 12 <= datetime.datetime.fromtimestamp(server_time / 1000, tz=pytz.utc).hour < 21 else None
            await send_start_message(bot, backtest_mode, current_session)

            exchange_info = client.futures_exchange_info()
            symbols_info = {s['symbol']: s for s in exchange_info['symbols']}
            position_mode = client.futures_get_position_mode()
            is_hedge_mode = position_mode.get('dualSidePosition')
            print(f"Futures position mode: {'Hedge Mode' if is_hedge_mode else 'One-way Mode'}")

            account_info = client.futures_account()
            if not account_info['canTrade']:
                print("Error: API key does not have permission to trade futures.")
                return
            if live_mode:
                balance = float(account_info['totalWalletBalance'])
                await bot.send_message(chat_id=keys.telegram_chat_id, text=f"Futures Account Balance: {balance:.2f} USDT")

        except Exception as e:
            print(f"Error initializing Binance client: {e}")
            return
    else: # mode == '3'
        backtest_mode = True
        print("Running in Backtest mode.")
        days_to_backtest = int(input("Enter the number of days to backtest: "))
        try:
            client = Client(keys.api_mainnet, keys.secret_mainnet)
            exchange_info = client.futures_exchange_info()
            symbols_info = {s['symbol']: s for s in exchange_info['symbols']}
        except Exception as e:
            print(f"Could not connect to Binance for backtest setup: {e}")
            print("Backtest will run without quantity calculation if symbols_info is not available.")


    # Start the Telegram bot
    def run_bot():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(application.run_polling())

    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()

    # Load trades from JSON
    if not backtest_mode and os.path.exists('trades.json'):
        with open('trades.json', 'r') as f:
            try:
                loaded_trades = json.load(f)
                with trades_lock:
                    trades.extend(loaded_trades)
                for trade in loaded_trades:
                    if trade['status'] in ['running', 'tp1_hit', 'tp2_hit', 'pending']:
                        virtual_orders[trade['symbol']] = trade
            except json.JSONDecodeError:
                pass

    if not backtest_mode:
        monitor_thread = threading.Thread(target=start_order_monitor, args=(client, application, backtest_mode, live_mode, symbols_info, is_hedge_mode), daemon=True)
        monitor_thread.start()

    if backtest_mode:
        backtest_trades = run_backtest(client, symbols, days_to_backtest, config, symbols_info)
        if backtest_trades:
            metrics = calculate_performance_metrics(backtest_trades, starting_balance)
            strategy_analysis = analyze_strategy_behavior(backtest_trades)
            generate_backtest_report(backtest_trades, config, starting_balance)
            await send_backtest_summary(bot, metrics, backtest_trades, starting_balance)
        await send_backtest_complete_message(bot)
    else:
        # Main scanning loop
        print("Entering main loop...")
        last_session = None
        while True:
            if not is_session_valid(client):
                print("Session is not valid. Halting new scans and cancelling pending orders.")
                with trades_lock:
                    pending_trades = [t for t in trades if t['status'] == 'pending']
                    for trade in pending_trades:
                        print(f"Cancelling pending trade for {trade['symbol']} due to invalid session.")
                        trade['status'] = 'cancelled_session_invalid'
                        if trade['symbol'] in virtual_orders:
                            del virtual_orders[trade['symbol']]
                    update_trade_report(trades)

                await send_telegram_alert(bot, "⚠️ Binance session is invalid. Bot is pausing new trade scans but will continue to monitor open positions.")
                print("Sleeping for 60 seconds before re-checking session.")
                await asyncio.sleep(60)
                continue

            print("Starting new scan cycle...")

            # Session Filtering
            server_time = get_binance_server_time(client)
            current_hour = -1
            if server_time:
                current_hour = datetime.datetime.fromtimestamp(server_time / 1000, tz=pytz.utc).hour

            active_session = None
            if 0 <= current_hour < 9:
                active_session = "Asia"
            elif 7 <= current_hour < 16:
                active_session = "London"
            elif 12 <= current_hour < 21:
                active_session = "New York"

            if active_session:
                if active_session != last_session:
                    await send_telegram_alert(bot, f"📈 New Session Started: {active_session}")
                    last_session = active_session
            else:
                if last_session is not None:
                    await send_telegram_alert(bot, "😴 Outside of all trading sessions. Bot is sleeping.")
                last_session = None
                print(f"Outside of trading session. Current UTC hour: {current_hour}. Sleeping...")
                await asyncio.sleep(60)
                continue

            for symbol in symbols:
                try:
                    if symbol in rejected_symbols and time.time() - rejected_symbols[symbol] < 4 * 60 * 60: # Avoid re-scanning recently rejected symbols
                        continue
                    if symbol in virtual_orders:
                        continue
                    with trades_lock:
                        open_trades = [t for t in trades if t['status'] in ['running', 'tp1_hit', 'tp2_hit', 'pending']]
                        if len(open_trades) >= max_open_positions:
                            continue

                    print(f"Scanning {symbol}...")
                    klines = get_klines(client, symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=lookback_candles)
                    if not klines:
                        continue

                    swing_highs, swing_lows = get_swing_points(klines, swing_window)
                    trend = get_trend(swing_highs, swing_lows)

                    current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                    if trend == "downtrend" and len(swing_highs) > 1 and len(swing_lows) > 1:
                        if time.time() * 1000 - swing_highs[-1][0] > 4 * 60 * 60 * 1000:
                            continue
                        last_swing_high = swing_highs[-1][1]
                        last_swing_low = swing_lows[-1][1]
                        entry_price = get_fib_retracement(last_swing_high, last_swing_low, trend)
                        if current_price < entry_price:
                            continue

                        sl = last_swing_high
                        tp1 = entry_price - (sl - entry_price)
                        tp2 = entry_price - (sl - entry_price) * 2

                        # --- ML Model Integration ---
                        if ml_model and ml_feature_columns:
                            features = generate_live_features(klines, ml_feature_columns)
                            if features is not None:
                                pred_prob = ml_model.predict(features)[0][0]
                                if pred_prob < model_confidence_threshold:
                                    log_ml_decision(symbol, 'short', 0, pred_prob, 'rejected')
                                    await send_telegram_alert(bot, f"❌ Signal Rejected by ML Model ❌\nSymbol: {symbol}\nSide: Short\nReason: Confidence {pred_prob:.2f} < {model_confidence_threshold:.2f}")
                                    rejected_symbols[symbol] = time.time()
                                    continue
                                else:
                                    log_ml_decision(symbol, 'short', 1, pred_prob)
                                    print(f"  - ML Model ACCEPTED signal for {symbol} Short with confidence {pred_prob:.2f}")
                            else:
                                print(f"Could not generate features for {symbol}. Skipping ML check.")
                        # -------------------------- # ML Model Integration End
                        
                        symbol_info = symbols_info[symbol]
                        quantity = calculate_quantity(client, symbol_info, risk_per_trade, sl, entry_price, leverage, risk_amount_usd, use_fixed_risk_amount)
                        if quantity is None or quantity == 0:
                            continue
                        
                        pred_label = 1 if pred_prob > 0.5 else 0
                        ml_info = f"🧠 ML Prediction: Win (Conf: {pred_prob:.1%})"
                        
                        image_buffer = generate_fib_chart(symbol, klines, trend, last_swing_high, last_swing_low, entry_price, sl, tp1, tp2)
                        caption = (f"🚀 NEW TRADE SIGNAL (ML ACCEPTED) 🚀\n{ml_info}\n\n"
                                   f"Symbol: {symbol}\nSide: Short\nLeverage: {leverage}x\n"
                                   f"Risk : {risk_per_trade}%\nProposed Entry: {entry_price:.8f}\n"
                                   f"Stop Loss: {sl:.8f}\nTake Profit 1: {tp1:.8f}\nTake Profit 2: {tp2:.8f}")
                        await send_market_analysis_image(bot, keys.telegram_chat_id, image_buffer, caption)

                        new_trade = {'symbol': symbol, 'side': 'short', 'entry_price': entry_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'status': 'pending', 'quantity': quantity, 'timestamp': klines[-1][0], 'sl_order_id': None, 'tp_order_id': None,
                                     'ml_prediction': pred_label,
                                     'ml_confidence': pred_prob
                                     }

                        with trades_lock:
                            trades.append(new_trade)
                        virtual_orders[symbol] = new_trade
                        update_trade_report(trades)

                    elif trend == "uptrend" and len(swing_highs) > 1 and len(swing_lows) > 1:
                        if time.time() * 1000 - swing_lows[-1][0] > 4 * 60 * 60 * 1000:
                            continue
                        last_swing_high = swing_highs[-1][1]
                        last_swing_low = swing_lows[-1][1]
                        entry_price = get_fib_retracement(last_swing_low, last_swing_high, trend)
                        if current_price > entry_price:
                            continue

                        sl = last_swing_low
                        tp1 = entry_price + (entry_price - last_swing_low)
                        tp2 = entry_price + (entry_price - last_swing_low) * 2

                        # --- ML Model Integration ---
                        if ml_model and ml_feature_columns:
                            features = generate_live_features(klines, ml_feature_columns)
                            if features is not None:
                                pred_prob = ml_model.predict(features)[0][0]
                                if pred_prob < model_confidence_threshold:
                                    log_ml_decision(symbol, 'long', 0, pred_prob, 'rejected')
                                    await send_telegram_alert(bot, f"❌ Signal Rejected by ML Model ❌\nSymbol: {symbol}\nSide: Long\nReason: Confidence {pred_prob:.2f} < {model_confidence_threshold:.2f}")
                                    rejected_symbols[symbol] = time.time()
                                    continue
                                else:
                                    log_ml_decision(symbol, 'long', 1, pred_prob)
                                    print(f"  - ML Model ACCEPTED signal for {symbol} Long with confidence {pred_prob:.2f}")
                            else:
                                print(f"Could not generate features for {symbol}. Skipping ML check.")
                        # -------------------------- # ML Model Integration End
                        
                        symbol_info = symbols_info[symbol]
                        quantity = calculate_quantity(client, symbol_info, risk_per_trade, sl, entry_price, leverage, risk_amount_usd, use_fixed_risk_amount)
                        if quantity is None or quantity == 0:
                            continue

                        pred_label = 1 if pred_prob > 0.5 else 0
                        ml_info = f"🧠 ML Prediction: Win (Conf: {pred_prob:.1%})"

                        image_buffer = generate_fib_chart(symbol, klines, trend, last_swing_high, last_swing_low, entry_price, sl, tp1, tp2)
                        caption = (f"🚀 NEW TRADE SIGNAL (ML ACCEPTED) 🚀\n{ml_info}\n\n"
                                   f"Symbol: {symbol}\nSide: Long\nLeverage: {leverage}x\n"
                                   f"Risk : {risk_per_trade}%\nProposed Entry: {entry_price:.8f}\n"
                                   f"Stop Loss: {sl:.8f}\nTake Profit 1: {tp1:.8f}\nTake Profit 2: {tp2:.8f}")
                        await send_market_analysis_image(bot, keys.telegram_chat_id, image_buffer, caption)

                        new_trade = {'symbol': symbol, 'side': 'long', 'entry_price': entry_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'status': 'pending', 'quantity': quantity, 'timestamp': klines[-1][0], 'sl_order_id': None, 'tp_order_id': None,
                                     'ml_prediction': pred_label,
                                     'ml_confidence': pred_prob
                                     }
                        with trades_lock:
                            trades.append(new_trade)
                        virtual_orders[symbol] = new_trade
                        update_trade_report(trades)
                except Exception as e:
                    print(f"Error scanning {symbol}: {e}")
                    rejected_symbols[symbol] = time.time()

            print("Scan cycle complete. Cooling down for 2 minutes...")
            await asyncio.sleep(120)

if __name__ == "__main__":
    asyncio.run(main())
