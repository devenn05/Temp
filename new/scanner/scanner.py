# scanner.py

import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
import os
import warnings

# --- Local Imports ---
import api_client
import main as strategy_engine
import backtester # We need this for pre-computation logic
from strategy_configs import OPTIMIZED_PARAMETERS, DEFAULT_SETTINGS

# Suppress pandas warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning, module='pandas_ta')

# --- Configuration ---
# The symbols to scan are the ones we have optimized parameters for.
SYMBOLS_TO_SCAN = list(OPTIMIZED_PARAMETERS.keys())
NUMBER_OF_SIGNALS_TO_SHOW = 10

# --- COLOR CODES for terminal output ---
class bcolors:
    HEADER = '\033[95m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def get_ema_slope(series: pd.Series, length: int) -> float:
    """Helper to calculate EMA slope for the quality score."""
    if len(series) < length + 6: return 0.0
    try:
        ema = series.ewm(span=length, adjust=False).mean()
        # Calculate the slope over the last 5 periods
        slope = ema.iloc[-1] - ema.iloc[-6]
        return slope
    except (IndexError, TypeError):
        return 0.0

def analyze_asset(symbol):
    """
    (WORKER FUNCTION) - Fetches data for a single asset, runs the analysis,
    and returns any valid signal found.
    """
    try:
        # Load optimized settings, or use default if not available
        settings = OPTIMIZED_PARAMETERS.get(symbol, DEFAULT_SETTINGS)
        
        # 1. Fetch live data for required timeframes
        data_frames = {
            '15m': api_client.fetch_ohlc_data(symbol, '15m', 'futures', limit=300),
            '1h': api_client.fetch_ohlc_data(symbol, '1h', 'futures', limit=300),
            '4h': api_client.fetch_ohlc_data(symbol, '4h', 'futures', limit=300)
        }

        for tf, df in data_frames.items():
            if df is None or len(df) < 250:
                # print(f"Could not fetch sufficient data for {symbol} on {tf}. Skipping.")
                return None

        # 2. Pre-compute all indicators
        entry_df = backtester.precompute_indicators(data_frames, settings)

        # 3. Run the strategy logic on the *second to last* candle.
        # This is crucial because the most recent candle is not yet closed.
        analysis_index = len(entry_df) - 2
        
        final_analysis = None
        trade_type = None

        # Check for both long and short signals
        long_analysis = strategy_engine.run_analysis_for_backtest(entry_df, None, "long", settings, analysis_index)
        if long_analysis['final_report']['recommendation'] == 'PROCEED':
            final_analysis = long_analysis
            trade_type = 'long'
        else:
            short_analysis = strategy_engine.run_analysis_for_backtest(entry_df, None, "short", settings, analysis_index)
            if short_analysis['final_report']['recommendation'] == 'PROCEED':
                final_analysis = short_analysis
                trade_type = 'short'

        # 4. If a signal is found, package it with a quality score
        if final_analysis and trade_type:
            plan = final_analysis['trade_plan']
            
            # --- Quality Score (Momentum) ---
            # A stronger slope on the 1-hour chart gives a higher quality score.
            # This helps rank the most explosive setups first.
            quality_score = get_ema_slope(data_frames['1h']['close'], 50)
            
            # Ensure score is directionally correct (positive for longs, negative for shorts)
            is_aligned = (trade_type == 'long' and quality_score > 0) or \
                         (trade_type == 'short' and quality_score < 0)
            
            quality_score = abs(quality_score) if is_aligned else -abs(quality_score)

            signal = {
                'symbol': symbol,
                'type': trade_type,
                'entry_price': plan['entry_price'],
                'stop_loss': plan['stop_loss'],
                'take_profit_final': plan['take_profit_final'],
                'risk_reward_ratio': plan['risk_reward_ratio'],
                'reason': final_analysis['final_report']['reason'],
                'quality_score': quality_score
            }
            return signal

    except Exception as e:
        print(f"An error occurred while analyzing {symbol}: {e}")
    return None

def run_scanner():
    """Main function to orchestrate the parallel scanning."""
    start_time = time.time()
    print(f"\n{bcolors.HEADER}{bcolors.BOLD}--- Crypto AI Scanner v2.0 --- ({time.ctime()}){bcolors.ENDC}")
    
    if not SYMBOLS_TO_SCAN:
        print(f"{bcolors.WARNING}No optimized parameters found in 'strategy_configs.py'.")
        print("Please run 'optimizer.py' first to generate configurations.")
        return
        
    print(f"Scanning {len(SYMBOLS_TO_SCAN)} assets for high-probability signals...")

    # Use ThreadPoolExecutor to analyze all symbols concurrently
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        results = list(executor.map(analyze_asset, SYMBOLS_TO_SCAN))

    # Filter out None results (where no signal was found)
    all_signals = [res for res in results if res is not None]

    print("\n--- Scan Complete ---")
    
    if not all_signals:
        print(f"{bcolors.WARNING}No valid trade signals found in this scan.{bcolors.ENDC}")
    else:
        # Sort signals by their quality score in descending order
        sorted_signals = sorted(all_signals, key=lambda x: x['quality_score'], reverse=True)
        
        print(f"{bcolors.OKCYAN}Found {len(sorted_signals)} signals. Displaying top {min(NUMBER_OF_SIGNALS_TO_SHOW, len(sorted_signals))}:{bcolors.ENDC}")
        
        # --- Print the report table ---
        print("-" * 125)
        print(f"{bcolors.BOLD}{'Rank':<5} {'Symbol':<10} {'Type':<8} {'Entry Price':<15} {'Stop Loss':<15} {'Take Profit':<15} {'R:R':<7} {'Reason'}{bcolors.ENDC}")
        print("-" * 125)
        
        for i, signal in enumerate(sorted_signals[:NUMBER_OF_SIGNALS_TO_SHOW]):
            trade_color = bcolors.OKGREEN if signal['type'] == 'long' else bcolors.FAIL
            
            print(f"{i+1:<5} "
                  f"{signal['symbol']:<10} "
                  f"{trade_color}{signal['type'].upper():<8}{bcolors.ENDC} "
                  f"{f'{signal['entry_price']:.4f}':<15} "
                  f"{f'{signal['stop_loss']:.4f}':<15} "
                  f"{f'{signal['take_profit_final']:.4f}':<15} "
                  f"{f'{signal['risk_reward_ratio']:.2f}':<7} "
                  f"{signal['reason']}")

    print("-" * 125)
    end_time = time.time()
    print(f"Total scan time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    while True:
        run_scanner()
        # Wait for the next 15-minute candle to start a new scan
        # We check every minute to be more responsive to the candle close
        print("\nScan complete. Waiting for the next 15-minute candle cycle...")
        try:
            while True:
                current_minute = datetime.now().minute
                if current_minute % 15 == 0:
                    print("New 15m candle detected. Pausing for 10s to let data settle, then re-scanning...")
                    time.sleep(10)
                    break
                time.sleep(30) # Check every 30 seconds
        except KeyboardInterrupt:
            print("\nScanner stopped by user.")
            break