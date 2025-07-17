# optimizer.py
# A multi-core, parallel backtest optimizer for the trading strategy.
# VERSION 3.1: Fixes SettingWithCopyWarning for robust indicator calculation.

import os
import pandas as pd
import numpy as np
from itertools import product
from multiprocessing import Pool
import functools
from tqdm import tqdm
import main as strategy_engine
import backtester

# --- 1. DEFINE THE HIGH-GRANULARITY SEARCH SPACE ---
OPTIMIZATION_CONFIG = {
    "coins": ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'BNBUSDT', 'XRPUSDT'],
    "min_rr_values": list(np.round(np.arange(1.5, 3.6, 0.5), 2)),
    "atr_buffer_values": list(np.round(np.arange(0.6, 2.3, 0.2), 2)),
    "risk_per_trade_values": [2.0],
    "partial_tp_rr_values": [1.1]
}

# --- 2. CONFIGURE PARALLEL PROCESSING ---
NUM_CORES = 7

# --- (analyze_run_performance function remains exactly the same) ---
def analyze_run_performance(trade_list, starting_equity=10000):
    if not trade_list: return {"net_profit": 0, "win_rate": 0, "profit_factor": 0, "max_drawdown": 0, "total_trades": 0, "total_r_gain": 0}
    df = pd.DataFrame(trade_list)
    equity = starting_equity
    equity_curve = [equity]
    for _, trade in df.iterrows(): equity += trade['profit_loss_dollars']; equity_curve.append(equity)
    equity_series = pd.Series(equity_curve)
    peak_equity = equity_series.expanding().max()
    drawdown = (peak_equity - equity_series) / peak_equity
    max_drawdown_percent = drawdown.max() * 100 if not drawdown.empty else 0.0
    net_profit = equity - starting_equity
    total_trades = len(df)
    wins = df[df['outcome'] == 'Win']
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    gross_profit = df[df['profit_loss_dollars'] > 0]['profit_loss_dollars'].sum()
    gross_loss = abs(df[df['profit_loss_dollars'] < 0]['profit_loss_dollars'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    total_r_gain = df['final_r_value'].sum()
    return {"net_profit": round(net_profit, 2), "win_rate": round(win_rate, 2), "profit_factor": round(profit_factor, 2), "max_drawdown": round(max_drawdown_percent, 2), "total_trades": total_trades, "total_r_gain": round(total_r_gain, 2)}

def run_worker(job, start_date, end_date):
    """
    The main worker function, now accepts a start and end date for slicing.
    """
    try:
        coin = job["coin"]
        risk_params = job["risk_params"]
        
        all_data = {}
        for timeframe, df_name in backtester.REQUIRED_TIMEFRAMES.items():
            file_path = os.path.join(backtester.DATA_FOLDER, f"{coin}-{timeframe}-data.csv")
            if not os.path.exists(file_path): return {"error": f"Data not found: {file_path}", **job}
            all_data[df_name] = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)

        # --- DYNAMIC DATA SLICING ---
        for df_name in all_data:
            df = all_data[df_name]
            # --- THE FIX IS HERE: Add .copy() to ensure we work on a new DataFrame ---
            # This explicitly tells pandas to create an independent copy, silencing the warning.
            all_data[df_name] = df.loc[start_date:end_date].copy()
            if all_data[df_name].empty: return {"error": f"No data for {coin} in range", **job}

        settings = strategy_engine.SMC_V2_SETTINGS.copy()
        settings['risk'].update(risk_params)
        
        data_frames = backtester.precompute_indicators(all_data, settings)
        trade_list = backtester.run_backtest_logic(data_frames, settings, progress_bar=None, quiet_mode=True)
        performance_metrics = analyze_run_performance(trade_list)
        
        return {"coin": coin, **risk_params, **performance_metrics}

    except Exception as e:
        return {"error": str(e), **job}

if __name__ == "__main__":
    # --- DYNAMIC DATE DETECTION ---
    print("Finding the latest available data to define a 1-year test period...")
    latest_end_date = None
    for coin in OPTIMIZATION_CONFIG["coins"]:
        try:
            file_path = os.path.join(backtester.DATA_FOLDER, f"{coin}-15m-data.csv")
            df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
            if latest_end_date is None or df.index[-1] > latest_end_date:
                latest_end_date = df.index[-1]
        except (FileNotFoundError, IndexError):
            pass

    if latest_end_date is None:
        raise ValueError("Could not find any data files to determine the backtest period. Aborting.")

    end_date = pd.to_datetime(latest_end_date)
    start_date = end_date - pd.Timedelta(days=365)
    
    print(f"--- Starting High-Granularity Optimization ---")
    print(f"--- Testing on the latest 1-year period: {start_date.date()} to {end_date.date()} ---")

    keys = OPTIMIZATION_CONFIG.keys()
    values = OPTIMIZATION_CONFIG.values()
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    jobs = []
    for params in param_combinations:
        job = { "coin": params["coins"], "risk_params": {"min_rr": params["min_rr_values"], "atr_buffer_multiplier": params["atr_buffer_values"], "risk_per_trade_percent": params["risk_per_trade_values"],"partial_tp_rr": params["partial_tp_rr_values"]}}
        jobs.append(job)

    print(f"Generated a total of {len(jobs)} unique backtest jobs to run.")
    print(f"Distributing workload across {NUM_CORES} CPU cores...")

    worker_with_dates = functools.partial(run_worker, start_date=start_date, end_date=end_date)
    
    with Pool(NUM_CORES) as p:
        results = list(tqdm(p.imap_unordered(worker_with_dates, jobs), total=len(jobs)))

    print("\n--- Optimization Complete ---")

    results_df = pd.DataFrame(results)

    # --- THIS IS THE CORRECTED LOGIC ---
    # First, check IF the 'error' column was even created.
    if 'error' in results_df.columns:
        error_df = results_df[results_df['error'].notna()]
        if not error_df.empty:
            print("\n--- Encountered Errors During Backtests ---")
            print(error_df)
        
        # Create a df with only the valid results by dropping rows that have an error.
        valid_results_df = results_df[results_df['error'].isna()].copy()
    else:
        # If the 'error' column doesn't exist, it means all jobs were successful.
        print("\n--- All backtest jobs completed without any errors! ---")
        valid_results_df = results_df
    # --- END OF CORRECTED LOGIC ---
    
    output_filename = f"optimization_results_latest_year2024-25.csv"
    
    # Ensure there are valid results before trying to save and print
    if not valid_results_df.empty:
        valid_results_df.to_csv(output_filename, index=False)
        print(f"\nValid results saved to '{output_filename}'")

        print(f"\n--- Top 5 Performing Parameter Sets per Coin for Latest Year (Sorted by Profit Factor) ---")
        for coin in OPTIMIZATION_CONFIG["coins"]:
            print(f"\n--- {coin} ---")
            coin_results = valid_results_df[valid_results_df['coin'] == coin]
            if not coin_results.empty:
                top_5 = coin_results.sort_values(by="profit_factor", ascending=False).head(5)
                print_cols = ['min_rr', 'atr_buffer_multiplier', 'profit_factor', 'net_profit', 'win_rate', 'max_drawdown', 'total_trades']
                existing_cols = [col for col in print_cols if col in top_5.columns]
                print(top_5[existing_cols].to_string())
            else:
                print(f"No valid results for {coin}.")
    else:
        print("\nNo valid results were generated from the optimization run.")