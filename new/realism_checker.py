# realism_checker_multicore_logging.py
# FINAL VERSION - Modified for extended parameter grids and detailed file logging.

import os
import time
import itertools
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import io
import contextlib

# We only need to import the core modules
# Make sure these modules do not have side effects on import
import backtester
import main as strategy_engine

# --- Master Configuration ---
ASSET_CONFIGS = {
    "BTCUSDT": { "mode_name": "SMC Grid-Optimized - BTC", "risk": {"atr_buffer_multiplier": 0.2, "min_rr": 3.0, "risk_per_trade_percent": 2.0, "partial_tp_rr": 1.2, "adx_threshold": 22}},
    "ETHUSDT": { "mode_name": "SMC Grid-Optimized - ETH", "risk": {"atr_buffer_multiplier": 0.2, "min_rr": 1.5, "risk_per_trade_percent": 2.0, "partial_tp_rr": 1.2, "adx_threshold": 22}},
    "ADAUSDT": { "mode_name": "SMC Grid-Optimized - ADA", "risk": {"atr_buffer_multiplier": 0.2, "min_rr": 1.5, "risk_per_trade_percent": 2.0, "partial_tp_rr": 1.2, "adx_threshold": 22}},
    "SOLUSDT": { "mode_name": "SMC Grid-Optimized - SOL", "risk": {"atr_buffer_multiplier": 0.2, "min_rr": 2.4, "risk_per_trade_percent": 2.0, "partial_tp_rr": 1.1, "adx_threshold": 22}},
    "XRPUSDT": { "mode_name": "SMC Grid-Optimized - XRP", "risk": {"atr_buffer_multiplier": 0.2, "min_rr": 2.4, "risk_per_trade_percent": 2.0, "partial_tp_rr": 1.2, "adx_threshold": 22}},
    "BNBUSDT": { "mode_name": "SMC Grid-Optimized - BNB", "risk": {"atr_buffer_multiplier": 0.2, "min_rr": 2.4, "risk_per_trade_percent": 3.0, "partial_tp_rr": 2.1, "adx_threshold": 22}},
}

# --- Parameter Grid for Optimization ---
# Define the exact variations you want to test. Add or remove as needed.
PARAM_GRID = {
    "atr_buffer_multiplier": [0.2, 0.6, 0.9, 1.5, 2.4],
    "min_rr": [1.5, 2.0, 2.5, 3.0],
    "partial_tp_rr": [1.2, 1.5, 1.8],
    "risk_per_trade_percent": [0,5, 1.0, 2.0]
}


def run_single_backtest(task_info):
    """
    Encapsulates the logic for a single backtest run.
    Instead of printing, it captures all output as a string and returns it.

    Args:
        task_info (tuple): A tuple containing (symbol, parameter_dictionary).
    """
    symbol, params = task_info
    
    # --- Create a unique configuration for this task ---
    config = ASSET_CONFIGS[symbol].copy()
    config['risk'] = config['risk'].copy()
    config['risk'].update(params) # Overwrite base config with the grid search params
    
    # Create a dynamic mode_name for clear identification in logs
    param_str = " | ".join([f"{key.split('_')[0]}:{val}" for key, val in params.items()])
    config['mode_name'] = f"{symbol} | {param_str}"

    # Use StringIO to capture all print statements into a variable
    output_capture = io.StringIO()
    with contextlib.redirect_stdout(output_capture):
        try:
            # 1. Load data
            data_frames = {}
            data_found = True
            for timeframe in ['15m', '1h', '4h']:
                file_path = os.path.join("data", f"{symbol}-{timeframe}-data.csv")
                if not os.path.exists(file_path):
                    print(f"FATAL ERROR: Data file not found for task: {config['mode_name']}. Path: {file_path}")
                    data_found = False; break
            if not data_found:
                return output_capture.getvalue()

            for tf, df_name in {'15m': 'entry_df', '1h': 'htf_1h', '4h': 'htf_4h'}.items():
                file_path = os.path.join("data", f"{symbol}-{tf}-data.csv")
                data_frames[df_name] = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)

            # 2. Pre-compute indicators
            data_frames = backtester.precompute_indicators(data_frames, config)

            # 3. Run the backtest logic (in quiet mode)
            frictionless_trade_list = backtester.run_backtest_logic(data_frames, config, progress_bar=None, start_index=200, quiet_mode=True)
            
            # --- Generate the full report text ---
            print("\n" + "="*80)
            print(f"###   RESULTS FOR: {config['mode_name']}   ###")
            
            # 4. Apply costs and generate the final report
            if frictionless_trade_list:
                realistic_list = []
                COST_PER_TRADE_PERCENT = 0.1
                for trade in frictionless_trade_list:
                    new_trade = trade.copy()
                    cost_in_price = new_trade['entry_price'] * (COST_PER_TRADE_PERCENT / 100.0)
                    risk_distance = abs(new_trade['entry_price'] - new_trade['stop_loss'])
                    cost_in_r = cost_in_price / risk_distance if risk_distance > 0 else 0
                    new_trade['final_r_value'] -= cost_in_r
                    realistic_list.append(new_trade)
                
                print(f"Applied {COST_PER_TRADE_PERCENT}% cost to {len(realistic_list)} trades.")
                
                # This function's print output will be captured by StringIO
                backtester.generate_detailed_report(
                    realistic_list, 10000, config['risk']['risk_per_trade_percent']
                )
                print("\n--- Parameters For This Run ---")
                for k, v in config['risk'].items():
                    print(f"{k.replace('_', ' ').title()}: {v}")
                print("="*80 + "\n")
            else:
                print("\nNo trades were executed for this configuration.")
                print("="*80 + "\n")
        
        except Exception as e:
            print("\n" + "!"*80)
            print(f"AN ERROR OCCURRED during backtest for task: {config.get('mode_name', 'Unknown Task')}")
            print(f"Error Type: {type(e).__name__}, Details: {e}")
            import traceback
            traceback.print_exc() # This will print the full error traceback to the log
            print("!"*80 + "\n")

    return output_capture.getvalue()


def run_portfolio_backtests_with_realism():
    """
    Sets up the multiprocessing pool, distributes backtesting tasks,
    and logs all results to a timestamped file.
    """
    # 1. Dynamically generate all unique combinations of parameters
    symbols_to_test = list(ASSET_CONFIGS.keys())
    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    param_combinations = list(itertools.product(*param_values))

    all_tasks = []
    for symbol in symbols_to_test:
        for combo in param_combinations:
            params_dict = dict(zip(param_names, combo))
            all_tasks.append((symbol, params_dict))
    
    total_tasks = len(all_tasks)
    
    # Setup for logging
    log_dir = "backtest_logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"backtest_run_{timestamp}.txt")

    header = f"""
################################################################################
###        STARTING MULTICORE PORTFOLIO REALISM CHECK - {timestamp}          ###
################################################################################
Generated {total_tasks} unique backtest configurations.
Saving all logs to: {log_filename}

--- PARAMETER GRID ---
{pd.Series(PARAM_GRID).to_string()}
----------------------
"""
    print(header)

    # 2. Set up and run the multiprocessing pool
    try:
        num_cores = cpu_count()
        print(f"Utilizing {num_cores} CPU cores for parallel processing...")
        with open(log_filename, 'w') as log_file:
            log_file.write(header)
            with Pool(processes=num_cores) as pool:
                # Use tqdm to create a master progress bar for the entire job
                with tqdm(total=total_tasks, desc="Overall Backtest Progress") as pbar:
                    # imap_unordered is efficient as it yields results as they are ready
                    for report_string in pool.imap_unordered(run_single_backtest, all_tasks):
                        if report_string:
                            print(report_string)      # Print to console for live feedback
                            log_file.write(report_string) # Write to the log file
                            log_file.flush()          # Ensure it's written to disk immediately
                        pbar.update(1) # Manually update the progress bar

    except Exception as e:
        final_error_msg = f"A critical error occurred during multiprocessing: {e}"
        print(final_error_msg)
        with open(log_filename, 'a') as log_file: # Append error to log
            log_file.write(f"\n\n{final_error_msg}\n")

    final_message = f"""
################################################################################
###                   ALL REALISM CHECKS FINISHED                          ###
################################################################################
All logs have been saved to: {log_filename}
"""
    print(final_message)


if __name__ == "__main__":
    run_portfolio_backtests_with_realism()