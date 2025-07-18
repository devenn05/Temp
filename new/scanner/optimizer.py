# optimizer.py (v2.0 - Robust and Corrected Logic)

import pandas as pd
import os
import itertools
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
import warnings

# --- Local Imports ---
import backtester
from strategy_configs import DEFAULT_SETTINGS

# Suppress irrelevant pandas warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Configuration ---
DATA_FOLDER = "data"
ASSETS_TO_OPTIMIZE = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']

PARAM_GRID = {
    'min_rr': [2.0, 2.5, 3.0, 3.5],
    'atr_buffer_multiplier': [1.2, 1.5, 1.8, 2.0, 2.5],
    'sweep_lookback': [15, 20, 25, 30]
}

def load_asset_data(symbol):
    """Loads all necessary raw dataframes for a single asset."""
    data_frames = {}
    for timeframe in ['15m', '1h', '4h']:
        file_path = os.path.join(DATA_FOLDER, f"{symbol}-{timeframe}-data.csv")
        if not os.path.exists(file_path):
            print(f"ERROR: Data file not found for {symbol}-{timeframe}. Skipping asset.")
            return None
        df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
        # This is critical: assign the name attribute right after loading.
        df.name = symbol
        data_frames[timeframe] = df
    return data_frames

def run_single_backtest_task(args):
    """
    This is the self-contained worker function for each thread.
    It takes the raw data and one parameter combination, and returns the backtest result.
    """
    raw_data_frames, param_combo = args
    
    # 1. Create the specific settings for this single run
    current_settings = json.loads(json.dumps(DEFAULT_SETTINGS))
    current_settings['risk'].update(param_combo)

    # 2. Pre-compute indicators using this run's specific settings
    # This is now done INSIDE the thread, making it safe and correct.
    entry_df = backtester.precompute_indicators(raw_data_frames, current_settings)
    
    # 3. Run the backtest
    result = backtester.run_backtest_logic(entry_df, current_settings)

    # 4. Attach parameters to the result for later identification
    result['params'] = param_combo
    return result

def run_optimization_for_asset(asset_symbol):
    """
    Manages the optimization process for a single asset.
    """
    print(f"\n--- Starting Optimization for {asset_symbol} ---")
    
    # Load raw data ONCE for the asset.
    raw_data = load_asset_data(asset_symbol)
    if not raw_data:
        return None

    # Generate all parameter combinations.
    keys, values = zip(*PARAM_GRID.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Testing {len(param_combinations)} parameter combinations for {asset_symbol}...")

    # Create a list of tasks. Each task is a tuple of (raw_data, param_combo).
    # We pass the same raw_data object to each task.
    tasks = [(raw_data, combo) for combo in param_combinations]

    best_result = {'fitness_score': -999}
    best_params = None

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Use tqdm to create a progress bar for the parallel execution.
        results_iterator = tqdm(executor.map(run_single_backtest_task, tasks), total=len(tasks), desc=f"Optimizing {asset_symbol}")
        
        for result in results_iterator:
            if result.get('total_trades', 0) < 30: # Filter out results with too few trades
                continue

            # Calculate a fitness score to rank results.
            # We want high win rate, high profit, and low drawdown.
            win_rate_score = result.get('win_rate', 0)
            drawdown_penalty = max(0, 100 - (result.get('max_drawdown_r', 100) * 5))
            total_r_score = result.get('total_r', 0)
            fitness_score = (win_rate_score * 1.5) + (drawdown_penalty * 1.0) + (total_r_score * 0.5)

            if fitness_score > best_result.get('fitness_score', -999):
                best_result = result
                best_result['fitness_score'] = fitness_score
                best_params = result['params'] # Get parameters from the result object

    if best_params:
        print(f"\n--- Best Result for {asset_symbol} ---")
        # Remove params from the dictionary for cleaner printing
        best_performance = {k: v for k, v in best_result.items() if k != 'params'}
        print(f"Parameters: {best_params}")
        print(f"Performance: {best_performance}")
        # The final dictionary to be saved needs the correct structure
        final_params = DEFAULT_SETTINGS
        final_params['risk'].update(best_params)
        return {asset_symbol: final_params}
    else:
        print(f"Could not find a profitable parameter set for {asset_symbol} after testing.")
        return None

def run_optimizer():
    """Main function to orchestrate the entire optimization process."""
    all_best_params = {}

    for asset in ASSETS_TO_OPTIMIZE:
        result = run_optimization_for_asset(asset)
        if result:
            all_best_params.update(result)

    if not all_best_params:
        print("\nOptimization complete, but no profitable parameters were found.")
        return

    # --- Save the best parameters to the strategy_configs.py file ---
    config_filepath = 'strategy_configs.py'
    try:
        with open(config_filepath, 'w') as f:
            f.write("# strategy_configs.py\n\n")
            f.write("# This file holds the 'golden parameters' discovered during optimization.\n")
            f.write("# It is automatically generated by optimizer.py\n\n")
            f.write("OPTIMIZED_PARAMETERS = ")
            f.write(json.dumps(all_best_params, indent=4))
            f.write("\n\n")
            f.write("DEFAULT_SETTINGS = ")
            f.write(json.dumps(DEFAULT_SETTINGS, indent=4))
            f.write("\n")
        print(f"\n\nOptimization process complete. Best parameters saved to '{config_filepath}'")
        print("You can now run scanner.py to get live signals.")
    except Exception as e:
        print(f"\nERROR: Could not write to {config_filepath}. Please check file permissions. Error: {e}")

if __name__ == "__main__":
    run_optimizer()