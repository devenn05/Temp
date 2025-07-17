# optimizer.py
# A Multi-Asset, Sequential Optimizer.
# This script is configured to test ONLY the REMAINING parameters:
# - partial_tp_rr
# - risk_per_trade_percent

import os
import itertools
from tqdm import tqdm
import pandas as pd
import numpy as np

import backtester
import main as strategy_engine

def run_multi_asset_sequential_optimization():
    """
    Main function that loops through assets and runs sequential optimizations on each.
    """
    print("--- Starting Sequential Strategy Parameter Optimization (Remaining Tasks) ---")

    # --- 1. Define the Assets and REMAINING Parameters to Test ---
    SYMBOLS_TO_TEST = ["SOLUSDT", "XRPUSDT", "BNBUSDT", "UNIUSDT"]

    # THE FIX IS HERE: We only list the two tests that still need to be run.
    optimization_tasks = {
        'partial_tp_rr': np.arange(1.0, 2.6, 0.1),
        'risk_per_trade_percent': np.arange(0.5, 3.1, 0.1),
    }
    
    print("\nTasks to run:")
    for param in optimization_tasks.keys():
        print(f"- Optimize '{param}'")

    # --- The main loop that iterates through each cryptocurrency ---
    for symbol in SYMBOLS_TO_TEST:
        print("\n\n" + "#"*80)
        print(f"### NOW RUNNING REMAINING OPTIMIZATIONS FOR: {symbol} ###")
        print("#"*80)
        
        # --- Temporarily override the load_local_data function for the current symbol ---
        original_load_local_data = backtester.load_local_data
        
        def patched_load_local_data(symbol_to_load=symbol):
            """A temporary override that loads data for a specific symbol."""
            all_data = {}
            for timeframe, df_name in backtester.REQUIRED_TIMEFRAMES.items():
                file_path = os.path.join(backtester.DATA_FOLDER, f"{symbol_to_load}-{timeframe}-data.csv")
                if not os.path.exists(file_path):
                    print(f"Error: Data file for {symbol_to_load} not found at '{file_path}'. Skipping this symbol.")
                    return None
                all_data[df_name] = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
            return all_data

        backtester.load_local_data = patched_load_local_data
        
        # --- Pre-computation for the current symbol ---
        print(f"\nLoading and pre-computing indicator data for {symbol}...")
        data_frames = backtester.load_local_data()
        if data_frames is None: 
            backtester.load_local_data = original_load_local_data 
            continue
            
        data_frames = backtester.precompute_indicators(data_frames)
    
        # --- Inner loop for each remaining parameter test ---
        for param_to_test, param_range in optimization_tasks.items():
            print("\n" + "="*80)
            print(f"--- Optimizing '{param_to_test.upper()}' for {symbol} ---")
            print("="*80)
            
            all_results = []
            
            for param_value in tqdm(param_range, desc=f"Testing {param_to_test}"):
                current_settings = strategy_engine.SMC_V2_SETTINGS.copy()
                params = {param_to_test: param_value}
                current_settings['risk'].update(params)

                try:
                    trade_list = backtester.run_backtest_logic(data_frames=data_frames, settings=current_settings, progress_bar=None)
                    if not trade_list: continue

                    portfolio_metrics = backtester.calculate_portfolio_metrics(
                        trade_list, 
                        risk_per_trade_percent=current_settings['risk']['risk_per_trade_percent']
                    )
                    total_r = sum(t.get('final_r_value', 0) for t in trade_list)
                    param_summary = {key: round(value, 2) for key, value in params.items()}

                    run_summary = {
                        **param_summary,
                        'Annualized Return %': float(portfolio_metrics.get('Annualized Return %', '0.0%').replace('%', '')),
                        'Total R-Gain': round(total_r, 2), 
                        'Max Drawdown %': float(portfolio_metrics.get('Max Drawdown %', '0.0%').replace('%', '')),
                        'Total Trades': len(trade_list), 
                        'Win Rate %': round((len([t for t in trade_list if t['outcome'] == 'Win']) / len(trade_list) * 100), 2) if trade_list else 0,
                    }
                    all_results.append(run_summary)
                except Exception as e:
                    print(f"\nError during run with param value {param_value}: {e}")

            if not all_results:
                print(f"\nOptimization for {param_to_test} on {symbol} finished, but no results were generated.")
                continue
                
            results_df = pd.DataFrame(all_results)
            results_filename = f"optimizer_{symbol}_{param_to_test}_results.csv"
            results_df.to_csv(results_filename, index=False)
            print(f"\nFull results saved to '{results_filename}'")

            sorted_results = results_df.sort_values(by="Annualized Return %", ascending=False)
            print(f"\n--- Top 10 '{param_to_test.title().replace('_', ' ')}' Settings for {symbol} ---")
            
            core_metrics = ['Annualized Return %', 'Total R-Gain', 'Max Drawdown %', 'Win Rate %', 'Total Trades']
            display_columns = core_metrics + [param_to_test]
            
            pd.set_option('display.width', 200)
            pd.set_option('display.max_columns', 20)
            print(sorted_results[display_columns].head(10).to_string())
        
        # Restore the original function after all tests for this symbol are done
        backtester.load_local_data = original_load_local_data

    print("\n\n" + "#"*80)
    print("### ALL REMAINING MULTI-ASSET OPTIMIZATIONS FINISHED ###")
    print("#"*80)


if __name__ == "__main__":
    run_multi_asset_sequential_optimization()