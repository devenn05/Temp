# optimizer.py
# This script is configured to test ONLY the 'atr_buffer_multiplier' parameter.

import os
import itertools
from tqdm import tqdm
import pandas as pd
import numpy as np

import backtester
import main as strategy_engine


def run_optimization():
    """
    Main function to run the optimization process.
    """
    print("--- Starting Strategy Parameter Optimization ---")
    print("--- Focusing on: atr_buffer_multiplier ---")

    # --- 1. Load Data Just ONCE ---
    print("Loading and pre-computing indicator data...")
    data_frames = backtester.load_local_data()
    if data_frames is None:
        print("Could not load data. Aborting optimization.")
        return
    # Note: precompute_indicators also calls precompute_smc_signals now
    data_frames = backtester.precompute_indicators(data_frames)

    # --- 2. Define Parameter Range for a Single Variable ---
    # All other settings are locked to the defaults in main.py.
    # We are only iterating through the ATR buffer.

    param_grid = {
        'adx_threshold': np.arange(18, 31, 1)  # From 15 to 30
    }

    # Generate all possible unique combinations
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    total_runs = len(param_combinations)
    print(f"\nParameter Grid Defined. Total combinations to test: {total_runs}")
    print("This should be a relatively quick process.")

    # --- 3. Loop Through All Combinations and Run Backtests ---
    all_results = []

    for params in tqdm(param_combinations, desc="Optimizing ATR Multiplier"):
        # Create a settings dictionary for this specific run
        current_settings = strategy_engine.SMC_V2_SETTINGS.copy()
        current_settings['risk'].update(params)

        try:
            # Run the core backtesting logic
            trade_list = backtester.run_backtest_logic(
                data_frames=data_frames,
                settings=current_settings,
                progress_bar=None
            )

            if not trade_list:
                continue

            # Calculate metrics
            portfolio_metrics = backtester.calculate_portfolio_metrics(trade_list)
            total_r = sum(t.get('final_r_value', 0) for t in trade_list)

            run_summary = {
                **params,  # This unpacks the current parameters (e.g., {'adx_threshold': 18}) into the summary
                'Annualized Return %': float(portfolio_metrics.get('Annualized Return %', '0.0%').replace('%', '')),
                'Total R-Gain': round(total_r, 2),
                'Max Drawdown %': float(portfolio_metrics.get('Max Drawdown %', '0.0%').replace('%', '')),
                'Total Trades': len(trade_list),
                'Win Rate %': round((len([t for t in trade_list if t['outcome'] == 'Win']) / len(trade_list) * 100),
                                    2) if trade_list else 0,
            }
            all_results.append(run_summary)

        except Exception as e:
            print(f"\nError during run with params {params}: {e}")

    # --- 4. Analyze Results and Show the Best ---
    if not all_results:
        print("\nOptimization finished, but no results were generated.")
        return

    print("\n\n--- Optimization Finished ---")

    results_df = pd.DataFrame(all_results)

    # Save the focused results to a new CSV file
    results_filename = "optimizer_atr_buffer_results.csv"
    results_df.to_csv(results_filename, index=False)
    print(f"Full results saved to '{results_filename}'")

    # Sort the DataFrame by Annualized Return to find the best settings
    sorted_results = results_df.sort_values(by="Annualized Return %", ascending=False)

    print("\n--- Top 10 ATR Buffer Multiplier Settings by Annualized Return ---")

    # We don't need to display all the parameter columns now, just the one we tuned
    display_columns = [
        'Annualized Return %', 'Total R-Gain', 'Max Drawdown %', 'Win Rate %', 'Total Trades', 'ATR Multiplier'
    ]
    # Set pandas display options to prevent wrapping
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 15)

    print(sorted_results[display_columns].head(10).to_string())


if __name__ == "__main__":
    run_optimization()