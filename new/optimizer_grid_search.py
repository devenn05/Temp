# realism_checker.py
# FINAL VERSION - Cleaned up and guaranteed to be quiet.

import os
import time
from tqdm import tqdm
import pandas as pd

# We only need to import the core modules
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
    # UNI was not included in the last results, so it's commented out.
    # "UNIUSDT": { "mode_name": "SMC Grid-Optimized - UNI", "risk": { ... }},
}


def run_portfolio_backtests_with_realism():
    for symbol, config in ASSET_CONFIGS.items():
        print("\n\n" + "#"*80)
        print(f"###   NOW RUNNING FINAL REALISM CHECK FOR: {symbol}   ###")
        print("#"*80)
        
        # 1. Load data for the specific symbol directly
        print(f"Loading local data for {symbol}...")
        data_frames = {}
        data_found = True
        for timeframe in ['15m', '1h', '4h']:
            file_path = os.path.join("data", f"{symbol}-{timeframe}-data.csv")
            if not os.path.exists(file_path):
                print(f"FATAL ERROR: Data file not found: {file_path}. Skipping asset.")
                data_found = False; break
        if not data_found: continue

        for tf, df_name in {'15m': 'entry_df', '1h': 'htf_1h', '4h': 'htf_4h'}.items():
            file_path = os.path.join("data", f"{symbol}-{tf}-data.csv")
            data_frames[df_name] = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
            
        # 2. Get the specific settings for this asset
        current_settings = config

        # 3. Pre-compute indicators
        data_frames = backtester.precompute_indicators(data_frames, current_settings)

        # 4. Run the backtest logic
        print(f"Running REALISM backtest with '{current_settings.get('mode_name', 'tuned')}' parameters...")
        start_index = 200
        total_candles = len(data_frames['entry_df']) - 200
        progress_bar = tqdm(total=total_candles, desc=f"Simulating {symbol} w/ Costs")
        
        # We pass quiet_mode=True to suppress the individual trade logs
        frictionless_trade_list = backtester.run_backtest_logic(data_frames, current_settings, progress_bar, start_index, quiet_mode=True)
        progress_bar.close()

        # 5. Apply costs and generate the final report
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
            
            print(f"\nApplied a {COST_PER_TRADE_PERCENT}% round-trip cost to {len(realistic_list)} trades.")
            
            # This now correctly calls the detailed report from the backtester module
            backtester.generate_detailed_report(
                realistic_list, 10000, current_settings['risk']['risk_per_trade_percent']
            )
            # Print the parameters used for this run for easy reference
            print("\n--- Strategy Parameters Used ---")
            for k,v in current_settings['risk'].items():
                print(f"{k.replace('_', ' ').title()}: {v}")
            print("------------------------------------------")

        else:
            print("\nNo trades were executed in the frictionless backtest.")
        time.sleep(2)
    
    print("\n\n" + "#"*80)
    print("###   ALL REALISM CHECKS FINISHED   ###")
    print("#"*80)


if __name__ == "__main__":
    run_portfolio_backtests_with_realism()