# advanced_portfolio_backtester.py
# A deterministic, parallelized, portfolio-level backtesting engine.

import pandas as pd
import os
import time
from itertools import product
from multiprocessing import Pool
from tqdm import tqdm

import main as strategy_engine
import backtester

# --- 1. SCRIPT CONFIGURATION ---

# --- Timeframe ---
# Set to 1, 2, or 3 for the latest N years. Set to None for the full historical backtest.
YEARS_TO_TEST = 1

# --- Portfolio Risk Management ---
MAX_CONCURRENT_TRADES = 20
STARTING_EQUITY = 10000.0

# --- Strategy Golden Parameters (per asset) ---
STRATEGY_CONFIG_MAP = {
    'BTCUSDT': {"min_rr": 3.0, "risk_per_trade_percent": 2.0, "atr_buffer_multiplier": 1.5, "partial_tp_rr": 1.1, "adx_threshold": 22},
    'SOLUSDT': {"min_rr": 3.0, "risk_per_trade_percent": 2.0, "atr_buffer_multiplier": 2.2, "partial_tp_rr": 1.5, "adx_threshold": 22},
    'ADAUSDT': {"min_rr": 3.0, "risk_per_trade_percent": 2.0, "atr_buffer_multiplier": 2.0, "partial_tp_rr": 1.1, "adx_threshold": 22},
}


# --- 2. HELPER & WORKER FUNCTIONS ---

def get_ema_slope(series: pd.Series, length: int) -> float:
    """Helper to calculate EMA slope for the quality score."""
    if len(series) < length: return 0.0
    ema = series.ewm(span=length, adjust=False).mean()
    return ema.iloc[-1] - ema.iloc[-6] if len(ema) > 5 else 0.0

def generate_signals_for_asset(asset):
    """
    (WORKER for Phase 1) - Takes one asset, loads its data, and finds all valid entry signals.
    This function is designed to be run in parallel for each asset.
    """
    try:
        # Load and prepare all required dataframes for this single asset
        data_frames = {}
        for timeframe, df_name_key in backtester.REQUIRED_TIMEFRAMES.items():
            file_path = os.path.join(backtester.DATA_FOLDER, f"{asset}-{timeframe}-data.csv")
            if not os.path.exists(file_path): raise FileNotFoundError(f"Data for {asset}-{timeframe} not found.")
            df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True).copy()
            data_frames[df_name_key] = df
            
        asset_settings = {"risk": STRATEGY_CONFIG_MAP[asset]}
        data_frames = backtester.precompute_indicators(data_frames, asset_settings)
        
        entry_df = data_frames['entry_df']
        signals = []
        for i in range(200, len(entry_df)):
            current_slice = entry_df.iloc[:i+1] # Provide historical context to function
            analysis = None
            
            # Check for Long/Short Signals using the strategy engine
            long_analysis = strategy_engine.run_analysis_for_backtest(entry_df, None, "long", asset_settings, i)
            if long_analysis['final_report']['recommendation'] == 'PROCEED':
                analysis = long_analysis; trade_type = 'long'
            else:
                short_analysis = strategy_engine.run_analysis_for_backtest(entry_df, None, "short", asset_settings, i)
                if short_analysis['final_report']['recommendation'] == 'PROCEED':
                    analysis = short_analysis; trade_type = 'short'
            
            if analysis:
                plan = analysis['trade_plan']
                # Calculate the Quality Score (Momentum) for this signal
                quality_score = get_ema_slope(current_slice['close'], 50)
                # Ensure score is directionally correct
                quality_score = abs(quality_score) if (trade_type == 'long' and quality_score > 0) or \
                                                       (trade_type == 'short' and quality_score < 0) else -abs(quality_score)
                
                signal_data = {'timestamp': entry_df.index[i], 'symbol': asset, 'type': trade_type, 'quality_score': quality_score, **plan}
                signals.append(signal_data)

        return pd.DataFrame(signals)
    except Exception as e:
        print(f"Error processing {asset}: {e}")
        return pd.DataFrame()


def run_portfolio_simulation(signals_df, unified_timeline):
    """
    (Phase 2) - Takes the pre-computed signals and runs the chronological portfolio simulation.
    This function is single-threaded as it must process events sequentially.
    """
    print("\nPhase 2: Running Deterministic Portfolio Simulation...")
    portfolio_equity = STARTING_EQUITY
    active_trades = {}
    closed_trades = []
    equity_curve = [(unified_timeline.index.min(), STARTING_EQUITY)]

    # Group signals by timestamp for efficient processing
    signals_by_time = signals_df.groupby('timestamp')
    
    for timestamp, candle_group in tqdm(unified_timeline.groupby(level=0), desc="Simulating Portfolio"):
        # --- 1. EXIT LOGIC ---
        exited_symbols = []
        for symbol, trade in active_trades.items():
            # Find the correct candle for this symbol at the current timestamp
            if symbol in candle_group['symbol'].values:
                candle = candle_group[candle_group['symbol'] == symbol].iloc[0]
                pnl = 0
                exit_price = 0
                if trade['type'] == 'long':
                    if candle['low'] <= trade['stop_loss']: pnl = -trade['risk_dollars']
                    elif candle['high'] >= trade['take_profit']: pnl = trade['risk_dollars'] * trade['rr']
                else: # Short
                    if candle['high'] >= trade['stop_loss']: pnl = -trade['risk_dollars']
                    elif candle['low'] <= trade['take_profit']: pnl = trade['risk_dollars'] * trade['rr']
                
                if pnl != 0:
                    portfolio_equity += pnl
                    equity_curve.append((timestamp, portfolio_equity))
                    trade.update({'exit_time': timestamp, 'pnl_dollars': pnl, 'outcome': 'Win' if pnl > 0 else 'Loss'})
                    closed_trades.append(trade)
                    exited_symbols.append(symbol)
        
        for symbol in exited_symbols: del active_trades[symbol]

        # --- 2. ENTRY LOGIC ---
        if timestamp in signals_by_time.groups:
            # Grab all signals for this exact timestamp
            potential_trades = signals_by_time.get_group(timestamp)
            
            # Sort them by our quality score to prioritize the best ones
            sorted_trades = potential_trades.sort_values(by='quality_score', ascending=False)
            
            for _, signal in sorted_trades.iterrows():
                # Check portfolio rules
                if signal['symbol'] not in active_trades and len(active_trades) < MAX_CONCURRENT_TRADES:
                    # All rules pass, open the trade
                    risk_dollars = portfolio_equity * (STRATEGY_CONFIG_MAP[signal['symbol']]['risk_per_trade_percent'] / 100.0)
                    active_trades[signal['symbol']] = {
                        'entry_time': timestamp, 'symbol': signal['symbol'], 'type': signal['type'],
                        'entry_price': signal['entry_price'], 'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit_final'], 'risk_dollars': risk_dollars, 'rr': signal['risk_reward_ratio']
                    }
    
    return closed_trades, portfolio_equity, pd.DataFrame(equity_curve, columns=['timestamp', 'equity'])

def generate_full_report(closed_trades, start_equity, end_equity, equity_curve_df, start_date, end_date):
    """Generates the final two-tiered portfolio report."""
    # (This function is the same as the one from the previous version)
    if not closed_trades: print("\n--- No trades executed. ---"); return

    df = pd.DataFrame(closed_trades)
    duration_days = (end_date - start_date).days if pd.notna(start_date) and pd.notna(end_date) else len(equity_curve_df['timestamp'].unique())
    duration_years = max(duration_days / 365.25, 1/365.25)
    
    print("\n" + "="*50); print("--- Overall Portfolio Performance ---"); print("="*50)
    total_return = (end_equity / start_equity - 1) * 100
    annualized_return = ((end_equity / start_equity) ** (1 / duration_years) - 1) * 100 if duration_years > 0 else 0
    
    equity_curve_df['peak'] = equity_curve_df['equity'].expanding().max()
    equity_curve_df['drawdown'] = (equity_curve_df['peak'] - equity_curve_df['equity']) / equity_curve_df['peak']
    max_drawdown = equity_curve_df['drawdown'].max() * 100
    
    print(f"Test Period:                 {start_date.date()} to {end_date.date()}")
    print(f"Start/End Equity:            ${start_equity:,.2f} -> ${end_equity:,.2f}")
    print(f"Net Profit:                  ${end_equity - start_equity:,.2f} | Total Return: {total_return:.2f}%")
    print(f"Annualized Return (CAGR):    {annualized_return:.2f}%")
    print(f"Maximum Portfolio Drawdown:  {max_drawdown:.2f}%")

    print("\n" + "="*50); print("--- Per-Asset Contribution ---"); print("="*50)
    asset_perf = df.groupby('symbol').agg(total_trades=('symbol', 'size'), net_profit=('pnl_dollars', 'sum')).round(2)
    asset_perf['win_rate_%'] = df[df['pnl_dollars'] > 0].groupby('symbol')['symbol'].count().reindex(asset_perf.index, fill_value=0) / df.groupby('symbol')['symbol'].count() * 100
    asset_perf['profit_factor'] = df[df['pnl_dollars'] > 0].groupby('symbol')['pnl_dollars'].sum() / abs(df[df['pnl_dollars'] < 0].groupby('symbol')['pnl_dollars'].sum())
    asset_perf.fillna(0, inplace=True)
    print(asset_perf.sort_values(by="net_profit", ascending=False))
    print("=" * 50)


# --- 3. MAIN SCRIPT EXECUTION ---
if __name__ == "__main__":
    start_time = time.time()
    
    # --- PHASE 1: Generate Signals in Parallel ---
    assets = list(STRATEGY_CONFIG_MAP.keys())
    with Pool(os.cpu_count()) as p:
        results = list(tqdm(p.imap(generate_signals_for_asset, assets), total=len(assets), desc="Phase 1: Generating Signals"))
    
    all_signals_df = pd.concat(results).sort_values(by='timestamp').reset_index(drop=True)
    
    # --- Data Slicing for Simulation ---
    end_date = all_signals_df['timestamp'].max()
    start_date = end_date - pd.Timedelta(days=365 * YEARS_TO_TEST) if YEARS_TO_TEST is not None else all_signals_df['timestamp'].min()
    
    signals_to_process = all_signals_df[all_signals_df['timestamp'] >= start_date].copy()

    # --- Prepare Unified Timeline for Simulation Period ---
    unified_list = []
    for asset in assets:
        # We need to re-load just the entry_df for the timeline
        file_path = os.path.join(backtester.DATA_FOLDER, f"{asset}-15m-data.csv")
        df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
        unified_list.append(df[df.index >= start_date].assign(symbol=asset))

    unified_timeline = pd.concat(unified_list).sort_index()

    print(f"\nFound {len(all_signals_df)} signals total. Processing {len(signals_to_process)} for the period.")

    if not signals_to_process.empty:
        # --- PHASE 2: Run Sequential Simulation ---
        final_trades, final_equity, equity_curve = run_portfolio_simulation(signals_to_process, unified_timeline)
        generate_full_report(final_trades, STARTING_EQUITY, final_equity, equity_curve, start_date, end_date)
    else:
        print("No signals found in the specified test period.")
        
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")