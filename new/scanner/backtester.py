# backtester.py (v2.0 - High-Performance with Numba)

import pandas as pd
import pandas_ta as ta
import numpy as np
import main as strategy_engine
from numba import jit

def precompute_indicators(data_frames, settings):
    """
    Calculates all necessary indicators. This part remains in Pandas
    as it is already fast and vectorized.
    """
    # (This function is identical to the previous version)
    entry_df_key = '15m'
    for df in data_frames.values():
        df.ta.ema(length=50, append=True); df.ta.ema(length=200, append=True)
        df.ta.rsi(length=14, append=True); df.ta.atr(length=14, append=True)
        df['volume_20_avg'] = df['volume'].rolling(window=20).mean()
    entry_df = data_frames[entry_df_key]
    sweep_lookback = settings['risk'].get('sweep_lookback', 20)
    entry_df['lookback_high'] = entry_df['high'].shift(1).rolling(window=sweep_lookback).max()
    entry_df['lookback_low'] = entry_df['low'].shift(1).rolling(window=sweep_lookback).min()
    entry_df['is_bullish_sweep'] = (entry_df['low'] < entry_df['lookback_low']) & (entry_df['close'] > entry_df['lookback_low'])
    entry_df['is_bearish_sweep'] = (entry_df['high'] > entry_df['lookback_high']) & (entry_df['close'] < entry_df['lookback_high'])
    htf_1h_df = data_frames['1h'].add_prefix('1h_')
    htf_4h_df = data_frames['4h'].add_prefix('4h_')
    merged_df = entry_df.copy()
    merged_df = pd.merge(merged_df, htf_1h_df[['1h_EMA_50', '1h_EMA_200']], left_index=True, right_index=True, how='left').ffill()
    merged_df = pd.merge(merged_df, htf_4h_df[['4h_EMA_50', '4h_EMA_200']], left_index=True, right_index=True, how='left').ffill()
    clean_df = merged_df.dropna()
    clean_df.name = entry_df.name
    return clean_df

@jit(nopython=True)
def _numba_backtest_kernel(high_prices, low_prices, sl_values, tp_values):
    """
    This is the core loop, compiled by Numba into machine code.
    It ONLY uses simple arrays and numbers, which is why it's so fast.
    """
    n = len(high_prices)
    trades = np.zeros((n, 2)) # Store PnL and type (1=long, -1=short)
    trade_count = 0
    
    in_trade = False
    trade_type = 0 # 1 for long, -1 for short
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    rr = 0.0

    for i in range(n):
        # --- Manage an active trade ---
        if in_trade:
            if trade_type == 1: # Long
                if low_prices[i] <= stop_loss:
                    trades[trade_count, 0] = -1.0 # PnL in R
                    trades[trade_count, 1] = trade_type
                    trade_count += 1
                    in_trade = False
                elif high_prices[i] >= take_profit:
                    trades[trade_count, 0] = rr
                    trades[trade_count, 1] = trade_type
                    trade_count += 1
                    in_trade = False
            elif trade_type == -1: # Short
                if high_prices[i] >= stop_loss:
                    trades[trade_count, 0] = -1.0
                    trades[trade_count, 1] = trade_type
                    trade_count += 1
                    in_trade = False
                elif low_prices[i] <= take_profit:
                    trades[trade_count, 0] = rr
                    trades[trade_count, 1] = trade_type
                    trade_count += 1
                    in_trade = False
        
        # --- Search for a new entry (if not in a trade) ---
        if not in_trade:
            if sl_values[i] != 0: # Signal exists on this candle
                in_trade = True
                trade_type = 1 if tp_values[i] > sl_values[i] else -1
                stop_loss = sl_values[i]
                take_profit = tp_values[i]
                
                # Approximate entry and rr
                entry = (take_profit + stop_loss) / 2
                if abs(entry - stop_loss) > 1e-9:
                     rr = abs(take_profit - entry) / abs(entry - stop_loss)
                else:
                    rr = 0

    return trades[:trade_count] # Return only the trades that happened


def run_backtest_logic(entry_df, settings):
    """
    The main backtesting function. It now prepares the data and then
    calls the high-speed Numba kernel to do the heavy lifting.
    """
    # 1. Prepare Signal Arrays: Use Pandas to quickly find all signals
    # A non-zero value in these arrays indicates a signal on that candle
    long_sl_signals = np.zeros(len(entry_df))
    long_tp_signals = np.zeros(len(entry_df))
    short_sl_signals = np.zeros(len(entry_df))
    short_tp_signals = np.zeros(len(entry_df))
    
    for i in range(200, len(entry_df)):
        long_analysis = strategy_engine.run_analysis_for_backtest(entry_df, None, "long", settings, i)
        if long_analysis['final_report']['recommendation'] == 'PROCEED':
            plan = long_analysis['trade_plan']
            long_sl_signals[i] = plan['stop_loss']
            long_tp_signals[i] = plan['take_profit_final']
            continue # Prioritize long signals

        short_analysis = strategy_engine.run_analysis_for_backtest(entry_df, None, "short", settings, i)
        if short_analysis['final_report']['recommendation'] == 'PROCEED':
            plan = short_analysis['trade_plan']
            short_sl_signals[i] = plan['stop_loss']
            short_tp_signals[i] = plan['take_profit_final']
    
    # Combine signals into final SL/TP arrays for the kernel
    sl_values = np.where(long_sl_signals != 0, long_sl_signals, short_sl_signals)
    tp_values = np.where(long_tp_signals != 0, long_tp_signals, short_tp_signals)

    # 2. Run the Numba kernel with pure NumPy arrays
    trades = _numba_backtest_kernel(
        entry_df['high'].values,
        entry_df['low'].values,
        sl_values,
        tp_values
    )
    
    # 3. Process the results back into a readable format
    if trades.shape[0] == 0:
        return {'total_trades': 0}

    results_df = pd.DataFrame(trades, columns=['pnl_r', 'type'])
    wins = results_df[results_df['pnl_r'] > 0]
    losses = results_df[results_df['pnl_r'] < 0]

    total_trades = len(results_df)
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    total_r = results_df['pnl_r'].sum()

    equity_r = results_df['pnl_r'].cumsum()
    peak_r = equity_r.expanding().max()
    drawdown_r = peak_r - equity_r
    max_drawdown_r = drawdown_r.max() if not drawdown_r.empty else 0
    
    profit_factor = wins['pnl_r'].sum() / abs(losses['pnl_r'].sum()) if not losses.empty and losses['pnl_r'].sum() != 0 else float('inf')

    return {
        'total_trades': total_trades, 'win_rate': win_rate, 'total_r': total_r,
        'max_drawdown_r': max_drawdown_r, 'profit_factor': profit_factor
    }