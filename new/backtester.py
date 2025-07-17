# backtester.py
# FINAL, COMPLETE, and ROBUST CORE ENGINE

import pandas as pd
from tqdm import tqdm
import os
import main as strategy_engine
import pandas_ta as ta

# --- Core Globals ---
DATA_FOLDER = "data"
REQUIRED_TIMEFRAMES = {'15m': 'entry_df', '1h': 'htf_1h', '4h': 'htf_4h'}


def load_local_data():
    """
    This function is now a generic loader for the simple backtest.
    The optimizer/realism checker scripts will load data directly.
    """
    if not os.path.exists(DATA_FOLDER):
        print(f"Error: '{DATA_FOLDER}' not found.")
        return None
    all_data = {}
    default_symbol = "ETHUSDT" 
    for timeframe, df_name in REQUIRED_TIMEFRAMES.items():
        file_path = os.path.join(DATA_FOLDER, f"{default_symbol}-{timeframe}-data.csv")
        if not os.path.exists(file_path):
            print(f"Warning: Default data file for {default_symbol} not found: '{file_path}'. This is ok if running an optimizer.")
            return {} 
        all_data[df_name] = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
    return all_data


def precompute_smc_signals(df, settings):
    """
    Uses vectorized operations to find all Sweeps.
    Is optimizer-aware and uses 'sweep_lookback' from settings.
    """
    lookback = settings['risk'].get('sweep_lookback', 20)
    
    df['lookback_high'] = df['high'].shift(1).rolling(window=lookback).max()
    df['lookback_low'] = df['low'].shift(1).rolling(window=lookback).min()
    df['is_bullish_sweep'] = (df['low'] < df['lookback_low']) & (df['close'] > df['lookback_low'])
    df['is_bearish_sweep'] = (df['high'] > df['lookback_high']) & (df['close'] < df['lookback_high'])
    return df


def precompute_indicators(data_frames, settings):
    """
    Calculates all indicators and creates a timestamp map for maximum speed.
    This version correctly accepts and passes the 'settings' dictionary.
    """
    print("Pre-computing indicators...")
    for df_name, df in data_frames.items():
        df.ta.ema(length=50, append=True); df.ta.ema(length=200, append=True)
        df.ta.adx(length=14, append=True); df.ta.atr(length=14, append=True)
        df.ta.rsi(length=14, append=True)
        df['volume_20_avg'] = df['volume'].rolling(window=20).mean()
        df['avg_body_size_20'] = (df['close'] - df['open']).abs().rolling(window=20).mean()

        if df_name == 'entry_df':
            df = precompute_smc_signals(df, settings)
            data_frames[df_name] = df
            
    print("Creating HTF timestamp map...")
    entry_df, htf_1h_df, htf_4h_df = data_frames['entry_df'], data_frames['htf_1h'], data_frames['htf_4h']
    mapped_1h = htf_1h_df.reindex(entry_df.index, method='ffill')
    mapped_4h = htf_4h_df.reindex(entry_df.index, method='ffill')
    for col in ['EMA_50', 'EMA_200']:
        entry_df[f'1h_{col}'] = mapped_1h[col]
        entry_df[f'4h_{col}'] = mapped_4h[col]
    data_frames['entry_df'] = entry_df
    print("Indicator pre-computation complete.")
    return data_frames


# In backtester.py

def generate_detailed_report(trade_list, starting_equity, settings): #<-- FIX: added 'settings' parameter
    """
    Takes a list of trade dictionaries and prints a comprehensive performance report.
    This is a core, stable function.
    """
    if not trade_list:
        print("\n--- Backtest Report --- \nNo trades were executed.")
        return
        
    # --- The rest of the function remains the same ---
    df = pd.DataFrame(trade_list)
    df['pnl_r'] = df['final_r_value']
    
    equity = starting_equity
    pnl_dollars = []
    equity_curve = [equity]

    # Use the pre-calculated risk_dollars for accurate P&L
    for _, trade in df.iterrows():
        dollar_pnl = trade['profit_loss_dollars']
        equity += dollar_pnl
        pnl_dollars.append(dollar_pnl)
        equity_curve.append(equity)

    df['profit_loss_dollars'] = pnl_dollars
    
    # --- Portfolio & Return Metrics ---
    print("\n" + "-"*40)
    print("--- Portfolio & Return Metrics (After Costs) ---")
    print("-" * 40)
    end_equity = equity
    total_return_percent = (end_equity / starting_equity - 1) * 100
    start_date, end_date = df['entry_time'].min(), df['entry_time'].max()
    duration_days = (end_date - start_date).days + 1
    duration_years = duration_days / 365.25 if duration_days > 0 else 0
    annualized_return_percent = 0.0
    if duration_years > 0 and end_equity > 0:
        annualized_return_percent = (((end_equity / starting_equity) ** (1 / duration_years)) - 1) * 100

    equity_series = pd.Series(equity_curve)
    peak_equity = equity_series.expanding().max()
    drawdown = (peak_equity - equity_series) / peak_equity
    max_drawdown_percent = drawdown.max() * 100 if not drawdown.empty else 0.0
    
    total_r = df['pnl_r'].sum()
    print(f"Starting Equity:         ${starting_equity:,.2f}")
    print(f"Ending Equity:           ${end_equity:,.2f}")
    print(f"Total Net Profit:        ${(end_equity - starting_equity):,.2f}")
    print(f"Annualized Return (CAGR):{annualized_return_percent:.2f}%")
    print(f"Max Drawdown:            {max_drawdown_percent:.2f}%")
    print(f"Total R-Gain:            {total_r:.2f}R")

    # --- Trade Statistics ---
    print("\n" + "-"*40)
    print("--- Trade Statistics (After Costs) ---")
    print("-" * 40)
    total_trades = len(df)
    wins = df[df['outcome'] == 'Win']
    losses = df[df['outcome'] == 'Loss']
    # The new logic doesn't produce 'Breakeven' outcomes anymore.
    breakevens = df[df['outcome'] == 'Breakeven'] 
    win_rate_percent = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    avg_trade_pnl_r = total_r / total_trades if total_trades > 0 else 0
    avg_win_r, avg_loss_r = wins['pnl_r'].mean() if not wins.empty else 0, losses['pnl_r'].mean() if not losses.empty else 0
    gross_profit = df[df['profit_loss_dollars'] > 0]['profit_loss_dollars'].sum()
    gross_loss = df[df['profit_loss_dollars'] < 0]['profit_loss_dollars'].sum()
    profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float('inf')
    
    print(f"Total Trades:            {total_trades}")
    print(f"Win Rate:                {win_rate_percent:.2f}% (W:{len(wins)}|L:{len(losses)}|B/E:{len(breakevens)})")
    print(f"Profit Factor:           {profit_factor:.2f}")
    print(f"Average Trade:           {avg_trade_pnl_r:.2f}R")
    print(f"Average Win / Loss (R):  +{avg_win_r:.2f}R / {avg_loss_r:.2f}R")
    print(f"Biggest Win:             ${df['profit_loss_dollars'].max():,.2f} ({df['pnl_r'].max():.2f}R)")
    print(f"Biggest Loss:            ${df['profit_loss_dollars'].min():,.2f} ({df['pnl_r'].min():.2f}R)")

    # --- Print Strategy & Cost Parameters ---
    print("\n--- Strategy Parameters Used ---")
    for k, v in settings['risk'].items():
        print(f"{k.replace('_', ' ').title()}: {v}")
    
    if 'costs' in settings:
        print("\n--- Cost Assumptions ---")
        for k, v in settings['costs'].items():
            print(f"{k.replace('_', ' ').title()}: {v}%")

    print("------------------------------------------")


# --- The Corrected Backtest Logic with Quiet Mode ---
def run_backtest_logic(data_frames, settings, progress_bar=None, start_index=200, quiet_mode=False):
    entry_df, trades, active_trade = data_frames['entry_df'], [], None
    equity = 10000
    
    # Extract cost settings once at the beginning
    fee_percent = settings.get('costs', {}).get('trading_fee_percent', 0)
    slippage_percent = settings.get('costs', {}).get('slippage_percent', 0)

    i = start_index
    while i < len(entry_df) - 1:
        if progress_bar: progress_bar.update(1)

        # --- Management of an Active Trade ---
        if active_trade:
            current_candle = entry_df.iloc[i]
            outcome, final_pnl = None, 0

            # --- Check for Partial Take-Profit (TP1) ---
            is_tp1_hit = active_trade.get('status') == 'active' and (
                (active_trade['type'] == 'long' and current_candle['high'] >= active_trade['take_profit_1']) or
                (active_trade['type'] == 'short' and current_candle['low'] <= active_trade['take_profit_1'])
            )
            
            if is_tp1_hit:
                partial_exit_price = active_trade['take_profit_1']
                partial_position_value = (active_trade['position_size'] / 2) * partial_exit_price
                partial_exit_fee = partial_position_value * (fee_percent / 100)
                
                partial_profit = 0.5 * (active_trade['risk_dollars'] * settings['risk']['partial_tp_rr'])
                net_partial_profit = partial_profit - partial_exit_fee
                
                active_trade['partial_profit_booked'] = net_partial_profit
                active_trade['partial_exit_fee'] = partial_exit_fee
                active_trade['stop_loss'] = active_trade['entry_price']
                active_trade['status'] = 'breakeven'

                if progress_bar and not quiet_mode:
                    print(f"\n>>> PARTIAL PROFIT: Net profit of ${net_partial_profit:,.2f} secured (after ${partial_exit_fee:,.2f} fee).")

            # --- Check for Trade-Ending Conditions ---
            is_sl_hit = (active_trade['type'] == 'long' and current_candle['low'] <= active_trade['stop_loss']) or \
                        (active_trade['type'] == 'short' and current_candle['high'] >= active_trade['stop_loss'])
            
            is_tp_final_hit = (active_trade['type'] == 'long' and current_candle['high'] >= active_trade['take_profit_final']) or \
                              (active_trade['type'] == 'short' and current_candle['low'] <= active_trade['take_profit_final'])
                              
            if is_sl_hit:
                final_exit_price = active_trade['stop_loss']
                if active_trade.get('status') == 'breakeven':
                    # Stopped at entry after partials. Close the second half.
                    remaining_pos_value = (active_trade['position_size'] / 2) * final_exit_price
                    final_exit_fee = remaining_pos_value * (fee_percent / 100)
                    
                    outcome = "Win" # Net positive PnL is still a win
                    final_pnl = active_trade.get('partial_profit_booked', 0) - active_trade['entry_fee'] - final_exit_fee
                else:
                    # Full loss. Close the entire position.
                    full_pos_value = active_trade['position_size'] * final_exit_price
                    final_exit_fee = full_pos_value * (fee_percent / 100)
                    
                    outcome = "Loss"
                    final_pnl = -active_trade['risk_dollars'] - active_trade['entry_fee'] - final_exit_fee
            
            elif is_tp_final_hit:
                outcome = "Win"
                final_exit_price = active_trade['take_profit_final']
                gross_profit = active_trade['risk_dollars'] * active_trade['risk_reward_ratio']

                if active_trade.get('status') == 'breakeven':
                    # Final TP hit after partials were taken. Close the second half.
                    remaining_pos_value = (active_trade['position_size'] / 2) * final_exit_price
                    final_exit_fee = remaining_pos_value * (fee_percent / 100)
                    total_fees = active_trade['entry_fee'] + active_trade.get('partial_exit_fee', 0) + final_exit_fee
                    final_pnl = gross_profit - total_fees
                else:
                    # Price went straight to final TP. Close the entire position.
                    full_pos_value = active_trade['position_size'] * final_exit_price
                    final_exit_fee = full_pos_value * (fee_percent / 100)
                    total_fees = active_trade['entry_fee'] + final_exit_fee
                    final_pnl = gross_profit - total_fees

            # --- Record the final trade outcome ---
            if outcome:
                finalized_trade = active_trade.copy()
                finalized_trade.update({'outcome': outcome, 'profit_loss_dollars': final_pnl})

                if finalized_trade['risk_dollars'] > 0:
                    finalized_trade['final_r_value'] = final_pnl / finalized_trade['risk_dollars']
                else:
                    finalized_trade['final_r_value'] = 0
                trades.append(finalized_trade)
                active_trade = None
            
            i += 1
            continue

        # --- Search for a New Trade Entry ---
        analysis = strategy_engine.run_analysis_for_backtest(entry_df, None, "long", settings, i)
        if analysis['final_report']['recommendation'] != 'PROCEED':
            analysis = strategy_engine.run_analysis_for_backtest(entry_df, None, "short", settings, i)

        if analysis['final_report']['recommendation'] == 'PROCEED':
            plan = analysis['trade_plan']
            ideal_entry = entry_df.iloc[i + 1]['open']
            trade_type = "long" if ideal_entry > plan['stop_loss'] else "short"

            # --- Apply Slippage to Entry Price ---
            slippage_mult = 1 + (slippage_percent / 100) if trade_type == 'long' else 1 - (slippage_percent / 100)
            entry_price = ideal_entry * slippage_mult
            
            # --- Recalculate everything based on the slipped entry price ---
            risk_per_trade_dollars = equity * (settings['risk']['risk_per_trade_percent'] / 100.0)
            price_risk = abs(entry_price - plan['stop_loss'])
            
            if price_risk < 1e-8:
                i += 1; continue
                
            position_size = risk_per_trade_dollars / price_risk
            entry_fee = (position_size * entry_price) * (fee_percent / 100) # Calculate entry fee
            
            if (position_size * entry_price) > (equity * 10):
                i += 1; continue
            
            final_rr = abs(plan['take_profit_final'] - entry_price) / price_risk
            if final_rr < settings['risk']['min_rr']:
                i += 1; continue
            
            # --- Create the new trade object with cost data ---
            active_trade = {
                'entry_time': entry_df.index[i + 1], 'type': trade_type,
                'entry_price': entry_price, 'stop_loss': plan['stop_loss'],
                'take_profit_1': plan['take_profit_1'], 'take_profit_final': plan['take_profit_final'],
                'risk_reward_ratio': round(final_rr, 2), 'status': 'active',
                'reason': analysis['final_report']['reason'],
                'risk_dollars': risk_per_trade_dollars, 'position_size': position_size,
                'entry_fee': entry_fee, 'partial_profit_booked': 0,
            }

            if progress_bar and not quiet_mode:
                print(f"\n>>> TRADE ENTERED: {active_trade['type'].upper()} on {active_trade['entry_time'].date()}. "
                      f"Slipped Entry: {entry_price:.4f}, Entry Fee: ${entry_fee:,.2f}")
            i += 1
        i += 1
    return trades

def run_simple_backtest():
    print("--- Running a single, simple backtest with default settings ---")
    data_frames = load_local_data()
    if not data_frames:
        print("Could not load default data.")
        return
        
    settings = strategy_engine.SMC_V2_SETTINGS
    data_frames = precompute_indicators(data_frames, settings)
    
    start_index = 200
    total_candles = len(data_frames['entry_df']) - 200
    progress_bar = tqdm(total=total_candles, desc="Simulating Trades")
    
    trade_list = run_backtest_logic(data_frames, settings, progress_bar, start_index)
    progress_bar.close()

    print("\n\n--- Backtest Summary ---")
    if trade_list:
        print(f"Data Period: {trade_list[0]['entry_time'].date()} to {trade_list[-1]['entry_time'].date()}")
        print("------------------------------------------")
        # FIX: Pass the entire 'settings' object instead of one value from it
        generate_detailed_report(trade_list, 10000, settings)
        
if __name__ == "__main__":
    run_simple_backtest()