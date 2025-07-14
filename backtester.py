# backtester.py

import pandas as pd
from tqdm import tqdm
import os
import main as strategy_engine
import pandas_ta as ta

DATA_FOLDER = "data"
REQUIRED_TIMEFRAMES = {'15m': 'entry_df', '1h': 'htf_1h', '4h': 'htf_4h'}


def load_local_data():
    if not os.path.exists(DATA_FOLDER):
        print(f"Error: '{DATA_FOLDER}' not found.")
        return None
    all_data = {}
    for timeframe, df_name in REQUIRED_TIMEFRAMES.items():
        file_path = os.path.join(DATA_FOLDER, f"BTCUSDT-{timeframe}-data.csv")
        if not os.path.exists(file_path):
            print(f"Error: Data file not found at '{file_path}'.")
            return None
        all_data[df_name] = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
    return all_data


def precompute_smc_signals(df):
    """
    Uses vectorized operations to find all Sweeps and FVGs at once. This is extremely fast.
    """
    print("Pre-computing SMC signals (Sweeps and FVGs)...")
    lookback = 20
    df['lookback_high'] = df['high'].shift(1).rolling(window=lookback).max()
    df['lookback_low'] = df['low'].shift(1).rolling(window=lookback).min()
    df['is_bullish_sweep'] = (df['low'] < df['lookback_low']) & (df['close'] > df['lookback_low'])
    df['is_bearish_sweep'] = (df['high'] > df['lookback_high']) & (df['close'] < df['lookback_high'])
    return df


def precompute_indicators(data_frames):
    """
    Calculates all indicators and creates a timestamp map for maximum speed.
    """
    print("Pre-computing indicators for all timeframes...")
    for df_name, df in data_frames.items():
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=200, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.rsi(length=14, append=True)
        df['volume_20_avg'] = df['volume'].rolling(window=20).mean()
        df['avg_body_size_20'] = (df['close'] - df['open']).abs().rolling(window=20).mean()

        if df_name == 'entry_df':
            df = precompute_smc_signals(df)
            data_frames[df_name] = df

    print("Creating HTF timestamp map for high-speed lookups...")
    entry_df = data_frames['entry_df']
    htf_1h_df = data_frames['htf_1h']
    htf_4h_df = data_frames['htf_4h']

    mapped_1h = htf_1h_df.reindex(entry_df.index, method='ffill')
    mapped_4h = htf_4h_df.reindex(entry_df.index, method='ffill')

    for col in ['EMA_50', 'EMA_200']:
        entry_df[f'1h_{col}'] = mapped_1h[col]
        entry_df[f'4h_{col}'] = mapped_4h[col]

    data_frames['entry_df'] = entry_df
    print("Indicator pre-computation complete.")
    return data_frames


def calculate_portfolio_metrics(trades, starting_equity=10000, risk_per_trade_percent=1.0):
    if not trades:
        return {}
    equity = starting_equity
    peak_equity = starting_equity
    max_drawdown = 0.0

    for trade in trades:
        risk_amount = equity * (risk_per_trade_percent / 100.0)
        profit_loss = trade.get('final_r_value', 0) * risk_amount
        equity += profit_loss
        if equity > peak_equity:
            peak_equity = equity
        drawdown = (peak_equity - equity) / peak_equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    start_date = trades[0]['entry_time']
    end_date = trades[-1]['entry_time']
    duration_days = (end_date - start_date).days + 1
    duration_years = duration_days / 365.25 if duration_days > 0 else 0
    total_return_percent = (equity / starting_equity - 1) * 100
    annualized_return_percent = (((equity / starting_equity) ** (
                1 / duration_years)) - 1) * 100 if duration_years > 0 else 0
    return {
        "Starting Equity": f"${starting_equity:,.2f}", "Ending Equity": f"${equity:,.2f}",
        "Total Return %": f"{total_return_percent:.2f}%", "Annualized Return %": f"{annualized_return_percent:.2f}%",
        "Max Drawdown %": f"{max_drawdown * 100:.2f}%",
    }


def run_backtest_logic(data_frames, settings, progress_bar=None, start_index=200):
    entry_df = data_frames['entry_df']
    trades = []
    active_trade = None

    i = start_index
    while i < len(entry_df) - 1:
        if progress_bar: progress_bar.update(1)

        if active_trade is not None:
            current_candle = entry_df.iloc[i]
            outcome = None
            if (active_trade['type'] == 'long' and current_candle['low'] <= active_trade['stop_loss']) or \
                    (active_trade['type'] == 'short' and current_candle['high'] >= active_trade['stop_loss']):
                outcome = "Breakeven" if active_trade.get('status') == 'breakeven' else "Loss"
            elif active_trade.get('status') == 'active' and \
                    ((active_trade['type'] == 'long' and current_candle['high'] >= active_trade['take_profit_1']) or \
                     (active_trade['type'] == 'short' and current_candle['low'] <= active_trade['take_profit_1'])):
                if progress_bar: print(
                    f"\n>>> [PARTIAL PROFIT] {active_trade['type'].upper()} @ {current_candle.name.date()} | TP1 Hit. Moving SL to Breakeven.")
                active_trade['status'] = 'breakeven'
                active_trade['stop_loss'] = active_trade['entry_price']
            elif (active_trade['type'] == 'long' and current_candle['high'] >= active_trade['take_profit_final']) or \
                    (active_trade['type'] == 'short' and current_candle['low'] <= active_trade['take_profit_final']):
                outcome = "Win"

            if outcome is not None:
                final_r_value = 0
                if outcome == "Win":
                    final_r_value = active_trade['risk_reward_ratio']
                elif outcome == "Loss":
                    final_r_value = -1.0
                elif outcome == "Breakeven":
                    final_r_value = 0.5 * settings['risk']['partial_tp_rr']
                finalized_trade = active_trade.copy()
                finalized_trade['outcome'] = outcome
                finalized_trade['final_r_value'] = final_r_value
                trades.append(finalized_trade)
                active_trade = None
            i += 1
            continue

        # --- THE FIX IS HERE: The function call now has the correct 5 arguments ---
        analysis = strategy_engine.run_analysis_for_backtest(entry_df, None, "long", settings, i)
        if analysis['final_report']['recommendation'] != 'PROCEED':
            analysis = strategy_engine.run_analysis_for_backtest(entry_df, None, "short", settings, i)

        if analysis['final_report']['recommendation'] == 'PROCEED':
            plan = analysis['trade_plan']
            entry_price_actual = entry_df.iloc[i + 1]['open']
            trade_type = "long" if plan['entry_price'] > plan['stop_loss'] else "short"
            new_sl = plan['stop_loss']
            if abs(entry_price_actual - new_sl) <= 1e-6:
                i += 1
                continue
            final_rr = abs(plan['take_profit_final'] - entry_price_actual) / abs(entry_price_actual - new_sl)
            active_trade = {
                'entry_time': entry_df.index[i + 1], 'type': trade_type, 'entry_price': entry_price_actual,
                'stop_loss': new_sl, 'take_profit_1': plan['take_profit_1'],
                'take_profit_final': plan['take_profit_final'],
                'risk_reward_ratio': round(final_rr, 2), 'status': 'active',
                'reason': analysis['final_report']['reason']
            }
            if progress_bar: print(
                f"\n>>> [TRADE ENTERED] {active_trade['type'].upper()} @ {active_trade['entry_time'].date()} | Reason: {active_trade['reason']}")
            i += 1
        i += 1

    return trades


def run_simple_backtest():
    print("--- Running a single, simple backtest with default settings ---")
    print("Loading local data...")
    data_frames = load_local_data()
    if data_frames is None: return

    data_frames = precompute_indicators(data_frames)
    entry_df = data_frames['entry_df']
    settings = strategy_engine.SMC_V2_SETTINGS
    start_index = 200
    total_candles = len(entry_df) - start_index
    progress_bar = tqdm(total=total_candles, desc="Simulating Trades", unit="candle")

    trade_list = run_backtest_logic(data_frames, settings, progress_bar, start_index)
    progress_bar.close()

    if not trade_list:
        print("\n\n--- Backtest Summary ---")
        print("No trades were executed.")
        return

    wins = len([t for t in trade_list if t['outcome'] == 'Win'])
    losses = len([t for t in trade_list if t['outcome'] == 'Loss'])
    breakevens = len([t for t in trade_list if t['outcome'] == 'Breakeven'])
    trades_count = len(trade_list)
    win_rate = (wins / trades_count * 100) if trades_count > 0 else 0
    total_r = sum(t.get('final_r_value', 0) for t in trade_list)

    print("\n\n--- Backtest Summary ---")

    print(f"Data Period: {entry_df.index[start_index].date()} to {entry_df.index[-1].date()}")
    print("------------------------------------------")
    portfolio_summary = calculate_portfolio_metrics(trade_list)
    for key, value in portfolio_summary.items():
        print(f"{key}: {value}")
    print("------------------------------------------")
    print(f"Total Trades: {trades_count}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Wins: {wins} | Losses: {losses} | Breakevens: {breakevens}")
    print(f"Total R-Gain: {total_r:.2f}R")
    print("------------------------------------------")
    print("\n--- Strategy Parameters Used ---")
    print(f"Min Risk/Reward Ratio: {settings['risk']['min_rr']}")
    print(f"ATR Buffer Multiplier for SL: {settings['risk']['atr_buffer_multiplier']}")
    print(f"Partial Profit TP Target: {settings['risk']['partial_tp_rr']}R")
    print("------------------------------------------")


if __name__ == "__main__":
    run_simple_backtest()