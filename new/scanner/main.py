# main.py

import pandas as pd
from typing import Dict, Any, Optional
import indicators

def check_zone_overlap(candle_low, candle_high, zone_bottom, zone_top) -> bool:
    """Checks if a candle's body overlaps with a given price zone."""
    if zone_bottom is None or zone_top is None: return False
    # Ensure zone bottom is always less than zone top
    if zone_bottom > zone_top: zone_bottom, zone_top = zone_top, zone_bottom
    return candle_high >= zone_bottom and candle_low <= zone_top

def get_confluence_score_optimized(entry_df: pd.DataFrame, current_index: int, trade_type: str) -> Dict:
    """A simplified, fast filter for trade confluence."""
    entry_candle = entry_df.iloc[current_index]
    reasons = []

    # 1. Volume Filter: Must have above-average volume
    if entry_candle['volume'] > entry_candle['volume_20_avg'] * 1.5:
        reasons.append("Volume Spike")
    else:
        return {"recommendation": "AVOID", "reason": "Failed: Low Volume"}

    # 2. RSI Filter: Should not be in extreme overbought/oversold territory already
    rsi_value = entry_candle['RSI_14']
    if pd.isna(rsi_value):
        return {"recommendation": "AVOID", "reason": "RSI is NaN"}

    if trade_type == "long" and rsi_value < 70: # Avoid buying if already overbought
        reasons.append(f"RSI {rsi_value:.1f}")
    elif trade_type == "short" and rsi_value > 30: # Avoid shorting if already oversold
        reasons.append(f"RSI {rsi_value:.1f}")
    else:
        return {"recommendation": "AVOID", "reason": f"Failed: RSI State ({rsi_value:.1f})"}

    final_reason = " | ".join(reasons)
    return {"recommendation": "PROCEED", "reason": final_reason}


def run_analysis_for_backtest(entry_df: pd.DataFrame, ltf_df: Optional[pd.DataFrame],
                              trade_type: str, settings: Dict[str, Any], i: int):
    """
    The core logic of the trading strategy. Analyzes a single candle (`i`) for a trade signal.
    """
    default_response = {'final_report': {'recommendation': 'AVOID'}, 'trade_plan': {}}
    if i < 200: return default_response

    entry_candle = entry_df.iloc[i]
    
    # --- 1. TREND FILTER ---
    # The 4-hour trend must align with the trade direction.
    # The 1-hour trend (slope) must confirm recent momentum.
    is_4h_bias_long = entry_candle['4h_EMA_50'] > entry_candle['4h_EMA_200']
    is_4h_bias_short = entry_candle['4h_EMA_50'] < entry_candle['4h_EMA_200']
    
    # Calculate slope of the 1-hour 50 EMA
    ema_50_1h_slope = entry_candle['1h_EMA_50'] - entry_df['1h_EMA_50'].iloc[i - 5]

    if trade_type == 'long' and not (is_4h_bias_long and ema_50_1h_slope > 0):
        return default_response
    if trade_type == 'short' and not (is_4h_bias_short and ema_50_1h_slope < 0):
        return default_response

    # --- 2. SMC (Smart Money Concepts) FILTER ---
    # We look for a liquidity sweep followed by a reaction candle.
    poi_zone = None
    # Look back up to 12 candles for a liquidity sweep setup
    for j in range(2, 12):
        setup_idx = i - j
        if setup_idx < 0: break
        setup_candle = entry_df.iloc[setup_idx]

        is_sweep = setup_candle['is_bullish_sweep'] if trade_type == 'long' else setup_candle['is_bearish_sweep']
        
        # If a sweep candle is found...
        if is_sweep:
            # The "Point of Interest" (POI) is the body of the sweep candle itself.
            poi_zone = (min(setup_candle['open'], setup_candle['close']), max(setup_candle['open'], setup_candle['close']))
            
            # Did the current entry candle touch this POI zone?
            if check_zone_overlap(entry_candle['low'], entry_candle['high'], poi_zone[0], poi_zone[1]):
                break # Valid setup found, exit the loop
            else:
                poi_zone = None # This sweep was not respected, reset and keep looking
            
    if poi_zone is None:
        return default_response # No valid sweep and reaction found
    
    # --- 3. CONFLUENCE & FINAL CHECK ---
    # Run final checks on the entry candle itself
    final_report = get_confluence_score_optimized(entry_df, i, trade_type)
    if final_report['recommendation'] == 'AVOID':
        return default_response

    # --- 4. TRADE PLAN GENERATION ---
    # If all filters pass, construct the trade plan
    try:
        structure = indicators.find_recent_swing_points(entry_df, i)
        recent_low, recent_high = structure.get('recent_low'), structure.get('recent_high')
        if not recent_low or not recent_high: return default_response

        entry_price = entry_candle['close']
        atr = entry_candle['ATRr_14']
        if pd.isna(atr) or atr == 0: return default_response

        # Define Stop Loss based on volatility and structure
        atr_buffer = atr * settings['risk']['atr_buffer_multiplier']
        sl = (recent_low - atr_buffer) if trade_type == 'long' else (recent_high + atr_buffer)

        # Define Take Profit based on the *next* major structural point
        tp = recent_high if trade_type == 'long' else recent_low

        # Basic validation
        if abs(entry_price - sl) <= 1e-8: return default_response # Avoid division by zero
        
        # Calculate Risk:Reward ratio
        rr = abs(tp - entry_price) / abs(entry_price - sl)
        
        # --- Final Filters on the Trade Plan ---
        if rr < settings['risk']['min_rr']:
            return default_response

        # Profit target must be meaningful (e.g., at least 3x ATR away)
        profit_distance = abs(tp - entry_price)
        if profit_distance < (3 * atr):
            return default_response

        # --- Success! Return the full plan ---
        trade_plan = {
            'entry_price': entry_price,
            'stop_loss': sl,
            'take_profit_final': tp,
            'risk_reward_ratio': round(rr, 2)
        }
        return {'final_report': final_report, 'trade_plan': trade_plan}

    except Exception:
        # Catch any unexpected errors during plan generation
        return default_response