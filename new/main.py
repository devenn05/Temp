# main.py
# FINAL CORRECTED VERSION

import pandas as pd
from typing import Dict, Any, Optional
import indicators

# --- Baseline Settings ---
# The optimizer will override these values during its runs.
SMC_V2_SETTINGS = {
    "mode_name": "SMC V2.1",
    "entry_timeframe": '15m',
    "score_threshold": 100,
    "risk": {
        "min_rr": 3.5,
"risk_per_trade_percent": 2.0,
"atr_buffer_multiplier": 1.2,
"partial_tp_rr": 1.1,
        "adx_threshold": 22
    },
    "costs": {
        "trading_fee_percent": 0.04,  # Standard taker fee (0.04%) on many exchanges
        "slippage_percent": 0.01       # A small slippage of 0.01%
    }
}


def check_zone_overlap(candle_low, candle_high, zone_bottom, zone_top) -> bool:
    if zone_bottom is None or zone_top is None: return False
    if zone_bottom > zone_top: zone_bottom, zone_top = zone_top, zone_bottom
    return candle_high >= zone_bottom and candle_low <= zone_top

def get_confluence_score_optimized(entry_df: pd.DataFrame, current_index: int, trade_type: str, settings: Dict) -> Dict:
    entry_candle, reasons = entry_df.iloc[current_index], []

    if entry_candle['volume'] > entry_candle['volume_20_avg'] * 1.5:
        reasons.append("Volume Spike")
    else: return {"recommendation": "AVOID", "reason": "Failed: Low Volume"}
    
    rsi_value = entry_candle['RSI_14']
    if pd.isna(rsi_value): return {"recommendation": "AVOID", "reason": "RSI is NaN"}

    if (trade_type == "long" and rsi_value < 65) or (trade_type == "short" and rsi_value > 35):
        reasons.append(f"RSI {rsi_value:.1f}")
    else: return {"recommendation": "AVOID", "reason": f"Failed: RSI State ({rsi_value:.1f})"}
    
    final_reason = " | ".join(reasons)
    return {"recommendation": "PROCEED", "reason": final_reason}


def run_analysis_for_backtest(entry_df: pd.DataFrame, ltf_df: Optional[pd.DataFrame],
                              trade_type: str, settings: Dict[str, Any], i: int):
    default_response = {'final_report': {'recommendation': 'AVOID'}, 'trade_plan': {}}
    if i < 200: return default_response

    entry_candle = entry_df.iloc[i]
    
    is_4h_bias_long = entry_candle['4h_EMA_50'] > entry_candle['4h_EMA_200']
    is_4h_bias_short = entry_candle['4h_EMA_50'] < entry_candle['4h_EMA_200']
    ema_50_1h_slope = entry_candle['1h_EMA_50'] - entry_df['1h_EMA_50'].iloc[i - 5]

    if (trade_type == 'long' and not (is_4h_bias_long and ema_50_1h_slope > 0)) or \
       (trade_type == 'short' and not (is_4h_bias_short and ema_50_1h_slope < 0)):
        return default_response
        
    poi_zone = None
    for j in range(2, 12):
        setup_idx = i - j
        if setup_idx < 0: break
        setup_candle = entry_df.iloc[setup_idx]
        is_sweep = setup_candle['is_bullish_sweep'] if trade_type == 'long' else setup_candle['is_bearish_sweep']
        if not is_sweep: continue
        
        poi_zone = (min(setup_candle['open'], setup_candle['close']), max(setup_candle['open'], setup_candle['close']))
        if check_zone_overlap(entry_candle['low'], entry_candle['high'], poi_zone[0], poi_zone[1]):
            break
        else:
            poi_zone = None
            
    if poi_zone is None: return default_response
    
    final_report = get_confluence_score_optimized(entry_df, i, trade_type, settings)
    if final_report['recommendation'] == 'AVOID': return default_response

    try:
        structure = indicators.find_recent_swing_points(entry_df, i)
        recent_low, recent_high = structure.get('recent_low'), structure.get('recent_high')
        if not recent_low or not recent_high: return default_response

        entry_price = entry_candle['close']
        atr = entry_candle['ATRr_14']
        if pd.isna(atr) or atr == 0: return default_response

        atr_buffer = atr * settings['risk']['atr_buffer_multiplier']
        sl = (recent_low - atr_buffer) if trade_type == 'long' else (recent_high + atr_buffer)
        tp = recent_high if trade_type == 'long' else recent_low

        if abs(entry_price - sl) <= 1e-8: return default_response
        rr = abs(tp - entry_price) / abs(entry_price - sl)
        
        if rr < settings['risk']['min_rr']: return default_response

        # --- Profit Potential Filter ---
        profit_distance = abs(tp - entry_price)
        if profit_distance < (3 * atr):
            return default_response

        tp1_dist = abs(entry_price - sl) * settings['risk']['partial_tp_rr']
        tp1 = entry_price + tp1_dist if trade_type == 'long' else entry_price - tp1_dist
        
        # --- THE CORRECTED LINE IS HERE ---
        trade_plan = {'entry_price': entry_price, 'stop_loss': sl, 'take_profit_1': tp1, 'take_profit_final': tp, 'risk_reward_ratio': round(rr, 2)}
        return {'final_report': final_report, 'trade_plan': trade_plan}

    except Exception:
        pass
    return default_response