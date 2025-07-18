# indicators.py
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
import random
# Using python's built-in Type Hinting for cleaner code
from typing import Dict, Any, Union, Optional, Tuple
from scipy.signal import argrelextrema
import pandas_ta as ta

SMC_DEBUG_PRINTED_ONCE = False

# --- DEPENDENCIES ---
# This file will now depend on our new api_client.
import api_client


# === START OF PHASE 1 UPGRADE: MARKET REGIME FILTER ===

def get_market_regime_advanced(df: pd.DataFrame) -> tuple[str, str]:
    if df is None or len(df) < 200: return "Ranging", "Sideways"
    try:
        ema_50 = df.ta.ema(length=50)
        ema_200 = df.ta.ema(length=200)
        if ema_50 is None or ema_200 is None or ema_50.isna().all() or ema_200.isna().all():
             return "Ranging", "Sideways"
        is_bullish = ema_50.iloc[-1] > ema_200.iloc[-1]
        is_bearish = ema_50.iloc[-1] < ema_200.iloc[-1]
        adx_indicator = df.ta.adx(length=14)
        if adx_indicator is None or adx_indicator.empty: return "Ranging", "Sideways"
        is_trending = adx_indicator.iloc[-1]['ADX_14'] > 20
        if is_trending:
            if is_bullish: return "Trending", "Bullish"
            if is_bearish: return "Trending", "Bearish"
        return "Ranging", "Sideways"
    except Exception:
        return "Ranging", "Sideways"
# === HELPER FUNCTIONS ===

def format_price(price):
    try:
        price = float(price)
        if pd.isna(price) or price == 0: return "0.00"
        abs_price = abs(price)
        if abs_price < 0.0001: return f"{price:.8f}".rstrip('0').rstrip('.')
        if abs_price < 1: return f"{price:.4f}".rstrip('0').rstrip('.')
        if abs_price < 1000: return f"{price:.2f}".rstrip('0').rstrip('.')
        return f"{int(price):,}"
    except (ValueError, TypeError):
        return str(price)


def find_key_levels(df, lookback=50):
    if df is None or len(df) < lookback:
        return {'recent_high': None, 'recent_low': None}
    df_slice = df.tail(lookback).copy()
    return {'recent_high': df_slice['high'].max(), 'recent_low': df_slice['low'].min()}


# === ALL INDICATOR VERDICT FUNCTIONS (UPGRADED FOR ROBUSTNESS) ===

WHALE_TRADE_THRESHOLD = 5


def adx_verdict(df: pd.DataFrame | None, trade_type: str) -> dict:
    if df is None or df.empty: return {"verdict": "error", "explanation": "ADX: No data provided."}
    try:
        adx_df = df.ta.adx()
        if adx_df is None or adx_df.empty: return {"verdict": "error", "explanation": "ADX calculation failed."}
        latest = adx_df.iloc[-1]
        adx, plus_di, minus_di = latest['ADX_14'], latest['DMP_14'], latest['DMN_14']
        verdict = "no"
        explanation = f"ADX: {format_price(adx)} (Weak Trend)"
        if pd.isna(adx) or pd.isna(plus_di) or pd.isna(minus_di): return {"verdict": "error",
                                                                          "explanation": "ADX contains NaN values."}
        if adx > 25:
            explanation = f"ADX: {format_price(adx)} (Strong Trend)"
            if trade_type == "long" and plus_di > minus_di:
                verdict = "yes"
            elif trade_type == "short" and minus_di > plus_di:
                verdict = "yes"
        return {"verdict": verdict, "explanation": explanation}
    except Exception as e:
        return {"verdict": "error", "explanation": f"ADX: An error occurred - {e}"}


def ema_verdict(df: pd.DataFrame | None, trade_type: str) -> dict:
    if df is None or len(df) < 200: return {"verdict": "error", "explanation": "EMA: Insufficient data."}
    try:
        ema50 = df.ta.ema(length=50).iloc[-1]
        ema200 = df.ta.ema(length=200).iloc[-1]
        if pd.isna(ema50) or pd.isna(ema200): return {"verdict": "error",
                                                      "explanation": "EMA calculation resulted in NaN."}
        verdict = "no"
        explanation = f"EMA50: {format_price(ema50)}, EMA200: {format_price(ema200)}"
        if trade_type == "long" and ema50 > ema200:
            verdict, explanation = "yes", explanation + " (Bullish Crossover)"
        elif trade_type == "short" and ema50 < ema200:
            verdict, explanation = "yes", explanation + " (Bearish Crossover)"
        return {"verdict": verdict, "explanation": explanation}
    except Exception as e:
        return {"verdict": "error", "explanation": f"EMA: An error occurred - {e}"}


def netflow_verdict(ticker_data: dict | None, trade_type: str) -> dict:
    if not ticker_data: return {"verdict": "error", "explanation": "Could not fetch 24hr stats."}
    try:
        asset_volume = float(ticker_data.get("volume", 0))
        netflow = asset_volume * random.uniform(-0.1, 0.1)  # Simulated netflow
        verdict = "no"
        explanation = f"Approx. Net Flow: {format_price(netflow)}"
        if trade_type == "long" and netflow < 0:
            verdict, explanation = "yes", explanation + " (Outflow suggests accumulation)"
        elif trade_type == "short" and netflow > 0:
            verdict, explanation = "yes", explanation + " (Inflow suggests distribution)"
        return {"verdict": verdict, "explanation": explanation}
    except Exception as e:
        return {"verdict": "error", "explanation": f"Netflow: An error occurred - {e}"}


def sentiment_verdict(fng_value: int | None, trade_type: str) -> dict:
    if not fng_value: return {"verdict": "error", "explanation": "Could not fetch F&G Index."}
    verdict = "no"
    if fng_value <= 25:
        explanation = f"F&G Index: {fng_value} (Extreme Fear)"
        if trade_type == "long": verdict = "yes"
    elif fng_value >= 75:
        explanation = f"F&G Index: {fng_value} (Extreme Greed)"
        if trade_type == "short": verdict = "yes"
    else:
        explanation = f"F&G Index: {fng_value} (Neutral)"
    return {"verdict": verdict, "explanation": explanation}


def miner_verdict() -> dict:
    try:
        netflow = random.uniform(-1000, 1000)
        explanation = f"Simulated Miner Flow: {format_price(netflow)}"
        verdict = "yes" if netflow < 0 else "no"
        explanation += " (Holding)" if verdict == "yes" else " (Selling)"
        return {"verdict": verdict, "explanation": explanation}
    except Exception as e:
        return {"verdict": "error", "explanation": f"Miner: An error occurred - {e}"}


def macd_verdict(df: pd.DataFrame | None, trade_type: str) -> dict:
    if df is None or df.empty: return {"verdict": "error", "explanation": "MACD: No data provided."}
    try:
        macd_df = df.ta.macd()
        if macd_df is None or macd_df.empty: return {"verdict": "error", "explanation": "MACD calculation failed."}
        latest = macd_df.iloc[-1]
        macd_line, signal_line = latest['MACD_12_26_9'], latest['MACDs_12_26_9']
        if pd.isna(macd_line) or pd.isna(signal_line): return {"verdict": "error", "explanation": "MACD contains NaN."}
        verdict = "no"
        explanation = f"MACD: {format_price(macd_line)}, Signal: {format_price(signal_line)}"
        if trade_type == "long" and macd_line > signal_line:
            verdict, explanation = "yes", explanation + " (Bullish Crossover)"
        elif trade_type == "short" and macd_line < signal_line:
            verdict, explanation = "yes", explanation + " (Bearish Crossover)"
        return {"verdict": verdict, "explanation": explanation}
    except Exception as e:
        return {"verdict": "error", "explanation": f"MACD: An error occurred - {e}"}


def rsi_verdict(df: pd.DataFrame, trade_type: str) -> dict:
    if df is None or df.empty: return {"verdict": "error", "explanation": "RSI: No data."}
    try:
        rsi = df.ta.rsi().iloc[-1]
        if pd.isna(rsi): return {"verdict": "error", "explanation": "RSI is NaN."}
        verdict = "no"
        if rsi < 30 and trade_type == "long": verdict = "yes"; explanation = f"RSI: {rsi:.2f} (Oversold)"
        elif rsi > 70 and trade_type == "short": verdict = "yes"; explanation = f"RSI: {rsi:.2f} (Overbought)"
        else: explanation = f"RSI: {rsi:.2f} (Neutral)"
        return {"verdict": verdict, "explanation": explanation}
    except Exception:
        return {"verdict": "error", "explanation": "RSI Error"}


def stoch_rsi_verdict(df: pd.DataFrame | None, trade_type: str) -> dict:
    if df is None or df.empty: return {"verdict": "error", "explanation": "StochRSI: No data provided."}
    try:
        stoch_rsi_series = df.ta.stochrsi()
        if stoch_rsi_series is None or stoch_rsi_series.empty: return {"verdict": "error",
                                                                       "explanation": "StochRSI calculation failed."}
        stoch_rsi = stoch_rsi_series.iloc[-1]['STOCHRSIk_14_14_3_3']
        if pd.isna(stoch_rsi): return {"verdict": "error", "explanation": "StochRSI calculation is NaN."}
        verdict = "no"
        if stoch_rsi < 20:
            explanation = f"StochRSI: {format_price(stoch_rsi)} (Oversold)"
            if trade_type == "long": verdict = "yes"
        elif stoch_rsi > 80:
            explanation = f"StochRSI: {format_price(stoch_rsi)} (Overbought)"
            if trade_type == "short": verdict = "yes"
        else:
            explanation = f"StochRSI: {format_price(stoch_rsi)} (Neutral)"
        return {"verdict": verdict, "explanation": explanation}
    except Exception as e:
        return {"verdict": "error", "explanation": f"StochRSI: An error occurred - {e}"}


def whale_verdict(order_book: dict | None, trade_type: str, market_type: str) -> dict:
    if not order_book: return {"verdict": "error", "explanation": "Could not fetch order book."}
    try:
        bids = [(float(p), float(q)) for p, q in order_book.get("bids", [])]
        asks = [(float(p), float(q)) for p, q in order_book.get("asks", [])]
        large_buys = sum(q for p, q in bids if q > WHALE_TRADE_THRESHOLD)
        large_sells = sum(q for p, q in asks if q > WHALE_TRADE_THRESHOLD)
        verdict = "no"
        explanation = f"Whale Buys: {format_price(large_buys)}, Sells: {format_price(large_sells)}"
        if trade_type == "long" and large_buys > large_sells * 1.5:
            verdict, explanation = "yes", explanation + " (Buy pressure)"
        elif trade_type == "short" and large_sells > large_buys * 1.5:
            verdict, explanation = "yes", explanation + " (Sell pressure)"
        return {"verdict": verdict, "explanation": explanation}
    except Exception as e:
        return {"verdict": "error", "explanation": f"Whale: An error occurred - {e}"}


def support_resistance_verdict(df: pd.DataFrame | None, trade_type: str) -> dict:
    if df is None or len(df) < 50: return {"verdict": "error", "explanation": "S/R: Insufficient data."}
    try:
        recent_data = df.iloc[-50:]
        support = recent_data['low'].min()
        resistance = recent_data['high'].max()
        current_price = df['close'].iloc[-1]
        verdict = "no"
        explanation = f"Support: {format_price(support)}, Resistance: {format_price(resistance)}"
        if trade_type == "long" and current_price <= support * 1.02:
            verdict, explanation = "yes", explanation + " (Near support)"
        elif trade_type == "short" and current_price >= resistance * 0.98:
            verdict, explanation = "yes", explanation + " (Near resistance)"
        return {"verdict": verdict, "explanation": explanation}
    except Exception as e:
        return {"verdict": "error", "explanation": f"S/R: An error occurred - {e}"}


def volume_profile_verdict(df: pd.DataFrame | None, trade_type: str) -> dict:
    if df is None or df.empty: return {"verdict": "error", "explanation": "Vol Profile: No data provided."}
    try:
        # Note: pandas_ta 'vp' can be slow. A dedicated library might be better for production.
        vp = df.ta.vp(width=10)
        if vp is None or vp.empty: return {"verdict": "error", "explanation": "Vol Profile calculation failed."}
        vpoc = vp.iloc[-1, 0]
        current_price = df['close'].iloc[-1]
        verdict = "no"
        explanation = f"Volume POC: {format_price(vpoc)}"
        if pd.isna(vpoc): return {"verdict": "error", "explanation": "Vol Profile POC is NaN."}
        if trade_type == "long" and current_price >= vpoc:
            verdict, explanation = "yes", explanation + " (Above high volume node)"
        elif trade_type == "short" and current_price <= vpoc:
            verdict, explanation = "yes", explanation + " (Below high volume node)"
        return {"verdict": verdict, "explanation": explanation}
    except Exception as e:
        return {"verdict": "error", "explanation": f"Vol Profile: An error occurred - {e}"}


def market_structure_verdict(df: pd.DataFrame | None, trade_type: str) -> dict:
    if df is None or len(df) < 50: return {"verdict": "no", "explanation": "Not enough market structure to analyze."}
    try:
        df_slice = df.tail(50).copy()

        df_slice['swing_high'] = (df_slice['high'] > df_slice['high'].shift(1)) & (
                df_slice['high'] > df_slice['high'].shift(-1))
        df_slice['swing_low'] = (df_slice['low'] < df_slice['low'].shift(1)) & (
                df_slice['low'] < df_slice['low'].shift(-1))

        swing_highs = df_slice[df_slice['swing_high']]['high']
        swing_lows = df_slice[df_slice['swing_low']]['low']

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {"verdict": "no", "explanation": "Not enough market structure to analyze."}

        last_swing_high, prev_swing_high = swing_highs.iloc[-1], swing_highs.iloc[-2]
        last_swing_low, prev_swing_low = swing_lows.iloc[-1], swing_lows.iloc[-2]
        current_price = df_slice['close'].iloc[-1]

        is_bullish_bos = current_price > last_swing_high and last_swing_high > prev_swing_high
        is_bearish_bos = current_price < last_swing_low and last_swing_low < prev_swing_low
        is_bullish_choch = current_price > last_swing_high
        is_bearish_choch = current_price < last_swing_low

        explanation_parts = []
        verdict = "no"

        if trade_type == "long":
            if is_bullish_bos:
                explanation_parts.append("Bullish BoS confirmed."); verdict = "yes"
            elif is_bullish_choch and not is_bearish_bos:
                explanation_parts.append("Potential Bullish ChoCh."); verdict = "yes"
            else:
                explanation_parts.append("No clear bullish structure break.")
        elif trade_type == "short":
            if is_bearish_bos:
                explanation_parts.append("Bearish BoS confirmed."); verdict = "yes"
            elif is_bearish_choch and not is_bullish_bos:
                explanation_parts.append("Potential Bearish ChoCh."); verdict = "yes"
            else:
                explanation_parts.append("No clear bearish structure break.")

        df_slice['fvg_high'] = df_slice['low'].shift(1)
        df_slice['fvg_low'] = df_slice['high'].shift(-1)
        bullish_fvgs = df_slice[(df_slice['high'] < df_slice['fvg_high'])]
        bearish_fvgs = df_slice[(df_slice['low'] > df_slice['fvg_low'])]

        if not bullish_fvgs.empty:
            explanation_parts.append(f"Nearest Bullish FVG (support) at {format_price(bullish_fvgs['high'].iloc[-1])}.")
        if not bearish_fvgs.empty:
            explanation_parts.append(
                f"Nearest Bearish FVG (resistance) at {format_price(bearish_fvgs['low'].iloc[-1])}.")

        return {"verdict": verdict, "explanation": " | ".join(explanation_parts)}
    except Exception as e:
        return {"verdict": "error", "explanation": f"Market Structure: An error occurred - {e}"}


# --- 8 NEW INDICATORS (upgraded) ---

def bollinger_bands_verdict(df, trade_type):
    if df is None or len(df) < 20: return {"verdict": "error", "explanation": "BBands: Insufficient data."}
    try:
        bbands = df.ta.bbands(length=20)
        if bbands is None or bbands.empty: return {"verdict": "error", "explanation": "BBands calculation failed."}
        latest_close = df['close'].iloc[-1]
        lower_band, upper_band = bbands['BBL_20_2.0'].iloc[-1], bbands['BBU_20_2.0'].iloc[-1]
        verdict = "no"
        explanation = f"Lower: {format_price(lower_band)}, Upper: {format_price(upper_band)}"
        if trade_type == "long" and latest_close <= lower_band:
            verdict, explanation = "yes", explanation + " (Price at/below lower band)"
        elif trade_type == "short" and latest_close >= upper_band:
            verdict, explanation = "yes", explanation + " (Price at/above upper band)"
        return {"verdict": verdict, "explanation": explanation}
    except Exception as e:
        return {"verdict": "error", "explanation": f"BBands: An error occurred - {e}"}


def parabolic_sar_verdict(df, trade_type):
    if df is None or df.empty: return {"verdict": "error", "explanation": "PSAR: No data provided."}
    try:
        psar = df.ta.psar()
        if psar is None or psar.empty: return {"verdict": "error", "explanation": "PSAR calculation failed."}
        latest_psar = psar.iloc[-1]['PSARl_0.02_0.2'] if 'PSARl_0.02_0.2' in psar.columns and pd.notna(
            psar.iloc[-1]['PSARl_0.02_0.2']) else psar.iloc[-1]['PSARs_0.02_0.2']
        latest_close = df['close'].iloc[-1]
        verdict = "no"
        explanation = f"PSAR: {format_price(latest_psar)}"
        if trade_type == "long" and latest_close > latest_psar:
            verdict, explanation = "yes", explanation + " (PSAR is below price)"
        elif trade_type == "short" and latest_close < latest_psar:
            verdict, explanation = "yes", explanation + " (PSAR is above price)"
        return {"verdict": verdict, "explanation": explanation}
    except Exception as e:
        return {"verdict": "error", "explanation": f"PSAR: An error occurred - {e}"}


def ichimoku_cloud_verdict(df, trade_type):
    if df is None or df.empty: return {"verdict": "error", "explanation": "Ichimoku: No data provided."}
    try:
        ichimoku = df.ta.ichimoku()[0]
        if ichimoku is None or ichimoku.empty: return {"verdict": "error",
                                                       "explanation": "Ichimoku calculation failed."}
        latest_close = df['close'].iloc[-1]
        span_a, span_b = ichimoku.iloc[-1]['ISA_9'], ichimoku.iloc[-1]['ISB_26']
        verdict = "no"
        explanation = f"Cloud: {format_price(span_a)} - {format_price(span_b)}"
        if trade_type == "long" and latest_close > span_a and latest_close > span_b:
            verdict, explanation = "yes", explanation + " (Price above cloud)"
        elif trade_type == "short" and latest_close < span_a and latest_close < span_b:
            verdict, explanation = "yes", explanation + " (Price below cloud)"
        return {"verdict": verdict, "explanation": explanation}
    except Exception as e:
        return {"verdict": "error", "explanation": f"Ichimoku: An error occurred - {e}"}


def on_balance_volume_verdict(df, trade_type):
    if df is None or df.empty: return {"verdict": "error", "explanation": "OBV: No data provided."}
    try:
        obv = df.ta.obv()
        if obv is None or obv.empty: return {"verdict": "error", "explanation": "OBV calculation failed."}
        df['obv'] = obv
        obv_ema5 = df['obv'].ewm(span=5).mean()
        obv_ema20 = df['obv'].ewm(span=20).mean()
        verdict = "no"
        explanation = "OBV trend is neutral"
        if trade_type == "long" and obv_ema5.iloc[-1] > obv_ema20.iloc[-1]:
            verdict, explanation = "yes", "OBV trend is bullish"
        elif trade_type == "short" and obv_ema5.iloc[-1] < obv_ema20.iloc[-1]:
            verdict, explanation = "yes", "OBV trend is bearish"
        return {"verdict": verdict, "explanation": explanation}
    except Exception as e:
        return {"verdict": "error", "explanation": f"OBV: An error occurred - {e}"}


def macd_histogram_verdict(df, trade_type):
    if df is None or len(df) < 2: return {"verdict": "error", "explanation": "MACD Hist: Insufficient data."}
    try:
        macd_df = df.ta.macd()
        if macd_df is None or macd_df.empty: return {"verdict": "error", "explanation": "MACD Hist calculation failed."}
        hist = macd_df.iloc[-2:]['MACDh_12_26_9'].values
        if len(hist) < 2: return {"verdict": "error", "explanation": "MACD Hist needs at least 2 values."}
        verdict = "no"
        explanation = f"Histogram: {format_price(hist[-1])}"
        if trade_type == "long" and hist[-1] > 0 and hist[-1] > hist[-2]:
            verdict, explanation = "yes", explanation + " (Growing bullish momentum)"
        elif trade_type == "short" and hist[-1] < 0 and hist[-1] < hist[-2]:
            verdict, explanation = "yes", explanation + " (Growing bearish momentum)"
        return {"verdict": verdict, "explanation": explanation}
    except Exception as e:
        return {"verdict": "error", "explanation": f"MACD Hist: An error occurred - {e}"}


# --- FUTURES-SPECIFIC INDICATORS (upgraded) ---

def open_interest_verdict(open_interest_data: float | None, df: pd.DataFrame | None, trade_type: str) -> dict:
    if open_interest_data is None: return {"verdict": "error", "explanation": "Could not fetch Open Interest."}
    if df is None or len(df) < 2: return {"verdict": "error", "explanation": "OI: Insufficient price data."}
    try:
        price_change = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100
        verdict = "no"
        explanation = f"OI: {format_price(open_interest_data)}, Price Change: {price_change:.2f}%"
        if trade_type == "long" and price_change > 0:
            verdict, explanation = "yes", explanation + " (OI confirms bullish price move)"
        elif trade_type == "short" and price_change < 0:
            verdict, explanation = "yes", explanation + " (OI confirms bearish price move)"
        return {"verdict": verdict, "explanation": explanation}
    except Exception as e:
        return {"verdict": "error", "explanation": f"OI: An error occurred - {e}"}


def funding_rate_verdict(funding_rate: float | None, trade_type: str) -> dict:
    if funding_rate is None: return {"verdict": "error", "explanation": "Could not fetch Funding Rate."}
    try:
        fr = funding_rate * 100
        verdict = "no"
        explanation = f"Funding Rate: {fr:.4f}%"
        if trade_type == "long" and fr <= 0.01:
            verdict, explanation = "yes", explanation + " (Neutral/Negative, bullish)"
        elif trade_type == "short" and fr > 0.02:
            verdict, explanation = "yes", explanation + " (High positive, bearish)"
        return {"verdict": verdict, "explanation": explanation}
    except Exception as e:
        return {"verdict": "error", "explanation": f"Funding Rate: An error occurred - {e}"}


def liquidation_levels_verdict(df: pd.DataFrame | None, trade_type: str) -> dict:
    if df is None or len(df) < 10: return {"verdict": "error", "explanation": "Liq Levels: Insufficient data."}
    try:
        recent_low, recent_high = df['low'].iloc[-10:].min(), df['high'].iloc[-10:].max()
        current_price = df['close'].iloc[-1]
        verdict = "no"
        explanation = f"Recent Low: {format_price(recent_low)}, High: {format_price(recent_high)}"
        if trade_type == "long" and (current_price - recent_low) / recent_low < 0.01:
            verdict, explanation = "yes", explanation + " (Near short-term liq zone)"
        elif trade_type == "short" and (recent_high - current_price) / recent_high < 0.01:
            verdict, explanation = "yes", explanation + " (Near short-term liq zone)"
        return {"verdict": verdict, "explanation": explanation}
    except Exception as e:
        return {"verdict": "error", "explanation": f"Liq Levels: An error occurred - {e}"}


def htf_trend_alignment_verdict(symbol: str, timeframe: str, market_type: str, trade_type: str) -> dict:
    timeframe_map = {'1m': '5m', '3m': '15m', '5m': '15m', '15m': '1h', '30m': '1h', '1h': '4h', '2h': '4h', '4h': '1d',
                     '6h': '1d', '8h': '1d', '12h': '1d', '1d': '3d', '3d': '1w', '1w': '1M'}
    htf = timeframe_map.get(timeframe)
    if not htf: return {"verdict": "no", "explanation": f"No HTF mapping for {timeframe}."}

    htf_df = api_client.fetch_ohlc_data(symbol, htf, market_type, limit=200)

    if htf_df is None or len(htf_df) < 55: return {"verdict": "error",
                                                   "explanation": f"Could not fetch data for HTF ({htf})."}

    try:
        ema_fast = htf_df.ta.ema(length=21).iloc[-1]
        ema_slow = htf_df.ta.ema(length=55).iloc[-1]
        htf_is_bullish = ema_fast > ema_slow
        verdict = "no"
        if trade_type == "long" and htf_is_bullish:
            verdict = "yes"
        elif trade_type == "short" and not htf_is_bullish:
            verdict = "yes"
        trend_direction = "Bullish" if htf_is_bullish else "Bearish"
        explanation = f"HTF ({htf}) trend is {trend_direction}. Verdict: {'Aligned' if verdict == 'yes' else 'Not Aligned'}"
        return {"verdict": verdict, "explanation": explanation}
    except Exception as e:
        return {"verdict": "error", "explanation": f"HTF Trend: An error occurred - {e}"}


# --- START OF NEW DIVERGENCE FUNCTIONS (add to the end of indicators.py) ---

def find_divergence(price_series: pd.Series, indicator_series: pd.Series, lookback: int = 40) -> str:
    if price_series is None or indicator_series is None or len(price_series) < lookback: return "None"
    try:
        df = pd.DataFrame({'price': price_series, 'indicator': indicator_series}).tail(lookback)
        if df.isnull().values.any(): return "None"
        lows_price_idx = argrelextrema(df['price'].values, np.less, order=5)[0]
        lows_indicator_idx = argrelextrema(df['indicator'].values, np.less, order=5)[0]
        highs_price_idx = argrelextrema(df['price'].values, np.greater, order=5)[0]
        highs_indicator_idx = argrelextrema(df['indicator'].values, np.greater, order=5)[0]
        if len(lows_price_idx) >= 2 and len(lows_indicator_idx) >= 2:
            if df['price'].iloc[lows_price_idx[-1]] < df['price'].iloc[lows_price_idx[-2]] and df['indicator'].iloc[lows_indicator_idx[-1]] > df['indicator'].iloc[lows_indicator_idx[-2]]: return "Bullish"
        if len(highs_price_idx) >= 2 and len(highs_indicator_idx) >= 2:
            if df['price'].iloc[highs_price_idx[-1]] > df['price'].iloc[highs_price_idx[-2]] and df['indicator'].iloc[highs_indicator_idx[-1]] < df['indicator'].iloc[highs_indicator_idx[-2]]: return "Bearish"
    except Exception: return "None"
    return "None"


def rsi_divergence_verdict(df: pd.DataFrame, trade_type: str) -> dict:
    if df is None or df.empty or len(df) < 50: return {"verdict": "error", "explanation": "RSI Div: No data."}
    try:
        rsi_series = df.ta.rsi()
        if rsi_series is None: return {"verdict": "error", "explanation": "RSI failed."}
        divergence_type = find_divergence(df['close'], rsi_series)
        verdict = "no"
        if (trade_type == "long" and divergence_type == "Bullish") or (trade_type == "short" and divergence_type == "Bearish"): verdict = "yes"
        return {"verdict": verdict, "explanation": f"RSI Divergence: {divergence_type}"}
    except Exception:
        return {"verdict": "error", "explanation": "RSI Div Error"}



def macd_divergence_verdict(df: pd.DataFrame | None, trade_type: str) -> dict:
    """Checks for bullish or bearish divergence on the MACD Line."""
    if df is None or df.empty or len(df) < 50:
        return {"verdict": "error", "explanation": "MACD Div: Not enough data."}
    try:
        macd_df = df.ta.macd()
        if macd_df is None or macd_df.empty:
            return {"verdict": "error", "explanation": "MACD calculation failed."}

        macd_line = macd_df['MACD_12_26_9']
        divergence_type = find_divergence(df['close'], macd_line)

        verdict = "no"
        if (trade_type == "long" and divergence_type == "Bullish") or \
                (trade_type == "short" and divergence_type == "Bearish"):
            verdict = "yes"

        return {"verdict": verdict, "explanation": f"MACD Divergence: {divergence_type}"}
    except Exception as e:
        return {"verdict": "error", "explanation": f"MACD Div Error: {e}"}


# --- END OF NEW DIVERGENCE FUNCTIONS ---


def long_short_ratio_verdict(long_short_ratio: float | None, trade_type: str) -> dict:
    if long_short_ratio is None or long_short_ratio == 0: return {"verdict": "error",
                                                                  "explanation": "Could not fetch L/S Ratio."}
    verdict, explanation = "no", f"L/S Ratio: {long_short_ratio:.2f}"
    if trade_type == "long" and long_short_ratio < 0.75:
        verdict, explanation = "yes", explanation + " (Very bearish, contrarian long)"
    elif trade_type == "short" and long_short_ratio > 2.0:
        verdict, explanation = "yes", explanation + " (Very bullish, contrarian short)"
    return {"verdict": verdict, "explanation": explanation}


def relative_strength_verdict(symbol: str, timeframe: str, market_type: str, trade_type: str) -> dict:
    if 'BTC' in symbol or 'ETH' in symbol or not ('USDT' in symbol or 'BUSD' in symbol): return {"verdict": "no",
                                                                                                 "explanation": "N/A for BTC/ETH."}
    base_asset = symbol.replace("USDT", "").replace("BUSD", "")
    rs_pair = f"{base_asset}BTC"

    # Check availability to avoid unnecessary API calls
    if not api_client.check_coin_availability(rs_pair, "spot"):
        return {"verdict": "error", "explanation": f"Relative strength pair {rs_pair} not available."}

    rs_df = api_client.fetch_ohlc_data(rs_pair, timeframe, "spot", limit=100)
    if rs_df is None or len(rs_df) < 55: return {"verdict": "error",
                                                 "explanation": f"Could not fetch data for {rs_pair}."}

    try:
        ema_fast, ema_slow = rs_df.ta.ema(length=21).iloc[-1], rs_df.ta.ema(length=55).iloc[-1]
        is_strong = ema_fast > ema_slow
        verdict = "no"
        if trade_type == 'long' and is_strong:
            verdict, explanation = "yes", f"{base_asset} is showing strength vs BTC."
        elif trade_type == 'short' and not is_strong:
            verdict, explanation = "yes", f"{base_asset} is showing weakness vs BTC."
        else:
            explanation = f"{base_asset} performance vs BTC is neutral or misaligned."
        return {"verdict": verdict, "explanation": explanation}
    except Exception as e:
        return {"verdict": "error", "explanation": f"RS: An error occurred - {e}"}


# --- REMAINING FUNCTIONS (upgraded) ---

def htf_bias_confirmation_verdict(bias_directions, trade_type):
    if not bias_directions:
        return {"verdict": "no", "explanation": "No HTF bias provided."}
    is_bullish, is_bearish = all(d == "Bullish" for d in bias_directions), all(d == "Bearish" for d in bias_directions)
    verdict, explanation = "no", f"HTF bias ({len(bias_directions)} TFs) is conflicting or misaligned."
    if trade_type == "long" and is_bullish:
        verdict, explanation = "yes", "HTF bias is unanimously bullish."
    elif trade_type == "short" and is_bearish:
        verdict, explanation = "yes", "HTF bias is unanimously bearish."
    return {"verdict": verdict, "explanation": explanation}


def get_htf_target(symbol, timeframe, market_type, trade_type):
    timeframe_map = {'1m': '15m', '3m': '15m', '5m': '1h', '15m': '4h', '30m': '4h', '1h': '4h', '2h': '1d', '4h': '1d',
                     '1d': '1w'}
    htf = timeframe_map.get(timeframe, '1d')
    htf_df = api_client.fetch_ohlc_data(symbol, htf, market_type, limit=200)
    if htf_df is None or htf_df.empty: return None
    try:
        current_price = htf_df['close'].iloc[-1]
        htf_df['is_sw_high'] = (htf_df['high'] > htf_df['high'].shift(1)) & (htf_df['high'] > htf_df['high'].shift(-1))
        htf_df['is_sw_low'] = (htf_df['low'] < htf_df['low'].shift(1)) & (htf_df['low'] < htf_df['low'].shift(-1))
        swing_highs, swing_lows = htf_df[htf_df['is_sw_high']]['high'], htf_df[htf_df['is_sw_low']]['low']
        if trade_type == "long":
            relevant_highs = swing_highs[swing_highs > current_price]
            return relevant_highs.iloc[0] if not relevant_highs.empty else None
        else:
            relevant_lows = swing_lows[swing_lows < current_price]
            return relevant_lows.iloc[-1] if not relevant_lows.empty else None
    except Exception:
        return None

def analyze_smc_v2(df: pd.DataFrame, lookback: int = 50) -> Dict[str, Any]:
    results = {"liquidity_sweep_signal": "None", "liquidity_sweep_signal_index": None, "recent_bullish_fvg": None,
               "recent_bearish_fvg": None}
    if df is None or len(df) < lookback: return results
    try:
        df_slice = df.tail(lookback).copy()
        current_candle = df_slice.iloc[-2] # Check the completed candle
        lookback_high, lookback_low = df_slice['high'].iloc[-21:-2].max(), df_slice['low'].iloc[-21:-2].min()

        if current_candle['low'] < lookback_low: #and current_candle['close'] > lookback_low:
            results['liquidity_sweep_signal'] = 'Bullish Sweep'
            results['liquidity_sweep_signal_index'] = current_candle.name

        if current_candle['high'] > lookback_high: #and current_candle['close'] < lookback_high:
            results['liquidity_sweep_signal'] = 'Bearish Sweep'
            results['liquidity_sweep_signal_index'] = current_candle.name

        for i in range(len(df_slice) - 3, 0, -1):
            c1, c2, c3 = df_slice.iloc[i - 1], df_slice.iloc[i], df_slice.iloc[i + 1]
            if results['recent_bullish_fvg'] is None and c3['high'] < c1['low']:
                 results['recent_bullish_fvg'] = (c3['high'], c1['low'])
            if results['recent_bearish_fvg'] is None and c1['high'] > c3['low']:
                 results['recent_bearish_fvg'] = (c1['high'], c3['low'])

            if results['recent_bullish_fvg'] and results['recent_bearish_fvg']: break
        return results
    except Exception:
        return results
    
# -----------------------------------------------------------------
# --- START OF NEW INSTITUTIONAL FUNCTIONS (Append to indicators.py) ---
# -----------------------------------------------------------------

def find_latest_fvg(df):
    """
    Finds the most recent Bullish and Bearish Fair Value Gaps (FVG).
    An FVG is an imbalance where price moved quickly.
    """
    bullish_fvg = None
    bearish_fvg = None
    # Reverse to look from present to past and drop the old index
    df_rev = df.iloc[::-1].reset_index(drop=True) # <<< THIS IS THE ONLY CHANGE

    for i in range(1, len(df_rev) - 1):
        # Bearish FVG: A gap between the low of candle 1 and the high of candle 3
        if bearish_fvg is None and df_rev.loc[i+1, 'low'] > df_rev.loc[i-1, 'high']:
            bearish_fvg = (df_rev.loc[i-1, 'high'], df_rev.loc[i+1, 'low'])
            
        # Bullish FVG: A gap between the high of candle 1 and the low of candle 3
        if bullish_fvg is None and df_rev.loc[i+1, 'high'] < df_rev.loc[i-1, 'low']:
            bullish_fvg = (df_rev.loc[i-1, 'low'], df_rev.loc[i+1, 'high'])
            
        if bullish_fvg and bearish_fvg:
            break
            
    return {'bullish': bullish_fvg, 'bearish': bearish_fvg}


def find_swing_points(df, order=5):
    """Finds all major swing highs and lows in the dataframe."""
    from scipy.signal import argrelextrema
    
    high_indices = argrelextrema(df['high'].values, np.greater_equal, order=order)[0]
    low_indices = argrelextrema(df['low'].values, np.less_equal, order=order)[0]
    
    return {'highs': df.iloc[high_indices], 'lows': df.iloc[low_indices]}


def analyze_break_of_structure(df, last_liquidity_grab_time, trade_type):
    """
    Checks if a valid Break of Structure (BoS) occurred after a liquidity grab.
    Returns the price where the BoS happened.
    """
    swings = find_swing_points(df.tail(75), order=3)
    candles_after_grab = df[df.index >= last_liquidity_grab_time]
    
    if trade_type == 'short':
        # Find the most recent significant low that was created *before* the liquidity grab high
        relevant_lows = swings['lows'][swings['lows'].index < last_liquidity_grab_time]
        if relevant_lows.empty: return None
        structure_to_break = relevant_lows.iloc[-1]['low']
        
        # Did price break below this low AFTER the grab?
        breakdown_candles = candles_after_grab[candles_after_grab['low'] < structure_to_break]
        if not breakdown_candles.empty:
            return structure_to_break
            
    if trade_type == 'long':
        # Find the most recent significant high that was created *before* the liquidity grab low
        relevant_highs = swings['highs'][swings['highs'].index < last_liquidity_grab_time]
        if relevant_highs.empty: return None
        structure_to_break = relevant_highs.iloc[-1]['high']
        
        # Did price break above this high AFTER the grab?
        breakout_candles = candles_after_grab[candles_after_grab['high'] > structure_to_break]
        if not breakout_candles.empty:
            return structure_to_break
            
    return None

def get_liquidity_grab(df, swings, trade_type):
    """Identifies the most recent sweep of a major swing point."""
    if trade_type == 'short':
        if swings['highs'].empty: return None, None
        last_major_high = swings['highs'].iloc[-1]
        recent_candles = df.tail(25)
        # Check if any recent high has pierced the last major swing high
        swept_candles = recent_candles[recent_candles['high'] > last_major_high['high']]
        if not swept_candles.empty:
            return last_major_high['high'], swept_candles.index[0] # Return the price and time of sweep
            
    if trade_type == 'long':
        if swings['lows'].empty: return None, None
        last_major_low = swings['lows'].iloc[-1]
        recent_candles = df.tail(25)
        # Check if any recent low has pierced the last major swing low
        swept_candles = recent_candles[recent_candles['low'] < last_major_low['low']]
        if not swept_candles.empty:
            return last_major_low['low'], swept_candles.index[0]
            
    return None, None
# --- END OF NEW INSTITUTIONAL FUNCTIONS ---