# api_client.py

import requests
import pandas as pd
import sys
import time

# --- API Configuration ---
BINANCE_SPOT_URL = "https://api.binance.com/api/v3"
BINANCE_FUTURES_URL = "https://fapi.binance.com/fapi/v1"
FNG_API_URL = "https://api.alternative.me/fng/"


def check_coin_availability(symbol, market_type):
    """Check if the trading pair exists on Binance."""
    base_url = BINANCE_FUTURES_URL if market_type == "futures" else BINANCE_SPOT_URL
    url = f"{base_url}/ticker/price?symbol={symbol.upper()}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException as e:
        # We don't need to log this for a simple check, it can be noisy.
        return False
    return False


def fetch_ohlc_data(symbol, interval, market_type, limit=300):
    """
    Fetches OHLCV data from Binance. This version includes robust data cleaning
    and proper error logging.
    """
    base_url = BINANCE_FUTURES_URL if market_type == "futures" else BINANCE_SPOT_URL
    url = f"{base_url}/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
    for attempt in range(3): # Retry mechanism
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()  # Will raise an exception for 4xx/5xx errors
            response_json = response.json()

            if not isinstance(response_json, list) or len(response_json) == 0:
                # print(f"API_CLIENT WARN in fetch_ohlc_data for {symbol}: API returned non-list or empty.", file=sys.stderr)
                return None

            df = pd.DataFrame(response_json, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "close_time",
                "quote_asset_volume", "trades", "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume", "ignore"
            ])
            
            # Use the more reliable 'timestamp' for the index
            df.set_index(pd.to_datetime(df['timestamp'], unit='ms'), inplace=True)

            numeric_cols = ["open", "high", "low", "close", "volume"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df.dropna(subset=numeric_cols, inplace=True)
            
            # Add metadata to the dataframe for later use
            df.name = symbol
            df.attrs['timeframe'] = interval
            df.attrs['market_type'] = market_type
            
            return df if not df.empty else None

        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"API_CLIENT ERROR in fetch_ohlc_data for {symbol} (attempt {attempt+1}): {e}", file=sys.stderr)
            time.sleep(2) # Wait before retrying
    return None