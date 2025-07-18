# api_client.py

import requests
import pandas as pd
import sys

# --- API Configuration ---
BINANCE_SPOT_URL = "https://api.binance.com/api/v3"
BINANCE_FUTURES_URL = "https://fapi.binance.com/fapi/v1"
FNG_API_URL = "https://api.alternative.me/fng/"


# === START OF MODIFIED FUNCTIONS ===

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
        # ADDED LOGGING
        print(f"API_CLIENT ERROR in check_coin_availability for {symbol}: {e}", file=sys.stderr)
        error_json = {}
        try:
            error_json = e.response.json()
        except (ValueError, AttributeError):
            pass
        if "Invalid symbol" in error_json.get('msg', ''):
            return False
        return False
    return False


def fetch_ohlc_data(symbol, interval, market_type, limit=200, startTime=None):
    """
    Fetches OHLCV data from Binance.
    This version is UPGRADED to accept an optional 'startTime' for historical fetches.
    """
    base_url = BINANCE_FUTURES_URL if market_type == "futures" else BINANCE_SPOT_URL
    
    # Construct the base URL
    url = f"{base_url}/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
    
    # Add the startTime parameter to the URL ONLY if it is provided
    if startTime:
        url += f"&startTime={startTime}"
        
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        response_json = response.json()

        if not isinstance(response_json, list) or len(response_json) == 0:
            return None

        df = pd.DataFrame(response_json, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "close_time",
            "quote_asset_volume", "trades", "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume", "ignore"
        ])
        numeric_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=numeric_cols, inplace=True)
        return df if not df.empty else None
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"API_CLIENT ERROR in fetch_ohlc_data for {symbol}: {e}", file=sys.stderr)
        return None


def get_current_price(symbol, market_type):
    """Gets the current price for a symbol."""
    base_url = BINANCE_FUTURES_URL if market_type == "futures" else BINANCE_SPOT_URL
    url = f"{base_url}/ticker/price?symbol={symbol.upper()}"
    try:
        response = requests.get(url, timeout=10).json()
        return float(response["price"])
    except Exception as e:
        # ADDED LOGGING
        print(f"API_CLIENT ERROR in get_current_price for {symbol}: {e}", file=sys.stderr)
        return None


def fetch_24hr_stats(symbol, market_type):
    """Fetches 24-hour ticker statistics."""
    base_url = BINANCE_FUTURES_URL if market_type == "futures" else BINANCE_SPOT_URL
    url = f"{base_url}/ticker/24hr?symbol={symbol.upper()}"
    try:
        return requests.get(url, timeout=10).json()
    except Exception as e:
        # ADDED LOGGING
        print(f"API_CLIENT ERROR in fetch_24hr_stats for {symbol}: {e}", file=sys.stderr)
        return None


def fetch_fng_index():
    """Fetches the Fear and Greed Index."""
    try:
        response = requests.get(FNG_API_URL, timeout=10).json()
        return int(response["data"][0]["value"])
    except Exception as e:
        # ADDED LOGGING
        print(f"API_CLIENT ERROR in fetch_fng_index: {e}", file=sys.stderr)
        return None


def fetch_order_book(symbol, market_type):  # <<< Added market_type parameter
    """Fetches the order book."""
    # --- BUG FIX: Use the correct URL for spot vs futures ---
    base_url = BINANCE_FUTURES_URL if market_type == "futures" else BINANCE_SPOT_URL
    url = f"{base_url}/depth?symbol={symbol.upper()}&limit=500"
    try:
        return requests.get(url, timeout=10).json()
    except Exception as e:
        # ADDED LOGGING
        print(f"API_CLIENT ERROR in fetch_order_book for {symbol}: {e}", file=sys.stderr)
        return None


def fetch_open_interest(symbol):
    """Fetches open interest for a futures symbol."""
    url = f"{BINANCE_FUTURES_URL}/openInterest?symbol={symbol.upper()}"
    try:
        response = requests.get(url, timeout=10).json()
        return float(response.get("openInterest", 0))
    except Exception as e:
        # ADDED LOGGING
        print(f"API_CLIENT ERROR in fetch_open_interest for {symbol}: {e}", file=sys.stderr)
        return None


def fetch_funding_rate(symbol):
    """Fetches the latest funding rate for a futures symbol."""
    url = f"{BINANCE_FUTURES_URL}/premiumIndex?symbol={symbol.upper()}"
    try:
        response = requests.get(url, timeout=10).json()
        # Handle cases where a specific symbol might not have a funding rate returned
        if isinstance(response, list) and len(response) > 0:
            return float(response[0].get("lastFundingRate", 0))
        elif isinstance(response, dict):
            return float(response.get("lastFundingRate", 0))
        else:
            print(f"API_CLIENT WARN in fetch_funding_rate for {symbol}: Unexpected response format {response}",
                  file=sys.stderr)
            return 0
    except Exception as e:
        # ADDED LOGGING
        print(f"API_CLIENT ERROR in fetch_funding_rate for {symbol}: {e}", file=sys.stderr)
        return None


def fetch_long_short_ratio(symbol):
    supported_periods = ['1h', '4h', '12h', '1d', '5m', '15m', '30m']
    for period in supported_periods:
        url = f"{BINANCE_FUTURES_URL}/globalLongShortAccountRatio?symbol={symbol.upper()}&period={period}&limit=1"
        try:
            response = requests.get(url, timeout=5).json()
            if response and isinstance(response, list) and 'longShortRatio' in response[0]:
                return float(response[0]["longShortRatio"])
        except (requests.exceptions.RequestException, IndexError, KeyError, ValueError) as e:
            # ADDED LOGGING (but only on the last attempt)
            if period == supported_periods[-1]:
                print(f"API_CLIENT ERROR in fetch_long_short_ratio for {symbol} after trying all periods: {e}",
                      file=sys.stderr)
            continue
    return None