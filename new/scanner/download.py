# data_downloader.py (v2.0 - Robust and Intelligent)

import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import time
from tqdm import tqdm

# --- Configuration ---
DATA_FOLDER = "data"
# We now use the futures endpoint, which is more relevant for these pairs
BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
REQUIRED_TIMEFRAMES = ['15m', '1h', '4h']

def get_top_symbols(limit=50):
    """Fetches the top symbols by 24h trading volume on Binance Futures."""
    tqdm.write("Fetching top symbols by volume from Binance...")
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()

        # Filter for USDT pairs, remove leveraged tokens and expiring futures
        usdt_pairs = [d for d in data if d['symbol'].endswith('USDT') and '_' not in d['symbol']]
        usdt_pairs = [d for d in usdt_pairs if not any(x in d['symbol'] for x in ['UP', 'DOWN', 'BEAR', 'BULL'])]

        sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)
        symbols = [d['symbol'] for d in sorted_pairs]
        tqdm.write(f"Found {len(symbols)} USDT pairs. Taking top {limit}.")
        return symbols[:limit]
    except requests.exceptions.RequestException as e:
        tqdm.write(f"Error fetching symbols: {e}")
        return []

def get_first_listing_timestamp(symbol, interval):
    """Fetches the timestamp of the very first candle for a symbol."""
    try:
        url_params = {'symbol': symbol, 'interval': interval, 'limit': 1, 'startTime': 0}
        res = requests.get(BASE_URL, params=url_params, timeout=10)
        res.raise_for_status()
        data = res.json()
        if data:
            return data[0][0] # Return the timestamp of the first candle
    except requests.exceptions.RequestException:
        return None
    return None

def download_data_for_symbol(params):
    """(Worker Function) Downloads and saves data for one symbol/timeframe combination."""
    symbol, timeframe, desired_start_ts, end_ts = params
    filepath = os.path.join(DATA_FOLDER, f"{symbol}-{timeframe}-data.csv")

    # --- THE CORE FIX: DYNAMIC START DATE ---
    actual_listing_ts = get_first_listing_timestamp(symbol, timeframe)
    if not actual_listing_ts:
        tqdm.write(f"SKIPPING: Could not get listing date for {symbol}-{timeframe}.")
        return None

    # Use the LATER of the two start dates
    start_ts = max(desired_start_ts, actual_listing_ts)
    # --- END OF FIX ---

    all_data = []

    while start_ts < end_ts:
        try:
            url_params = {
                'symbol': symbol, 'interval': timeframe, 'startTime': start_ts, 'limit': 1500
            }
            res = requests.get(BASE_URL, params=url_params, timeout=20)
            res.raise_for_status()
            data = res.json()

            if not data:
                break

            all_data.extend(data)
            start_ts = data[-1][0] + 1
            time.sleep(0.1) # Be respectful

        except requests.exceptions.RequestException as e:
            tqdm.write(f"ERROR downloading {symbol}-{timeframe}: {e}. Skipping.")
            return None

    if not all_data:
        return None

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'trades', 'taker_buy_base_asset_volume',
        'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.to_csv(filepath, index=False)
    # Use tqdm.write for thread-safe printing
    tqdm.write(f"SUCCESS: Saved {len(df)} rows to {filepath} (Data from {df['timestamp'].min().date()})")
    return filepath


def run_downloader():
    """Main function to orchestrate the parallel download."""
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    symbols_to_download = get_top_symbols()
    if not symbols_to_download:
        return

    end_date = datetime.now()
    start_date = end_date - timedelta(days=3 * 365)
    desired_start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)

    print(f"\nConfiguration:")
    print(f"Data folder: '{DATA_FOLDER}'")
    print(f"Desired Start Date: {start_date.strftime('%Y-%m-%d')}")
    
    tasks = []
    for symbol in symbols_to_download:
        for timeframe in REQUIRED_TIMEFRAMES:
            filepath = os.path.join(DATA_FOLDER, f"{symbol}-{timeframe}-data.csv")
            if not os.path.exists(filepath):
                tasks.append((symbol, timeframe, desired_start_ts, end_ts))

    if not tasks:
        print("\nAll required data files seem to be present. No download needed.")
        return

    print(f"\nStarting intelligent parallel download for {len(tasks)} new files...")
    # Wrap the executor map with tqdm for a master progress bar
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(tqdm(executor.map(download_data_for_symbol, tasks), total=len(tasks), desc="Downloading Data"))

    print(f"\n--- Download Complete! ---")
    print("All tasks finished. Check logs above for any errors.")


if __name__ == "__main__":
    run_downloader()