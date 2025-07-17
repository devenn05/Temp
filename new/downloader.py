import requests
import pandas as pd
import os
from datetime import datetime
import time
from tqdm import tqdm

# Symbols and timeframes
SYMBOLS = ["BTCUSDT","ETHUSDT","ADAUSDT","SOLUSDT", "XRPUSDT", "BNBUSDT", "UNIUSDT"]
TIMEFRAMES = ['5m', '15m', '30m', '1h', '4h', '1d']
START_DATE = "2021-01-01"
OUTPUT_FOLDER = "data"

# Binance Futures API
BINANCE_API_URL = "https://fapi.binance.com/fapi/v1/klines"

# Column headers
COLUMN_NAMES = [
    "timestamp", "open", "high", "low", "close", "volume", "close_time",
    "quote_asset_volume", "trades", "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume", "ignore"
]

def download_data_for_timeframe(symbol, timeframe, start_date_str):
    print(f"\n--- Starting download for {symbol} on {timeframe} ---")

    all_data = []
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = int(start_dt.timestamp() * 1000)

    pbar = tqdm(desc=f"Downloading {symbol} {timeframe}", unit=" batch")

    while start_ts < end_ts:
        params = {
            'symbol': symbol,
            'interval': timeframe,
            'startTime': start_ts,
            'endTime': end_ts,
            'limit': 1500
        }

        try:
            response = requests.get(BINANCE_API_URL, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            all_data.extend(data)

            last_time = data[-1][0]
            # Binance returns startTime of each candle, so advance by 1 ms after last
            start_ts = last_time + 1

            pbar.update(1)
            time.sleep(0.25)  # Respectful rate limit

        except requests.exceptions.RequestException as e:
            print(f"\nAn error occurred: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            continue

    pbar.close()

    if not all_data:
        print(f"No data found for {symbol} on {timeframe}")
        return

    print(f"Processing data for {symbol}...")

    df = pd.DataFrame(all_data, columns=COLUMN_NAMES)
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.drop_duplicates(subset='timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    filename = os.path.join(OUTPUT_FOLDER, f"{symbol}-{timeframe}-data.csv")
    df.to_csv(filename)
    print(f"Saved {len(df)} rows to {filename}")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created '{OUTPUT_FOLDER}' directory.")

    for symbol in SYMBOLS:
        print(f"\n{'='*25} {symbol} {'='*25}")
        for tf in TIMEFRAMES:
            download_data_for_timeframe(symbol, tf, START_DATE)

    print("\nAll data download complete.")
