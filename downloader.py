import requests
import pandas as pd
import os
from datetime import datetime
import time
from tqdm import tqdm

# --- Configuration ---
SYMBOL = "BTCUSDT"
TIMEFRAMES = ['15m', '1h', '4h']
START_DATE = "2021-01-01"  # Let's get data starting from 2021
OUTPUT_FOLDER = "data"

# Binance API endpoint for futures klines
BINANCE_API_URL = "https://fapi.binance.com/fapi/v1/klines"

# These are the standard column names from Binance
COLUMN_NAMES = [
    "timestamp", "open", "high", "low", "close", "volume", "close_time",
    "quote_asset_volume", "trades", "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume", "ignore"
]


def download_data_for_timeframe(timeframe, start_date_str):
    """
    Downloads, processes, and saves historical data for a given timeframe.
    """
    print(f"\n--- Starting download for {timeframe} timeframe ---")

    all_data = []
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.now()

    # We will fetch data month by month
    current_start = start_date

    # Create a progress bar
    date_range = pd.date_range(start=current_start, end=end_date, freq='MS')
    pbar = tqdm(total=len(date_range), desc=f'Downloading {timeframe} data')

    while current_start < end_date:
        # Calculate the end of the current month
        current_end = current_start + pd.offsets.MonthEnd(1)
        if current_end > end_date:
            current_end = end_date

        # Convert dates to Unix timestamps in milliseconds
        start_ts = int(current_start.timestamp() * 1000)
        end_ts = int(current_end.timestamp() * 1000)

        params = {
            'symbol': SYMBOL,
            'interval': timeframe,
            'startTime': start_ts,
            'endTime': end_ts,
            'limit': 1500  # Max limit per request
        }

        try:
            # Make the API request
            response = requests.get(BINANCE_API_URL, params=params, timeout=20)
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()
            if data:
                all_data.extend(data)
        except requests.exceptions.RequestException as e:
            print(f"\nAn error occurred: {e}. Skipping this period.")

        pbar.update(1)
        # Move to the next month
        current_start = current_start + pd.DateOffset(months=1)
        # Be respectful of the API rate limit
        time.sleep(0.5)

    pbar.close()

    if not all_data:
        print(f"No data was downloaded for timeframe {timeframe}. Aborting.")
        return

    # --- Process the downloaded data ---
    print(f"Processing downloaded {timeframe} data...")
    df = pd.DataFrame(all_data, columns=COLUMN_NAMES)

    # Keep only the columns we need
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    # Convert data types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove duplicates and set the timestamp as the index
    df.drop_duplicates(subset='timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    # --- Save the final, clean file ---
    output_filename = os.path.join(OUTPUT_FOLDER, f"{SYMBOL}-{timeframe}-data.csv")
    df.to_csv(output_filename)
    print(f"Successfully saved {len(df)} candles to '{output_filename}'")


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created '{OUTPUT_FOLDER}' directory.")

    for tf in TIMEFRAMES:
        download_data_for_timeframe(tf, START_DATE)

    print("\n\nAll data has been downloaded and processed.")
    print("You can now move on to Part 2 and run the backtester.")