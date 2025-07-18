# download_backtest_data.py

import pandas as pd
import api_client # Assumes api_client.py is in the same folder
import backtest_config as config # Uses the backtest specific config
import os
import sys
import time
from datetime import datetime, timedelta

def download_full_asset_history(symbol, timeframe, years=1):
    """
    Downloads a full historical dataset by stitching together multiple
    1000-candle API calls.
    """
    print(f"  Downloading {years} year(s) of {timeframe} data for {symbol}...")
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=years*365)
    
    all_data = []
    
    # Binance API uses milliseconds for timestamps
    end_time_ms = int(end_time.timestamp() * 1000)
    current_start_time_ms = int(start_time.timestamp() * 1000)
    
    pbar = tqdm(total=(end_time - start_time).days, unit=" days")
    
    last_known_timestamp = 0

    while current_start_time_ms < end_time_ms:
        chunk = api_client.fetch_ohlc_data(symbol, timeframe, 'futures', limit=1000, startTime=current_start_time_ms)
        
        if chunk is None or chunk.empty:
            break
        
        last_timestamp_in_chunk = int(chunk['close_time'].iloc[-1])
        if last_timestamp_in_chunk == last_known_timestamp:
            break # Break if we are not getting new data to avoid infinite loops
        last_known_timestamp = last_timestamp_in_chunk
        
        all_data.append(chunk)
        pbar.update((pd.to_datetime(last_known_timestamp, unit='ms') - pd.to_datetime(current_start_time_ms, unit='ms')).days)
        current_start_time_ms = last_timestamp_in_chunk + 1
        
        time.sleep(0.2) # Be kind to the API
        
    pbar.close()

    if not all_data:
        print(f"    > Could not download any data for {symbol}-{timeframe}.")
        return None

    full_df = pd.concat(all_data)
    full_df.drop_duplicates(subset=['timestamp'], inplace=True)
    full_df.sort_values('timestamp', inplace=True)
    return full_df

def run_full_download():
    """
    Orchestrates the download process for all assets defined in the config.
    """
    if not os.path.exists(config.DATA_DIRECTORY):
        os.makedirs(config.DATA_DIRECTORY)
        print(f"Created data directory: '{config.DATA_DIRECTORY}'")
        
    for asset in config.ASSETS_TO_TEST:
        print(f"\n--- Processing Asset: {asset} ---")
        for tf in ['1d', '4h', '1h', '15m']:
            file_path = os.path.join(config.DATA_DIRECTORY, f"{asset}_{tf}.csv")
            if os.path.exists(file_path):
                print(f"  > Data for {asset}-{tf} already exists. Skipping.")
                continue
            
            df = download_full_asset_history(asset, tf, years=1)
            if df is not None:
                df.to_csv(file_path, index=False)
                print(f"  > Saved {asset}-{tf} data to '{file_path}'.")

if __name__ == "__main__":
    run_full_download()
    print("\n--- Backtest data download complete. ---")