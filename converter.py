import pandas as pd
import os

# --- Configuration ---
# Place the raw, combined CSV files you downloaded from Binance here.
RAW_FILES = {
    '15m': 'raw_15m.csv',
    '1h': 'raw_1h.csv',
    '4h': 'raw_4h.csv'
}

# The folder where the processed, ready-to-use files will be saved.
OUTPUT_FOLDER = 'data'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Created '{OUTPUT_FOLDER}' directory.")

# --- THE FINAL, CORRECTED SCRIPT ---
for timeframe, raw_filename in RAW_FILES.items():
    if not os.path.exists(raw_filename):
        print(f"Error: Raw file '{raw_filename}' not found. Please place it in the project folder.")
        continue

    print(f"Processing '{raw_filename}' for timeframe '{timeframe}'...")

    try:
        # Read the CSV, letting pandas detect the header.
        df = pd.read_csv(raw_filename)

        # Standardize the timestamp column name.
        # This handles cases where the column is named 'open_time' or 'timestamp'.
        df.rename(columns={'open_time': 'timestamp'}, inplace=True)
        if 'timestamp' not in df.columns:
            print(f"Critical Error: 'timestamp' column not found in '{raw_filename}'.")
            continue

        # --- THE CRITICAL FIX IS HERE ---
        # We REMOVE `unit='ms'`. Pandas is now smart enough to figure out
        # the format of your human-readable date string automatically.
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # --- ENSURE DATA IS SORTED BY TIME ---
        # This is a critical step to ensure the backtest works correctly.
        df.sort_values(by='timestamp', inplace=True)

        # Keep only the columns our backtester needs.
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df_processed = df[required_cols]

        # Set the datetime object as the index.
        df_processed.set_index('timestamp', inplace=True)

        output_filename = os.path.join(OUTPUT_FOLDER, f"BTCUSDT-{timeframe}-data.csv")
        df_processed.to_csv(output_filename)

        print(f"Successfully converted and saved {len(df_processed)} rows to '{output_filename}'")

    except Exception as e:
        print(f"\nAn error occurred while processing '{raw_filename}': {e}")
        print("Please ensure the CSV is a standard format with columns like 'timestamp', 'open', 'high', etc.")

print("\nData preparation complete. Your data is now ready.")