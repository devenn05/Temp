import pandas as pd
import os

def invert_single_csv(input_csv_path): # Removed output_csv_path argument
    """
    Reads a single CSV file, inverts the order of its rows, and saves to the SAME CSV file.

    Args:
        input_csv_path (str): The full path to the input CSV file.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(input_csv_path)

        # Invert the order of the rows
        inverted_df = df.iloc[::-1].reset_index(drop=True)

        # Save the inverted DataFrame back to the ORIGINAL CSV file
        inverted_df.to_csv(input_csv_path, index=False) # <--- Changed here: saving to input_csv_path
        print(f"Successfully inverted and OVERWRITTEN '{os.path.basename(input_csv_path)}'.")

    except FileNotFoundError:
        print(f"Error: The file '{os.path.basename(input_csv_path)}' was not found.")
    except pd.errors.EmptyDataError:
        print(f"Warning: '{os.path.basename(input_csv_path)}' is empty. Skipping inversion.")
    except Exception as e:
        print(f"An error occurred while processing '{os.path.basename(input_csv_path)}': {e}")

def invert_all_csvs_in_folder_overwrite():
    """
    Searches for all CSV files in the script's directory,
    inverts each one, and OVERWRITES the original file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Searching for CSV files in: {script_dir} to OVERWRITE")
    print("-" * 50)

    found_csv = False
    for filename in os.listdir(script_dir):
        # Exclude the script itself and any previously created '_inverted' files if they exist
        # This prevents accidental processing of files that were meant to be separate backups
        if filename.endswith(".csv") and not filename.endswith("_inverted.csv") and filename != os.path.basename(__file__):
            found_csv = True
            input_path = os.path.join(script_dir, filename)

            # Call the function to invert and overwrite
            invert_single_csv(input_path)
            print("-" * 50)

    if not found_csv:
        print("No CSV files found in the script's directory to invert.")
    else:
        print("\nAll found CSV files processed (original files were overwritten).")


if __name__ == "__main__":
    # IMPORTANT: Running this will OVERWRITE your original CSV files.
    # Make sure you have backups if you need them.
    invert_all_csvs_in_folder_overwrite()