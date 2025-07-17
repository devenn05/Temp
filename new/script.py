import os

# --- Configuration ---
# The name of the output file
OUTPUT_FILENAME = "python_backend_dump.txt"

# The specific list of 9 core backend files to extract.
FILES_TO_EXTRACT = [
    'api_client.py',
    'indicators.py',
    'main.py',
    'backtester.py',
    'requirements.txt',
    'parameters.txt',
    'portfolio.py'
]

# --- Script Logic ---
def extract_backend_files():
    """
    Reads the content of specified backend files and concatenates
    them into a single text file.
    """
    print(f"Starting backend extraction. Output will be saved to '{OUTPUT_FILENAME}'...")
    count = 0
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as outfile:
        for filename in FILES_TO_EXTRACT:
            if not os.path.exists(filename):
                print(f"  [!] Warning: File not found, skipping: {filename}")
                continue

            print(f"  -> Processing: {filename}")
            
            try:
                with open(filename, 'r', encoding='utf-8', errors='ignore') as infile:
                    content = infile.read()

                # Write a clear header for each file
                outfile.write(f"--- START OF FILE {filename} ---\n\n")
                outfile.write(content)
                # Write a clear footer for each file
                outfile.write(f"\n\n--- END OF FILE {filename} ---\n\n\n")
                count += 1
            
            except Exception as e:
                print(f"    [!] Error reading file {filename}: {e}")
                
    print("-" * 50)
    print(f"Extraction complete! Successfully processed {count} out of {len(FILES_TO_EXTRACT)} files.")
    print(f"Backend dump saved as '{OUTPUT_FILENAME}'")

if __name__ == "__main__":
    extract_backend_files()