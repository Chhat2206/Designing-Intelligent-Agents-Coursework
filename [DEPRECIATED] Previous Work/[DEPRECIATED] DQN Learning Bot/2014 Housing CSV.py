import pandas as pd

# Redfin's public dataset URL (gzipped TSV updated weekly)
# File size: ~500MB compressed, expands to ~2GB uncompressed
url = "https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_market_tracker/zip_code_market_tracker.tsv000.gz"

# Download and load data (typically takes 3-8 minutes depending on internet speed)
# This makes an HTTP GET request to fetch the compressed file
print("Downloading dataset from Redfin servers...")
df = pd.read_csv(
    url,
    sep='\t',                # TSV format
    compression='gzip',      # Auto-decompress gzip
    low_memory=False         # Prevent mixed-type warnings
)

# Convert date column to datetime format
print("Processing timestamps...")
df['period_begin'] = pd.to_datetime(df['period_begin'])

# Filter for records from 2016 onward (original data starts in 2012)
print("Filtering historical data...")
df_filtered = df[df['period_begin'] >= '2016-01-01']

# Save filtered data to CSV (expect 1-2GB file, takes 1-5 minutes)
print("Saving results...")
df_filtered.to_csv(
    'redfin_data_2016_present.csv',
    index=False,             # Exclude pandas index
    encoding='utf-8-sig'     # For Excel compatibility
)

print(f"Complete! Saved {len(df_filtered):,} records.")