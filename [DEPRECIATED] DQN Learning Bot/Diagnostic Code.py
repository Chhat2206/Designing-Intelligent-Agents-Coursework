import pandas as pd

# Load CSV with whitespace handling
df = pd.read_csv('property_listings.csv', skipinitialspace=True)

# Clean column names
df.columns = df.columns.str.strip()

# Verify
print(df.columns.tolist())  # Check for 'yearbuilt'

# Access the column safely
if 'yearBuilt' in df.columns:
    print(df['yearBuilt'])
else:
    print("Column 'yearBuilt' not found.")
